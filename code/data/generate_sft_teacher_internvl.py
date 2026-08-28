"""Generate placeholder-based VRT teacher trajectories with InternVL3-78B.

The teacher is asked to produce plan/think/answer text with symbolic placeholders only; numeric coordinates are injected later from ground truth."""

import os
import re
import json
import math
import torch
import random
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel


# ==========================================
# 0. Paths and knobs
# ==========================================
INPUT_JSONL = os.environ.get("INPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl")
OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/teacher_reasoning.jsonl")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/root/autodl-tmp/SPIN/image")
INTERNVL_MODEL_PATH = os.environ.get("INTERNVL_MODEL_PATH", "OpenGVLab/InternVL3-78B")

NUM_SAMPLES = 2149  # SFT warmup subset size; set None for all
IMAGE_SIZE = 448
MAX_TILES = 12
MAX_NEW_TOKENS = 512
LOAD_MODE = "8bit_single_gpu"  # "8bit_single_gpu", "8bit_multi_gpu", "multi_gpu_bf16", or "single_gpu_bf16"
USE_FLASH_ATTN = True
LOCAL_FILES_ONLY = True

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "1"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep backend operations deterministic where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Apply the global seed on import.
set_global_seed(42)


# ==========================================
# 1. Prompt
# ==========================================
def build_prompt(item):
    n_objects = len(item.get("gt_anchor_boxes", []) or [])
    object_placeholders = ", ".join(f"OBJECT_{i}" for i in range(1, n_objects + 1))
    answer_objects = ", ".join(f"object: OBJECT_{i}" for i in range(1, n_objects + 1))

    return f"""
Localize objects in the image based on the following description: {item["instruction"].replace("<image>", "").strip()}

[Global Structure Framework]
You must output your response using exactly this structure:
1. <plan>: Concisely generate a step-by-step reasoning plan based on the description.
2. <think>: Follow the plan to detail your search process. Whenever an object, region, or subject is identified, tag it naturally in your sentences like <object> OBJECT_1 or <region> REGION or <subject> SUBJECT.
3. <answer>: Summarize the placeholders from the think block.

[Allowed Placeholders]
Objects: {object_placeholders}
Region: REGION
Subject: SUBJECT

[Hard Rules]
1. Concept Distinction: Use "object" for existing reference entities, "region" for deduced virtual spaces, and "subject" for the target.
2. Answer Consistency: The placeholders within the final <answer> must be exactly identical to those in the <think> block.
3. You MUST ALWAYS have one placeholder for <subject>.
4. Do NOT output numeric coordinates. Use placeholders only.
5. Do NOT invent extra placeholders.
6. Do NOT output JSON, Markdown, or extra explanations outside the three required parts.

[Perfect Example] (You must strictly imitate this format!)
User: Localize objects in the image based on the following description: a basketball court on the right of a lake.
Assistant:
<plan>
First, locate the lake as the reference object. Then, deduce the search region to the right of the lake. Finally, find the basketball court within this region.
</plan>
<think>
I should first look for the lake in the image. I found the reference lake <object> OBJECT_1. Next, based on the description, the target is to the right of this lake, so I define the search area <region> REGION. Within this region, I lock onto the target basketball court <subject> SUBJECT.
</think>
<answer>
{answer_objects}, region: REGION, subject: SUBJECT
</answer>
"""


# ==========================================
# 2. Image loading for InternVL
# ==========================================
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # Match InternVL dynamic tiling so teacher inference sees the same image layout as the model expects.
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    grid_width = target_width // image_size
    for i in range(blocks):
        box = (
            (i % grid_width) * image_size,
            (i // grid_width) * image_size,
            ((i % grid_width) + 1) * image_size,
            ((i // grid_width) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ==========================================
# 3. Helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_image_path(item):
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]

    if item.get("image_relpath"):
        for root in [IMAGE_ROOT, os.path.join(os.path.dirname(IMAGE_ROOT), "images")]:
            path = os.path.join(root, item["image_relpath"])
            if os.path.exists(path):
                return path

    image_id = str(item["image_id"])
    for root in [IMAGE_ROOT, os.path.join(os.path.dirname(IMAGE_ROOT), "images")]:
        for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                path = os.path.join(root, "RSRG_ME_datasets_ori", dataset, "images", image_id + ext)
                if os.path.exists(path):
                    return path
    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


def valid_box(box):
    return isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def valid_item(item):
    if not item.get("gt_anchor_boxes"):
        return False
    if not valid_box(item.get("gt_subject_box")):
        return False
    if not valid_box(item.get("pseudo_region_box")):
        return False
    return all(valid_box(box) for box in item.get("gt_anchor_boxes", []))


def normalize_output(text):
    text = text.strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return text.strip()


def validate_teacher_output(text, n_objects):
    if not re.search(r"<plan>.*?</plan>", text, re.IGNORECASE | re.DOTALL):
        return False
    if not re.search(r"<think>.*?</think>", text, re.IGNORECASE | re.DOTALL):
        return False
    if not re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL):
        return False
    # Numeric coordinates are intentionally forbidden; GT boxes are injected in the next stage.
    if re.search(r"\[[\d\.\,\s]+\]", text):
        return False

    required = [f"OBJECT_{i}" for i in range(1, n_objects + 1)]
    required.extend(["REGION", "SUBJECT"])
    return all(tok in text for tok in required)


# Compatibility patch: expose tied-weight keys through a dict-like interface.
class TiedWeightsKeysCompat(dict):
    def __init__(self, keys_list):
        # Store a dict-like value so .items() and .keys() remain available.
        super().__init__({k: k for k in keys_list})
        self._keys_list = keys_list

    def __iter__(self):
        return iter(self._keys_list)


def patch_transformers_for_internvl():
    current_attr = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(current_attr, property):
        return

    def get_all_tied_weights_keys(self):
        value = getattr(self, "_all_tied_weights_keys_compat", None)
        if value is not None:
            return value
            
        old_value = getattr(self, "_tied_weights_keys", [])
        if old_value is None:
            old_value = []
            
        if isinstance(old_value, dict):
            return old_value
            
        compat_value = TiedWeightsKeysCompat(old_value)
        setattr(self, "_all_tied_weights_keys_compat", compat_value)
        return compat_value

    def set_all_tied_weights_keys(self, value):
        if not isinstance(value, dict) and value is not None:
            value = TiedWeightsKeysCompat(value)
        self.__dict__["_all_tied_weights_keys_compat"] = value

    PreTrainedModel.all_tied_weights_keys = property(
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA device found for InternVL inference.")

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    num_layers = config.llm_config.num_hidden_layers

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for gpu_id, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_id
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_internvl_model():
    common_kwargs = dict(
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=USE_FLASH_ATTN,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )

    if LOAD_MODE == "8bit_single_gpu":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModel.from_pretrained(
            INTERNVL_MODEL_PATH,
            quantization_config=quant_config,
            device_map={"": 0},
            **common_kwargs,
        ).eval()

    if LOAD_MODE == "8bit_multi_gpu":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModel.from_pretrained(
            INTERNVL_MODEL_PATH,
            quantization_config=quant_config,
            device_map=split_model(INTERNVL_MODEL_PATH),
            **common_kwargs,
        ).eval()

    if LOAD_MODE == "multi_gpu_bf16":
        return AutoModel.from_pretrained(
            INTERNVL_MODEL_PATH,
            device_map=split_model(INTERNVL_MODEL_PATH),
            **common_kwargs,
        ).eval()

    if LOAD_MODE == "single_gpu_bf16":
        return AutoModel.from_pretrained(
            INTERNVL_MODEL_PATH,
            **common_kwargs,
        ).eval().cuda()

    raise ValueError('LOAD_MODE must be "8bit_single_gpu", "8bit_multi_gpu", "multi_gpu_bf16", or "single_gpu_bf16"')


def main():
    data = load_jsonl(INPUT_JSONL)
    
    # ==========================================
    # Random sampling for the SFT warmup subset.
    # ==========================================
    if NUM_SAMPLES is not None:
        # Keep the subset deterministic for reproducibility.
        random.seed(42) 
        
        # Do not sample more rows than the dataset contains.
        actual_samples = min(NUM_SAMPLES, len(data))
        
        print(f"Random sampling mode: selecting {actual_samples} rows from {len(data)} total rows...")
        data = random.sample(data, actual_samples)
    else:
        print(f"Full-data mode: using all {len(data)} rows...")
    # ==========================================

    print(f"Loading InternVL tokenizer: {INTERNVL_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        INTERNVL_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
        local_files_only=LOCAL_FILES_ONLY,
    )

    patch_transformers_for_internvl()
    print(f"Loading InternVL model: {INTERNVL_MODEL_PATH} ({LOAD_MODE})")
    model = load_internvl_model()

    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    saved = 0
    skipped = 0
    invalid = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="generating sft teacher reasoning"):
            if not valid_item(item):
                skipped += 1
                continue

            try:
                image_path = find_image_path(item)
                pixel_values = load_image(image_path, input_size=IMAGE_SIZE, max_num=MAX_TILES)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

                prompt = "<image>\n" + build_prompt(item).strip()
                raw_output = model.chat(tokenizer, pixel_values, prompt, generation_config)
                raw_output = normalize_output(raw_output)

                n_objects = len(item.get("gt_anchor_boxes", []) or [])
                is_valid = validate_teacher_output(raw_output, n_objects)
                invalid += 0 if is_valid else 1

                new_item = dict(item)
                new_item["image_path"] = image_path
                new_item["sft_teacher_raw_output"] = raw_output
                new_item["sft_teacher_valid"] = is_valid
                out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                out_f.flush()
                saved += 1

            except Exception as e:
                skipped += 1
                print(f"skip {item.get('image_id')}: {e}")

    print("=" * 50)
    print(f"input: {INPUT_JSONL}")
    print(f"output: {OUTPUT_JSONL}")
    print(f"saved: {saved}")
    print(f"skipped: {skipped}")
    print(f"invalid teacher outputs: {invalid}")
    print("=" * 50)


if __name__ == "__main__":
    main()


# python code/data/generate_sft_teacher_internvl.py
