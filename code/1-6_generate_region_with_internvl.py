"""Construct virtual-region pseudo-labels for VRT training.

The default mode is deterministic geometry: it uses ground-truth subject and reference boxes to build the intermediate search region. InternVL is loaded only when optional anchor-phrase alignment is enabled."""

import os
import re
import json
import math
import torch
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel


# ==========================================
# 0. Paths and knobs
# ==========================================
INPUT_JSONL = os.environ.get("INPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1.jsonl")
OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/root/autodl-tmp/SPIN/image")
INTERNVL_MODEL_PATH = os.environ.get("INTERNVL_MODEL_PATH", "OpenGVLab/InternVL3-78B")
RUN_INTERNVL_ENTITY_ALIGNMENT = False  # False: only generate geometric regions.
OPPOSITE_SIDE_PADDING_RATIO = 0.0  # Padding behind anchors along the main direction.


START_INDEX = 0
NUM_SAMPLES = None # None means all
IMAGE_SIZE = 448
MAX_TILES = 12
MAX_NEW_TOKENS = 512
LOAD_MODE = "8bit_single_gpu"  # "8bit_single_gpu", "8bit_multi_gpu", "multi_gpu_bf16", or "single_gpu_bf16"
USE_FLASH_ATTN = False

LOCAL_FILES_ONLY = False

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==========================================
# 1. Prompt: entity alignment only
# ==========================================
def build_prompt(item):
    object_lines = []
    object_format_lines = []
    
    for i, box in enumerate(item.get("gt_anchor_boxes", []) or [], start=1):
        object_lines.append(f"object_{i}: {box}")
        object_format_lines.append(f"object_{i}: [noun phrase from instruction]")

    object_text = "\n".join(object_lines) if object_lines else "none"
    object_format_text = "\n".join(object_format_lines) if object_format_lines else "none"

    return f"""You are an expert remote-sensing spatial annotation assistant.

[Task]
Entity Alignment: Match each provided reference object box (Object boxes) to the exact corresponding noun phrase in the instruction.

[Rules]
- Strict Formatting: You must output your response strictly following the format provided below. Do not output any conversational text.

[Inputs]
Instruction: {item['instruction']}
Object boxes:
{object_text}

[Output Format]
{object_format_text}
"""


# ==========================================
# 2. Core geometry: directional virtual-region generator
# ==========================================
def valid_box(box):
    return isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]

def union_box(boxes):
    valid_boxes = [box for box in boxes if valid_box(box)]
    if not valid_boxes:
        return None
    return [
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    ]

def box_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

def intervals_overlap(a_min, a_max, b_min, b_max):
    return min(a_max, b_max) > max(a_min, b_min)

def generate_directional_region(
    anchor_boxes,
    subject_box,
    img_width,
    img_height,
    padding_ratio=0.15,
    opposite_padding_ratio=OPPOSITE_SIDE_PADDING_RATIO,
):
    """
    Build a region from the subject-anchor union and open it toward the
    subject side relative to the anchors.
    """
    if not subject_box or not valid_box(subject_box):
        return None

    valid_anchors = [box for box in (anchor_boxes or []) if valid_box(box)]
    if not valid_anchors:
        # No anchor is available; fall back to a padded subject box.
        return apply_padding(subject_box, img_width, img_height, padding_ratio)

    anchor_union = union_box(valid_anchors)
    search_union = union_box(valid_anchors + [subject_box])
    if not anchor_union or not search_union:
        return apply_padding(subject_box, img_width, img_height, padding_ratio)

    A_cx, A_cy = box_center(anchor_union)
    S_cx, S_cy = box_center(subject_box)
    X_min, Y_min, X_max, Y_max = search_union

    # A non-overlapping axis directly indicates the spatial direction between anchor and subject.
    x_overlap = intervals_overlap(subject_box[0], subject_box[2], anchor_union[0], anchor_union[2])
    y_overlap = intervals_overlap(subject_box[1], subject_box[3], anchor_union[1], anchor_union[3])

    use_x_direction = not x_overlap
    use_y_direction = not y_overlap

    if not use_x_direction and not use_y_direction:
        # If boxes overlap on both axes, use the dominant center displacement as a stable tie-breaker.
        use_x_direction = abs(S_cx - A_cx) >= abs(S_cy - A_cy)
        use_y_direction = not use_x_direction

    # Open the union region from the anchor side toward the subject side.
    if use_x_direction:
        if S_cx >= A_cx:
            X_max = img_width
        else:
            X_min = 0

    if use_y_direction:
        if S_cy >= A_cy:
            Y_max = img_height
        else:
            Y_min = 0

    # Keep padding asymmetric so the region does not grow behind the anchors.
    left_ratio = padding_ratio
    right_ratio = padding_ratio
    top_ratio = padding_ratio
    bottom_ratio = padding_ratio

    if use_x_direction:
        if S_cx >= A_cx:
            left_ratio = opposite_padding_ratio
        else:
            right_ratio = opposite_padding_ratio

    if use_y_direction:
        if S_cy >= A_cy:
            top_ratio = opposite_padding_ratio
        else:
            bottom_ratio = opposite_padding_ratio

    return apply_side_padding(
        [X_min, Y_min, X_max, Y_max],
        img_width,
        img_height,
        left_ratio=left_ratio,
        right_ratio=right_ratio,
        top_ratio=top_ratio,
        bottom_ratio=bottom_ratio,
    )

def apply_side_padding(
    box,
    img_width,
    img_height,
    left_ratio,
    right_ratio,
    top_ratio,
    bottom_ratio,
):
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]

    final_x_min = max(0, box[0] - box_w * left_ratio)
    final_y_min = max(0, box[1] - box_h * top_ratio)
    final_x_max = min(img_width, box[2] + box_w * right_ratio)
    final_y_max = min(img_height, box[3] + box_h * bottom_ratio)

    if final_x_min >= final_x_max or final_y_min >= final_y_max:
        return box
    return [final_x_min, final_y_min, final_x_max, final_y_max]

def apply_padding(box, img_width, img_height, padding_ratio):
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    final_x_min = max(0, box[0] - pad_x)
    final_y_min = max(0, box[1] - pad_y)
    final_x_max = min(img_width, box[2] + pad_x)
    final_y_max = min(img_height, box[3] + pad_y)
    
    if final_x_min >= final_x_max or final_y_min >= final_y_max:
        return box
    return [final_x_min, final_y_min, final_x_max, final_y_max]


# ==========================================
# 3. Image loading for InternVL
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

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
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
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_from_pil(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


# ==========================================
# 4. Small helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def find_image_path(item):
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]

    image_id = str(item["image_id"])
    roots = [
        IMAGE_ROOT,
        os.path.join(os.path.dirname(IMAGE_ROOT), "image"),
        os.path.join(os.path.dirname(IMAGE_ROOT), "images"),
    ]
    for root in roots:
        for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                path = os.path.join(root, "RSRG_ME_datasets_ori", dataset, "images", image_id + ext)
                if os.path.exists(path):
                    return path
    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


def parse_object_names(text, n_objects):
    names = []
    for i in range(1, n_objects + 1):
        match = re.search(rf"object_{i}\s*:\s*([^\n<]+)", text, re.IGNORECASE)
        names.append(match.group(1).strip() if match else "")
    return names

def box_contains(child, parent):
    if not valid_box(child) or not valid_box(parent):
        return False
    return child[0] >= parent[0] and child[1] >= parent[1] and child[2] <= parent[2] and child[3] <= parent[3]


class TiedWeightsKeysCompat(list):
    def keys(self):
        return list(self)

def patch_transformers_for_internvl():
    current_attr = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(current_attr, property):
        return
    def get_all_tied_weights_keys(self):
        value = self.__dict__.get("_all_tied_weights_keys_compat", getattr(self, "_tied_weights_keys", []) or [])
        if value is None:
            value = []
        if not isinstance(value, dict):
            value = {k: None for k in value}
        return value
    def set_all_tied_weights_keys(self, value):
        self.__dict__["_all_tied_weights_keys_compat"] = value
    PreTrainedModel.all_tied_weights_keys = property(get_all_tied_weights_keys, set_all_tied_weights_keys)

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA device found for InternVL inference.")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)
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
    device_map.update({"vision_model": 0, "mlp1": 0, "language_model.model.tok_embeddings": 0, "language_model.model.embed_tokens": 0, "language_model.output": 0, "language_model.model.norm": 0, "language_model.model.rotary_emb": 0, "language_model.lm_head": 0, f"language_model.model.layers.{num_layers - 1}": 0})
    return device_map

def load_internvl_model():
    common_kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=USE_FLASH_ATTN, trust_remote_code=True, local_files_only=LOCAL_FILES_ONLY)
    if LOAD_MODE == "8bit_single_gpu":
        return AutoModel.from_pretrained(INTERNVL_MODEL_PATH, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0}, **common_kwargs).eval()
    if LOAD_MODE == "8bit_multi_gpu":
        return AutoModel.from_pretrained(INTERNVL_MODEL_PATH, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=split_model(INTERNVL_MODEL_PATH), **common_kwargs).eval()
    if LOAD_MODE == "multi_gpu_bf16":
        return AutoModel.from_pretrained(INTERNVL_MODEL_PATH, device_map=split_model(INTERNVL_MODEL_PATH), **common_kwargs).eval()
    if LOAD_MODE == "single_gpu_bf16":
        return AutoModel.from_pretrained(INTERNVL_MODEL_PATH, **common_kwargs).eval().cuda()
    raise ValueError('Invalid LOAD_MODE')


# ==========================================
# 5. Main generation
# ==========================================
def main():
    data = load_jsonl(INPUT_JSONL)
    if NUM_SAMPLES is None:
        data = data[START_INDEX:]
    else:
        data = data[START_INDEX:START_INDEX + NUM_SAMPLES]

    tokenizer = None
    model = None
    generation_config = None
    if RUN_INTERNVL_ENTITY_ALIGNMENT:
        print(f"Loading InternVL tokenizer: {INTERNVL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            INTERNVL_MODEL_PATH, trust_remote_code=True, use_fast=False, fix_mistral_regex=True, local_files_only=LOCAL_FILES_ONLY
        )

        patch_transformers_for_internvl()
        print(f"Loading InternVL model: {INTERNVL_MODEL_PATH} ({LOAD_MODE})")
        model = load_internvl_model()

        generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    else:
        print("Skipping InternVL entity alignment; only geometric regions will be generated.")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Processing Entities & Generating Regions"):
            image_path = find_image_path(item)
            
            # 1. Load image dimensions for geometric construction.
            pil_img = Image.open(image_path).convert("RGB")
            img_width, img_height = pil_img.size
            
            gt_anchor_boxes = item.get("gt_anchor_boxes", []) or []
            gt_subject_box = item.get("gt_subject_box")

            # 2. Optional entity alignment. Region generation itself is pure geometry.
            if not RUN_INTERNVL_ENTITY_ALIGNMENT or len(gt_anchor_boxes) == 0:
                raw_output = ""
                object_names = []
            else:
                pixel_values = load_image_from_pil(pil_img, input_size=IMAGE_SIZE, max_num=MAX_TILES).to(torch.bfloat16).cuda()
                prompt = "<image>\n" + build_prompt(item).strip()
                raw_output = model.chat(tokenizer, pixel_values, prompt, generation_config)
                object_names = parse_object_names(raw_output, len(gt_anchor_boxes))

            # 3. Generate the virtual region with deterministic geometry.
            region_box = generate_directional_region(
                anchor_boxes=gt_anchor_boxes, 
                subject_box=gt_subject_box, 
                img_width=img_width, 
                img_height=img_height, 
                padding_ratio=0.15
            )

            new_item = dict(item)
            new_item["image_path"] = image_path
            new_item["anchor_object_names"] = object_names
            new_item["pseudo_region_box"] = region_box
            new_item["region_teacher_raw_output"] = raw_output
            new_item["region_contains_subject"] = box_contains(gt_subject_box, region_box) if region_box else False

            out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Saved region-augmented data to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()



# python code/1-6_generate_region_with_internvl.py
