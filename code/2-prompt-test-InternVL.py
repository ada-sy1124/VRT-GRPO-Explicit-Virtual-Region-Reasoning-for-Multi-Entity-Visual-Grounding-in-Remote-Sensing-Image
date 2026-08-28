"""Legacy InternVL prompt-only VRT inference script.

It is retained for reproducing earlier exploratory runs and should not be treated as the main README entry point."""

import os
import re
import json
import glob
import torch
import torch.distributed as dist
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


# ==========================================
# 0. Paths and knobs
# ==========================================
DATA_PATH = "/root/autodl-tmp/SPIN/vrt_eval_experiments/vrt_eval_test.jsonl"
IMAGE_DIR = "/root/autodl-tmp/SPIN/image"

BASE_MODEL_PATH = "/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master"

OUTPUT_JSONL = "/root/autodl-tmp/SPIN/output_InternVL3_5/vrt_test_baseline.jsonl"

START_INDEX = 0
NUM_SAMPLES = None  # None means all
MAX_NEW_TOKENS = 2048
TORCH_DTYPE = torch.bfloat16
IMAGE_SIZE = 448
MAX_TILES = 12

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==========================================
# 1. Prompt
# ==========================================

PROMPT_TEMPLATE = """Localize the subject and objects in the image with description: {instruction}. Generate a step-by-step reasoning process in <think></think> tags and output the box coordinates of one subject and one or more objects in the format of <answer>subject: [x1, y1, x2, y2], object: [x1, y1, x2, y2], ...</answer>"""



# PROMPT_TEMPLATE = """Localize objects in the image based on the following description: {instruction}

# [Global Structure Framework]
# You must output your response using exactly this structure:
# 1. <plan>: Concisely generate a step-by-step reasoning plan based on the description.
# 2. <think>: Follow the plan to detail your search process. Whenever an object, region, or subject is identified, tag it naturally in your sentences like <object> [x1, y1, x2, y2] or <region> [x1, y1, x2, y2] or <subject> [x1, y1, x2, y2].
# 3. <answer>: Summarize the box coordinates from the think block.

# [Hard Rules]
# 1. Concept Distinction: Use "object" for existing reference entities, "region" for deduced virtual spaces, and "subject" for the target.
# 2. Answer Consistency: The box coordinates within the final <answer> must be exactly identical to those in the <think> block.
# 3. You MUST ALWAYS have at least one bounding box for <subject>.
# """

# ==========================================
# 2. Helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_pretrained_path(path):
    path = os.path.expanduser(path)
    if os.path.exists(os.path.join(path, "config.json")):
        return path

    candidates = []
    for config_path in glob.glob(os.path.join(path, "**", "config.json"), recursive=True):
        model_dir = os.path.dirname(config_path)
        has_model_files = any(
            os.path.exists(os.path.join(model_dir, name))
            for name in [
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
            ]
        )
        if has_model_files:
            candidates.append((os.path.getmtime(config_path), model_dir))

    if candidates:
        return sorted(candidates)[-1][1]
    return path


def is_internvl_model(path):
    if "internvl" in path.lower():
        return True

    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return False

    values = [
        str(config.get("model_type", "")),
        str(config.get("architectures", "")),
        str(config.get("_name_or_path", "")),
    ]
    return any("internvl" in value.lower() for value in values)


def patch_transformers_for_internvl():
    def normalize_tied_keys(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, (list, tuple, set)):
            return {str(key): None for key in value}
        if hasattr(value, "keys"):
            return value
        return {}

    def get_all_tied_weights_keys(self):
        value = self.__dict__.get("_all_tied_weights_keys_compat", None)
        if value is None:
            value = getattr(self, "_tied_weights_keys", None)
        return normalize_tied_keys(value)

    def set_all_tied_weights_keys(self, value):
        self.__dict__["_all_tied_weights_keys_compat"] = normalize_tied_keys(value)

    current_attr = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if not isinstance(current_attr, property) or current_attr.fset is None:
        PreTrainedModel.all_tied_weights_keys = property(
            get_all_tied_weights_keys,
            set_all_tied_weights_keys,
        )


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRT evaluation")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return world_size, rank, local_rank


def merge_rank_outputs(output_path, world_size):
    results = []
    for rank in range(world_size):
        results.extend(load_jsonl(f"{output_path}.rank{rank}.tmp"))
    results.sort(key=lambda item: item["index"])

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for rank in range(world_size):
        os.remove(f"{output_path}.rank{rank}.tmp")


def find_image_path(item):
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]

    if item.get("images"):
        image_path = item["images"][0]
        if os.path.exists(image_path):
            return image_path

    if item.get("image_relpath"):
        path = os.path.join(IMAGE_DIR, item["image_relpath"])
        if os.path.exists(path):
            return path

    image_id = str(item["image_id"])
    base_dir = os.path.join(IMAGE_DIR, "RSRG_ME_datasets_ori")
    for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            path = os.path.join(base_dir, dataset, "images", image_id + ext)
            if os.path.exists(path):
                return path

    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


def valid_box(box):
    return isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def parse_boxes_from_pattern(text, pattern):
    boxes = []
    for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
        try:
            box = [float(x.strip()) for x in match.group(1).split(",")]
            if valid_box(box):
                boxes.append(box)
        except Exception:
            pass
    return boxes


def parse_xml_tag_boxes(text, tag):
    return parse_boxes_from_pattern(text, rf"<{tag}>\s*:?\s*\[([\d\.\,\s]+)\]")


def parse_answer_role_boxes(text, role):
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    answer_text = answer_match.group(1) if answer_match else text
    return parse_boxes_from_pattern(
        answer_text,
        rf"{role}\s*:\s*(?:[^\[\n]*?)\[([\d\.\,\s]+)\]",
    )


def parse_role_boxes(text, role):
    boxes = parse_xml_tag_boxes(text, role)
    if boxes:
        return boxes
    return parse_answer_role_boxes(text, role)


def compute_iou(box1, box2):
    if not valid_box(box1) or not valid_box(box2):
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1e-6)
    area2 = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1e-6)

    return inter / (area1 + area2 - inter)


def build_messages(instruction):
    clean_instruction = instruction.replace("<image>", "").strip()
    prompt = PROMPT_TEMPLATE.format(instruction=clean_instruction)

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


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
    for i in range(blocks):
        cols = target_width // image_size
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def load_internvl_tokenizer(model_path):
    kwargs = dict(
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_internvl_model(model_path, local_rank):
    patch_transformers_for_internvl()

    common_kwargs = dict(
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
    )
    try:
        return AutoModel.from_pretrained(
            model_path,
            dtype=TORCH_DTYPE,
            **common_kwargs,
        ).eval()
    except TypeError:
        return AutoModel.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE,
            **common_kwargs,
        ).eval()


def generate_one_qwen(model, processor, image, instruction):
    messages = build_messages(instruction)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_one_internvl(model, tokenizer, image_path, instruction):
    clean_instruction = instruction.replace("<image>", "").strip()
    prompt = "<image>\n" + PROMPT_TEMPLATE.format(instruction=clean_instruction).strip()

    pixel_values = load_internvl_image(
        image_path,
        input_size=IMAGE_SIZE,
        max_num=MAX_TILES,
    )
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device=device, dtype=TORCH_DTYPE)

    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
    }
    return model.chat(tokenizer, pixel_values, prompt, generation_config).strip()


# ==========================================
# 3. Main
# ==========================================
def main():
    world_size, rank, local_rank = init_distributed()
    data = load_jsonl(DATA_PATH)

    if NUM_SAMPLES is None:
        selected_data = data[START_INDEX:]
    else:
        selected_data = data[START_INDEX:START_INDEX + NUM_SAMPLES]
    indexed_data = list(enumerate(selected_data, start=START_INDEX))
    rank_data = indexed_data[rank::world_size]

    model_path = resolve_pretrained_path(BASE_MODEL_PATH)
    backend = "internvl" if is_internvl_model(model_path) else "qwen"
    if rank == 0:
        print(f"World size: {world_size}; total samples: {len(indexed_data)}")
        if model_path != BASE_MODEL_PATH:
            print(f"Resolved model path: {model_path}")
        print(f"Backend: {backend}")

    if backend == "internvl":
        if rank == 0:
            print(f"Loading InternVL tokenizer: {model_path}")
        processor = load_internvl_tokenizer(model_path)

        if rank == 0:
            print(f"Loading InternVL model: {model_path}")
        model = load_internvl_model(model_path, local_rank)
    else:
        if rank == 0:
            print(f"Loading processor: {model_path}")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if rank == 0:
            print(f"Loading model: {model_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE,
            device_map={"": local_rank},
            trust_remote_code=True,
            local_files_only=True,
        )

    model.eval()

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    shard_path = f"{OUTPUT_JSONL}.rank{rank}.tmp"

    sub_hit = 0
    sub_total = 0

    with open(shard_path, "w", encoding="utf-8") as out_f:
        for index, item in tqdm(rank_data, desc=f"testing rank {rank}", disable=rank != 0):
            image_id = str(item["image_id"])
            image_path = find_image_path(item)
            if backend == "internvl":
                output_text = generate_one_internvl(
                    model=model,
                    tokenizer=processor,
                    image_path=image_path,
                    instruction=item["instruction"],
                )
            else:
                image = Image.open(image_path).convert("RGB")
                output_text = generate_one_qwen(
                    model=model,
                    processor=processor,
                    image=image,
                    instruction=item["instruction"],
                )

            pred_subjects = parse_role_boxes(output_text, "subject")
            pred_objects = parse_role_boxes(output_text, "object")
            pred_regions = parse_role_boxes(output_text, "region")

            pred_subject_box = pred_subjects[-1] if pred_subjects else None
            gt_subject_box = item.get("gt_subject_box")
            iou = compute_iou(pred_subject_box, gt_subject_box)

            if valid_box(gt_subject_box):
                sub_total += 1
                if iou >= 0.5:
                    sub_hit += 1

            result = {
                "index": index,
                "image_id": image_id,
                "image_path": image_path,
                "instruction": item["instruction"],
                "raw_output": output_text,
                "pred_subject_box": pred_subject_box,
                "pred_object_boxes": pred_objects,
                "pred_region_boxes": pred_regions,
                "gt_subject_box": gt_subject_box,
                "gt_anchor_boxes": item.get("gt_anchor_boxes", []),
                "subject_iou": iou,
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    stats = torch.tensor([sub_hit, sub_total], dtype=torch.long, device=f"cuda:{local_rank}")
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.barrier()
    total_sub_hit, total_sub_total = stats.cpu().tolist()

    if rank == 0:
        merge_rank_outputs(OUTPUT_JSONL, world_size)
        acc_sub = total_sub_hit / total_sub_total if total_sub_total else 0.0
        print("=" * 60)
        print(f"Saved test outputs to: {OUTPUT_JSONL}")
        print(f"Samples: {len(indexed_data)}")
        print(f"Subject Acc@0.5: {acc_sub:.4f} ({total_sub_hit}/{total_sub_total})")
        print("=" * 60)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()



# python ./2-prompt-test-InternVL.py


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./2-prompt-test-InternVL.py

