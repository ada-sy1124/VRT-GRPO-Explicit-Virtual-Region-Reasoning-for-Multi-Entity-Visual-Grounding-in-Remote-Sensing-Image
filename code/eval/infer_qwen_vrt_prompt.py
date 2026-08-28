"""Run prompt-only Qwen VRT inference without LoRA adapters."""

import os
import re
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ==========================================
# 0. Paths and knobs
# ==========================================
DATA_PATH = os.environ.get("DATA_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/autodl-tmp/SPIN/image")

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct")

OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_prompt.jsonl")

START_INDEX = 0
NUM_SAMPLES = None  # None means all
MAX_NEW_TOKENS = 2048
TORCH_DTYPE = torch.bfloat16

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==========================================
# 1. Prompt
# ==========================================
# PROMPT_TEMPLATE = """Localize objects in the image based on the following description: {instruction}

# Output exactly three XML blocks: <plan>, <think>, and <answer>.
# In <think>, mark boxes as <object> [x1, y1, x2, y2], <region> [x1, y1, x2, y2], and <subject> [x1, y1, x2, y2].
# In <answer>, summarize the same boxes as object: [x1, y1, x2, y2], region: [x1, y1, x2, y2], subject: [x1, y1, x2, y2].
# """


PROMPT_TEMPLATE = """Localize objects in the image based on the following description: {instruction}

[Global Structure Framework]
You must output your response using exactly this structure:
1. <plan>: Concisely generate a step-by-step reasoning plan based on the description.
2. <think>: Follow the plan to detail your search process. Whenever an object, region, or subject is identified, tag it naturally in your sentences like <object> [x1, y1, x2, y2] or <region> [x1, y1, x2, y2] or <subject> [x1, y1, x2, y2].
3. <answer>: Summarize the box coordinates from the think block.

[Hard Rules]
1. Concept Distinction: Use "object" for existing reference entities, "region" for deduced virtual spaces, and "subject" for the target.
2. Answer Consistency: The box coordinates within the final <answer> must be exactly identical to those in the <think> block.
3. You MUST ALWAYS have at least one bounding box for <subject>.
"""

# ==========================================
# 2. Helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    # Each distributed rank writes a shard; rank 0 restores original sample order.
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


def generate_one(model, processor, image, instruction):
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

    if rank == 0:
        print(f"World size: {world_size}; total samples: {len(indexed_data)}")
        print(f"Loading processor: {BASE_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    if rank == 0:
        print(f"Loading model: {BASE_MODEL_PATH}")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
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
            image = Image.open(image_path).convert("RGB")

            output_text = generate_one(
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


# python code/eval/infer_qwen_vrt_prompt.py
