"""Build the chat-format SFT dataset from placeholder teacher trajectories.

Teacher text is validated first, then OBJECT/REGION/SUBJECT placeholders are replaced by deterministic ground-truth boxes to avoid coordinate drift in the SFT targets."""

import os
import re
import json
from collections import Counter
from tqdm import tqdm


# ==========================================
# 0. Paths and knobs
# ==========================================

INPUT_JSONL = os.environ.get("INPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/teacher_reasoning.jsonl")
OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_sft_train.jsonl")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/root/autodl-tmp/SPIN/image")

START_INDEX = 0
NUM_SAMPLES = None  # None means all


SYSTEM_PROMPT = """Localize objects in the image based on the following description: {instruction}

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
# 1. Helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def fmt_box(box):
    vals = []
    for x in box:
        x = float(x)
        vals.append(str(int(x)) if x.is_integer() else f"{x:.2f}".rstrip("0").rstrip("."))
    return "[" + ", ".join(vals) + "]"


def valid_box(box):
    return isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def validate_teacher_output(text, n_objects):
    if not isinstance(text, str) or not text.strip():
        return False
    if "<plan>" not in text or "</plan>" not in text:
        return False
    if "<think>" not in text or "</think>" not in text:
        return False
    if "<answer>" not in text or "</answer>" not in text:
        return False

    # Require exactly the placeholders that correspond to the available reference objects.
    object_ids = {int(x) for x in re.findall(r"OBJECT_(\d+)", text)}
    expected_ids = set(range(1, n_objects + 1))
    if object_ids != expected_ids:
        return False

    return "REGION" in text and "SUBJECT" in text


def build_assistant_output(item):
    objects = item.get("gt_anchor_boxes", []) or []
    region_box = item.get("pseudo_region_box")
    subject_box = item.get("gt_subject_box")
    teacher = item.get("sft_teacher_raw_output", "")

    if not objects:
        raise ValueError("missing gt_anchor_boxes")
    if not item.get("region_contains_subject", False):
        raise ValueError("region does not contain subject")
    if not item.get("sft_teacher_valid", False):
        raise ValueError("invalid teacher output flag")
    if not validate_teacher_output(teacher, len(objects)):
        raise ValueError("invalid teacher output format")
    if not valid_box(region_box):
        raise ValueError("missing valid pseudo_region_box")
    if not valid_box(subject_box):
        raise ValueError("missing valid gt_subject_box")
    for box in objects:
        if not valid_box(box):
            raise ValueError("missing valid gt_anchor_boxes")

    assistant = teacher.strip()
    # Inject numeric boxes only after validation so the SFT target remains deterministic.
    for i in range(len(objects), 0, -1):
        assistant = assistant.replace(f"OBJECT_{i}", fmt_box(objects[i - 1]))
    assistant = assistant.replace("REGION", fmt_box(region_box))
    assistant = assistant.replace("SUBJECT", fmt_box(subject_box))

    if re.search(r"OBJECT_\d+|REGION|SUBJECT", assistant):
        raise ValueError("unreplaced placeholder remains")

    return assistant

def build_user_prompt(instruction):
    clean_instruction = instruction.replace("<image>", "").strip()
    return SYSTEM_PROMPT.format(instruction=clean_instruction)



def build_sft_item(item):
    image_path = find_image_path(item)
    assistant = build_assistant_output(item)

    return {
        "image_id": str(item["image_id"]),
        "image_path": image_path,
        "images": [image_path],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_user_prompt(item["instruction"])},
                ],
            },
            {"role": "assistant", "content": assistant},
        ],
        "gt_subject_box": item.get("gt_subject_box"),
        "gt_anchor_boxes": item.get("gt_anchor_boxes", []),
        "sft_region_box": item.get("pseudo_region_box"),
        "sft_teacher_valid": item.get("sft_teacher_valid", False),
    }


# ==========================================
# 2. Main
# ==========================================
def main():
    data = load_jsonl(INPUT_JSONL)
    if NUM_SAMPLES is None:
        data = data[START_INDEX:]
    else:
        data = data[START_INDEX:START_INDEX + NUM_SAMPLES]

    output = []
    skipped = 0
    for item in tqdm(data, desc="building sft data"):
        try:
            output.append(build_sft_item(item))
        except Exception as e:
            skipped += 1
            print(f"skip {item.get('image_id')}: {e}")

    save_jsonl(OUTPUT_JSONL, output)
    print("=" * 50)
    print(f"input: {INPUT_JSONL}")
    print(f"output: {OUTPUT_JSONL}")
    print(f"saved: {len(output)}")
    print(f"skipped: {skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()


# python code/data/build_vrt_sft_dataset.py
