"""Evaluate subject and reference-object grounding accuracy at IoU 0.5."""

import os
import re
import json
from PIL import Image


JSONL_PATH = os.environ.get(
    "JSONL_PATH",
    "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl",
)
GT_JSONL_PATH = os.environ.get(
    "GT_JSONL_PATH",
    "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl",
)
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/root/autodl-tmp/SPIN/image")

MODEL_OUTPUT_COORD_SYSTEM = "pixel"  # SFT/GRPO outputs use original image pixels
IOU_THRESHOLD = 0.5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gt_index(path):
    return {str(item["image_id"]): item for item in load_jsonl(path)}


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


def parse_subject_boxes(text):
    boxes = parse_xml_tag_boxes(text, "subject")
    if boxes:
        return boxes
    return parse_answer_role_boxes(text, "subject")


def parse_object_boxes(text):
    boxes = parse_xml_tag_boxes(text, "object")
    if boxes:
        return boxes
    return parse_answer_role_boxes(text, "object")


def valid_box(box):
    return isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def find_image_path(item):
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]

    image_id = str(item.get("image_id", ""))
    for image_root in [IMAGE_ROOT, os.path.join(os.path.dirname(IMAGE_ROOT), "image")]:
        for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                path = os.path.join(image_root, "RSRG_ME_datasets_ori", dataset, "images", image_id + ext)
                if os.path.exists(path):
                    return path
    return None


def get_image_size(item):
    image_path = find_image_path(item)
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        return image.size


def model_box_to_pixel(box, image_size):
    if MODEL_OUTPUT_COORD_SYSTEM != "norm1000" or image_size is None or not valid_box(box):
        return box
    width, height = image_size
    return [
        box[0] / 1000.0 * width,
        box[1] / 1000.0 * height,
        box[2] / 1000.0 * width,
        box[3] / 1000.0 * height,
    ]


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
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def hit_any(pred_boxes, gt_box):
    return any(compute_iou(pred_box, gt_box) >= IOU_THRESHOLD for pred_box in pred_boxes)


def normalize_pred_boxes(value):
    if valid_box(value):
        return [value]
    if isinstance(value, list):
        return [box for box in value if valid_box(box)]
    return []


def count_one_to_one_hits(pred_boxes, gt_boxes):
    """Maximum one-to-one matches whose IoU reaches the threshold."""
    # One predicted object can satisfy at most one ground-truth reference object.
    adjacency = [
        [
            pred_idx
            for pred_idx, pred_box in enumerate(pred_boxes)
            if compute_iou(pred_box, gt_box) >= IOU_THRESHOLD
        ]
        for gt_box in gt_boxes
    ]
    pred_matches = [-1] * len(pred_boxes)

    def match_gt(gt_idx, visited_preds):
        for pred_idx in adjacency[gt_idx]:
            if pred_idx in visited_preds:
                continue
            visited_preds.add(pred_idx)
            if pred_matches[pred_idx] == -1 or match_gt(pred_matches[pred_idx], visited_preds):
                pred_matches[pred_idx] = gt_idx
                return True
        return False

    return sum(match_gt(gt_idx, set()) for gt_idx in range(len(gt_boxes)))


def main():
    data = load_jsonl(JSONL_PATH)
    gt_index = load_gt_index(GT_JSONL_PATH)
    sub_hit = 0
    sub_total = 0
    obj_sample_score_sum = 0.0
    obj_sample_total = 0

    for item in data:
        image_id = str(item["image_id"])
        gt_item = gt_index[image_id]
        raw_output = item.get("raw_output", "")
        image_size = get_image_size(item)

        pred_subjects = normalize_pred_boxes(item.get("pred_subject_box"))
        if not pred_subjects:
            pred_subjects = normalize_pred_boxes(item.get("pred_subject_boxes"))
        if not pred_subjects:
            pred_subjects = parse_subject_boxes(raw_output)
        pred_subjects = [
            model_box_to_pixel(box, image_size)
            for box in pred_subjects
        ]
        gt_subject = gt_item.get("gt_subject_box")
        if valid_box(gt_subject):
            sub_total += 1
            if hit_any(pred_subjects, gt_subject):
                sub_hit += 1

        pred_objects = normalize_pred_boxes(item.get("pred_object_boxes"))
        if not pred_objects:
            pred_objects = parse_object_boxes(raw_output)
        pred_objects = [
            model_box_to_pixel(box, image_size)
            for box in pred_objects
        ]
        gt_objects = [
            box
            for box in (gt_item.get("gt_anchor_boxes", []) or [])
            if valid_box(box)
        ]
        if gt_objects:
            matched_objects = count_one_to_one_hits(pred_objects, gt_objects)
            obj_sample_score_sum += matched_objects / len(gt_objects)
            obj_sample_total += 1

    acc_sub = sub_hit / sub_total if sub_total else 0.0
    acc_obj = obj_sample_score_sum / obj_sample_total if obj_sample_total else 0.0
    macc = (acc_sub + acc_obj) / 2.0

    print("=" * 50)
    print(f"JSONL: {JSONL_PATH}")
    print(f"GT JSONL: {GT_JSONL_PATH}")
    print(f"Acc@0.5_sub: {acc_sub:.4f} ({sub_hit}/{sub_total})")
    print(
        f"Acc@0.5_obj (sample mean): {acc_obj:.4f} "
        f"({obj_sample_score_sum:.2f}/{obj_sample_total} samples)"
    )
    print(f"mAcc@0.5: {macc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()




# python code/eval/evaluate_grounding.py
