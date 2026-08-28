"""Evaluate generated virtual regions for validity, subject coverage, direction alignment, and optional IoU against pseudo regions."""

import json
import os
import re


# ==========================================
# 0. Paths and knobs
# ==========================================
# Fill these paths manually.
# JSONL_PATHS can contain one file or several files.
JSONL_PATHS = [
    "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_prompt.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_sft.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_qwen2_5/vrt_test_prompt.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_qwen2_5/vrt_test_sft.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_qwen2_5/vrt_test_grpo.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_test_prompt.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_test_sft.jsonl",
    "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_test_grpo.jsonl",
]

# Test set with gt_subject_box and gt_anchor_boxes.
GT_JSONL_PATH = "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl"

# Full set with pseudo_region_box. The script will use test image_id + instruction
# to find the corresponding pseudo_region_box from this file.
REGION_GT_JSONL_PATH = "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl"

REGION_IOU_THRESHOLD = float(os.environ.get("REGION_IOU_THRESHOLD", "0.5"))
SUBJECT_COVER_THRESHOLD = float(os.environ.get("SUBJECT_COVER_THRESHOLD", "0.5"))
STRICT_SUBJECT_COVER_THRESHOLD = float(os.environ.get("STRICT_SUBJECT_COVER_THRESHOLD", "0.8"))
DIRECTION_ALIGNMENT_THRESHOLD = float(os.environ.get("DIRECTION_ALIGNMENT_THRESHOLD", "1.0"))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def item_image_id(item):
    value = item.get("image_id")
    if value:
        return str(value)
    for key in ["image_path", "image_relpath"]:
        value = item.get(key)
        if value:
            return os.path.splitext(os.path.basename(str(value)))[0]
    images = item.get("images") or []
    if images:
        return os.path.splitext(os.path.basename(str(images[0])))[0]
    return ""


def item_instruction(item):
    instruction = item.get("instruction")
    if instruction:
        return str(instruction)

    for message in item.get("messages", []) or []:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return " ".join(texts)
    return ""


def normalize_instruction(text):
    text = str(text or "").replace("<image>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".").strip().lower()


def load_index(path):
    # Keep both strict pair lookup and image-only fallback for datasets with repeated image ids.
    if not path:
        return {"by_pair": {}, "by_id": {}, "path": ""}
    if not os.path.exists(path):
        return {"by_pair": {}, "by_id": {}, "path": path}

    by_pair = {}
    by_id = {}
    for item in load_jsonl(path):
        image_id = item_image_id(item)
        if not image_id:
            continue
        instruction = normalize_instruction(item_instruction(item))
        if instruction:
            by_pair.setdefault((image_id, instruction), []).append(item)
        by_id.setdefault(image_id, []).append(item)
    return {"by_pair": by_pair, "by_id": by_id, "path": path}


def valid_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def normalize_boxes(value):
    if valid_box(value):
        return [value]
    if isinstance(value, list):
        return [box for box in value if valid_box(box)]
    return []


def parse_boxes_from_pattern(text, pattern):
    boxes = []
    for match in re.finditer(pattern, text or "", re.IGNORECASE | re.DOTALL):
        values = [
            float(value)
            for value in re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                match.group(1),
            )
        ]
        if len(values) == 4 and valid_box(values):
            boxes.append(values)
    return boxes


def parse_xml_tag_boxes(text, tag):
    return parse_boxes_from_pattern(
        text,
        rf"<{tag}(?:_[^>]*)?>\s*:?\s*\[([^\]]+)\]",
    )


def parse_answer_role_boxes(text, role):
    answer_match = re.search(r"<answer>(.*?)</answer>", text or "", re.IGNORECASE | re.DOTALL)
    answer_text = answer_match.group(1) if answer_match else text
    return parse_boxes_from_pattern(
        answer_text,
        rf"{role}\s*:\s*(?:[^\[\n]*?)\[([^\]]+)\]",
    )


def parse_region_boxes(text):
    boxes = parse_xml_tag_boxes(text, "region")
    if boxes:
        return boxes
    return parse_answer_role_boxes(text, "region")


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


def compute_ioc(child_box, parent_box):
    if not valid_box(child_box) or not valid_box(parent_box):
        return 0.0
    x1 = max(child_box[0], parent_box[0])
    y1 = max(child_box[1], parent_box[1])
    x2 = min(child_box[2], parent_box[2])
    y2 = min(child_box[3], parent_box[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    child_area = max((child_box[2] - child_box[0]) * (child_box[3] - child_box[1]), 1e-6)
    return inter / child_area


def box_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def relation_dimensions(instruction):
    text = str(instruction).lower()
    dimensions = []
    if re.search(r"\b(left|right|east|west)\b", text):
        dimensions.append("horizontal")
    if re.search(r"\b(above|below|upper|lower|top|bottom|over|under|north|south)\b", text):
        dimensions.append("vertical")
    return dimensions


def relation_alignment(region_box, anchor_box, subject_box, dimensions):
    # A region is directionally aligned when its center lies on the same side of the anchor as the subject.
    region_cx, region_cy = box_center(region_box)
    anchor_cx, anchor_cy = box_center(anchor_box)
    subject_cx, subject_cy = box_center(subject_box)
    comparisons = []

    if "horizontal" in dimensions and abs(subject_cx - anchor_cx) > 1e-6:
        comparisons.append((region_cx - anchor_cx) * (subject_cx - anchor_cx) > 0)
    if "vertical" in dimensions and abs(subject_cy - anchor_cy) > 1e-6:
        comparisons.append((region_cy - anchor_cy) * (subject_cy - anchor_cy) > 0)

    if not comparisons:
        return None
    return sum(comparisons) / len(comparisons)


def boxes_close(box1, box2, tol=1e-3):
    return (
        valid_box(box1)
        and valid_box(box2)
        and all(abs(float(a) - float(b)) <= tol for a, b in zip(box1, box2))
    )


def box_lists_close(boxes1, boxes2):
    boxes1 = normalize_boxes(boxes1)
    boxes2 = normalize_boxes(boxes2)
    return len(boxes1) == len(boxes2) and all(
        boxes_close(box1, box2)
        for box1, box2 in zip(boxes1, boxes2)
    )


def pick_best_lookup_match(item, candidates):
    if not candidates:
        return {}
    if len(candidates) == 1:
        return candidates[0]

    gt_subject = item.get("gt_subject_box")
    gt_anchors = item.get("gt_anchor_boxes")
    for candidate in candidates:
        if boxes_close(gt_subject, candidate.get("gt_subject_box")) and box_lists_close(
            gt_anchors,
            candidate.get("gt_anchor_boxes"),
        ):
            return candidate
    return candidates[0]


def lookup_item(item, index):
    image_id = item_image_id(item)
    instruction = normalize_instruction(item_instruction(item))
    if image_id and instruction:
        candidates = index.get("by_pair", {}).get((image_id, instruction), [])
        if candidates:
            return pick_best_lookup_match(item, candidates)
    if image_id:
        candidates = index.get("by_id", {}).get(image_id, [])
        if candidates:
            return pick_best_lookup_match(item, candidates)
    return {}


def get_gt_item(item, gt_index):
    return lookup_item(item, gt_index)


def first_valid_box(*values):
    for value in values:
        if valid_box(value):
            return value
    return None


def valid_boxes_from_item(item, gt_item, key):
    boxes = normalize_boxes(item.get(key))
    if boxes:
        return boxes
    return normalize_boxes(gt_item.get(key))


def region_gt_box(item, gt_item, region_gt_index):
    region_item = lookup_item(item, region_gt_index)
    return first_valid_box(
        item.get("pseudo_region_box"),
        item.get("sft_region_box"),
        item.get("gt_region_box"),
        gt_item.get("pseudo_region_box"),
        gt_item.get("sft_region_box"),
        gt_item.get("gt_region_box"),
        region_item.get("pseudo_region_box"),
        region_item.get("sft_region_box"),
        region_item.get("gt_region_box"),
    )


def evaluate_file(path, gt_index, region_gt_index):
    data = load_jsonl(path)
    total = 0
    has_region = 0
    exactly_one_region = 0

    cover_total = 0
    cover_hit = 0
    strict_cover_hit = 0
    cover_sum = 0.0

    direction_total = 0
    direction_hit = 0
    direction_score_sum = 0.0

    region_iou_total = 0
    region_iou_hit = 0
    region_iou_sum = 0.0
    region_lookup_miss = 0

    combined_total = 0
    combined_hit = 0

    for item in data:
        total += 1
        gt_item = get_gt_item(item, gt_index)
        raw_output = item.get("raw_output", "")
        pred_regions = normalize_boxes(item.get("pred_region_boxes"))
        if not pred_regions:
            pred_regions = parse_region_boxes(raw_output)
        if pred_regions:
            has_region += 1
        if len(pred_regions) == 1:
            exactly_one_region += 1
        pred_region = pred_regions[-1] if pred_regions else None
        if not valid_box(pred_region):
            continue

        gt_subject = first_valid_box(item.get("gt_subject_box"), gt_item.get("gt_subject_box"))
        if valid_box(gt_subject):
            # Coverage metrics are computed over samples with a valid predicted region and valid subject box.
            cover_total += 1
            coverage = compute_ioc(gt_subject, pred_region)
            cover_sum += coverage
            if coverage >= SUBJECT_COVER_THRESHOLD:
                cover_hit += 1
            if coverage >= STRICT_SUBJECT_COVER_THRESHOLD:
                strict_cover_hit += 1

        gt_region = region_gt_box(item, gt_item, region_gt_index)
        if valid_box(gt_region):
            region_iou_total += 1
            region_iou = compute_iou(pred_region, gt_region)
            region_iou_sum += region_iou
            if region_iou >= REGION_IOU_THRESHOLD:
                region_iou_hit += 1
        elif region_gt_index.get("by_id"):
            region_lookup_miss += 1

        gt_anchors = valid_boxes_from_item(item, gt_item, "gt_anchor_boxes")
        dimensions = relation_dimensions(item.get("instruction", gt_item.get("instruction", "")))
        alignment_scores = [
            relation_alignment(pred_region, anchor, gt_subject, dimensions)
            for anchor in gt_anchors
            if valid_box(gt_subject)
        ]
        alignment_scores = [score for score in alignment_scores if score is not None]
        if alignment_scores:
            direction_total += 1
            direction_score = sum(alignment_scores) / len(alignment_scores)
            direction_score_sum += direction_score
            if direction_score >= DIRECTION_ALIGNMENT_THRESHOLD:
                direction_hit += 1

            if valid_box(gt_subject):
                combined_total += 1
                coverage = compute_ioc(gt_subject, pred_region)
                if (
                    coverage >= SUBJECT_COVER_THRESHOLD
                    and direction_score >= DIRECTION_ALIGNMENT_THRESHOLD
                ):
                    combined_hit += 1

    def ratio(hit, denom):
        return hit / denom if denom else 0.0

    print("=" * 70)
    print(f"JSONL: {path}")
    print(f"Samples: {total}")
    print(f"Region valid rate: {ratio(has_region, total):.4f} ({has_region}/{total})")
    print(f"Exactly-one-region rate: {ratio(exactly_one_region, total):.4f} ({exactly_one_region}/{total})")
    print(
        f"Region subject-cover@{SUBJECT_COVER_THRESHOLD:g}: "
        f"{ratio(cover_hit, cover_total):.4f} ({cover_hit}/{cover_total}); "
        f"mean IoC={ratio(cover_sum, cover_total):.4f}"
    )
    print(
        f"Region subject-cover@{STRICT_SUBJECT_COVER_THRESHOLD:g}: "
        f"{ratio(strict_cover_hit, cover_total):.4f} ({strict_cover_hit}/{cover_total})"
    )
    print(
        f"Region direction acc@{DIRECTION_ALIGNMENT_THRESHOLD:g}: "
        f"{ratio(direction_hit, direction_total):.4f} ({direction_hit}/{direction_total}); "
        f"mean direction score={ratio(direction_score_sum, direction_total):.4f}"
    )
    print(
        f"Region cover+direction acc: "
        f"{ratio(combined_hit, combined_total):.4f} ({combined_hit}/{combined_total})"
    )
    if region_iou_total:
        print(
            f"Region IoU@{REGION_IOU_THRESHOLD:g} vs pseudo/GT region: "
            f"{ratio(region_iou_hit, region_iou_total):.4f} ({region_iou_hit}/{region_iou_total}); "
            f"mean IoU={ratio(region_iou_sum, region_iou_total):.4f}"
        )
    else:
        print("Region IoU vs pseudo/GT region: unavailable (no matching pseudo_region_box found)")
    if region_lookup_miss:
        print(f"Region GT lookup misses among valid predicted regions: {region_lookup_miss}")


def main():
    gt_index = load_index(GT_JSONL_PATH)
    region_gt_index = load_index(REGION_GT_JSONL_PATH)
    paths = [path for path in JSONL_PATHS if path and os.path.exists(path)]
    if not paths:
        raise FileNotFoundError("No JSONL files found. Please fill JSONL_PATHS at the top.")

    if gt_index.get("path"):
        print(f"GT JSONL: {gt_index['path']}")
    if region_gt_index.get("path"):
        print(f"Region GT JSONL: {region_gt_index['path']}")
    for path in paths:
        evaluate_file(path, gt_index, region_gt_index)


if __name__ == "__main__":
    main()


# Evaluate all default local outputs:
# python code/eval/evaluate_region.py
