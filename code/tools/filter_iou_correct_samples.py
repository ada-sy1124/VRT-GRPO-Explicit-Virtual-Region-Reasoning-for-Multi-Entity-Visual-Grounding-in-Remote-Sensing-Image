"""Filter prediction JSONL files to cases where subject and all reference objects are correct at an IoU threshold."""

import argparse
import json
import os


DEFAULT_INPUT_JSONL = "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl"
DEFAULT_OUTPUT_JSONL = "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo_subject_object_correct.jsonl"
DEFAULT_IOU_THRESHOLD = 0.5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def valid_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def normalize_boxes(value):
    if valid_box(value):
        return [value]
    if isinstance(value, list):
        return [box for box in value if valid_box(box)]
    return []


def compute_iou(box1, box2):
    if not valid_box(box1) or not valid_box(box2):
        return 0.0

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
    area2 = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
    return inter / (area1 + area2 - inter)


def count_one_to_one_hits(pred_boxes, gt_boxes, threshold):
    # Build a bipartite graph and solve maximum matching so duplicate predictions are not double-counted.
    adjacency = [
        [
            pred_idx
            for pred_idx, pred_box in enumerate(pred_boxes)
            if compute_iou(pred_box, gt_box) >= threshold
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


def best_subject_iou(item):
    pred_subjects = normalize_boxes(item.get("pred_subject_box"))
    if not pred_subjects:
        pred_subjects = normalize_boxes(item.get("pred_subject_boxes"))

    gt_subject = item.get("gt_subject_box")
    if not valid_box(gt_subject) or not pred_subjects:
        return 0.0
    return max(compute_iou(pred_box, gt_subject) for pred_box in pred_subjects)


def object_match_score(item, threshold):
    pred_objects = normalize_boxes(item.get("pred_object_boxes"))
    gt_objects = normalize_boxes(item.get("gt_anchor_boxes"))
    if not pred_objects or not gt_objects:
        return 0, len(gt_objects)
    return count_one_to_one_hits(pred_objects, gt_objects, threshold), len(gt_objects)


def keep_item(item, threshold):
    subject_iou = best_subject_iou(item)
    object_hits, object_total = object_match_score(item, threshold)
    return subject_iou >= threshold and object_total > 0 and object_hits == object_total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter JSONL samples whose subject and all reference objects have IoU >= threshold."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL, help="Input JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU threshold.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_jsonl(args.input)
    kept = [item for item in data if keep_item(item, args.threshold)]
    save_jsonl(args.output, kept)

    print("=" * 60)
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"IoU threshold: {args.threshold}")
    print(f"total samples: {len(data)}")
    print(f"kept samples: {len(kept)}")
    print(f"kept ratio: {len(kept) / len(data):.4f}" if data else "kept ratio: 0.0000")
    print("=" * 60)


if __name__ == "__main__":
    main()


# Example:
# python code/tools/filter_iou_correct_samples.py \
#   --input /root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl \
#   --output /root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo_correct.jsonl
