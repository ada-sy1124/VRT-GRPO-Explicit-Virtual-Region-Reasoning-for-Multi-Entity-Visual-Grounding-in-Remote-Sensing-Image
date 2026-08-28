"""Extract ME-RSRG raw annotation files from the dataset archive into compact JSONL files.

The resulting records keep only the fields used by this project: image id, instruction, subject box, and reference-object boxes."""

import argparse
import json
import os
import re
import zipfile

from tqdm import tqdm


DEFAULT_ZIP_PATH = "/root/autodl-tmp/SPIN/ME-RSRG_datasets.zip"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/SPIN/ME-RSRG-Data"
DEFAULT_SPLITS = ["train", "test"]


def clean_instruction(text):
    text = str(text or "")
    match = re.search(r"description:\s*(.*?)\.\s*Generate", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    text = text.replace("Localize the subject and objects in the image with description:", "")
    return text.split("Generate")[0].strip()


def extract_image_id(item, fallback):
    images = item.get("images") or []
    if images:
        return os.path.splitext(os.path.basename(images[0]))[0]
    return str(fallback)


def extract_prompt(item):
    for message in item.get("messages", []) or []:
        if message.get("role") == "user":
            return clean_instruction(message.get("content", ""))
    return clean_instruction(item.get("instruction", ""))


def extract_boxes(item):
    gt_subject_box = None
    gt_anchor_boxes = []
    objects = item.get("objects", {})

    # ME-RSRG stores boxes as parallel ref/bbox arrays; "subject" is the target and others are anchors.
    refs = objects.get("ref", [])
    bboxes = objects.get("bbox", [])
    for ref_name, bbox in zip(refs, bboxes):
        if ref_name == "subject":
            gt_subject_box = bbox
        else:
            gt_anchor_boxes.append(bbox)

    return gt_subject_box, gt_anchor_boxes


def convert_split(zip_file, split, output_dir):
    member = f"RSRG_ME_datasets_ori/{split}.json"
    output_path = os.path.join(output_dir, f"{split}_cleaned_stage1.jsonl")
    converted = 0
    skipped = 0

    with zipfile.ZipFile(zip_file, "r") as archive:
        with archive.open(member) as file:
            lines = file.readlines()

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for row_index, line in enumerate(tqdm(lines, desc=f"converting {split}")):
            if not line.strip():
                continue
            try:
                item = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                skipped += 1
                continue

            gt_subject_box, gt_anchor_boxes = extract_boxes(item)
            if not gt_subject_box:
                skipped += 1
                continue

            output_item = {
                "image_id": extract_image_id(item, row_index),
                "instruction": extract_prompt(item),
                "gt_subject_box": gt_subject_box,
                "gt_anchor_boxes": gt_anchor_boxes,
            }
            out_file.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            converted += 1

    print(f"{split}: saved {converted} samples to {output_path}; skipped {skipped}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ME-RSRG raw JSON files into project JSONL files.")
    parser.add_argument("--zip-path", default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    return parser.parse_args()


def main():
    args = parse_args()
    for split in args.splits:
        convert_split(args.zip_path, split, args.output_dir)


if __name__ == "__main__":
    main()
