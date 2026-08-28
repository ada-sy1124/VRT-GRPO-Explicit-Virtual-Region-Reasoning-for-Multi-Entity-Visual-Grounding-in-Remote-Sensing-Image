"""Attach resolved image paths to JSONL records and drop samples whose image file is missing."""

import os
import json
from tqdm import tqdm


INPUT_JSONL = os.environ.get("INPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/train_cleaned_stage1.jsonl")
OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1.jsonl")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/autodl-tmp/SPIN/image")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_image_index():
    image_index = {}
    duplicate_count = 0

    for root, _, files in os.walk(IMAGE_DIR):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue

            path = os.path.join(root, name)
            if stem in image_index:
                duplicate_count += 1
                continue
            image_index[stem] = path

    print(f"Image scan completed: {len(image_index)} unique image ids, {duplicate_count} duplicate file stems.")
    if not image_index:
        raise RuntimeError(f"No image files found under IMAGE_DIR: {IMAGE_DIR}")
    return image_index


def main():
    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")

    # Build one global image-id index first; this avoids repeated directory scans per sample.
    image_index = build_image_index()
    kept = 0
    dropped = 0

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    tmp_output = OUTPUT_JSONL + ".tmp"

    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f, open(tmp_output, "w", encoding="utf-8") as out_f:
        for line in tqdm(in_f, desc="filtering"):
            if not line.strip():
                continue

            item = json.loads(line)
            image_id = str(item["image_id"])
            image_path = image_index.get(image_id)

            if image_path is None:
                dropped += 1
                continue

            item["image_path"] = image_path
            item["image_relpath"] = os.path.relpath(image_path, IMAGE_DIR)
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    if kept == 0:
        os.remove(tmp_output)
        raise RuntimeError("No samples were kept after filtering; aborted to avoid writing an empty dataset.")

    os.replace(tmp_output, OUTPUT_JSONL)

    print("=" * 50)
    print(f"input: {INPUT_JSONL}")
    print(f"output: {OUTPUT_JSONL}")
    print(f"kept samples: {kept}")
    print(f"dropped samples without images: {dropped}")
    print("=" * 50)


if __name__ == "__main__":
    main()


# python code/data/filter_existing_images.py
