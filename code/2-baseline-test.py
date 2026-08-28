"""Legacy entity-aware baseline inference script.

This file is kept for experiment traceability; the organized evaluation entry points live under code/eval/."""

import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


DATA_PATH = "/root/autodl-tmp/SPIN/output/vrt_eval_test.jsonl"
IMAGE_ROOT = "/root/autodl-tmp/SPIN/images"
BASE_MODEL_PATH = "/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master"
OUTPUT_JSONL = "/root/autodl-tmp/SPIN/output_InternVL3_5/baseline-test.jsonl"

START_INDEX = 10
NUM_SAMPLES = None  # Set None to run the full set.
MAX_NEW_TOKENS = 1024

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


PROMPT_TEMPLATE = """Localize the subject and objects in the image with description: {instruction}. Generate a step-by-step reasoning process in <think></think> tags and output the box coordinates of one subject and one or more objects in the format of <answer>subject: [x1, y1, x2, y2], object: [x1, y1, x2, y2], ...</answer>"""


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


def build_messages(instruction):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction.replace("<image>", "").strip())
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
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
    ).to(model.device)

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


def main():
    data = load_jsonl(DATA_PATH)
    if NUM_SAMPLES is None:
        data = data[START_INDEX:]
    else:
        data = data[START_INDEX:START_INDEX + NUM_SAMPLES]

    print(f"Loading processor: {BASE_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Loading model: {BASE_MODEL_PATH}")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for offset, item in enumerate(tqdm(data, desc="baseline testing")):
            image_id = str(item["image_id"])
            image_path = find_image_path(item)
            image = Image.open(image_path).convert("RGB")

            raw_output = generate_one(
                model=model,
                processor=processor,
                image=image,
                instruction=item["instruction"],
            )

            result = {
                "index": START_INDEX + offset,
                "image_id": image_id,
                "image_path": image_path,
                "instruction": item["instruction"],
                "raw_output": raw_output,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Saved baseline outputs to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()


# python code/2-baseline-test.py
