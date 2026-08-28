"""Visualize subject, reference-object, and virtual-region boxes for qualitative analysis."""

import os
import re
import json
import cv2


# ==========================================
# 0. Paths and knobs
# ==========================================
JSONL_PATH = os.environ.get("JSONL_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl")
DASHED_REGION_JSONL_PATH = os.environ.get("DASHED_REGION_JSONL_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/center_pass_regions.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/autodl-tmp/SPIN/figures")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/root/autodl-tmp/SPIN/image")

def env_bool(name, default):
    return os.environ.get(name, "1" if default else "0") == "1"


# Draw samples by image_id. This is the stable key across different JSONL files.
# Pass comma-separated ids through IMAGE_IDS, for example: IMAGE_IDS=12147,12220.
IMAGE_IDS = [
    image_id.strip()
    for image_id in os.environ.get("IMAGE_IDS", "").split(",")
    if image_id.strip()
]

# Used only when IMAGE_IDS is empty.
START_SAMPLE_NUMBER = int(os.environ.get("START_SAMPLE_NUMBER", "0"))
NUM_SAMPLES_TEXT = os.environ.get("NUM_SAMPLES", "1").strip()
NUM_SAMPLES = int(NUM_SAMPLES_TEXT) if NUM_SAMPLES_TEXT else None
MODEL_OUTPUT_COORD_SYSTEM = os.environ.get("MODEL_OUTPUT_COORD_SYSTEM", "pixel")  # "pixel" or "norm1000"

# Choose which box types to draw.
# subject = final target, object = reference/anchor, region = virtual region.
DRAW_SUBJECT = env_bool("DRAW_SUBJECT", True)
DRAW_OBJECT = env_bool("DRAW_OBJECT", True)
DRAW_REGION = env_bool("DRAW_REGION", True)
DRAW_DASHED_REGION = env_bool("DRAW_DASHED_REGION", False)

IMAGE_ROOT_CANDIDATES = [
    IMAGE_ROOT,
    os.path.join(os.path.dirname(IMAGE_ROOT), "image"),
    os.path.join(os.path.dirname(IMAGE_ROOT), "images"),
]

# OpenCV uses BGR color.
MODEL_OBJECT_COLOR = (0, 255, 255)    # yellow
MODEL_REGION_COLOR = (0, 255, 0)      # green
MODEL_SUBJECT_COLOR = (0, 0, 255)     # red

GT_SUBJECT_COLOR = (0, 0, 255)        # red
GT_ANCHOR_COLOR = (0, 255, 255)       # yellow
GT_REGION_COLOR = (0, 255, 0)         # green
DASHED_REGION_COLOR = (0, 255, 0)     # green
DASHED_REGION_THICKNESS = 5
DASH_LENGTH = 18
GAP_LENGTH = 12


# ==========================================
# 1. IO helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_image_id(item, fallback=""):
    return str(item.get("image_id", item.get("file_name", fallback)))


def build_image_id_lookup(data):
    lookup = {}
    for item in data:
        lookup.setdefault(get_image_id(item), item)
    return lookup


def find_dashed_region_item(item, lookup):
    # The dashed overlay comes from a second JSONL but is matched by the stable image id.
    return lookup.get(get_image_id(item))


def find_image_path(item):
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]

    if item.get("images"):
        image_path = item["images"][0]
        if os.path.exists(image_path):
            return image_path

    if item.get("image_relpath"):
        for image_root in IMAGE_ROOT_CANDIDATES:
            path = os.path.join(image_root, item["image_relpath"])
            if os.path.exists(path):
                return path

    image_id = get_image_id(item)
    for image_root in IMAGE_ROOT_CANDIDATES:
        for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                path = os.path.join(
                    image_root,
                    "RSRG_ME_datasets_ori",
                    dataset,
                    "images",
                    image_id + ext,
                )
                if os.path.exists(path):
                    return path

    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


# ==========================================
# 2. Box parsing
# ==========================================
def valid_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def valid_region(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except Exception:
        return False
    return x2 >= x1 and y2 >= y1


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
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        text or "",
        re.IGNORECASE | re.DOTALL,
    )
    answer_text = answer_match.group(1) if answer_match else text
    return parse_boxes_from_pattern(
        answer_text,
        rf"{role}\s*:\s*(?:[^\[\n]*?)\[([^\]]+)\]",
    )


def parse_model_boxes(text, role):
    boxes = parse_xml_tag_boxes(text, role)
    if boxes:
        return boxes
    return parse_answer_role_boxes(text, role)


def model_box_to_pixel(box, width, height):
    if MODEL_OUTPUT_COORD_SYSTEM != "norm1000":
        return box
    return [
        box[0] / 1000.0 * width,
        box[1] / 1000.0 * height,
        box[2] / 1000.0 * width,
        box[3] / 1000.0 * height,
    ]


# ==========================================
# 3. Drawing
# ==========================================
def clip_box(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    ]


def draw_dashed_line(image, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return

    unit_x = dx / length
    unit_y = dy / length
    distance = 0
    while distance < length:
        end_distance = min(distance + DASH_LENGTH, length)
        p1 = (
            int(round(x1 + unit_x * distance)),
            int(round(y1 + unit_y * distance)),
        )
        p2 = (
            int(round(x1 + unit_x * end_distance)),
            int(round(y1 + unit_y * end_distance)),
        )
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
        distance += DASH_LENGTH + GAP_LENGTH


def draw_dashed_box(image, box, color):
    if not valid_region(box):
        return

    height, width = image.shape[:2]
    box = clip_box(box, width, height)
    if not valid_region(box):
        return

    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        draw_dashed_line(
            image,
            start,
            end,
            color,
            DASHED_REGION_THICKNESS,
        )


def draw_box(image, box, label, color):
    if not valid_box(box):
        return

    height, width = image.shape[:2]
    box = clip_box(box, width, height)
    if not valid_box(box):
        return

    x1, y1, x2, y2 = [int(round(float(value))) for value in box]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 7)

    # text_y = max(15, y1 - 6)
    # cv2.putText(
    #     image,
    #     label,
    #     (x1, text_y),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.3,
    #     color,
    #     0.5,
    #     cv2.LINE_AA,
    # )


def collect_gt_boxes(item):
    boxes = []

    if DRAW_SUBJECT and valid_box(item.get("gt_subject_box")):
        boxes.append(("gt_subject", item["gt_subject_box"], GT_SUBJECT_COLOR))

    if DRAW_OBJECT:
        for index, box in enumerate(item.get("gt_anchor_boxes", []) or [], start=1):
            if valid_box(box):
                boxes.append((f"gt_object_{index}", box, GT_ANCHOR_COLOR))

    if DRAW_REGION:
        if valid_box(item.get("pseudo_region_box")):
            boxes.append(("pseudo_region", item["pseudo_region_box"], GT_REGION_COLOR))

        if valid_box(item.get("gt_region_box")):
            boxes.append(("gt_region", item["gt_region_box"], GT_REGION_COLOR))

        if valid_box(item.get("sft_region_box")):
            boxes.append(("sft_region", item["sft_region_box"], GT_REGION_COLOR))

    return boxes


def collect_model_boxes(item, width, height):
    boxes = []
    raw_output = item.get("raw_output", "")

    if DRAW_OBJECT:
        pred_objects = normalize_boxes(item.get("pred_object_boxes"))
        if pred_objects:
            for index, box in enumerate(pred_objects, start=1):
                boxes.append((f"pred_object_{index}", box, MODEL_OBJECT_COLOR))
        else:
            for index, box in enumerate(parse_model_boxes(raw_output, "object"), start=1):
                boxes.append(
                    (
                        f"pred_object_{index}",
                        model_box_to_pixel(box, width, height),
                        MODEL_OBJECT_COLOR,
                    )
                )

    if DRAW_REGION:
        pred_regions = normalize_boxes(item.get("pred_region_boxes"))
        if pred_regions:
            for index, box in enumerate(pred_regions, start=1):
                boxes.append((f"pred_region_{index}", box, MODEL_REGION_COLOR))
        else:
            for index, box in enumerate(parse_model_boxes(raw_output, "region"), start=1):
                boxes.append(
                    (
                        f"pred_region_{index}",
                        model_box_to_pixel(box, width, height),
                        MODEL_REGION_COLOR,
                    )
                )

    if DRAW_SUBJECT:
        pred_subjects = normalize_boxes(item.get("pred_subject_box"))
        if pred_subjects:
            for index, box in enumerate(pred_subjects, start=1):
                boxes.append((f"pred_subject_{index}", box, MODEL_SUBJECT_COLOR))
        else:
            for index, box in enumerate(parse_model_boxes(raw_output, "subject"), start=1):
                boxes.append(
                    (
                        f"pred_subject_{index}",
                        model_box_to_pixel(box, width, height),
                        MODEL_SUBJECT_COLOR,
                    )
                )

    return boxes


def safe_name(text):
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(text))[:120]


def select_items(data):
    indexed_items = [
        (row_number, item)
        for row_number, item in enumerate(data)
    ]

    if IMAGE_IDS:
        wanted = {str(image_id) for image_id in IMAGE_IDS}
        return [
            (sample_number, item)
            for sample_number, item in indexed_items
            if get_image_id(item) in wanted
        ]

    end_sample_number = (
        None
        if NUM_SAMPLES is None
        else START_SAMPLE_NUMBER + NUM_SAMPLES
    )
    return [
        (sample_number, item)
        for sample_number, item in indexed_items
        if sample_number >= START_SAMPLE_NUMBER
        and (end_sample_number is None or sample_number < end_sample_number)
    ]


def get_dashed_region_box(item):
    if item is None:
        return None
    for key in ["pseudo_region_box", "gt_region_box", "sft_region_box"]:
        if valid_region(item.get(key)):
            return item[key]
    return None


def draw_boxes_on_image(image, boxes, dashed_region_box=None):
    output = image.copy()
    for label, box, color in boxes:
        draw_box(output, box, label, color)
    if DRAW_DASHED_REGION and valid_region(dashed_region_box):
        draw_dashed_box(output, dashed_region_box, DASHED_REGION_COLOR)
    return output


# ==========================================
# 4. Main
# ==========================================
def main():
    data = load_jsonl(JSONL_PATH)
    dashed_region_lookup = {}
    if DRAW_DASHED_REGION:
        dashed_region_lookup = build_image_id_lookup(load_jsonl(DASHED_REGION_JSONL_PATH))

    selected_data = select_items(data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    saved_images = 0
    skipped = 0

    for sample_number, item in selected_data:
        item_index = item.get("index", "")
        image_id = get_image_id(item, sample_number)

        try:
            dashed_region_item = find_dashed_region_item(item, dashed_region_lookup)
            dashed_region_box = get_dashed_region_box(dashed_region_item)

            image_path = find_image_path(item)
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"OpenCV failed to read image: {image_path}")

            height, width = image.shape[:2]

            gt_boxes = collect_gt_boxes(item)
            model_boxes = collect_model_boxes(item, width, height)
            has_dashed_region = DRAW_DASHED_REGION and valid_region(dashed_region_box)

            base_name = f"sample_{sample_number:06d}_{safe_name(image_id)}"
            saved_paths = []

            if gt_boxes or has_dashed_region:
                gt_path = os.path.join(OUTPUT_DIR, base_name + "_gt.jpg")
                gt_image = draw_boxes_on_image(image, gt_boxes, dashed_region_box)
                cv2.imwrite(gt_path, gt_image)
                saved_paths.append(gt_path)

            if model_boxes:
                model_path = os.path.join(OUTPUT_DIR, base_name + "_model.jpg")
                model_image = draw_boxes_on_image(image, model_boxes)
                cv2.imwrite(model_path, model_image)
                saved_paths.append(model_path)

            if not saved_paths:
                # If all draw flags are disabled, still save the unmodified image for figure composition.
                original_path = os.path.join(OUTPUT_DIR, base_name + "_original.jpg")
                cv2.imwrite(original_path, image)
                saved_paths.append(original_path)

            saved += 1
            saved_images += len(saved_paths)
            print(
                f"sample_number={sample_number}: item_index={item_index}, "
                f"image_id={image_id}, "
                f"gt_boxes={len(gt_boxes)}, model_boxes={len(model_boxes)}"
            )
            print(f"instruction: {item.get('instruction', '')}")
            for path in saved_paths:
                print(f"saved: {path}")

        except Exception as error:
            skipped += 1
            print(
                f"skip sample_number={sample_number}, "
                f"item_index={item_index}, image_id={image_id}: {error}"
            )

    print("=" * 60)
    print(f"input JSONL: {JSONL_PATH}")
    print(f"output dir: {OUTPUT_DIR}")
    print(f"requested image ids: {IMAGE_IDS if IMAGE_IDS else 'range mode'}")
    print(f"saved samples: {saved}")
    print(f"saved images: {saved_images}")
    print(f"skipped: {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# python code/tools/draw_boxes.py
