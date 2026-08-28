"""Run InternVL VRT inference with a LoRA adapter on the ME-RSRG test split."""

import glob
import inspect
import json
import os
import re
import types

import torch
import torch.distributed as dist
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from transformers.modeling_utils import PreTrainedModel


# ==========================================
# 0. Paths and knobs
# ==========================================


DATA_PATH = os.environ.get("DATA_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/autodl-tmp/SPIN/image")

BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master")
LORA_ADAPTER_PATH = os.environ.get("LORA_ADAPTER_PATH", "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_grpo")
AUTO_FIND_LATEST_LORA = os.environ.get("AUTO_FIND_LATEST_LORA", "1") == "1"

OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_test_grpo.jsonl")


START_INDEX = int(os.environ.get("START_INDEX", "0"))
NUM_SAMPLES_TEXT = os.environ.get("NUM_SAMPLES", "").strip()
NUM_SAMPLES = int(NUM_SAMPLES_TEXT) if NUM_SAMPLES_TEXT else None
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
TORCH_DTYPE = torch.bfloat16

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==========================================
# 1. Prompt
# ==========================================
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
# 2. InternVL compatibility helpers
# ==========================================
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


def patch_qwen2_tokenizer_for_internvl():
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    except Exception:
        Qwen2Tokenizer = None
    try:
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    except Exception:
        Qwen2TokenizerFast = None

    def token_getter(attr_name, default_value):
        def getter(self):
            extra_tokens = self.init_kwargs.get("extra_special_tokens", {}) or {}
            return (
                getattr(self, f"_{attr_name}", None)
                or self.init_kwargs.get(attr_name)
                or extra_tokens.get(attr_name)
                or default_value
            )

        return getter

    def token_setter(attr_name):
        def setter(self, value):
            setattr(self, f"_{attr_name}", value)

        return setter

    def id_getter(token_attr):
        def getter(self):
            token = getattr(self, token_attr)
            token_id = self.convert_tokens_to_ids(token)
            if token_id is None:
                return self.unk_token_id
            return token_id

        return getter

    token_defaults = {
        "start_image_token": "<img>",
        "end_image_token": "</img>",
        "context_image_token": "<IMG_CONTEXT>",
        "image_token": "<image>",
        "video_token": "<video>",
    }
    id_attrs = {
        "start_image_token_id": "start_image_token",
        "end_image_token_id": "end_image_token",
        "context_image_token_id": "context_image_token",
        "image_token_id": "image_token",
        "video_token_id": "video_token",
    }

    for tokenizer_cls in [Qwen2Tokenizer, Qwen2TokenizerFast]:
        if tokenizer_cls is None:
            continue
        for attr_name, default_value in token_defaults.items():
            if not isinstance(getattr(tokenizer_cls, attr_name, None), property):
                setattr(
                    tokenizer_cls,
                    attr_name,
                    property(token_getter(attr_name, default_value), token_setter(attr_name)),
                )
        for attr_name, token_attr in id_attrs.items():
            if not isinstance(getattr(tokenizer_cls, attr_name, None), property):
                setattr(tokenizer_cls, attr_name, property(id_getter(token_attr)))


def patch_chat_template_processor_kwargs(processor):
    original_apply_chat_template = processor.apply_chat_template

    def apply_chat_template_compat(self, *args, **kwargs):
        direct_processor_kwargs = {
            key: kwargs.pop(key)
            for key in ("padding", "padding_side", "return_tensors")
            if key in kwargs
        }
        if direct_processor_kwargs:
            processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
            processor_kwargs.update(direct_processor_kwargs)
            kwargs["processor_kwargs"] = processor_kwargs
        return original_apply_chat_template(*args, **kwargs)

    processor.apply_chat_template = types.MethodType(apply_chat_template_compat, processor)


def patch_internvl_forward_contract_for_peft(model):
    original_forward = model.forward
    if "inputs_embeds" in inspect.signature(original_forward).parameters:
        return

    def forward_compat(self, *args, **kwargs):
        del self
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        if inputs_embeds is not None:
            raise RuntimeError(
                "InternVL does not support non-empty inputs_embeds; expected input_ids."
            )
        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(forward_compat, model)


def set_internvl_img_context_token_id(model, processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    context_token = getattr(tokenizer, "context_image_token", "<IMG_CONTEXT>")
    token_id = tokenizer.convert_tokens_to_ids(context_token)
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    if token_id is None or token_id < 0:
        raise RuntimeError(f"cannot resolve InternVL context token id for {context_token!r}")

    patched = 0
    for module in model.modules():
        if hasattr(module, "img_context_token_id"):
            module.img_context_token_id = int(token_id)
            patched += 1
    if patched == 0 and hasattr(model, "img_context_token_id"):
        model.img_context_token_id = int(token_id)
        patched = 1
    if patched == 0:
        raise RuntimeError("cannot find img_context_token_id on InternVL model")


def register_internvl_vision_dtype_guard(model):
    vision_model = getattr(model, "vision_model", None)
    if not isinstance(vision_model, torch.nn.Module):
        raise RuntimeError("cannot find vision_model on InternVL model")

    try:
        vision_dtype = next(
            parameter.dtype
            for parameter in vision_model.parameters()
            if parameter.is_floating_point()
        )
    except StopIteration as error:
        raise RuntimeError("InternVL vision_model has no floating-point parameters") from error

    def align_pixel_values_dtype(module, args, kwargs):
        del module
        args = list(args)
        if (
            args
            and torch.is_tensor(args[0])
            and args[0].is_floating_point()
            and args[0].dtype != vision_dtype
        ):
            args[0] = args[0].to(dtype=vision_dtype)

        pixel_values = kwargs.get("pixel_values")
        if (
            torch.is_tensor(pixel_values)
            and pixel_values.is_floating_point()
            and pixel_values.dtype != vision_dtype
        ):
            kwargs = dict(kwargs)
            kwargs["pixel_values"] = pixel_values.to(dtype=vision_dtype)
        return tuple(args), kwargs

    vision_model.register_forward_pre_hook(align_pixel_values_dtype, with_kwargs=True)


# ==========================================
# 3. General helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


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


def find_latest_lora_adapter(path):
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path

    candidates = []
    for config_path in glob.glob(os.path.join(path, "**", "adapter_config.json"), recursive=True):
        adapter_dir = os.path.dirname(config_path)
        match = re.search(r"checkpoint-(\d+)", adapter_dir)
        step = int(match.group(1)) if match else -1
        candidates.append((step, os.path.getmtime(config_path), adapter_dir))

    if not candidates:
        raise FileNotFoundError(f"cannot find adapter_config.json under {path}")

    return sorted(candidates)[-1][2]


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
        shard_path = f"{output_path}.rank{rank}.tmp"
        if os.path.exists(shard_path):
            results.extend(load_jsonl(shard_path))
    results.sort(key=lambda item: item["index"])

    with open(output_path, "w", encoding="utf-8") as out_file:
        for item in results:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    for rank in range(world_size):
        shard_path = f"{output_path}.rank{rank}.tmp"
        if os.path.exists(shard_path):
            os.remove(shard_path)


def find_image_path(item):
    if item.get("image_path") and os.path.exists(item["image_path"]):
        return item["image_path"]

    if item.get("images"):
        image_path = item["images"][0]
        if os.path.exists(image_path):
            return image_path

    if item.get("image_relpath"):
        for root in [IMAGE_DIR, os.path.join(os.path.dirname(IMAGE_DIR), "images")]:
            path = os.path.join(root, item["image_relpath"])
            if os.path.exists(path):
                return path

    image_id = str(item.get("image_id", item.get("file_name", "")))
    base_dir = os.path.join(IMAGE_DIR, "RSRG_ME_datasets_ori")
    for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
        image_dir = os.path.join(base_dir, dataset, "images")
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            path = os.path.join(image_dir, image_id + ext)
            if os.path.exists(path):
                return path

    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


def valid_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def parse_boxes_from_pattern(text, pattern):
    boxes = []
    for match in re.finditer(pattern, text or "", re.IGNORECASE | re.DOTALL):
        try:
            values = [
                float(value)
                for value in re.findall(
                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                    match.group(1),
                )
            ]
            if len(values) == 4 and valid_box(values):
                boxes.append(values)
        except Exception:
            pass
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
    clean_instruction = str(instruction).replace("<image>", "").strip()
    prompt = PROMPT_TEMPLATE.format(instruction=clean_instruction).strip()
    if "<IMG_CONTEXT>" not in prompt:
        prompt = "<IMG_CONTEXT>\n" + prompt

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def decode_tokens(processor, token_ids):
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        decoder = processor.tokenizer.batch_decode
    return decoder(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def extract_completion_ids(generated_ids, input_ids):
    sequences = generated_ids
    if not torch.is_tensor(sequences):
        sequences = getattr(generated_ids, "sequences", None)
    if not torch.is_tensor(sequences):
        raise RuntimeError(f"unsupported generate output type: {type(generated_ids)!r}")
    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)

    prompt_length = input_ids.size(1)
    includes_prompt = (
        sequences.size(1) >= prompt_length
        and torch.equal(sequences[:, :prompt_length].to(input_ids.device), input_ids)
    )
    if includes_prompt:
        return sequences[:, prompt_length:]
    return sequences


def generate_one(model, processor, image, instruction):
    messages = build_messages(instruction)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=[image],
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    inputs = move_batch_to_device(inputs, device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    completion_ids = extract_completion_ids(generated_ids, inputs["input_ids"])
    return decode_tokens(processor, completion_ids)


def load_internvl_processor(model_path):
    patch_transformers_for_internvl()
    patch_qwen2_tokenizer_for_internvl()
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    patch_chat_template_processor_kwargs(processor)
    return processor


def load_internvl_model(model_path, adapter_path, local_rank):
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "device_map": {"": local_rank},
    }
    try:
        model = AutoModel.from_pretrained(
            model_path,
            dtype=TORCH_DTYPE,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE,
            **model_kwargs,
        )

    model.config.use_cache = True
    patch_internvl_forward_contract_for_peft(model)
    register_internvl_vision_dtype_guard(model)

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )
        model.config.use_cache = True
    return model


# ==========================================
# 4. Main
# ==========================================
def main():
    world_size, rank, local_rank = init_distributed()
    data = load_jsonl(DATA_PATH)

    if NUM_SAMPLES is None:
        selected_data = data[START_INDEX:]
    else:
        selected_data = data[START_INDEX : START_INDEX + NUM_SAMPLES]
    indexed_data = list(enumerate(selected_data, start=START_INDEX))
    rank_data = indexed_data[rank::world_size]

    model_path = resolve_pretrained_path(BASE_MODEL_PATH)
    adapter_path = ""
    if LORA_ADAPTER_PATH:
        adapter_path = (
            find_latest_lora_adapter(LORA_ADAPTER_PATH)
            if AUTO_FIND_LATEST_LORA
            else os.path.expanduser(LORA_ADAPTER_PATH)
        )

    if rank == 0:
        print(f"World size: {world_size}; total samples: {len(indexed_data)}")
        if model_path != BASE_MODEL_PATH:
            print(f"Resolved InternVL model path: {model_path}")
        print(f"Loading InternVL processor: {model_path}")
    processor = load_internvl_processor(model_path)

    if rank == 0:
        print(f"Loading InternVL model: {model_path}")
        if adapter_path:
            print(f"Loading InternVL LoRA adapter: {adapter_path}")
        else:
            print("No LoRA adapter path was provided; evaluating the base model.")
    model = load_internvl_model(model_path, adapter_path, local_rank)
    set_internvl_img_context_token_id(model, processor)
    model.eval()

    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    shard_path = f"{OUTPUT_JSONL}.rank{rank}.tmp"

    sub_hit = 0
    sub_total = 0

    with open(shard_path, "w", encoding="utf-8") as out_file:
        for index, item in tqdm(rank_data, desc=f"testing rank {rank}", disable=rank != 0):
            image_id = str(item.get("image_id", item.get("file_name", "")))
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

            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()

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


# 4-GPU:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 code/eval/infer_internvl_vrt_lora.py
#
# Example for a GRPO checkpoint:
# LORA_ADAPTER_PATH=/root/autodl-tmp/SPIN/output_InternVL3_5/vrt_grpo_shared_lora/checkpoint-100 \
# OUTPUT_JSONL=/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_test_grpo_ckpt100.jsonl \
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 code/eval/infer_internvl_vrt_lora.py
