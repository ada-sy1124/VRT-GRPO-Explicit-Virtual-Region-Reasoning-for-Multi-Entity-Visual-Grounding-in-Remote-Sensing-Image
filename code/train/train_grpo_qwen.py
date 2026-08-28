"""Train a Qwen policy with GRPO using VRT rewards and an SFT LoRA warm start."""

import os
import re
import json
import glob
import torch
import random
import inspect
import time
import types
import requests
from datasets import load_dataset, Image as DatasetImage
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel


# ==========================================
# 0. Global hyperparameters and config
# ==========================================
# Paths
DATA_PATH = os.environ.get("DATA_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/autodl-tmp/SPIN/image")
POLICY_MODEL_ID = os.environ.get("POLICY_MODEL_ID", "/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_grpo")
SFT_LORA_ADAPTER_PATH = os.environ.get("SFT_LORA_ADAPTER_PATH", "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_sft_lora")

# GRPO continues training the same LoRA initialized by SFT warmup.
# A frozen copy of the initial SFT adapter is loaded as "ref" for KL.
AUTO_FIND_LATEST_SFT_LORA = os.environ.get("AUTO_FIND_LATEST_SFT_LORA", "1") == "1"

# Debug and feature flags
DEBUG_SAMPLE_SIZE = None       # Set an integer for a debug subset; keep None for full training.
USE_VLM_CRITIC = True          # Enable the external VLM critic.
CRITIC_API_URL = os.environ.get("CRITIC_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
CRITIC_MODEL_NAME = os.environ.get("CRITIC_MODEL_NAME", "qwen3vl-30b-critic")
CRITIC_TIMEOUT = int(os.environ.get("CRITIC_TIMEOUT", "60"))
CRITIC_MAX_TOKENS = int(os.environ.get("CRITIC_MAX_TOKENS", "256"))
CRITIC_RETRIES = int(os.environ.get("CRITIC_RETRIES", "2"))

# GRPO training hyperparameters
LEARNING_RATE = 1e-5
MAX_STEPS = -1                 # -1 runs the full dataset for NUM_TRAIN_EPOCHS.
NUM_TRAIN_EPOCHS = 1           # First GRPO pass after SFT warmup.
SAVE_STEPS = 100               # Save a checkpoint every 100 optimizer updates.
SAVE_TOTAL_LIMIT = 25          # Keep checkpoints for offline checkpoint selection.
PER_DEVICE_BATCH_SIZE = 1      # Per-device batch size.
EXPECTED_WORLD_SIZE = 4        # Expected number of GPUs for the main run.
TARGET_GENERATION_BATCH_SIZE = 32  # 4 GPUs: grad_acc=8 covers 4 prompts per update.
NUM_GENERATIONS = 8            # Paper setting.
MAX_COMPLETION_LENGTH = 512    # Max generated tokens for plan/think/answer.
WARMUP_RATIO = 0.01            # Paper setting.
KL_BETA = 0.0025               # Paper setting.
GENERATION_TEMPERATURE = 0.8   # Keep group exploration while limiting coordinate drift.
GENERATION_TOP_P = 0.95
MAX_GRAD_NORM = 1.0
SEED = 42
SCALE_REWARDS = "batch"        # Group centering with batch-level variance scaling.
LOSS_TYPE = "dapo"             # Reduce completion-length bias.
MASK_TRUNCATED_COMPLETIONS = True
DISABLE_DROPOUT = True
GRADIENT_CHECKPOINTING = False # H20 96GB + LoRA + batch=1 usually does not need checkpointing.

# Reward weights. The final benchmark is subject/object mAcc@0.5, so grounding is the main signal.
# Plan/format/region are auxiliary shaping terms and should not dominate the policy update.
VLM_PLAN_REWARD_WEIGHT = 0.15
LOGIC_REWARD_WEIGHT = 0.25
ENTITY_REWARD_WEIGHT = 1.00
REGION_REWARD_WEIGHT = 0.25
SPATIAL_REWARD_WEIGHT = 0.20

# LoRA adapter parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# WandB logging
WANDB_PROJECT_NAME = "SPARC-VRT-GRPO"
WANDB_RUN_NAME = os.environ.get(
    "QWEN_WANDB_RUN_NAME",
    "Qwen3-VL-8B-VRT-GRPO-SFTInit-SharedLoRA-r16-G8-H20x4-Final",
).strip()

# Environment variables
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(SEED)


def distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    return world_size, local_rank


def prefer_bf16():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def policy_torch_dtype():
    return torch.bfloat16 if prefer_bf16() else torch.float16


def resolved_wandb_run_name():
    return WANDB_RUN_NAME


def patch_chat_template_processor_kwargs(processor):
    """Route TRL's legacy processor call kwargs through processor_kwargs."""
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

    processor.apply_chat_template = types.MethodType(
        apply_chat_template_compat,
        processor,
    )
    print("Processor argument interface patched: padding/padding_side/return_tensors -> processor_kwargs")

# ==========================================
# 1. Policy model prompt
# ==========================================
SYSTEM_PROMPT = """
Localize objects in the image based on the following description: {instruction}

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
# [Perfect Example] (You must strictly imitate this format!)
# User: Localize objects in the image based on the following description: a basketball court on the right of a lake.
# Assistant:
# <plan>
# First, locate the lake as the reference object. Then, deduce the search region to the right of the lake. Finally, find the basketball court within this region.
# </plan>
# <think>
# I should first look for the lake in the image. I found a blue lake <object> [100.5, 200.0, 300.2, 400.1]. Next, based on the description, the target is to the right of this lake. I will define a search area to the right <region> [300.2, 200.0, 600.0, 400.1]. Within this region, I clearly see a basketball court. I will lock onto this target <subject> [400.0, 250.0, 500.0, 350.0].
# </think>
# <answer>
# object: [100.5, 200.0, 300.2, 400.1], region: [300.2, 200.0, 600.0, 400.1], subject: [400.0, 250.0, 500.0, 350.0]
# </answer>
# """

# ==========================================
# 2. Critic model prompt and scoring helpers
# ==========================================
CRITIC_SYSTEM_PROMPT = """You are a strict and objective text-logic judge.

Your task is to evaluate whether the model-generated `<plan>` covers the constraints in the user instruction, and whether the actual execution trajectory follows that plan.

You will not see the image. Do not evaluate coordinate accuracy, bbox correctness, or output format.
Only evaluate the following two text-level dimensions:

1. constraint_coverage
Evaluate whether the Plan covers the target, reference objects, attributes, and spatial relations in the user instruction.
- 9 points: Fully covers the key constraints.
- 6 points: Covers the main constraints, but misses a few minor conditions.
- 3 points: Misses key constraints and may lead to the wrong region.
- 0 points: Barely covers the instruction, or identifies the wrong target category.

2. execution_consistency
Evaluate whether the execution trajectory involving object, region, and target generally follows the Plan.
- 9 points: The execution is highly consistent with the Plan.
- 6 points: Mostly consistent, but with minor omissions or unclear correspondence.
- 3 points: The execution is clearly disconnected from the Plan.
- 0 points: The execution largely ignores the Plan.

Strictly output JSON only. Do not use Markdown or provide any extra explanation:

output json:
{
  "reasoning": "Briefly explain the reason for any deductions",
  "constraint_coverage": <integer>,
  "execution_consistency": <integer>
}
"""


def compute_critic_reward(parsed_json):
    """Map 0-9 critic scores to a continuous, roughly zero-centered reward."""
    try:
        cc = max(0.0, min(9.0, float(parsed_json.get("constraint_coverage", 0))))
        ec = max(0.0, min(9.0, float(parsed_json.get("execution_consistency", 0))))
        # Full score 18 -> 1.0, acceptable 12 -> 0.0, all wrong 0 -> -2.0.
        return float((cc + ec - 12.0) / 6.0)
    except Exception as e:
        raise ValueError(f"invalid critic score JSON: {e}") from e


def parse_critic_json(text):
    text = (text or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def effective_gradient_accumulation():
    world_size, _ = distributed_info()
    denom = PER_DEVICE_BATCH_SIZE * world_size
    if TARGET_GENERATION_BATCH_SIZE % denom != 0:
        raise ValueError(
            f"TARGET_GENERATION_BATCH_SIZE ({TARGET_GENERATION_BATCH_SIZE}) must be divisible by "
            f"per_device_train_batch_size * world_size ({denom})."
        )
    return TARGET_GENERATION_BATCH_SIZE // denom


def validate_training_config():
    world_size, local_rank = distributed_info()
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"This final config requires world_size={EXPECTED_WORLD_SIZE}, got {world_size}. "
            "Launch with torchrun --standalone --nproc_per_node=4, not plain python."
        )
    if local_rank < 0 or not torch.cuda.is_available():
        raise RuntimeError("CUDA torchrun environment is not initialized correctly.")
    if not prefer_bf16():
        raise RuntimeError("The final 4xH20 config requires CUDA bf16 support.")
    grad_acc = effective_gradient_accumulation()
    generation_batch_size = PER_DEVICE_BATCH_SIZE * grad_acc * world_size
    if generation_batch_size < NUM_GENERATIONS:
        raise ValueError(
            f"generation_batch_size ({generation_batch_size}) must be >= num_generations ({NUM_GENERATIONS}). "
            f"Current: per_device_train_batch_size={PER_DEVICE_BATCH_SIZE}, "
            f"gradient_accumulation_steps={grad_acc}, world_size={world_size}."
        )
    if generation_batch_size % NUM_GENERATIONS != 0:
        raise ValueError(
            f"generation_batch_size ({generation_batch_size}) must be divisible by num_generations ({NUM_GENERATIONS}). "
            f"Set GRADIENT_ACCUMULATION to a multiple that makes "
            f"{PER_DEVICE_BATCH_SIZE} * grad_acc * {world_size} divisible by {NUM_GENERATIONS}."
        )
    if MAX_STEPS == -1 and NUM_TRAIN_EPOCHS <= 0:
        raise ValueError("NUM_TRAIN_EPOCHS must be > 0 when MAX_STEPS is -1.")
    adapter_path = (
        find_latest_lora_adapter(SFT_LORA_ADAPTER_PATH)
        if AUTO_FIND_LATEST_SFT_LORA
        else SFT_LORA_ADAPTER_PATH
    )
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(f"missing adapter_config.json under: {adapter_path}")
    validate_sft_lora_adapter(adapter_path)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")
    print(
        "GRPO config preflight passed: "
        f"world_size={world_size}, generation_batch_size={generation_batch_size}, "
        f"num_generations={NUM_GENERATIONS}, grad_acc={grad_acc}, "
        f"dtype={'bf16' if prefer_bf16() else 'fp16'}"
    )


def make_grpo_config(**kwargs):
    valid_keys = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    if "eval_strategy" in kwargs and "eval_strategy" not in valid_keys and "evaluation_strategy" in valid_keys:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    adapter_kwargs = {
        key: kwargs.pop(key)
        for key in ["model_adapter_name", "ref_adapter_name"]
        if key in kwargs
    }
    required_keys = {
        "output_dir",
        "learning_rate",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "max_completion_length",
        "temperature",
        "warmup_ratio",
        "beta",
        "scale_rewards",
        "loss_type",
        "mask_truncated_completions",
        "disable_dropout",
    }
    missing_required = sorted(k for k in required_keys if k not in valid_keys)
    if missing_required:
        raise RuntimeError(
            "The current TRL version is missing required GRPOConfig arguments, so the training semantics are not guaranteed: "
            f"{missing_required}. Please upgrade/change TRL or explicitly adapt these argument names."
        )

    filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print(f"The current TRL version does not support these optional GRPOConfig arguments; skipped: {dropped}")
    config = GRPOConfig(**filtered)

    missing_adapter_keys = sorted(set(adapter_kwargs) - valid_keys)
    for key, value in adapter_kwargs.items():
        setattr(config, key, value)
    if missing_adapter_keys:
        print(
            "The current TRL GRPOConfig does not declare adapter-switching arguments; "
            f"dynamic attributes were attached to training_args: {missing_adapter_keys}."
        )
    return config


def build_training_args():
    return make_grpo_config(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        max_steps=MAX_STEPS if DEBUG_SAMPLE_SIZE is None else 10,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=effective_gradient_accumulation(),
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        warmup_ratio=WARMUP_RATIO,
        beta=KL_BETA,
        scale_rewards=SCALE_REWARDS,
        loss_type=LOSS_TYPE,
        mask_truncated_completions=MASK_TRUNCATED_COMPLETIONS,
        disable_dropout=DISABLE_DROPOUT,
        model_adapter_name="default",
        ref_adapter_name="ref",
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
        data_seed=SEED,
        report_to="wandb",
        run_name=resolved_wandb_run_name(),
        bf16=prefer_bf16(),
        fp16=not prefer_bf16(),
        tf32=prefer_bf16(),
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
    )


def check_critic_server():
    if not USE_VLM_CRITIC:
        return
    payload = {
        "model": CRITIC_MODEL_NAME,
        "messages": [{"role": "user", "content": "Return exactly OK."}],
        "temperature": 0,
        "max_tokens": 8,
    }
    try:
        resp = requests.post(CRITIC_API_URL, json=payload, timeout=min(CRITIC_TIMEOUT, 15))
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        _ = resp.json()["choices"][0]["message"]["content"]
        print(f"External critic preflight passed: {CRITIC_API_URL} ({CRITIC_MODEL_NAME})")
    except Exception as e:
        raise RuntimeError(
            f"External critic preflight failed: {e}. First verify that curl http://127.0.0.1:8000/v1/models works, "
            "and make sure CRITIC_API_URL/CRITIC_MODEL_NAME match the model served by vLLM."
        )


def resolve_image_path(example):
    if example.get("image_path") and os.path.exists(example["image_path"]):
        return example["image_path"]

    if example.get("image_relpath"):
        for root in [IMAGE_DIR, os.path.join(os.path.dirname(IMAGE_DIR), "images")]:
            candidate = os.path.join(root, example["image_relpath"])
            if os.path.exists(candidate):
                return candidate

    image_id = str(example.get("image_id", example.get("file_name", "")))
    base_dir = os.path.join(IMAGE_DIR, "RSRG_ME_datasets_ori")
    for dataset_name in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
        image_dir = os.path.join(base_dir, dataset_name, "images")
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            candidate = os.path.join(image_dir, image_id + ext)
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


def format_vlm_dataset(example):
    image_path = resolve_image_path(example)
    clean_instruction = str(example["instruction"]).replace("<image>", "").strip()
    prompt_text = SYSTEM_PROMPT.format(instruction=clean_instruction).strip()
    example["prompt"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    # Datasets.Image decodes lazily, avoiding an expensive full-image map before training.
    example["image"] = image_path
    return example

# ==========================================
# 3. Helpers
# ==========================================

def valid_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def boxes_equal(box1, box2, tolerance=1e-4):
    return valid_box(box1) and valid_box(box2) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(box1, box2)
    )


def box_lists_equal(boxes1, boxes2):
    if len(boxes1) != len(boxes2):
        return False
    normalized1 = sorted(tuple(round(float(v), 4) for v in box) for box in boxes1)
    normalized2 = sorted(tuple(round(float(v), 4) for v in box) for box in boxes2)
    return normalized1 == normalized2


def valid_training_example(example):
    # GRPO rewards assume the pseudo region is a valid intermediate state containing the subject.
    subject = example.get("gt_subject_box")
    anchors = example.get("gt_anchor_boxes") or []
    region = example.get("pseudo_region_box")
    return (
        valid_box(subject)
        and bool(anchors)
        and all(valid_box(box) for box in anchors)
        and valid_box(region)
        and compute_ioc(subject, region) >= 0.999
    )


def parse_boxes_from_pattern(text, pattern):
    boxes = []
    for match in re.finditer(pattern, text or "", re.IGNORECASE | re.DOTALL):
        try:
            box = [float(x.strip()) for x in match.group(1).split(',')]
            if valid_box(box):
                boxes.append(box)
        except Exception:
            pass
    return boxes


def parse_xml_tag_boxes(text, tag):
    return parse_boxes_from_pattern(
        text,
        rf'<{tag}(?:_[^>]*)?>\s*:?\s*\[([\d\.\,\s]+)\]',
    )


def parse_answer_role_boxes(text, role):
    answer_match = re.search(r'<answer>(.*?)</answer>', text or "", re.IGNORECASE | re.DOTALL)
    answer_text = answer_match.group(1) if answer_match else text
    return parse_boxes_from_pattern(
        answer_text,
        rf'{role}\s*:\s*(?:[^\[\n]*?)\[([\d\.\,\s]+)\]',
    )


def parse_role_boxes(text, role):
    boxes = parse_xml_tag_boxes(text, role)
    if boxes:
        return boxes
    return parse_answer_role_boxes(text, role)


def parse_box(text, tag):
    boxes = parse_role_boxes(text, tag)
    return boxes[-1] if boxes else None


def extract_boxes_with_prefix(text, prefix):
    return parse_role_boxes(text, prefix)

def extract_text_from_completion(comp):
    if isinstance(comp, str): return comp
    if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[-1], dict) and "content" in comp[-1]:
        return comp[-1]["content"]
    return str(comp)

def find_latest_lora_adapter(path):
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path

    candidates = []
    for config_path in glob.glob(os.path.join(path, "**", "adapter_config.json"), recursive=True):
        adapter_dir = os.path.dirname(config_path)
        match = re.search(r"checkpoint-(\d+)", adapter_dir)
        step = int(match.group(1)) if match else -1
        candidates.append((step, os.path.getmtime(config_path), adapter_dir))
    if not candidates:
        raise FileNotFoundError(f"cannot find adapter_config.json under: {path}")
    return max(candidates)[2]


def validate_sft_lora_adapter(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    rank = int(config.get("r", -1))
    alpha = int(config.get("lora_alpha", -1))
    targets = set(config.get("target_modules") or [])
    if rank <= 0 or alpha <= 0 or not targets:
        raise ValueError(f"invalid SFT LoRA config: {config_path}")
    print(
        "SFT LoRA config is valid and will be used as the trainable GRPO warmup adapter: "
        f"r={rank}, alpha={alpha}, targets={sorted(targets)}"
    )


def load_policy_processor():
    print(f"Loading Qwen processor: {POLICY_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        POLICY_MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    patch_chat_template_processor_kwargs(processor)
    return processor


def load_policy_model_for_grpo(processor):
    adapter_path = find_latest_lora_adapter(SFT_LORA_ADAPTER_PATH) if AUTO_FIND_LATEST_SFT_LORA else SFT_LORA_ADAPTER_PATH
    print(f"Loading trainable SFT warmup LoRA: {adapter_path}")

    world_size, local_rank = distributed_info()
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model_kwargs = dict(
        dtype=policy_torch_dtype(),
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"Loading Qwen policy model: {POLICY_MODEL_ID}")
    if world_size == 1:
        model_kwargs["device_map"] = "auto"

    try:
        base_model = AutoModelForImageTextToText.from_pretrained(
            POLICY_MODEL_ID,
            **model_kwargs,
        )
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        base_model = AutoModelForImageTextToText.from_pretrained(
            POLICY_MODEL_ID,
            **model_kwargs,
        )
    base_model.config.use_cache = False

    policy_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        adapter_name="default",
        is_trainable=True,
    )
    policy_model.config.use_cache = False
    policy_model.load_adapter(
        adapter_path,
        adapter_name="ref",
        is_trainable=False,
    )
    policy_model.set_adapter("default")

    trainable_names = [
        name for name, parameter in policy_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("SFT warmup adapter has no trainable parameters")
    invalid_names = [
        name for name in trainable_names
        if "lora_" not in name or ".default." not in name
    ]
    if invalid_names:
        raise RuntimeError(
            "parameters outside the default SFT/GRPO LoRA are trainable: "
            f"{invalid_names[:10]}"
        )

    print(
        "SFT warmup LoRA loaded as the default trainable adapter; "
        "the ref adapter is a frozen copy of the initial SFT adapter."
    )
    return policy_model, None

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    b1_a = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1e-6)
    b2_a = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 1e-6)
    return inter / (b1_a + b2_a - inter)

def compute_ioc(box_child, box_parent):
    x1, y1 = max(box_child[0], box_parent[0]), max(box_child[1], box_parent[1])
    x2, y2 = min(box_child[2], box_parent[2]), min(box_child[3], box_parent[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    child_a = max((box_child[2] - box_child[0]) * (box_child[3] - box_child[1]), 1e-6)
    return inter / child_a

def box_area(box):
    return max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)

def box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def relation_dimensions(instruction):
    text = str(instruction).lower()
    dimensions = []
    if re.search(r"\b(left|right)\b", text):
        dimensions.append("horizontal")
    if re.search(r"\b(above|below|upper|lower|top|bottom|over|under)\b", text):
        dimensions.append("vertical")
    return dimensions


def relation_alignment(region_box, anchor_box, gt_subject_box, dimensions):
    region_cx, region_cy = box_center(region_box)
    anchor_cx, anchor_cy = box_center(anchor_box)
    subject_cx, subject_cy = box_center(gt_subject_box)
    comparisons = []

    if "horizontal" in dimensions and abs(subject_cx - anchor_cx) > 1e-6:
        comparisons.append((region_cx - anchor_cx) * (subject_cx - anchor_cx) > 0)
    if "vertical" in dimensions and abs(subject_cy - anchor_cy) > 1e-6:
        comparisons.append((region_cy - anchor_cy) * (subject_cy - anchor_cy) > 0)

    if not comparisons:
        return None
    return sum(comparisons) / len(comparisons)


def greedy_one_to_one_ious(pred_boxes, gt_boxes):
    matched_pred = set()
    matched_gt = set()
    matches = [0.0] * len(gt_boxes)
    pairs = sorted(
        (
            (compute_iou(pred_box, gt_box), pred_idx, gt_idx)
            for pred_idx, pred_box in enumerate(pred_boxes)
            for gt_idx, gt_box in enumerate(gt_boxes)
        ),
        reverse=True,
    )
    for iou, pred_idx, gt_idx in pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches[gt_idx] = iou
    return matches

# ==========================================
# 4. Five reward functions
# ==========================================

def extract_xml_block(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


# Critic 1: plan coverage and whether think/answer follow the plan.
def vlm_plan_reward_func(prompts, completions, **kwargs):
    instructions = kwargs.get("instruction", [""] * len(completions))
    rewards = []

    for comp, instruction in zip(completions, instructions):
        comp_str = extract_text_from_completion(comp)
        plan_text = extract_xml_block(comp_str, "plan")
        if not plan_text:
            rewards.append(-1.0 * VLM_PLAN_REWARD_WEIGHT)
            continue

        think_text = extract_xml_block(comp_str, "think")
        answer_text = extract_xml_block(comp_str, "answer")
        execution_text = f"<think>\n{think_text}\n</think>\n<answer>\n{answer_text}\n</answer>"
        # Hide exact numbers from the text critic so it judges planning logic, not coordinate precision.
        clean_execution = re.sub(r"\[[\d\.\,\s]+\]", "[coordinates]", execution_text)
        clean_instruction = str(instruction).replace("<image>", "").strip()
        critic_user_content = (
            f"Instruction:\n{clean_instruction}\n\n"
            f"Model plan:\n{plan_text}\n\n"
            f"Execution trajectory:\n{clean_execution}"
        )
        payload = {
            "model": CRITIC_MODEL_NAME,
            "messages": [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": critic_user_content},
            ],
            "temperature": 0,
            "max_tokens": CRITIC_MAX_TOKENS,
        }

        last_error = None
        for attempt in range(CRITIC_RETRIES + 1):
            try:
                response = requests.post(CRITIC_API_URL, json=payload, timeout=CRITIC_TIMEOUT)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                result_text = response.json()["choices"][0]["message"]["content"]
                reward = compute_critic_reward(parse_critic_json(result_text))
                rewards.append(reward * VLM_PLAN_REWARD_WEIGHT)
                break
            except Exception as error:
                last_error = error
                if attempt < CRITIC_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(
                f"critic failed after {CRITIC_RETRIES + 1} attempts: {last_error}"
            )

    return rewards


# Critic 2: format, count, and think/answer consistency.
def logic_reward_func(prompts, completions, **kwargs):
    gt_anchors_list = kwargs.get("gt_anchor_boxes", [[] for _ in completions])
    rewards = []

    for completion, gt_anchors in zip(completions, gt_anchors_list):
        text = extract_text_from_completion(completion)
        expected_objects = len([box for box in (gt_anchors or []) if valid_box(box)])
        score = 0.0

        strict_structure = re.fullmatch(
            r"\s*<plan>.*?</plan>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if strict_structure:
            score += 1.0
        else:
            for tag in ["plan", "think", "answer"]:
                score += 0.15 if extract_xml_block(text, tag) else -0.25

        think_subjects = parse_xml_tag_boxes(text, "subject")
        think_objects = parse_xml_tag_boxes(text, "object")
        think_regions = parse_xml_tag_boxes(text, "region")
        answer_subjects = parse_answer_role_boxes(text, "subject")
        answer_objects = parse_answer_role_boxes(text, "object")
        answer_regions = parse_answer_role_boxes(text, "region")

        score += 0.50 if len(think_subjects) == 1 else -0.75
        score += 0.40 if len(think_regions) == 1 else -0.60
        if len(think_objects) == expected_objects:
            score += 0.40
        else:
            score -= 0.40 + 0.10 * abs(len(think_objects) - expected_objects)

        for think_boxes, answer_boxes in [
            (think_subjects, answer_subjects),
            (think_objects, answer_objects),
            (think_regions, answer_regions),
        ]:
            score += 0.20 if box_lists_equal(think_boxes, answer_boxes) else -0.40

        rewards.append(score * LOGIC_REWARD_WEIGHT)

    return rewards


# Critic 3: subject/object grounding, using the same matching logic as evaluation.
def entity_grounding_reward_func(prompts, completions, **kwargs):
    gt_subjects = kwargs.get("gt_subject_box", [])
    gt_anchors_list = kwargs.get("gt_anchor_boxes", [])
    rewards = []

    for completion, gt_subject, gt_anchors in zip(completions, gt_subjects, gt_anchors_list):
        text = extract_text_from_completion(completion)
        score = 0.0

        think_subjects = parse_xml_tag_boxes(text, "subject")
        answer_subjects = parse_answer_role_boxes(text, "subject")
        eval_subjects = think_subjects if think_subjects else answer_subjects
        eval_subject = eval_subjects[-1] if eval_subjects else None

        if len(eval_subjects) == 1:
            score += 0.20
        else:
            score -= 0.50 * abs(len(eval_subjects) - 1)

        if len(think_subjects) == 1 and len(answer_subjects) == 1:
            score += 0.20 if boxes_equal(think_subjects[0], answer_subjects[0]) else -0.75
        else:
            score -= 0.50

        if valid_box(eval_subject) and valid_box(gt_subject):
            subject_iou = compute_iou(eval_subject, gt_subject)
            score += 2.0 * subject_iou
            if subject_iou >= 0.7:
                score += 0.50
            elif subject_iou >= 0.5:
                score += 0.30
            elif subject_iou < 0.1:
                score -= 0.40
        else:
            score -= 1.0

        pred_objects = parse_role_boxes(text, "object")
        valid_gt_anchors = [box for box in (gt_anchors or []) if valid_box(box)]
        count_difference = abs(len(pred_objects) - len(valid_gt_anchors))
        score += 0.20 if count_difference == 0 else -0.40 * count_difference

        if valid_gt_anchors:
            matched_ious = greedy_one_to_one_ious(pred_objects, valid_gt_anchors)
            object_score = 0.0
            for object_iou in matched_ious:
                object_score += 1.2 * object_iou
                if object_iou >= 0.7:
                    object_score += 0.30
                elif object_iou >= 0.5:
                    object_score += 0.20
                elif object_iou < 0.1:
                    object_score -= 0.20
            score += object_score / len(valid_gt_anchors)
        elif pred_objects:
            score -= 0.40 * len(pred_objects)

        rewards.append(score * ENTITY_REWARD_WEIGHT)

    return rewards


# Critic 4: a single region should align with the pseudo region and cover the subject.
def predicate_region_reward_func(prompts, completions, **kwargs):
    pseudo_regions = kwargs.get("pseudo_region_box", [])
    gt_subjects = kwargs.get("gt_subject_box", [])
    rewards = []

    for completion, pseudo_region, gt_subject in zip(completions, pseudo_regions, gt_subjects):
        text = extract_text_from_completion(completion)
        regions = parse_role_boxes(text, "region")
        if len(regions) != 1:
            penalty = -1.5 - 0.25 * abs(len(regions) - 1)
            rewards.append(penalty * REGION_REWARD_WEIGHT)
            continue
        if not valid_box(pseudo_region) or not valid_box(gt_subject):
            raise ValueError("invalid pseudo_region_box or gt_subject_box reached region reward")

        region = regions[0]
        # Region reward balances similarity to the pseudo region with actual target containment.
        region_iou = compute_iou(region, pseudo_region)
        subject_containment = compute_ioc(gt_subject, region)
        score = 1.5 * region_iou + 0.5 * subject_containment
        if region_iou >= 0.5:
            score += 0.50
        elif region_iou < 0.1:
            score -= 0.50
        if subject_containment < 0.5:
            score -= 0.50
        rewards.append(score * REGION_REWARD_WEIGHT)

    return rewards


# Critic 5: a single region should contain the subject and match anchor-side geometry.
def spatial_reasoning_reward_func(prompts, completions, **kwargs):
    gt_subjects = kwargs.get("gt_subject_box", [])
    gt_anchors_list = kwargs.get("gt_anchor_boxes", [])
    instructions = kwargs.get("instruction", [])
    rewards = []

    for completion, gt_subject, gt_anchors, instruction in zip(
        completions, gt_subjects, gt_anchors_list, instructions
    ):
        text = extract_text_from_completion(completion)
        regions = parse_role_boxes(text, "region")
        if len(regions) != 1:
            rewards.append(-1.25 * SPATIAL_REWARD_WEIGHT)
            continue
        if not valid_box(gt_subject):
            raise ValueError("invalid gt_subject_box reached spatial reward")

        region = regions[0]
        containment = compute_ioc(gt_subject, region)
        score = containment
        if containment >= 0.8:
            score += 0.50
        elif containment < 0.5:
            score -= 0.75

        # Direction alignment is derived only from relation words present in the instruction.
        dimensions = relation_dimensions(instruction)
        alignments = [
            relation_alignment(region, anchor, gt_subject, dimensions)
            for anchor in (gt_anchors or [])
            if valid_box(anchor)
        ]
        alignments = [value for value in alignments if value is not None]
        if alignments:
            alignment = sum(alignments) / len(alignments)
            score += 0.50 * (2.0 * alignment - 1.0)

        rewards.append(score * SPATIAL_REWARD_WEIGHT)

    return rewards

# ==========================================
# 5. Training entry point
# ==========================================
def validate_adapter_topology(trainer):
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    if not isinstance(unwrapped_model, PeftModel):
        raise RuntimeError("GRPO policy is not a PEFT model")

    adapter_names = set(getattr(unwrapped_model, "peft_config", {}).keys())
    if "default" not in adapter_names or "ref" not in adapter_names:
        raise RuntimeError(
            "shared-LoRA GRPO requires both default(trainable) and ref(frozen) adapters; "
            f"found adapters: {sorted(adapter_names)}"
        )

    reference_mode = "frozen SFT ref adapter" if KL_BETA != 0 else "disabled (beta=0)"

    trainable_names = [
        name for name, parameter in unwrapped_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("the SFT/GRPO shared adapter has no trainable parameters")
    invalid_names = [
        name for name in trainable_names
        if "lora_" not in name or ".default." not in name or ".ref." in name
    ]
    if invalid_names:
        raise RuntimeError(
            "parameters outside the default SFT/GRPO LoRA are trainable: "
            f"{invalid_names[:10]}"
        )
    if not any(".default." in name for name in trainable_names):
        raise RuntimeError("the default SFT/GRPO adapter is not trainable")

    trainable_count = sum(
        parameter.numel()
        for parameter in unwrapped_model.parameters()
        if parameter.requires_grad
    )
    print(
        "Adapter topology check passed: "
        f"continuing from the SFT warmup default LoRA ({trainable_count:,} parameters); "
        f"adapters={sorted(adapter_names)}; KL reference={reference_mode}"
    )


def run_training():
    print("Running GRPO preflight checks...")
    validate_training_config()
    training_args = build_training_args()
    check_critic_server()

    print("Loading and auditing the training dataset...")
    train_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    if DEBUG_SAMPLE_SIZE is not None:
        print(f"[debug] Using only the first {DEBUG_SAMPLE_SIZE} rows...")
        train_dataset = train_dataset.select(range(min(DEBUG_SAMPLE_SIZE, len(train_dataset))))

    original_count = len(train_dataset)
    train_dataset = train_dataset.filter(
        valid_training_example,
        load_from_cache_file=False,
        desc="filtering invalid GRPO samples",
    )
    skipped_count = original_count - len(train_dataset)
    if len(train_dataset) == 0:
        raise RuntimeError("no valid GRPO samples remain after data validation")
    print(
        f"Dataset audit completed: valid={len(train_dataset)}, skipped={skipped_count}. "
        "Requires valid subject/object/region boxes and a pseudo region that fully contains the subject."
    )

    # This map only resolves paths and text. Image decoding stays lazy through Datasets.Image.
    train_dataset = train_dataset.map(
        format_vlm_dataset,
        load_from_cache_file=False,
        desc="building current GRPO prompts",
    )
    train_dataset = train_dataset.cast_column("image", DatasetImage())

    generation_batch_size = (
        PER_DEVICE_BATCH_SIZE
        * effective_gradient_accumulation()
        * distributed_info()[0]
    )
    prompts_per_update = generation_batch_size // NUM_GENERATIONS
    estimated_updates = (len(train_dataset) + prompts_per_update - 1) // prompts_per_update
    print(
        f"Each optimizer update uses {generation_batch_size} completions = "
        f"{prompts_per_update} prompts x {NUM_GENERATIONS} generations; "
        f"estimated {estimated_updates} updates/epoch."
    )

    print(
        f"Policy backend: qwen; model={POLICY_MODEL_ID}"
    )
    print(f"WandB run name: {resolved_wandb_run_name()}")
    processor = load_policy_processor()
    policy_model, peft_config = load_policy_model_for_grpo(processor)

    active_reward_funcs = [
        logic_reward_func,
        entity_grounding_reward_func,
        predicate_region_reward_func,
        spatial_reasoning_reward_func,
    ]
    if USE_VLM_CRITIC:
        active_reward_funcs.insert(0, vlm_plan_reward_func)
        print("External VLM critic enabled.")
    else:
        print("External VLM critic disabled; using local geometry and logic rewards only.")

    print("Initializing the VRT-GRPO trainer...")
    trainer_kwargs = {
        "model": policy_model,
        "reward_funcs": active_reward_funcs,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer_signature = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = processor
    else:
        raise RuntimeError("current GRPOTrainer accepts neither processing_class nor tokenizer")

    trainer = GRPOTrainer(**trainer_kwargs)
    validate_adapter_topology(trainer)

    resume_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT")
    if resume_checkpoint and not os.path.isdir(resume_checkpoint):
        raise FileNotFoundError(
            f"RESUME_FROM_CHECKPOINT does not exist: {resume_checkpoint}"
        )
    if resume_checkpoint:
        output_root = os.path.realpath(OUTPUT_DIR)
        resume_path = os.path.realpath(resume_checkpoint)
        if os.path.commonpath([output_root, resume_path]) != output_root:
            raise RuntimeError(
                "Cannot resume the shared-LoRA GRPO run from an unrelated checkpoint: "
                f"{resume_checkpoint}"
            )

    print("Reward functions are ready. Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint or None)
    trainer.save_model(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        processor.save_pretrained(OUTPUT_DIR)
        resolved_sft_path = (
            find_latest_lora_adapter(SFT_LORA_ADAPTER_PATH)
            if AUTO_FIND_LATEST_SFT_LORA
            else SFT_LORA_ADAPTER_PATH
        )
        composition = {
            "base_model": POLICY_MODEL_ID,
            "policy_backend": "qwen",
            "wandb_run_name": resolved_wandb_run_name(),
            "sft_adapter": resolved_sft_path,
            "continue_training_sft_adapter": True,
            "merge_sft_before_grpo": False,
            "trainable_adapter": "default",
            "reference_adapter": "ref",
            "grpo_adapter": OUTPUT_DIR,
            "grpo_lora_rank": LORA_R,
            "grpo_lora_alpha": LORA_ALPHA,
            "grpo_target_modules": LORA_TARGET_MODULES,
        }
        with open(os.path.join(OUTPUT_DIR, "adapter_stack.json"), "w", encoding="utf-8") as file:
            json.dump(composition, file, ensure_ascii=False, indent=2)
        print(f"Training completed. The shared SFT warmup + GRPO LoRA was saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_training()




# Critic server terminal:

# conda activate qwen3vl_critic_h20

# export HF_ENDPOINT=https://hf-mirror.com
# export TOKENIZERS_PARALLELISM=false

# CUDA_VISIBLE_DEVICES=0 \
# python -m vllm.entrypoints.openai.api_server \
#   --model /root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-30B-A3B-Instruct \
#   --served-model-name qwen3vl-30b-critic \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --trust-remote-code \
#   --dtype bfloat16 \
#   --max-model-len 4096 \
#   --gpu-memory-utilization 0.90



# Training terminals

# Terminal 1:
# ssh -N -4 \
#   -L 127.0.0.1:8000:127.0.0.1:8000 \
#   -o ServerAliveInterval=30 \
#   -o ServerAliveCountMax=3 \
#   -p 34954 \
#   root@region-42.seetacloud.com

# Terminal 2:
# conda activate swift
# cd SPIN

# export CRITIC_API_URL=http://127.0.0.1:8000/v1/chat/completions
# export CRITIC_MODEL_NAME=qwen3vl-30b-critic
# export CRITIC_TIMEOUT=60
# export CRITIC_RETRIES=2
# export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./6-vrt-GRPO.py
