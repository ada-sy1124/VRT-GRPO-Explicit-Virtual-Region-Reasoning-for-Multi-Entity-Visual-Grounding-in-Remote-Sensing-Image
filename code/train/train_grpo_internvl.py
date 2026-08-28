"""Train an InternVL policy with GRPO using VRT rewards and an SFT LoRA warm start."""

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
from transformers import AutoModel, AutoProcessor
from transformers.modeling_utils import PreTrainedModel
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import (
    entropy_from_logits,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_tensor_dict,
)
from peft import PeftModel


_TRL_LOGP_PARAMETERS = inspect.signature(
    GRPOTrainer._get_per_token_logps_and_entropies
).parameters
_TRL_LOGP_RETURNS_AUX_LOSS = "compute_aux_loss" in _TRL_LOGP_PARAMETERS


# ==========================================
# 0. Global hyperparameters and config
# ==========================================
# Paths
DATA_PATH = os.environ.get("DATA_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/autodl-tmp/SPIN/image")
POLICY_MODEL_ID = os.environ.get("POLICY_MODEL_ID", "/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_grpo")
SFT_LORA_ADAPTER_PATH = os.environ.get("SFT_LORA_ADAPTER_PATH", "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_sft_lora")

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
    "INTERNVL_WANDB_RUN_NAME",
    "InternVL3.5-8B-VRT-GRPO-SFTInit-SharedLoRA-r16-G8-H20x4-Final",
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


def patch_transformers_for_internvl():
    """Patch transformer compatibility gaps seen with InternVL3.5 in this env."""
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
    """Expose InternVL visual token attributes on Qwen2 tokenizer classes."""
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
    if "<IMG_CONTEXT>" not in prompt_text:
        prompt_text = "<IMG_CONTEXT>\n" + prompt_text
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
    patch_transformers_for_internvl()
    patch_qwen2_tokenizer_for_internvl()
    print(f"Loading InternVL processor: {POLICY_MODEL_ID}")
    try:
        processor = AutoProcessor.from_pretrained(
            POLICY_MODEL_ID,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    except TypeError:
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
        raise RuntimeError("cannot find img_context_token_id on InternVL policy model")
    print(f"InternVL img_context_token_id set: {context_token} -> {int(token_id)}")


def register_internvl_vision_dtype_guard(model):
    """Align InternVL image tensors at the vision tower's actual call boundary."""
    vision_model = getattr(model, "vision_model", None)
    if not isinstance(vision_model, torch.nn.Module):
        raise RuntimeError("cannot find vision_model on InternVL policy model")

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

    vision_model.register_forward_pre_hook(
        align_pixel_values_dtype,
        with_kwargs=True,
    )
    print(f"InternVL vision-input dtype guard registered: pixel_values -> {vision_dtype}")


def patch_internvl_generate_contract_for_trl(model):
    """Make InternVL's generated-only output match TRL's prompt-plus-output contract."""
    original_generate = model.generate
    normalization_reported = False

    def generate_compat(self, *args, **kwargs):
        nonlocal normalization_reported
        del self
        input_ids = kwargs.get("input_ids")
        outputs = original_generate(*args, **kwargs)
        if input_ids is None:
            return outputs

        sequences = outputs if torch.is_tensor(outputs) else getattr(outputs, "sequences", None)
        if not torch.is_tensor(sequences) or sequences.ndim != 2 or input_ids.ndim != 2:
            raise RuntimeError(
                "InternVL generate returned an unsupported output shape for TRL: "
                f"input_ids={getattr(input_ids, 'shape', None)}, "
                f"sequences={getattr(sequences, 'shape', None)}"
            )
        if sequences.size(0) % input_ids.size(0) != 0:
            raise RuntimeError(
                "InternVL generate batch cannot be aligned with TRL prompts: "
                f"prompts={input_ids.size(0)}, returned_sequences={sequences.size(0)}"
            )

        num_return_sequences = sequences.size(0) // input_ids.size(0)
        expanded_prompt_ids = input_ids.repeat_interleave(num_return_sequences, dim=0).to(sequences.device)
        prompt_length = expanded_prompt_ids.size(1)
        includes_prompt = (
            sequences.size(1) >= prompt_length
            and torch.equal(sequences[:, :prompt_length], expanded_prompt_ids)
        )
        if includes_prompt:
            return outputs
        if sequences.size(1) == 0:
            raise RuntimeError("InternVL generate returned zero completion tokens before TRL slicing")

        normalized_sequences = torch.cat([expanded_prompt_ids, sequences], dim=1)
        if torch.is_tensor(outputs):
            outputs = normalized_sequences
        else:
            outputs.sequences = normalized_sequences

        if not normalization_reported:
            print(
                "InternVL generate output aligned with TRL: "
                f"prompt_tokens={prompt_length}, completion_tokens={sequences.size(1)}, "
                f"num_return_sequences={num_return_sequences}"
            )
            normalization_reported = True
        return outputs

    model.generate = types.MethodType(generate_compat, model)
    print("InternVL generate/TRL contract patch registered")


def patch_internvl_forward_contract_for_peft(model):
    """Drop PEFT's unsupported empty inputs_embeds argument at the InternVL boundary."""
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
    print("InternVL forward/PEFT contract patch registered: dropping inputs_embeds=None")


def find_internvl_chat_module(model):
    """Resolve the remote-code InternVL wrapper through PEFT/DDP containers."""
    for module in model.modules():
        if (
            isinstance(getattr(module, "num_image_token", None), int)
            and hasattr(module, "img_context_token_id")
            and isinstance(getattr(module, "vision_model", None), torch.nn.Module)
        ):
            return module
    raise RuntimeError(
        "cannot find the InternVL chat module carrying num_image_token, "
        "img_context_token_id, and vision_model"
    )


class InternVLGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with InternVL-aware dynamic image-tile forwarding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        internvl_module = find_internvl_chat_module(self.model)
        self._internvl_num_image_token = int(internvl_module.num_image_token)
        self._internvl_img_context_token_id = int(internvl_module.img_context_token_id)
        self._internvl_logp_contract_reported = False
        self._internvl_buffer_contract_reported = False

        if self._internvl_num_image_token <= 0:
            raise RuntimeError(
                f"invalid InternVL num_image_token: {self._internvl_num_image_token}"
            )
        if self._internvl_img_context_token_id < 0:
            raise RuntimeError(
                "InternVL img_context_token_id was not initialized before trainer creation"
            )

    def _internvl_tile_counts_from_prompt_ids(self, prompt_ids):
        if not torch.is_tensor(prompt_ids) or prompt_ids.ndim != 2:
            raise RuntimeError(
                "InternVL dynamic-tile grouping expects 2D prompt_ids, "
                f"got {getattr(prompt_ids, 'shape', None)}"
            )

        context_counts = (prompt_ids == self._internvl_img_context_token_id).sum(dim=1)
        remainders = context_counts.remainder(self._internvl_num_image_token)
        if torch.any(context_counts == 0) or torch.any(remainders != 0):
            raise RuntimeError(
                "InternVL prompt/image contract mismatch: each sample must contain a positive "
                f"multiple of {self._internvl_num_image_token} <IMG_CONTEXT> tokens; "
                f"got counts={context_counts.detach().cpu().tolist()}"
            )

        tile_counts = torch.div(
            context_counts,
            self._internvl_num_image_token,
            rounding_mode="floor",
        ).detach().cpu().tolist()
        return [int(value) for value in tile_counts]

    def _internvl_group_dynamic_tiles(self, generation_batch):
        """Convert flattened InternVL tile tensors into sample-aligned lists."""
        prompt_ids = generation_batch.get("prompt_ids")
        pixel_values = generation_batch.get("pixel_values")
        if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
            raise RuntimeError(
                "InternVL generation output must contain flattened pixel_values shaped "
                f"[tiles, C, H, W], got {getattr(pixel_values, 'shape', None)}"
            )

        tile_counts = self._internvl_tile_counts_from_prompt_ids(prompt_ids)
        total_tiles = sum(tile_counts)
        if pixel_values.size(0) != total_tiles:
            raise RuntimeError(
                "InternVL generation buffer mismatch before TRL slicing: prompt tokens require "
                f"{total_tiles} tiles ({tile_counts} per sample), but pixel_values contains "
                f"{pixel_values.size(0)} tiles."
            )

        grouped_batch = dict(generation_batch)
        grouped_batch["pixel_values"] = list(
            torch.split(pixel_values, tile_counts, dim=0)
        )

        image_flags = grouped_batch.get("image_flags")
        if image_flags is not None:
            if not torch.is_tensor(image_flags) or image_flags.size(0) != total_tiles:
                raise RuntimeError(
                    "InternVL image_flags must remain tile-aligned before buffering: "
                    f"flags={getattr(image_flags, 'shape', None)}, total_tiles={total_tiles}"
                )
            grouped_batch["image_flags"] = list(
                torch.split(image_flags, tile_counts, dim=0)
            )

        existing_num_tiles = grouped_batch.get("num_tiles")
        if existing_num_tiles is not None:
            if torch.is_tensor(existing_num_tiles):
                existing_num_tiles = existing_num_tiles.detach().cpu().tolist()
            else:
                existing_num_tiles = list(existing_num_tiles)
            if [int(value) for value in existing_num_tiles] != tile_counts:
                raise RuntimeError(
                    "InternVL num_tiles disagrees with <IMG_CONTEXT> tokens: "
                    f"num_tiles={existing_num_tiles}, inferred={tile_counts}"
                )
        grouped_batch["num_tiles"] = tile_counts
        return grouped_batch

    @staticmethod
    def _internvl_ungroup_dynamic_tiles(batch):
        """Restore one buffered micro-batch to InternVL's flattened tile format."""
        restored_batch = dict(batch)
        for key in ("pixel_values", "image_flags"):
            value = restored_batch.get(key)
            if isinstance(value, list):
                if not value:
                    raise RuntimeError(f"InternVL buffered {key} is unexpectedly empty")
                restored_batch[key] = torch.cat(value, dim=0)
        return restored_batch

    def _internvl_validate_buffered_batch(self, batch):
        prompt_ids = batch.get("prompt_ids")
        pixel_values = batch.get("pixel_values")
        tile_counts = self._internvl_tile_counts_from_prompt_ids(prompt_ids)
        total_tiles = sum(tile_counts)
        if (
            not torch.is_tensor(pixel_values)
            or pixel_values.ndim != 4
            or pixel_values.size(0) != total_tiles
        ):
            raise RuntimeError(
                "InternVL buffered micro-batch lost image tiles: prompt tokens require "
                f"{total_tiles} tiles ({tile_counts} per sample), got "
                f"{getattr(pixel_values, 'shape', None)}"
            )

        num_tiles = batch.get("num_tiles")
        if num_tiles is not None:
            if torch.is_tensor(num_tiles):
                num_tiles = num_tiles.detach().cpu().tolist()
            else:
                num_tiles = list(num_tiles)
            if [int(value) for value in num_tiles] != tile_counts:
                raise RuntimeError(
                    "InternVL buffered num_tiles no longer matches its prompt: "
                    f"num_tiles={num_tiles}, inferred={tile_counts}"
                )
        return tile_counts

    def _internvl_build_buffered_inputs(self, generation_batch):
        grouped_batch = self._internvl_group_dynamic_tiles(generation_batch)
        grouped_batch = shuffle_sequence_dict(grouped_batch)

        num_chunks = int(self.args.steps_per_generation)
        batch_size = int(grouped_batch["prompt_ids"].size(0))
        if num_chunks <= 0 or batch_size % num_chunks != 0:
            raise RuntimeError(
                "InternVL generation batch cannot be divided into optimizer micro-batches: "
                f"batch_size={batch_size}, steps_per_generation={num_chunks}"
            )

        chunks = split_tensor_dict(grouped_batch, num_chunks)
        buffered_inputs = [
            self._internvl_ungroup_dynamic_tiles(chunk)
            for chunk in chunks
        ]
        tile_layouts = [
            self._internvl_validate_buffered_batch(batch)
            for batch in buffered_inputs
        ]
        if not self._internvl_buffer_contract_reported:
            if getattr(self.accelerator, "is_main_process", True):
                print(
                    "InternVL generation buffer contract check passed: "
                    f"generation_samples={batch_size}, micro_batches={num_chunks}, "
                    f"tiles_per_micro_batch={tile_layouts}"
                )
            self._internvl_buffer_contract_reported = True
        return buffered_inputs

    def _prepare_inputs(self, generation_batch):
        """Preserve complete dynamic-tile groups across TRL's rollout buffer."""
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = self._internvl_build_buffered_inputs(
                    generation_batch
                )
            return self._buffered_inputs[
                self._step % self.args.steps_per_generation
            ]

        inputs = self._generate_and_score_completions(generation_batch)
        self._internvl_validate_buffered_batch(inputs)
        return inputs

    def _internvl_tile_layout(self, input_ids, logits_to_keep, pixel_values):
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise RuntimeError(
                f"InternVL log-prob expects 2D input_ids, got {getattr(input_ids, 'shape', None)}"
            )
        if not torch.is_tensor(pixel_values) or pixel_values.ndim != 4:
            raise RuntimeError(
                "InternVL log-prob expects flattened image tiles shaped [tiles, C, H, W], "
                f"got {getattr(pixel_values, 'shape', None)}"
            )

        completion_length = int(logits_to_keep)
        prompt_length = input_ids.size(1) - completion_length
        if prompt_length <= 0:
            raise RuntimeError(
                "InternVL log-prob cannot recover the prompt prefix: "
                f"sequence_length={input_ids.size(1)}, logits_to_keep={completion_length}"
            )

        prompt_ids = input_ids[:, :prompt_length]
        tile_counts = self._internvl_tile_counts_from_prompt_ids(prompt_ids)
        total_tiles = sum(tile_counts)
        if pixel_values.size(0) != total_tiles:
            raise RuntimeError(
                "InternVL dynamic-tile batch mismatch: prompt tokens require "
                f"{total_tiles} tiles ({tile_counts} per sample), but processor returned "
                f"{pixel_values.size(0)} tiles. Generic TRL batch slicing is invalid for InternVL."
            )

        boundaries = [0]
        for tile_count in tile_counts:
            boundaries.append(boundaries[-1] + tile_count)
        return tile_counts, boundaries

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        num_tiles=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
        image_flags=None,
        **extra_model_kwargs,
    ):
        del (
            image_grid_thw,
            num_images,
            pixel_attention_mask,
            spatial_shapes,
            num_tiles,
            image_sizes,
            token_type_ids,
            mm_token_type_ids,
            image_position_ids,
            extra_model_kwargs,
        )
        if pixel_values is None:
            raise RuntimeError(
                "InternVL GRPO log-prob received no pixel_values. The image column was lost "
                "between generation and the policy/reference forward pass."
            )
        if compute_aux_loss:
            raise RuntimeError("InternVL remote-code model does not expose a GRPO MoE auxiliary loss")

        tile_counts, tile_boundaries = self._internvl_tile_layout(
            input_ids,
            logits_to_keep,
            pixel_values,
        )
        if image_flags is None:
            image_flags = torch.ones(
                (pixel_values.size(0), 1),
                dtype=torch.long,
                device=pixel_values.device,
            )
        else:
            if not torch.is_tensor(image_flags) or image_flags.numel() != pixel_values.size(0):
                raise RuntimeError(
                    "InternVL image_flags must contain one value per image tile: "
                    f"flags={getattr(image_flags, 'shape', None)}, tiles={pixel_values.size(0)}"
                )
            image_flags = image_flags.reshape(-1, 1).to(
                device=pixel_values.device,
                dtype=torch.long,
            )
            if torch.any(image_flags != 1):
                raise RuntimeError(
                    "This GRPO dataset contains real images only, so every InternVL image flag must be 1"
                )

        effective_batch_size = int(batch_size or input_ids.size(0))
        if effective_batch_size <= 0:
            raise RuntimeError(f"invalid log-prob batch_size: {effective_batch_size}")

        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), effective_batch_size):
            end = min(start + effective_batch_size, input_ids.size(0))
            tile_start = tile_boundaries[start]
            tile_end = tile_boundaries[end]
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            model_inputs = {
                "pixel_values": pixel_values[tile_start:tile_end],
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "image_flags": image_flags[tile_start:tile_end],
                "use_cache": False,
            }

            outputs = model(**model_inputs)
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -int(logits_to_keep):, :] / self.temperature
            completion_ids = input_ids_batch[:, -int(logits_to_keep):]
            all_logps.append(selective_log_softmax(logits, completion_ids))

            if compute_entropy:
                if getattr(self, "_entropy_bonus_enabled", False):
                    entropies = entropy_from_logits(logits)
                else:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        if not self._internvl_logp_contract_reported:
            if getattr(self.accelerator, "is_main_process", True):
                print(
                    "InternVL GRPO log-prob contract check passed: "
                    f"tiles_per_sample={tile_counts}, total_tiles={pixel_values.size(0)}, "
                    "image_flags covered ref/policy forward"
                )
            self._internvl_logp_contract_reported = True

        if _TRL_LOGP_RETURNS_AUX_LOSS:
            return logps, entropies, None
        return logps, entropies


def validate_internvl_runtime_contract(trainer, processor):
    if not isinstance(trainer, InternVLGRPOTrainer):
        raise RuntimeError("InternVL training must use InternVLGRPOTrainer")
    if "pixel_values" not in _TRL_LOGP_PARAMETERS:
        raise RuntimeError(
            "The installed TRL GRPOTrainer has no multimodal pixel_values log-prob path"
        )

    processor_image_tokens = int(getattr(processor, "image_seq_length", -1))
    if processor_image_tokens != trainer._internvl_num_image_token:
        raise RuntimeError(
            "InternVL processor/model image token length mismatch: "
            f"processor={processor_image_tokens}, model={trainer._internvl_num_image_token}"
        )

    tokenizer = getattr(processor, "tokenizer", processor)
    processor_context_id = int(getattr(tokenizer, "context_image_token_id", -1))
    if processor_context_id != trainer._internvl_img_context_token_id:
        raise RuntimeError(
            "InternVL processor/model context token mismatch: "
            f"processor={processor_context_id}, model={trainer._internvl_img_context_token_id}"
        )

    return_arity = 3 if _TRL_LOGP_RETURNS_AUX_LOSS else 2
    print(
        "InternVL/TRL runtime contract preflight passed: "
        f"context_token_id={processor_context_id}, tokens_per_tile={processor_image_tokens}, "
        f"TRL_logp_return_arity={return_arity}"
    )


def run_internvl_forward_preflight(trainer, processor, example):
    """Exercise both InternVL forward and TRL rollout-buffer paths."""
    image = example.get("image")
    prompt = example.get("prompt")
    if image is None or not isinstance(prompt, list):
        raise RuntimeError("InternVL forward preflight requires one decoded image and one chat prompt")

    prompt_text = processor.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = processor(
        images=[image],
        text=[prompt_text],
        padding=True,
        return_tensors="pt",
    )
    required_keys = {"input_ids", "attention_mask", "pixel_values"}
    missing_keys = sorted(required_keys - set(model_inputs.keys()))
    if missing_keys:
        raise RuntimeError(
            f"InternVL processor omitted required preflight tensors: {missing_keys}"
        )

    model_device = next(
        parameter.device
        for parameter in trainer.model.parameters()
        if parameter.device.type != "meta"
    )
    prompt_ids = model_inputs["input_ids"].to(model_device)
    prompt_attention_mask = model_inputs["attention_mask"].to(model_device)
    pixel_values = model_inputs["pixel_values"].to(model_device)
    eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)
    if eos_token_id is None:
        raise RuntimeError("InternVL tokenizer has no eos_token_id for forward preflight")

    completion_token = prompt_ids.new_full(
        (prompt_ids.size(0), 1),
        int(eos_token_id),
    )
    completion_attention = prompt_attention_mask.new_ones(
        (prompt_attention_mask.size(0), 1)
    )
    input_ids = torch.cat([prompt_ids, completion_token], dim=1)
    attention_mask = torch.cat(
        [prompt_attention_mask, completion_attention],
        dim=1,
    )

    with torch.inference_mode():
        result = trainer._get_per_token_logps_and_entropies(
            trainer.model,
            input_ids,
            attention_mask,
            logits_to_keep=1,
            batch_size=1,
            compute_entropy=False,
            pixel_values=pixel_values,
        )
    logps = result[0]
    if logps.shape != (1, 1) or not torch.isfinite(logps).all():
        raise RuntimeError(
            "InternVL forward preflight returned invalid log-probs: "
            f"shape={tuple(logps.shape)}, finite={bool(torch.isfinite(logps).all())}"
        )
    print(
        "InternVL real-sample forward preflight passed: "
        f"prompt_tokens={input_ids.size(1) - 1}, image_tiles={pixel_values.size(0)}"
    )

    steps_per_generation = int(trainer.args.steps_per_generation)
    synthetic_batch = {
        "prompt_ids": prompt_ids.repeat(steps_per_generation, 1),
        "prompt_mask": prompt_attention_mask.repeat(steps_per_generation, 1),
        "completion_ids": completion_token.repeat(steps_per_generation, 1),
        "completion_mask": completion_attention.repeat(steps_per_generation, 1),
        "advantages": torch.zeros(steps_per_generation, device=model_device),
        "ref_per_token_logps": torch.zeros(
            (steps_per_generation, 1),
            device=model_device,
        ),
        "pixel_values": pixel_values.repeat(steps_per_generation, 1, 1, 1),
        "num_images": [1] * steps_per_generation,
        "num_items_in_batch": torch.tensor(
            steps_per_generation,
            device=model_device,
        ),
    }

    generate_method_name = "_generate_and_score_completions"
    had_instance_generate = generate_method_name in trainer.__dict__
    original_instance_generate = trainer.__dict__.get(generate_method_name)
    original_buffer = trainer._buffered_inputs
    original_step = trainer._step
    original_buffer_reported = trainer._internvl_buffer_contract_reported
    model_was_training = trainer.model.training
    cpu_rng_state = torch.random.get_rng_state()

    def return_synthetic_batch(self, generation_batch):
        del self, generation_batch
        return dict(synthetic_batch)

    try:
        trainer._generate_and_score_completions = types.MethodType(
            return_synthetic_batch,
            trainer,
        )
        trainer._buffered_inputs = None
        trainer._step = 0
        trainer.model.train()
        selected_batch = trainer._prepare_inputs([None] * steps_per_generation)
        buffered_inputs = trainer._buffered_inputs
    finally:
        if had_instance_generate:
            trainer.__dict__[generate_method_name] = original_instance_generate
        else:
            trainer.__dict__.pop(generate_method_name, None)
        trainer._buffered_inputs = original_buffer
        trainer._step = original_step
        trainer._internvl_buffer_contract_reported = original_buffer_reported
        torch.random.set_rng_state(cpu_rng_state)
        if not model_was_training:
            trainer.model.eval()

    if len(buffered_inputs) != steps_per_generation:
        raise RuntimeError(
            "InternVL buffer preflight created the wrong number of micro-batches: "
            f"expected={steps_per_generation}, actual={len(buffered_inputs)}"
        )
    expected_tiles = int(pixel_values.size(0))
    for index, buffered_batch in enumerate(buffered_inputs):
        tile_counts = trainer._internvl_validate_buffered_batch(buffered_batch)
        if tile_counts != [expected_tiles]:
            raise RuntimeError(
                "InternVL buffer preflight did not preserve one full image per micro-batch: "
                f"micro_batch={index}, expected={[expected_tiles]}, actual={tile_counts}"
            )
    if selected_batch is not buffered_inputs[0]:
        raise RuntimeError("InternVL buffer preflight selected an unexpected first micro-batch")
    print(
        "InternVL policy-loss cache preflight passed: "
        f"{steps_per_generation * expected_tiles} tiles -> "
        f"{steps_per_generation} micro-batches x {expected_tiles} tiles"
    )


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
        low_cpu_mem_usage=True,
    )
    patch_transformers_for_internvl()
    patch_qwen2_tokenizer_for_internvl()
    print(f"Loading InternVL policy model: {POLICY_MODEL_ID}")
    if world_size == 1:
        model_kwargs["device_map"] = "auto"

    try:
        base_model = AutoModel.from_pretrained(
            POLICY_MODEL_ID,
            **model_kwargs,
        )
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        base_model = AutoModel.from_pretrained(
            POLICY_MODEL_ID,
            **model_kwargs,
        )
    base_model.config.use_cache = False
    patch_internvl_forward_contract_for_peft(base_model)
    register_internvl_vision_dtype_guard(base_model)
    set_internvl_img_context_token_id(base_model, processor)

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
    set_internvl_img_context_token_id(policy_model, processor)
    patch_internvl_generate_contract_for_trl(policy_model)

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
        f"Policy backend: internvl; model={POLICY_MODEL_ID}"
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

    trainer = InternVLGRPOTrainer(**trainer_kwargs)
    validate_adapter_topology(trainer)
    validate_internvl_runtime_contract(trainer, processor)
    run_internvl_forward_preflight(trainer, processor, train_dataset[0])

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
            "policy_backend": "internvl",
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
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./6-vrt-GRPO-InternVL.py


# export RESUME_FROM_CHECKPOINT=/root/autodl-tmp/SPIN/output_InternVL3_5/vrt_grpo_old/checkpoint-1000
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./6-vrt-GRPO-InternVL.py
