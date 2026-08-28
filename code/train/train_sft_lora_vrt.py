"""Supervised fine-tune a Qwen VRT policy with LoRA on chat-format teacher trajectories."""

import os
import json
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model



# ==========================================
# 0. Paths and knobs
# ==========================================
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_sft_train.jsonl")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_sft_lora")


MAX_STEPS = 200
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # single-GPU setting; auto-scaled under torchrun by default
AUTO_SCALE_GRAD_ACCUMULATION = True
DATALOADER_NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.05
MAX_MODEL_LENGTH = 2048
SAVE_STEPS = 50
LOGGING_STEPS = 1

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

WANDB_PROJECT_NAME = "SPARC-VRT-SFT-qwen2_5"
WANDB_RUN_NAME = "Qwen-8B-VRT-SFT-Warmup-qwen2_5"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed = world_size > 1
    return is_distributed, local_rank, world_size


def effective_gradient_accumulation(world_size):
    if AUTO_SCALE_GRAD_ACCUMULATION and world_size > 1:
        return max(1, GRADIENT_ACCUMULATION // world_size)
    return GRADIENT_ACCUMULATION



# ==========================================
# 1. Collator
# ==========================================
def load_image(path):
    return Image.open(path).convert("RGB")


def collate_fn(examples, processor):
    full_texts = []
    prompt_texts = []
    images = []

    for example in examples:
        messages = example["messages"]
        prompt_messages = messages[:1]

        full_texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
        prompt_texts.append(
            processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        images.append(load_image(example["images"][0]))

    batch = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_MODEL_LENGTH,
    )
    prompt_batch = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_MODEL_LENGTH,
    )

    labels = batch["input_ids"].clone()
    # Mask prompt and padding tokens so LoRA learns only the assistant trajectory.
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :min(int(prompt_len), labels.size(1))] = -100

    batch["labels"] = labels
    return batch


# ==========================================
# 2. Main
# ==========================================
def main():
    is_distributed, local_rank, world_size = distributed_info()
    grad_accumulation = effective_gradient_accumulation(world_size)
    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    print(f"Loading dataset: {TRAIN_JSONL}")
    dataset = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    print(
        f"SFT setup: world_size={world_size}, per_device_batch={PER_DEVICE_BATCH_SIZE}, "
        f"grad_accumulation={grad_accumulation}, effective_batch={world_size * PER_DEVICE_BATCH_SIZE * grad_accumulation}"
    )

    print(f"Loading processor: {BASE_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "right"

    print(f"Loading model: {BASE_MODEL_PATH}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not is_distributed:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        **model_kwargs,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        ddp_find_unused_parameters=False if is_distributed else None,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda examples: collate_fn(examples, processor),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    if trainer.is_world_process_zero():
        processor.save_pretrained(OUTPUT_DIR)
        print(f"Saved SFT LoRA adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



