# VRT-GRPO: Explicit Virtual-Region Reasoning for Multi-Entity Visual Grounding in Remote-Sensing Images

This repository contains the code used to train and evaluate Virtual Region Trajectory (VRT) models on the ME-RSRG benchmark. VRT extends entity-aware remote-sensing grounding with an explicit virtual-region reasoning step and optimizes the resulting structured trajectory through supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

## Contents

- [Overview](#overview)
- [Reproduction Workflow](#reproduction-workflow)
- [Repository Layout](#repository-layout)
- [Model Roles](#model-roles)
- [Environment Setup](#environment-setup)
- [Path Configuration](#path-configuration)
- [Dataset Preparation](#dataset-preparation)
- [Model Download](#model-download)
- [Virtual-Region Pseudo-Labels](#virtual-region-pseudo-labels)
- [SFT Data Construction](#sft-data-construction)
- [SFT Training](#sft-training)
- [GRPO Training](#grpo-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Visualization and Case Filtering](#visualization-and-case-filtering)
- [Generated Files](#generated-files)
- [Reproduction Notes](#reproduction-notes)
- [References](#references)

## Overview

The VRT pipeline represents each prediction as a structured `plan-object-region-subject` trajectory:

1. generate a semantic plan from the language query;
2. localize the reference objects;
3. infer a virtual region that encodes the spatial search constraint;
4. localize the subject within the inferred region; and
5. summarize all predicted coordinates in the final answer.

The repository provides entry points for dataset preprocessing, virtual-region pseudo-label construction, teacher-trajectory generation, SFT data construction, LoRA SFT, GRPO training, inference, grounding evaluation, region evaluation, and qualitative visualization.

## Reproduction Workflow

| Stage | Entry point | Main input | Main output |
| --- | --- | --- | --- |
| Annotation extraction | `code/data/extract_me_rsrg.py` | ME-RSRG archive | Compact train/test JSONL files |
| Image validation | `code/data/filter_existing_images.py` | Extracted JSONL and image root | Samples with valid image paths |
| Region construction | `code/1-6_generate_region_with_internvl.py` | Training JSONL | JSONL with virtual-region pseudo-labels |
| Teacher generation | `code/data/generate_sft_teacher_internvl.py` | Region-annotated JSONL | Placeholder-based teacher trajectories |
| SFT data construction | `code/data/build_vrt_sft_dataset.py` | Teacher trajectories | Chat-format SFT dataset |
| SFT | `code/train/train_sft_lora_vrt.py` | SFT dataset and base model | SFT LoRA adapter |
| GRPO | `code/train/train_grpo_qwen.py` or `code/train/train_grpo_internvl.py` | Region-annotated data and SFT adapter | GRPO LoRA adapter |
| Inference | Scripts under `code/eval/` | Test data and model/adapter | Prediction JSONL |
| Evaluation | `code/eval/evaluate_grounding.py` and `code/eval/evaluate_region.py` | Prediction and ground-truth JSONL | Grounding and region metrics |

## Repository Layout

```text
.
├── README.md
├── requirements.txt
└── code/
    ├── data/
    │   ├── extract_me_rsrg.py
    │   │   # Convert ME-RSRG annotation files to compact JSONL.
    │   ├── filter_existing_images.py
    │   │   # Attach image paths and remove samples with missing images.
    │   ├── generate_sft_teacher_internvl.py
    │   │   # Generate placeholder-based VRT teacher trajectories with InternVL.
    │   └── build_vrt_sft_dataset.py
    │       # Replace placeholders and build final chat-format SFT samples.
    ├── train/
    │   ├── train_sft_lora_vrt.py
    │   │   # Run LoRA supervised fine-tuning for VRT.
    │   ├── train_grpo_qwen.py
    │   │   # Run GRPO for Qwen-VL backbones.
    │   └── train_grpo_internvl.py
    │       # Run GRPO for InternVL backbones.
    ├── eval/
    │   ├── infer_qwen_vrt_prompt.py
    │   │   # Prompt-only VRT inference for Qwen-VL.
    │   ├── infer_qwen_vrt_lora.py
    │   │   # SFT LoRA inference for Qwen-VL.
    │   ├── infer_qwen_vrt_grpo.py
    │   │   # GRPO LoRA inference for Qwen-VL.
    │   ├── infer_internvl_entity_baseline.py
    │   │   # Entity-aware baseline inference for InternVL.
    │   ├── infer_internvl_vrt_lora.py
    │   │   # SFT or GRPO LoRA inference for InternVL.
    │   ├── evaluate_grounding.py
    │   │   # Compute subject/object grounding metrics.
    │   └── evaluate_region.py
    │       # Compute virtual-region metrics.
    ├── tools/
    │   ├── draw_boxes.py
    │   │   # Draw subject, reference-object, and virtual-region boxes.
    │   └── filter_iou_correct_samples.py
    │       # Select samples with correctly localized subjects and objects.
    └── 1-6_generate_region_with_internvl.py
        # Construct geometric virtual-region pseudo-labels and optionally
        # perform InternVL-based anchor-phrase alignment.
```

Some exploratory scripts may remain at the top level of `code/` for traceability. The organized entry points listed above define the intended reproduction path.

> Although `1-6_generate_region_with_internvl.py` contains `internvl` in its filename, its default mode constructs virtual regions geometrically and does not load InternVL. InternVL is used only when entity alignment is explicitly enabled.

## Model Roles

| Role | Model used in the experiments | Required for |
| --- | --- | --- |
| Policy backbone | Qwen3-VL-8B-Instruct | Prompt, SFT, GRPO, and inference |
| Policy backbone | Qwen2.5-VL-7B-Instruct | Prompt, SFT, GRPO, and inference |
| Policy backbone | InternVL3.5-8B | SFT/GRPO and inference |
| SFT teacher | InternVL3-78B | Placeholder-based teacher-trajectory generation |
| External critic | Qwen3-VL-30B critic endpoint | GRPO planning reward |

The teacher and critic are auxiliary models. They are not used during final greedy-decoding evaluation. The external critic receives only the instruction, plan, and execution trajectory; it does not receive the image or evaluate coordinate accuracy.

## Environment Setup

Clone the repository, create a Python environment, and install the dependencies:

```bash
git clone <repository-url>
cd <repository-directory>

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Replace `<repository-url>` and `<repository-directory>` with the actual GitHub URL and local directory name before publishing or running these commands.

FlashAttention is optional and should only be installed when it is compatible with the local CUDA and PyTorch environment:

```bash
pip install flash-attn --no-build-isolation
```

The experiments are designed for CUDA GPUs. The main Qwen and InternVL runs use four GPUs. The InternVL3.5 model card recommends `transformers>=4.52.1`, which is reflected in `requirements.txt`.

## Path Configuration

The scripts use the following AutoDL directory layout by default:

```text
/root/autodl-tmp/SPIN/
├── image/
│   └── RSRG_ME_datasets_ori/
│       ├── dior_rsvg/images/
│       ├── opt_rsvg/images/
│       └── rsvg_hr/images/
├── ME-RSRG-Data/
└── outputs/
```

Files may be stored elsewhere. Override the script defaults through the relevant environment variables.

| Variable | Purpose |
| --- | --- |
| `INPUT_JSONL` | Input annotations or intermediate data |
| `OUTPUT_JSONL` | Generated JSONL output |
| `TRAIN_JSONL` | Chat-format SFT training data |
| `DATA_PATH` | GRPO or inference dataset |
| `GT_JSONL_PATH` | Ground-truth evaluation data |
| `JSONL_PATH` | Prediction file to evaluate or visualize |
| `IMAGE_DIR` / `IMAGE_ROOT` | Root directory containing the images |
| `BASE_MODEL_PATH` / `POLICY_MODEL_ID` | Base policy checkpoint |
| `INTERNVL_MODEL_PATH` | InternVL teacher or alignment checkpoint |
| `SFT_LORA_ADAPTER_PATH` | SFT LoRA adapter |
| `GRPO_LORA_ADAPTER_PATH` / `LORA_ADAPTER_PATH` | GRPO or generic LoRA adapter |
| `OUTPUT_DIR` | Training checkpoints or visualization outputs |
| `CRITIC_API_URL` | OpenAI-compatible critic endpoint |
| `CRITIC_MODEL_NAME` | Model name exposed by the critic server |

Not every script uses every variable. The commands below show the variables required by each stage.

## Dataset Preparation

Download ME-RSRG from either of the following sources:

- [ME-RSRG on Hugging Face](https://huggingface.co/datasets/AlleyOop26/ME-RSRG)
- [Official ME-RSRG repository](https://github.com/CV-ShuchangLyu/ME-RSRG)

ME-RSRG contains 7,162 remote-sensing images and 12,091 image-text instances. The official experimental setting uses 10,305 train/validation instances for training and 1,786 test instances for evaluation.

### 1. Download the dataset

```bash
mkdir -p /root/autodl-tmp/SPIN

huggingface-cli download AlleyOop26/ME-RSRG \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/SPIN/me-rsrg-hf
```

### 2. Extract the images

If the archive is named `ME-RSRG_datasets.zip`, extract it so that the image files appear under `/root/autodl-tmp/SPIN/image/RSRG_ME_datasets_ori/.../images/`:

```bash
mkdir -p /root/autodl-tmp/SPIN/image

unzip /root/autodl-tmp/SPIN/me-rsrg-hf/ME-RSRG_datasets.zip \
  -d /root/autodl-tmp/SPIN/image
```

### 3. Convert the annotations to JSONL

```bash
python code/data/extract_me_rsrg.py \
  --zip-path /root/autodl-tmp/SPIN/me-rsrg-hf/ME-RSRG_datasets.zip \
  --output-dir /root/autodl-tmp/SPIN/ME-RSRG-Data \
  --splits train test train_with_think
```

The official `train_with_think` subset is used as the source pool for SFT teacher construction. The extracted training split is used to construct the GRPO candidate pool; after image-path filtering and virtual-region construction, the reported run uses 8,156 GRPO candidate rows. If the downloaded release exposes `train` and `val` separately, merge them before creating the GRPO pool.

### 4. Attach image paths and remove missing files

SFT teacher data (`train_with_think` subset):

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/train_with_think_cleaned_stage1.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
python code/data/filter_existing_images.py
```

GRPO training pool:

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/train_cleaned_stage1.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/GRPO_cleaned_stage1.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
python code/data/filter_existing_images.py
```

Test data:

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/test_cleaned_stage1.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
python code/data/filter_existing_images.py
```

The implementation reported in the accompanying report uses 2,149 structured-trajectory samples for SFT teacher/SFT construction, 8,156 region-annotated GRPO candidate rows, and 1,786 test samples. The GRPO trainer further filters out rows whose pseudo region does not fully cover the subject.

## Model Download

Download the three policy backbones used in the experiments:

```bash
mkdir -p /root/autodl-tmp/model_cache/modelscope/models

huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct

huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen2.5-VL-7B-Instruct

huggingface-cli download OpenGVLab/InternVL3_5-8B \
  --local-dir /root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master

huggingface-cli download OpenGVLab/InternVL3-78B \
  --local-dir /root/autodl-tmp/model_cache/modelscope/models/OpenGVLab/InternVL3-78B
```

The SFT teacher script uses `OpenGVLab/InternVL3-78B` with local files by default, so `INTERNVL_MODEL_PATH` should point to the downloaded local directory unless the script setting is changed. The GRPO stage requires an OpenAI-compatible critic endpoint. Make sure that the local checkpoint identifiers and the model name exposed by the critic server match the values supplied to the scripts.

## Virtual-Region Pseudo-Labels

Generate geometric virtual-region pseudo-labels for the SFT teacher subset:

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl \
IMAGE_ROOT=/root/autodl-tmp/SPIN/image \
python code/1-6_generate_region_with_internvl.py
```

Generate the same labels for the GRPO training pool:

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/GRPO_cleaned_stage1.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/GRPO_raw.jsonl \
IMAGE_ROOT=/root/autodl-tmp/SPIN/image \
python code/1-6_generate_region_with_internvl.py
```

By default, `RUN_INTERNVL_ENTITY_ALIGNMENT=False`. In this mode, the script constructs virtual regions using deterministic geometry without loading InternVL. To enable anchor-phrase alignment, set `RUN_INTERNVL_ENTITY_ALIGNMENT=True` in the script and provide `INTERNVL_MODEL_PATH`.

## SFT Data Construction

### 1. Generate teacher trajectories

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/teacher_reasoning.jsonl \
IMAGE_ROOT=/root/autodl-tmp/SPIN/image \
INTERNVL_MODEL_PATH=/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab/InternVL3-78B \
python code/data/generate_sft_teacher_internvl.py
```

The teacher produces structured trajectories containing `OBJECT_i`, `REGION`, and `SUBJECT` placeholders rather than directly predicting numeric coordinates.

### 2. Build the final chat-format SFT dataset

```bash
INPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/teacher_reasoning.jsonl \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_sft_train.jsonl \
IMAGE_ROOT=/root/autodl-tmp/SPIN/image \
python code/data/build_vrt_sft_dataset.py
```

The builder validates the teacher output and replaces the placeholders with the corresponding ground-truth reference-object boxes, virtual-region pseudo-label, and ground-truth subject box.

## SFT Training

### Qwen3-VL-8B-Instruct

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_sft_train.jsonl \
BASE_MODEL_PATH=/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_sft_lora \
torchrun --standalone --nproc_per_node=4 code/train/train_sft_lora_vrt.py
```

### Qwen2.5-VL-7B-Instruct

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRAIN_JSONL=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_sft_train.jsonl \
BASE_MODEL_PATH=/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=/root/autodl-tmp/SPIN/outputs/output_qwen2_5/vrt_sft_lora \
torchrun --standalone --nproc_per_node=4 code/train/train_sft_lora_vrt.py
```

The later InternVL GRPO and LoRA-inference commands assume that an InternVL SFT adapter has already been prepared and saved at the path supplied through `SFT_LORA_ADAPTER_PATH` or `LORA_ADAPTER_PATH`. For InternVL SFT, use the same chat-format SFT data with an InternVL-compatible training stack; `code/run_swift_sft_with_internvl_patch.py` is provided as a compatibility launcher for ms-swift SFT.

### Main SFT hyperparameters

| Parameter | Value |
| --- | --- |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | `1e-4` |
| Max steps | `200` |
| Per-device batch size | `1` |
| Gradient accumulation | `16`, automatically scaled under `torchrun` |
| Max model length | `2048` |

## GRPO Training

GRPO continues from an SFT LoRA adapter and uses an OpenAI-compatible external critic for the planning reward.

### 1. Start the critic server

For example, using vLLM. Install vLLM separately if it is not already available:

```bash
vllm serve Qwen/Qwen3-VL-30B-Instruct \
  --served-model-name qwen3vl-30b-critic \
  --host 127.0.0.1 \
  --port 8000
```

The value of `CRITIC_MODEL_NAME` must match a model name accepted by the server. If the server exposes a different identifier, update the environment variable in the training commands.

### 2. Train Qwen-VL with GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/GRPO_raw.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
POLICY_MODEL_ID=/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct \
SFT_LORA_ADAPTER_PATH=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_sft_lora \
OUTPUT_DIR=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_grpo \
CRITIC_API_URL=http://127.0.0.1:8000/v1/chat/completions \
CRITIC_MODEL_NAME=qwen3vl-30b-critic \
torchrun --standalone --nproc_per_node=4 code/train/train_grpo_qwen.py
```

For Qwen2.5-VL, change `POLICY_MODEL_ID`, `SFT_LORA_ADAPTER_PATH`, and `OUTPUT_DIR` to the corresponding Qwen2.5-VL paths.

### 3. Train InternVL with GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/GRPO_raw.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
POLICY_MODEL_ID=/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master \
SFT_LORA_ADAPTER_PATH=/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_sft_lora \
OUTPUT_DIR=/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_grpo \
CRITIC_API_URL=http://127.0.0.1:8000/v1/chat/completions \
CRITIC_MODEL_NAME=qwen3vl-30b-critic \
torchrun --standalone --nproc_per_node=4 code/train/train_grpo_internvl.py
```

### Main GRPO hyperparameters

| Parameter | Value |
| --- | --- |
| Number of generations per prompt | `8` |
| Learning rate | `1e-5` |
| Epochs | `1` |
| Per-device batch size | `1` |
| Target generation batch size | `32` |
| Max completion length | `512` |
| Warmup ratio | `0.01` |
| KL beta | `0.0025` |
| Generation temperature | `0.8` |
| Top-p | `0.95` |
| Plan reward weight | `0.15` |
| Format/logic reward weight | `0.25` |
| Entity reward weight | `1.00` |
| Region reward weight | `0.25` |
| Spatial reward weight | `0.20` |

## Inference

All VRT evaluation runs use the same processed test set. The examples below use the Qwen3-VL paths; replace the model and adapter paths to evaluate Qwen2.5-VL.

### Prompt-only Qwen-VL

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
BASE_MODEL_PATH=/root/autodl-tmp/model_cache/modelscope/models/Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_prompt.jsonl \
torchrun --standalone --nproc_per_node=4 code/eval/infer_qwen_vrt_prompt.py
```

### Qwen-VL after SFT

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
SFT_LORA_ADAPTER_PATH=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_sft_lora \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_sft.jsonl \
torchrun --standalone --nproc_per_node=4 code/eval/infer_qwen_vrt_lora.py
```

### Qwen-VL after GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
GRPO_LORA_ADAPTER_PATH=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_grpo \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl \
torchrun --standalone --nproc_per_node=4 code/eval/infer_qwen_vrt_grpo.py
```

### InternVL LoRA inference

The same entry point accepts an SFT or GRPO adapter through `LORA_ADAPTER_PATH`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_DIR=/root/autodl-tmp/SPIN/image \
LORA_ADAPTER_PATH=/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_grpo \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/vrt_test_grpo.jsonl \
torchrun --standalone --nproc_per_node=4 code/eval/infer_internvl_vrt_lora.py
```

### InternVL entity-aware baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_ROOT=/root/autodl-tmp/SPIN/image \
START_INDEX=0 \
BASE_MODEL_PATH=/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master \
OUTPUT_JSONL=/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/entity_test.jsonl \
torchrun --standalone --nproc_per_node=4 code/eval/infer_internvl_entity_baseline.py
```

## Evaluation

### Subject/object grounding metrics

```bash
JSONL_PATH=/root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl \
GT_JSONL_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl \
IMAGE_ROOT=/root/autodl-tmp/SPIN/image \
python code/eval/evaluate_grounding.py
```

The grounding evaluator reports subject accuracy, reference-object accuracy, and their mean at the configured IoU threshold.

### Virtual-region metrics

```bash
python code/eval/evaluate_region.py
```

The region evaluator reports:

- valid-region rate;
- subject coverage at the configured thresholds;
- strict subject coverage; and
- cover+direction consistency.

If the prediction filenames differ from the defaults, update `JSONL_PATHS` at the top of `evaluate_region.py` before running the command.

## Visualization and Case Filtering

### Draw subject, reference-object, and virtual-region boxes

```bash
JSONL_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/cleaned_stage1_with_regions.jsonl \
OUTPUT_DIR=/root/autodl-tmp/SPIN/figures \
IMAGE_IDS=695131_4318256_3072_32615_sport_baseball \
python code/tools/draw_boxes.py
```

### Overlay a dashed region from a second JSONL file

```bash
DRAW_DASHED_REGION=1 \
DASHED_REGION_JSONL_PATH=/root/autodl-tmp/SPIN/ME-RSRG-Data/center_pass_regions.jsonl \
python code/tools/draw_boxes.py
```

### Filter correctly grounded qualitative cases

The following command retains samples for which the subject and all reference objects are correct at IoU 0.5:

```bash
python code/tools/filter_iou_correct_samples.py \
  --input /root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo.jsonl \
  --output /root/autodl-tmp/SPIN/outputs/output_qwen3/vrt_test_grpo_correct.jsonl \
  --threshold 0.5
```

## Generated Files

| File or directory | Description |
| --- | --- |
| `train_cleaned_stage1.jsonl` | Extracted training annotations |
| `train_with_think_cleaned_stage1.jsonl` | Extracted official train-with-think subset |
| `test_cleaned_stage1.jsonl` | Extracted test annotations |
| `cleaned_stage1.jsonl` | SFT teacher subset with valid image paths |
| `GRPO_cleaned_stage1.jsonl` | GRPO candidate pool with valid image paths |
| `vrt_eval_test.jsonl` | Processed official test set |
| `cleaned_stage1_with_regions.jsonl` | SFT teacher subset with virtual-region pseudo-labels |
| `GRPO_raw.jsonl` | GRPO candidate pool with virtual-region pseudo-labels |
| `teacher_reasoning.jsonl` | Placeholder-based teacher trajectories |
| `vrt_sft_train.jsonl` | Final chat-format SFT dataset |
| `vrt_sft_lora/` | SFT LoRA adapter and checkpoints |
| `vrt_grpo/` | GRPO LoRA adapter and checkpoints |
| `vrt_test_prompt.jsonl` | Prompt-only VRT predictions |
| `vrt_test_sft.jsonl` | SFT VRT predictions |
| `vrt_test_grpo.jsonl` | GRPO VRT predictions |
| `figures/` | Qualitative visualizations |

Large model checkpoints, LoRA adapters, generated datasets, and prediction files should normally be excluded from regular Git tracking. Use Git LFS or external storage when these artifacts must be distributed.

## Reproduction Notes

- Most scripts expose commonly modified path and training settings through environment variables or constants near the top of the file.
- The `*_prompt.py`, `*_sft.py`, and `*_grpo.py` outputs are designed to remain compatible with the same evaluation scripts.
- Use identical data splits, model backbones, decoding settings, and training budgets when comparing the entity-aware baseline and VRT variants.
- VRT test inference uses greedy decoding with `do_sample=False` and a maximum generation length of 2,048 tokens.
- The entity-aware baseline uses a maximum generation length of 1,024 tokens.
- GRPO uses eight generations per prompt and a target generation batch size of 32.
- The external critic evaluates text-level planning and execution consistency only; it does not receive the image.
- Before publishing the repository, replace the clone-command placeholders and verify that all model identifiers match the checkpoints exposed in the target environment.

## References

- [ME-RSRG dataset and EAR framework](https://github.com/CV-ShuchangLyu/ME-RSRG)
- [ME-RSRG dataset on Hugging Face](https://huggingface.co/datasets/AlleyOop26/ME-RSRG)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)

