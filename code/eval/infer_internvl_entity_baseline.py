"""Run InternVL entity-aware baseline inference on the ME-RSRG test split."""

import glob
import inspect
import json
import os
import re
import types

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from transformers.modeling_utils import PreTrainedModel



# ==========================================
# 0. Paths and knobs
# ==========================================
DATA_PATH = os.environ.get("DATA_PATH", "/root/autodl-tmp/SPIN/ME-RSRG-Data/vrt_eval_test.jsonl")
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "/root/autodl-tmp/SPIN/image")
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/root/autodl-tmp/model_cache/modelscope/models/OpenGVLab--InternVL3_5-8B/snapshots/master")
OUTPUT_JSONL = os.environ.get("OUTPUT_JSONL", "/root/autodl-tmp/SPIN/outputs/output_InternVL3_5/baseline_test.jsonl")

# START_INDEX = int(os.environ.get("START_INDEX", "10"))
# NUM_SAMPLES_TEXT = os.environ.get("NUM_SAMPLES", "").strip()
# NUM_SAMPLES = int(NUM_SAMPLES_TEXT) if NUM_SAMPLES_TEXT else None
# MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
# TORCH_DTYPE = torch.bfloat16

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# # ==========================================
# # 1. Prompt
# # ==========================================
# PROMPT_TEMPLATE = (
#     "Localize the subject and objects in the image with description: {instruction}. "
#     "Generate a step-by-step reasoning process in <think></think> tags and output the "
#     "box coordinates of one subject and one or more objects in the format of "
#     "<answer>subject: [x1, y1, x2, y2], object: [x1, y1, x2, y2], ...</answer>"
# )


# # ==========================================
# # 2. InternVL compatibility helpers
# # ==========================================
# def patch_transformers_for_internvl():
#     def normalize_tied_keys(value):
#         if value is None:
#             return {}
#         if isinstance(value, dict):
#             return value
#         if isinstance(value, (list, tuple, set)):
#             return {str(key): None for key in value}
#         if hasattr(value, "keys"):
#             return value
#         return {}

#     def get_all_tied_weights_keys(self):
#         value = self.__dict__.get("_all_tied_weights_keys_compat", None)
#         if value is None:
#             value = getattr(self, "_tied_weights_keys", None)
#         return normalize_tied_keys(value)

#     def set_all_tied_weights_keys(self, value):
#         self.__dict__["_all_tied_weights_keys_compat"] = normalize_tied_keys(value)

#     current_attr = getattr(PreTrainedModel, "all_tied_weights_keys", None)
#     if not isinstance(current_attr, property) or current_attr.fset is None:
#         PreTrainedModel.all_tied_weights_keys = property(
#             get_all_tied_weights_keys,
#             set_all_tied_weights_keys,
#         )


# def patch_qwen2_tokenizer_for_internvl():
#     try:
#         from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
#     except Exception:
#         Qwen2Tokenizer = None
#     try:
#         from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
#     except Exception:
#         Qwen2TokenizerFast = None

#     def token_getter(attr_name, default_value):
#         def getter(self):
#             extra_tokens = self.init_kwargs.get("extra_special_tokens", {}) or {}
#             return (
#                 getattr(self, f"_{attr_name}", None)
#                 or self.init_kwargs.get(attr_name)
#                 or extra_tokens.get(attr_name)
#                 or default_value
#             )

#         return getter

#     def token_setter(attr_name):
#         def setter(self, value):
#             setattr(self, f"_{attr_name}", value)

#         return setter

#     def id_getter(token_attr):
#         def getter(self):
#             token = getattr(self, token_attr)
#             token_id = self.convert_tokens_to_ids(token)
#             if token_id is None:
#                 return self.unk_token_id
#             return token_id

#         return getter

#     token_defaults = {
#         "start_image_token": "<img>",
#         "end_image_token": "</img>",
#         "context_image_token": "<IMG_CONTEXT>",
#         "image_token": "<image>",
#         "video_token": "<video>",
#     }
#     id_attrs = {
#         "start_image_token_id": "start_image_token",
#         "end_image_token_id": "end_image_token",
#         "context_image_token_id": "context_image_token",
#         "image_token_id": "image_token",
#         "video_token_id": "video_token",
#     }

#     for tokenizer_cls in [Qwen2Tokenizer, Qwen2TokenizerFast]:
#         if tokenizer_cls is None:
#             continue
#         for attr_name, default_value in token_defaults.items():
#             if not isinstance(getattr(tokenizer_cls, attr_name, None), property):
#                 setattr(
#                     tokenizer_cls,
#                     attr_name,
#                     property(token_getter(attr_name, default_value), token_setter(attr_name)),
#                 )
#         for attr_name, token_attr in id_attrs.items():
#             if not isinstance(getattr(tokenizer_cls, attr_name, None), property):
#                 setattr(tokenizer_cls, attr_name, property(id_getter(token_attr)))


# def patch_chat_template_processor_kwargs(processor):
#     original_apply_chat_template = processor.apply_chat_template

#     def apply_chat_template_compat(self, *args, **kwargs):
#         direct_processor_kwargs = {
#             key: kwargs.pop(key)
#             for key in ("padding", "padding_side", "return_tensors")
#             if key in kwargs
#         }
#         if direct_processor_kwargs:
#             processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
#             processor_kwargs.update(direct_processor_kwargs)
#             kwargs["processor_kwargs"] = processor_kwargs
#         return original_apply_chat_template(*args, **kwargs)

#     processor.apply_chat_template = types.MethodType(apply_chat_template_compat, processor)


# def patch_internvl_forward_contract_for_peft(model):
#     original_forward = model.forward
#     if "inputs_embeds" in inspect.signature(original_forward).parameters:
#         return

#     def forward_compat(self, *args, **kwargs):
#         del self
#         inputs_embeds = kwargs.pop("inputs_embeds", None)
#         if inputs_embeds is not None:
#             raise RuntimeError(
#                 "InternVL does not support non-empty inputs_embeds; expected input_ids."
#             )
#         return original_forward(*args, **kwargs)

#     model.forward = types.MethodType(forward_compat, model)


# def set_internvl_img_context_token_id(model, processor):
#     tokenizer = getattr(processor, "tokenizer", processor)
#     context_token = getattr(tokenizer, "context_image_token", "<IMG_CONTEXT>")
#     token_id = tokenizer.convert_tokens_to_ids(context_token)
#     if isinstance(token_id, list):
#         token_id = token_id[0] if token_id else None
#     if token_id is None or token_id < 0:
#         raise RuntimeError(f"cannot resolve InternVL context token id for {context_token!r}")

#     patched = 0
#     for module in model.modules():
#         if hasattr(module, "img_context_token_id"):
#             module.img_context_token_id = int(token_id)
#             patched += 1
#     if patched == 0 and hasattr(model, "img_context_token_id"):
#         model.img_context_token_id = int(token_id)
#         patched = 1
#     if patched == 0:
#         raise RuntimeError("cannot find img_context_token_id on InternVL model")


# def register_internvl_vision_dtype_guard(model):
#     vision_model = getattr(model, "vision_model", None)
#     if not isinstance(vision_model, torch.nn.Module):
#         raise RuntimeError("cannot find vision_model on InternVL model")

#     try:
#         vision_dtype = next(
#             parameter.dtype
#             for parameter in vision_model.parameters()
#             if parameter.is_floating_point()
#         )
#     except StopIteration as error:
#         raise RuntimeError("InternVL vision_model has no floating-point parameters") from error

#     def align_pixel_values_dtype(module, args, kwargs):
#         del module
#         args = list(args)
#         if (
#             args
#             and torch.is_tensor(args[0])
#             and args[0].is_floating_point()
#             and args[0].dtype != vision_dtype
#         ):
#             args[0] = args[0].to(dtype=vision_dtype)

#         pixel_values = kwargs.get("pixel_values")
#         if (
#             torch.is_tensor(pixel_values)
#             and pixel_values.is_floating_point()
#             and pixel_values.dtype != vision_dtype
#         ):
#             kwargs = dict(kwargs)
#             kwargs["pixel_values"] = pixel_values.to(dtype=vision_dtype)
#         return tuple(args), kwargs

#     vision_model.register_forward_pre_hook(align_pixel_values_dtype, with_kwargs=True)


# # ==========================================
# # 3. Helpers
# # ==========================================
# def load_jsonl(path):
#     with open(path, "r", encoding="utf-8") as file:
#         return [json.loads(line) for line in file if line.strip()]


# def resolve_pretrained_path(path):
#     path = os.path.expanduser(path)
#     if os.path.isfile(os.path.join(path, "config.json")):
#         return path

#     candidates = []
#     for config_path in glob.glob(os.path.join(path, "**", "config.json"), recursive=True):
#         model_dir = os.path.dirname(config_path)
#         has_processor = any(
#             os.path.exists(os.path.join(model_dir, name))
#             for name in [
#                 "preprocessor_config.json",
#                 "processor_config.json",
#                 "tokenizer_config.json",
#                 "tokenizer.json",
#                 "vocab.json",
#             ]
#         )
#         if has_processor:
#             candidates.append((os.path.getmtime(config_path), model_dir))

#     if candidates:
#         return sorted(candidates)[-1][1]
#     return path


# def validate_internvl_model_path(model_path):
#     config_path = os.path.join(model_path, "config.json")
#     if not os.path.isfile(config_path):
#         raise FileNotFoundError(f"missing config.json under BASE_MODEL_PATH: {model_path}")

#     with open(config_path, "r", encoding="utf-8") as config_file:
#         config = json.load(config_file)
#     descriptors = " ".join(
#         str(value)
#         for value in [
#             model_path,
#             config.get("model_type", ""),
#             config.get("architectures", ""),
#             config.get("auto_map", ""),
#         ]
#     ).lower()
#     if "internvl" not in descriptors:
#         raise RuntimeError(
#             "This baseline script is InternVL-only, but BASE_MODEL_PATH does not look like "
#             f"an InternVL checkpoint: {model_path}"
#         )


# def find_image_path(item):
#     if item.get("image_path") and os.path.exists(item["image_path"]):
#         return item["image_path"]

#     if item.get("images"):
#         image_path = item["images"][0]
#         if os.path.exists(image_path):
#             return image_path

#     if item.get("image_relpath"):
#         for root in [
#             IMAGE_ROOT,
#             os.path.join(os.path.dirname(IMAGE_ROOT), "image"),
#             os.path.join(os.path.dirname(IMAGE_ROOT), "images"),
#         ]:
#             path = os.path.join(root, item["image_relpath"])
#             if os.path.exists(path):
#                 return path

#     image_id = str(item.get("image_id", item.get("file_name", "")))
#     roots = [
#         IMAGE_ROOT,
#         os.path.join(os.path.dirname(IMAGE_ROOT), "image"),
#         os.path.join(os.path.dirname(IMAGE_ROOT), "images"),
#     ]
#     for root in roots:
#         for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
#             for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
#                 path = os.path.join(
#                     root,
#                     "RSRG_ME_datasets_ori",
#                     dataset,
#                     "images",
#                     image_id + ext,
#                 )
#                 if os.path.exists(path):
#                     return path

#     raise FileNotFoundError(f"cannot find image for image_id={image_id}")


# def build_messages(instruction):
#     clean_instruction = str(instruction).replace("<image>", "").strip()
#     prompt = PROMPT_TEMPLATE.format(instruction=clean_instruction)
#     if "<IMG_CONTEXT>" not in prompt:
#         prompt = "<IMG_CONTEXT>\n" + prompt
#     return [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": prompt},
#             ],
#         }
#     ]


# def move_batch_to_device(batch, device):
#     return {
#         key: value.to(device) if torch.is_tensor(value) else value
#         for key, value in batch.items()
#     }


# def decode_tokens(processor, token_ids):
#     decoder = getattr(processor, "batch_decode", None)
#     if decoder is None:
#         decoder = processor.tokenizer.batch_decode
#     return decoder(
#         token_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )[0].strip()


# def extract_completion_ids(generated_ids, input_ids):
#     sequences = generated_ids
#     if not torch.is_tensor(sequences):
#         sequences = getattr(generated_ids, "sequences", None)
#     if not torch.is_tensor(sequences):
#         raise RuntimeError(f"unsupported generate output type: {type(generated_ids)!r}")
#     if sequences.ndim == 1:
#         sequences = sequences.unsqueeze(0)

#     prompt_length = input_ids.size(1)
#     includes_prompt = (
#         sequences.size(1) >= prompt_length
#         and torch.equal(sequences[:, :prompt_length].to(input_ids.device), input_ids)
#     )
#     if includes_prompt:
#         return sequences[:, prompt_length:]
#     return sequences


# def load_internvl_processor(model_path):
#     patch_transformers_for_internvl()
#     patch_qwen2_tokenizer_for_internvl()
#     try:
#         processor = AutoProcessor.from_pretrained(
#             model_path,
#             trust_remote_code=True,
#             local_files_only=True,
#             fix_mistral_regex=True,
#         )
#     except TypeError:
#         processor = AutoProcessor.from_pretrained(
#             model_path,
#             trust_remote_code=True,
#             local_files_only=True,
#         )

#     tokenizer = getattr(processor, "tokenizer", None)
#     if tokenizer is not None:
#         tokenizer.padding_side = "left"
#     patch_chat_template_processor_kwargs(processor)
#     return processor


# def load_internvl_model(model_path):
#     model_kwargs = {
#         "trust_remote_code": True,
#         "local_files_only": True,
#         "low_cpu_mem_usage": True,
#         "device_map": "auto",
#     }
#     try:
#         model = AutoModel.from_pretrained(
#             model_path,
#             dtype=TORCH_DTYPE,
#             **model_kwargs,
#         )
#     except TypeError:
#         model = AutoModel.from_pretrained(
#             model_path,
#             torch_dtype=TORCH_DTYPE,
#             **model_kwargs,
#         )

#     model.config.use_cache = True
#     patch_internvl_forward_contract_for_peft(model)
#     register_internvl_vision_dtype_guard(model)
#     return model


# def generation_token_ids(processor):
#     tokenizer = getattr(processor, "tokenizer", processor)
#     eos_token_id = getattr(tokenizer, "eos_token_id", None)
#     pad_token_id = getattr(tokenizer, "pad_token_id", None)
#     if pad_token_id is None:
#         pad_token_id = eos_token_id
#     return pad_token_id, eos_token_id


# def generate_one(model, processor, image, instruction):
#     messages = build_messages(instruction)
#     text = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     inputs = processor(
#         text=[text],
#         images=[image],
#         return_tensors="pt",
#         padding=True,
#     )

#     device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
#     inputs = move_batch_to_device(inputs, device)
#     pad_token_id, eos_token_id = generation_token_ids(processor)

#     with torch.inference_mode():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#         )

#     completion_ids = extract_completion_ids(generated_ids, inputs["input_ids"])
#     return decode_tokens(processor, completion_ids)


# # ==========================================
# # 4. Main
# # ==========================================
# def main():
#     data = load_jsonl(DATA_PATH)
#     if NUM_SAMPLES is None:
#         data = data[START_INDEX:]
#     else:
#         data = data[START_INDEX : START_INDEX + NUM_SAMPLES]

#     model_path = resolve_pretrained_path(BASE_MODEL_PATH)
#     validate_internvl_model_path(model_path)

#     print(f"Loading InternVL processor: {model_path}")
#     processor = load_internvl_processor(model_path)

#     print(f"Loading InternVL base model: {model_path}")
#     model = load_internvl_model(model_path)
#     set_internvl_img_context_token_id(model, processor)
#     model.eval()

#     output_dir = os.path.dirname(OUTPUT_JSONL)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#     with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_file:
#         for offset, item in enumerate(tqdm(data, desc="InternVL baseline testing")):
#             image_id = str(item.get("image_id", item.get("file_name", "")))
#             image_path = find_image_path(item)
#             image = Image.open(image_path).convert("RGB")

#             raw_output = generate_one(
#                 model=model,
#                 processor=processor,
#                 image=image,
#                 instruction=item["instruction"],
#             )

#             result = {
#                 "index": START_INDEX + offset,
#                 "image_id": image_id,
#                 "image_path": image_path,
#                 "instruction": item["instruction"],
#                 "raw_output": raw_output,
#             }
#             out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
#             out_file.flush()

#     print(f"Saved InternVL baseline outputs to: {OUTPUT_JSONL}")


# if __name__ == "__main__":
#     main()


# # python code/eval/infer_internvl_entity_baseline.py


# # CUDA_VISIBLE_DEVICES=0,1,2,3 \
# # torchrun --standalone --nproc_per_node=4 ./2-baseline-test-InternVL.py


START_INDEX = int(os.environ.get("START_INDEX", "10"))
NUM_SAMPLES_TEXT = os.environ.get("NUM_SAMPLES", "").strip()
NUM_SAMPLES = int(NUM_SAMPLES_TEXT) if NUM_SAMPLES_TEXT else None
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
TORCH_DTYPE = torch.bfloat16
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto").strip()
MAX_MEMORY_PER_GPU = os.environ.get("MAX_MEMORY_PER_GPU", "").strip()
MAX_MEMORY = os.environ.get("MAX_MEMORY", "").strip()
CPU_MAX_MEMORY = os.environ.get("CPU_MAX_MEMORY", "").strip()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==========================================
# 1. Prompt
# ==========================================
PROMPT_TEMPLATE = (
    "Localize the subject and objects in the image with description: {instruction}. "
    "Generate a step-by-step reasoning process in <think></think> tags and output the "
    "box coordinates of one subject and one or more objects in the format of "
    "<answer>subject: [x1, y1, x2, y2], object: [x1, y1, x2, y2], ...</answer>"
)


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
# 3. Helpers
# ==========================================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_pretrained_path(path):
    path = os.path.expanduser(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path

    candidates = []
    for config_path in glob.glob(os.path.join(path, "**", "config.json"), recursive=True):
        model_dir = os.path.dirname(config_path)
        has_processor = any(
            os.path.exists(os.path.join(model_dir, name))
            for name in [
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
            ]
        )
        if has_processor:
            candidates.append((os.path.getmtime(config_path), model_dir))

    if candidates:
        return sorted(candidates)[-1][1]
    return path


def validate_internvl_model_path(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"missing config.json under BASE_MODEL_PATH: {model_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    descriptors = " ".join(
        str(value)
        for value in [
            model_path,
            config.get("model_type", ""),
            config.get("architectures", ""),
            config.get("auto_map", ""),
        ]
    ).lower()
    if "internvl" not in descriptors:
        raise RuntimeError(
            "This baseline script is InternVL-only, but BASE_MODEL_PATH does not look like "
            f"an InternVL checkpoint: {model_path}"
        )


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
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
        for root in [
            IMAGE_ROOT,
            os.path.join(os.path.dirname(IMAGE_ROOT), "image"),
            os.path.join(os.path.dirname(IMAGE_ROOT), "images"),
        ]:
            path = os.path.join(root, item["image_relpath"])
            if os.path.exists(path):
                return path

    image_id = str(item.get("image_id", item.get("file_name", "")))
    roots = [
        IMAGE_ROOT,
        os.path.join(os.path.dirname(IMAGE_ROOT), "image"),
        os.path.join(os.path.dirname(IMAGE_ROOT), "images"),
    ]
    for root in roots:
        for dataset in ["dior_rsvg", "opt_rsvg", "rsvg_hr"]:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                path = os.path.join(
                    root,
                    "RSRG_ME_datasets_ori",
                    dataset,
                    "images",
                    image_id + ext,
                )
                if os.path.exists(path):
                    return path

    raise FileNotFoundError(f"cannot find image for image_id={image_id}")


def build_messages(instruction):
    clean_instruction = str(instruction).replace("<image>", "").strip()
    prompt = PROMPT_TEMPLATE.format(instruction=clean_instruction)
    if "<IMG_CONTEXT>" not in prompt:
        prompt = "<IMG_CONTEXT>\n" + prompt
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
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


def resolve_device_map(world_size=1, local_rank=0):
    if world_size > 1:
        return {"": int(local_rank)}

    value = DEVICE_MAP.lower()
    if value in {"", "none", "false", "no"}:
        return None
    if value.isdigit():
        return {"": int(value)}
    return DEVICE_MAP


def parse_max_memory_items(text):
    memory = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "MAX_MEMORY must use comma-separated device:memory entries, "
                f"got {item!r}"
            )
        device, limit = item.split(":", 1)
        device = device.strip()
        limit = limit.strip()
        if device.lower() != "cpu" and device.isdigit():
            device = int(device)
        memory[device] = limit
    return memory


def build_max_memory():
    if MAX_MEMORY:
        memory = parse_max_memory_items(MAX_MEMORY)
    elif MAX_MEMORY_PER_GPU:
        memory = {
            gpu_index: MAX_MEMORY_PER_GPU
            for gpu_index in range(torch.cuda.device_count())
        }
    else:
        memory = {}

    if CPU_MAX_MEMORY:
        memory["cpu"] = CPU_MAX_MEMORY
    return memory or None


def summarize_device_map(model):
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return ""

    counts = {}
    for device in device_map.values():
        device_name = str(device)
        counts[device_name] = counts.get(device_name, 0) + 1
    return ", ".join(
        f"{device}:{counts[device]}" for device in sorted(counts)
    )


def load_internvl_model(model_path, world_size=1, local_rank=0, max_memory=None):
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    device_map = resolve_device_map(world_size=world_size, local_rank=local_rank)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

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
    return model


def generation_token_ids(processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return pad_token_id, eos_token_id


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
    )

    device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    inputs = move_batch_to_device(inputs, device)
    pad_token_id, eos_token_id = generation_token_ids(processor)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    completion_ids = extract_completion_ids(generated_ids, inputs["input_ids"])
    return decode_tokens(processor, completion_ids)


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
    validate_internvl_model_path(model_path)

    if rank == 0:
        print(f"World size: {world_size}; total samples: {len(indexed_data)}")
        print(f"Rank 0 shard samples: {len(rank_data)}")
        print(f"Loading InternVL processor: {model_path}")
    processor = load_internvl_processor(model_path)

    max_memory = build_max_memory()
    device_map = resolve_device_map(world_size=world_size, local_rank=local_rank)
    if rank == 0:
        print(f"Loading InternVL base model: {model_path}")
        print(f"Device map: {device_map}")
        if max_memory:
            print(f"Max memory: {max_memory}")
    model = load_internvl_model(
        model_path,
        world_size=world_size,
        local_rank=local_rank,
        max_memory=max_memory,
    )
    set_internvl_img_context_token_id(model, processor)
    model.eval()
    device_summary = summarize_device_map(model)
    if rank == 0 and device_summary:
        print(f"Loaded module device map summary: {device_summary}")

    output_dir = os.path.dirname(OUTPUT_JSONL)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    shard_path = f"{OUTPUT_JSONL}.rank{rank}.tmp" if world_size > 1 else OUTPUT_JSONL
    with open(shard_path, "w", encoding="utf-8") as out_file:
        for index, item in tqdm(
            rank_data,
            desc=f"InternVL baseline rank {rank}",
            disable=rank != 0,
        ):
            image_id = str(item.get("image_id", item.get("file_name", "")))
            image_path = find_image_path(item)
            image = Image.open(image_path).convert("RGB")

            raw_output = generate_one(
                model=model,
                processor=processor,
                image=image,
                instruction=item["instruction"],
            )

            result = {
                "index": index,
                "image_id": image_id,
                "image_path": image_path,
                "instruction": item["instruction"],
                "raw_output": raw_output,
            }
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()

    if world_size > 1:
        dist.barrier()
        if rank == 0:
            merge_rank_outputs(OUTPUT_JSONL, world_size)
            print(f"Saved InternVL baseline outputs to: {OUTPUT_JSONL}")
        dist.barrier()
        dist.destroy_process_group()
    else:
        print(f"Saved InternVL baseline outputs to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()


# 4-GPU data-parallel baseline:
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 code/eval/infer_internvl_entity_baseline.py
#
# Small 4-GPU smoke test:
# CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_SAMPLES=8 torchrun --standalone --nproc_per_node=4 code/eval/infer_internvl_entity_baseline.py
#
# Optional single-process model-parallel prompt test:
# CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_MEMORY_PER_GPU=10GiB NUM_SAMPLES=5 python code/eval/infer_internvl_entity_baseline.py
#
# More explicit per-GPU limits:
# CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_MEMORY=0:8GiB,1:8GiB,2:8GiB,3:8GiB NUM_SAMPLES=5 python code/eval/infer_internvl_entity_baseline.py
