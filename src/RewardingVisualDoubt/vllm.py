import os
import typing as t
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from LLAVA_Biovil import llava
from LLAVA_Biovil.biovil_t.model import ImageModel
from LLAVA_Biovil.biovil_t.pretrained import \
    _download_biovil_t_image_model_weights
from LLAVA_Biovil.biovil_t.types import ImageEncoderType
from LLAVA_Biovil.llava.constants import (DEFAULT_IM_END_TOKEN,
                                          DEFAULT_IM_START_TOKEN,
                                          DEFAULT_IMAGE_PATCH_TOKEN)
from LLAVA_Biovil.llava.model.multimodal_projector.builder import \
    build_vision_projector
from peft import LoraModel, PeftModel

LLAVA_BASE_MODEL_NAME = "liuhaotian/llava-v1.5-7b"
LLAVA_LORA_ADAPTER = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
FINETUNED_LLAVA_REPO_ID = "ChantalPellegrini/RaDialog-interactive-radiology-report-generation"


##################### HELPER FUNCTIONS #####################


def _get_hf_model_path(repo_id) -> Path:
    return Path(snapshot_download(repo_id=repo_id, revision="main"))


def _resolve_model_configuration(
    kwargs, device: str, device_map: str, load_8bit: bool, load_4bit: bool
) -> dict:

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        # kwargs['torch_dtype'] = torch.float16
        kwargs["torch_dtype"] = torch.bfloat16

    kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


##################### IMAGE INPUT SUPPORT #####################


def _modify_tokenizer_for_image_input(
    tokenizer: transformers.LlamaTokenizer,
    mm_use_im_start_end: bool = False,
    mm_use_im_patch_token: bool = True,
) -> transformers.LlamaTokenizer:
    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    return tokenizer


def _modify_model_for_image_input(
    model: llava.model.LlavaLlamaForCausalLM, tokenizer_length: int
) -> llava.model.LlavaLlamaForCausalLM:
    model.resize_token_embeddings(tokenizer_length)
    return model


def _merge_llm_with_vision_tower(
    model: llava.model.LlavaLlamaForCausalLM,
    device: torch.device,
    non_lora_trainables,
    tokenizer_length: int,
) -> llava.model.LlavaLlamaForCausalLM:

    model = _modify_model_for_image_input(model, tokenizer_length=tokenizer_length)

    biovilt_checkpoint_path = _download_biovil_t_image_model_weights()
    model_type = ImageEncoderType.RESNET50_MULTI_IMAGE
    vision_tower = ImageModel(
        img_encoder_type=model_type,
        joint_feature_size=128,
        pretrained_model_path=biovilt_checkpoint_path,
    )
    model.model.vision_tower = vision_tower

    vision_tower.to(device=device, dtype=torch.bfloat16)

    # if non_lora_trainables contains something about vision_tower, load it
    if non_lora_trainables is not None and any(
        k.startswith("model.vision_tower.") for k in non_lora_trainables
    ):
        new_vision_tower_state_dict = {}
        for (
            k,
            v,
        ) in (
            non_lora_trainables.items()
        ):  # we need remapping, because state_dict from model is always like model.vision_tower. It should be vision_tower.
            if "model.vision_tower.vision_tower." in k:  # original CLIP
                new_k = k.replace("model.vision_tower.", "")
                new_vision_tower_state_dict[new_k] = v
            elif "model.vision_tower" in k:  # biovil
                new_k = k.replace("model.vision_tower.", "")
                new_vision_tower_state_dict[new_k] = v
        print("Loaded additional vision tower weights...")
        vision_tower.load_state_dict(new_vision_tower_state_dict, strict=False)
    return model


##################### ENTRY POINTS #####################


def load_pretrained_text_only_llava_tokenizer(model_base: str) -> transformers.LlamaTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_base, use_fast=False)
    assert isinstance(tokenizer, transformers.LlamaTokenizer)
    return tokenizer


def load_pretrained_llava_tokenizer_with_image_support(
    model_base: str,
) -> transformers.LlamaTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_base, use_fast=False)
    assert isinstance(tokenizer, transformers.LlamaTokenizer)
    tokenizer = _modify_tokenizer_for_image_input(tokenizer)
    return tokenizer


def get_llava_image_processor(
    model: llava.model.LlavaLlamaForCausalLM,
) -> transformers.CLIPImageProcessor:
    assert model.model and model.model.vision_tower
    return model.model.vision_tower.image_processor


def get_model_context_length(model: llava.model.LlavaLlamaForCausalLM) -> int:
    context_len = (
        model.config.max_sequence_length if hasattr(model.config, "max_sequence_length") else 2048
    )
    return context_len


def load_pretrained_llava_model(
    model_path: Path = _get_hf_model_path(repo_id=FINETUNED_LLAVA_REPO_ID),
    model_base=LLAVA_BASE_MODEL_NAME,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map="auto",
    device="cuda",
    **kwargs
) -> llava.model.LlavaLlamaForCausalLM:

    kwargs = _resolve_model_configuration(kwargs, device, device_map, load_8bit, load_4bit)

    print("Model base: ", model_base)

    lora_cfg_pretrained = transformers.AutoConfig.from_pretrained(model_path)

    print("Loading LLaVA from base model...")
    model = t.cast(
        transformers.LlamaForCausalLM,
        llava.model.LlavaLlamaForCausalLM.from_pretrained(
            model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
        ),
    )
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features

    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(
            torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
        )
        model.model.embed_tokens.weight = torch.nn.Parameter(
            torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
        )

    if model.config.mm_vision_tower == "biovil":
        # reset mm_projector as wrong shape is loaded from pretrained base model
        model.model.mm_projector = build_vision_projector(model.config)
        model.model.mm_projector.to(device=model.device, dtype=model.dtype)

    print("Loading additional LLaVA weights...")
    assert os.path.exists(os.path.join(model_path, "non_lora_trainables.bin"))
    non_lora_trainables = torch.load(
        os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
    )
    non_lora_trainables = {
        (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
    }
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {
            (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
        }
    model.load_state_dict(non_lora_trainables, strict=False)

    print("Loading LoRA weights...")
    model = t.cast(LoraModel, PeftModel.from_pretrained(model, model_path))
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("Model is loaded...")

    # Merge model with the vision tower
    assert model.config.mm_vision_tower == "biovil"
    tokenizer_length = len(load_pretrained_llava_tokenizer_with_image_support(model_base))
    model = _merge_llm_with_vision_tower(model, model.device, non_lora_trainables, tokenizer_length)
    model.config.tokenizer_padding_side = "left"  # TODO why?
    return model
