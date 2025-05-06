import typing as t

import torch
from LLAVA_Biovil import llava
from LLAVA_Biovil.biovil_t.model import ImageModel
from LLAVA_Biovil.biovil_t.pretrained import \
    _download_biovil_t_image_model_weights
from LLAVA_Biovil.biovil_t.types import ImageEncoderType
from LLAVA_Biovil.llava.model.multimodal_projector.builder import \
    build_vision_projector


def reset_mm_projector_to_fix_wrong_shape(
    model: llava.LlavaLlamaForCausalLM,
) -> llava.LlavaLlamaForCausalLM:
    if model.config.mm_vision_tower == "biovil":
        # reset mm_projector as wrong shape is loaded from pretrained base model
        model.model.mm_projector = build_vision_projector(model.config)
        model.model.mm_projector.to(device=model.device, dtype=model.dtype)
    return model


def _modify_model_for_image_input(
    model: llava.model.LlavaLlamaForCausalLM, tokenizer_length: int
) -> llava.model.LlavaLlamaForCausalLM:
    model.resize_token_embeddings(tokenizer_length)
    return model


def merge_llm_with_vision_tower(
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
