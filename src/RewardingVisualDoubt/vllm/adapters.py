import typing as t

import bitsandbytes
import peft
import torch
import trl
from LLAVA_Biovil import llava

from RewardingVisualDoubt import shared


def _get_lora_config() -> peft.LoraConfig:
    return peft.LoraConfig(
        r=128,
        target_modules=[
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "up_proj",
            "q_proj",
        ],
        lora_alpha=256,
        lora_dropout=0.05,
        fan_in_fan_out=False,
        bias="none",
        modules_to_save=None,
        init_lora_weights=True,
        layers_to_transform=None,
        layers_pattern=None,
        task_type="CAUSAL_LM",
    )


def _load_lora_weights_in_state_dict_format_for_radialog(
    lora_weights_path: str,
    is_target_model_with_value_head: bool = True,
) -> dict[str, torch.Tensor]:

    adapter_state_dict = torch.load(lora_weights_path, map_location="cpu")
    mapped_state_dict = {}
    if is_target_model_with_value_head:
        for k, v in adapter_state_dict.items():
            # Replace the prefix to match the loaded model structure
            new_key = k.replace("base_model.", "pretrained_model.base_model.")
            new_key = new_key.split(".weight")[0] + ".default" + ".weight"
            mapped_state_dict[new_key] = v
    else:
        for k, v in adapter_state_dict.items():
            new_key = k.split(".weight")[0] + ".default" + ".weight"
            mapped_state_dict[new_key] = v
    return mapped_state_dict


def add_finetuned_lora_adapters_to_LlavaLlamaForCausalLM_model(
    model: llava.model.LlavaLlamaForCausalLM,
    radialog_lora_weights_path: str,
) -> peft.PeftModel:

    is_model_quantized: bool = any(
        isinstance(m, bitsandbytes.nn.Linear4bit) for m in model.modules()
    ) or any(isinstance(m, bitsandbytes.nn.Linear8bitLt) for m in model.modules())

    if is_model_quantized:
        model = t.cast(
            llava.model.LlavaLlamaForCausalLM, peft.prepare_model_for_kbit_training(model)
        )

    lora_config = _get_lora_config()
    print("Adding LoRA adapters to the model...")
    lora_model = peft.get_peft_model(model, lora_config)
    radialog_mapped_lora_state_dict = _load_lora_weights_in_state_dict_format_for_radialog(
        lora_weights_path=radialog_lora_weights_path, is_target_model_with_value_head=False
    )

    missing_keys, unexpected_keys = lora_model.load_state_dict(
        radialog_mapped_lora_state_dict, strict=False
    )
    assert unexpected_keys == [], "Some LoRA weights could not be mapped to the RaDialog model"

    if is_model_quantized:
        # Cast vision related modules back to bfloat16, as lora loading messes them about to float32
        model_ = t.cast(llava.LlavaLlamaForCausalLM, lora_model.base_model.model.model)
        model_.vision_tower.to(
            device=torch.device(shared.torch_devices.cuda.value), dtype=torch.bfloat16
        )
        model_.mm_projector.to(
            device=torch.device(shared.torch_devices.cuda.value), dtype=torch.bfloat16
        )

    return lora_model


def add_finetuned_or_fresh_lora_adapters_and_fresh_value_head_to_LlavaLlamaForCausalLM_model(
    model: llava.model.LlavaLlamaForCausalLM,
    radialog_lora_weights_path: str | None,
) -> trl.models.modeling_value_head.AutoModelForCausalLMWithValueHead:
    """
    Adds LoRA adapters and a value head to the model. If `radialog_lora_weights_path` is provided, it loads the
    LoRA weights from that path. If not, it initializes the LoRA adapters and value head with fresh weights.
    Args:
        model: The base model to add LoRA adapters and value head to.
        radialog_lora_weights_path: The path to the LoRA weights to load. If None, fresh weights are used.
    Returns:
        The model with LoRA adapters and value head added.
    """

    is_model_quantized: bool = any(
        isinstance(m, bitsandbytes.nn.Linear4bit) for m in model.modules()
    ) or any(isinstance(m, bitsandbytes.nn.Linear8bitLt) for m in model.modules())

    lora_config = _get_lora_config()

    if is_model_quantized:
        model = peft.prepare_model_for_kbit_training(model)
    print("Adding LoRA adapters and value head to the model...")
    trl_lora_model: trl.models.modeling_value_head.AutoModelForCausalLMWithValueHead = (
        trl.AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=lora_config)
    )
    if radialog_lora_weights_path:
        radialog_mapped_lora_state_dict = _load_lora_weights_in_state_dict_format_for_radialog(
            lora_weights_path=radialog_lora_weights_path, is_target_model_with_value_head=True
        )

        missing_keys, unexpected_keys = trl_lora_model.load_state_dict(
            radialog_mapped_lora_state_dict, strict=False
        )
        assert unexpected_keys == [], "Some LoRA weights could not be mapped to the RaDialog model"

    if is_model_quantized:
        # Cast vision related modules back to bfloat16, as lora loading messes them about to float32
        model_ = t.cast(
            llava.LlavaLlamaForCausalLM, trl_lora_model.pretrained_model.base_model.model.model
        )
        model_.vision_tower.to(
            device=torch.device(shared.torch_devices.cuda.value), dtype=torch.bfloat16
        )
        model_.mm_projector.to(
            device=torch.device(shared.torch_devices.cuda.value), dtype=torch.bfloat16
        )

    return trl_lora_model
