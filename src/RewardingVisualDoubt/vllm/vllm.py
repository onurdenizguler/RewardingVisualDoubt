import enum
import os
import pathlib as path
import typing as t
from pathlib import Path

import huggingface_hub
import peft
import torch
import transformers
import trl

from RewardingVisualDoubt import shared

from . import adapters, tokenizer, vision

LLAVA_BASE_MODEL_NAME = "liuhaotian/llava-v1.5-7b"
LLAVA_LORA_ADAPTER = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
RADIALOG_BASELINE_LORA_ADAPTER_REPO_ID = (
    "ChantalPellegrini/RaDialog-interactive-radiology-report-generation"
)
MODEL_SAVING_DIR = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models"


class RadialogLoraWeightsPath(enum.Enum):
    ORIGINAL = (
        "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/data/RaDialog_adapter_model.bin"
    )
    BINARY_QA_WITH_CONFIDENCE_SFT = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radialog_binary_qa_with_confidence_sft_resulting_adapter.pth/adapter_model.bin"


class RadialogMergedLlavaModelPath(enum.Enum):
    BINARY_QA_WITH_CONFIDENCE_SFT = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radialog_binary_qa_with_confidence_sft_full_merged_model"


class BinaryQAPPOAdapterPath(enum.Enum):
    BINARY_QA_CONFIDENCE_PPO_001_05_05_2025 = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radialog_binary_qa_ppo_adapter_1_05-05-2025/adapter_model.bin"


Precision = t.Literal["16bit", "8bit", "4bit"]

##################### HELPER FUNCTIONS #####################


def _get_hf_model_path(repo_id) -> Path:
    return Path(huggingface_hub.snapshot_download(repo_id=repo_id, revision="main"))


def _resolve_model_configuration(
    kwargs, device: str, device_map: str, precision: Precision
) -> dict:
    print(f"Model will be loaded at precision: {precision}")

    kwargs = {"device_map": device_map, **kwargs}
    kwargs["torch_dtype"] = torch.bfloat16

    if device != "cuda":
        kwargs["device_map"] = {"": device}
    if precision == "8bit":
        kwargs["load_in_8bit"] = True
    elif precision == "4bit":
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def _get_finetuned_llava_config(model_path: Path) -> shared.llava.model.LlavaConfig:
    return transformers.AutoConfig.from_pretrained(model_path)


##################### PREPARE FOR INFERENCE OR TRAINING #####################


def _freeze_all_params(
    model: shared.llava.model.LlavaLlamaForCausalLM,
) -> shared.llava.model.LlavaLlamaForCausalLM:
    for param in model.parameters():
        param.requires_grad = False
    return model


def _freeze_all_non_lora_params(
    model: shared.llava.model.LlavaLlamaForCausalLM,
) -> shared.llava.model.LlavaLlamaForCausalLM:
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
    return model


######################### LOAD NON-LORA TRAINABLES #####################


def _read_non_lora_trainables(model_path: Path) -> dict:
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
    return non_lora_trainables


def _load_additional_non_lora_llava_weights(
    model: shared.llava.LlavaLlamaForCausalLM,
    model_path: Path,
    skip_vision_related_weights: bool = False,
) -> shared.llava.LlavaLlamaForCausalLM:

    print("Loading additional LLaVA weights...")
    non_lora_trainables = _read_non_lora_trainables(model_path)
    if skip_vision_related_weights:
        non_lora_trainables = {
            k: v
            for k, v in non_lora_trainables.items()
            if not ("mm_projector" in k or "vision_tower" in k)
        }
    model.load_state_dict(non_lora_trainables, strict=False)
    return model


########################## BASELINE MODEL LOADING #####################


def _load_base_LlavaLamaForCausalLM_model(
    model_base: str | Path, model_path: Path, **kwargs
) -> shared.llava.model.LlavaLlamaForCausalLM:

    print(f"Loading LLaVA from base {model_base}")
    model = t.cast(
        transformers.LlamaForCausalLM,
        shared.llava.model.LlavaLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_base,
            low_cpu_mem_usage=True,
            config=_get_finetuned_llava_config(model_path=model_path),
            **kwargs,
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

    return t.cast(shared.llava.model.LlavaLlamaForCausalLM, model)


def load_baseline_llava_model_with_vision_modules(
    model_path: Path = _get_hf_model_path(repo_id=RADIALOG_BASELINE_LORA_ADAPTER_REPO_ID),
    device_map="auto",
    device: str = shared.torch_devices.cuda.value,
    precision: Precision = "16bit",
    **kwargs,
) -> shared.llava.model.LlavaLlamaForCausalLM:
    """
    Load the baseline LLaVA model with the vision modules finetuned for RaDialog.
    We do not load any LLM LoRA adapters here due to pecularities with the loading logic.
    We have additional methods in vllm.adapters module to load LoRA adapters manually.
    args:
        model_path: Path to the model checkpoint
        model_base: Base model name
        device_map: Device map
        device: Device  (default: "cuda")
        is_lora_trainable: Whether to load LoRA weights as trainable (default: False)
        precision: Precision of the model (default: "16bit")
        skip_lora_adapters: Whether to skip loading LoRA adapters (default: False)
    """
    print("Loading LLaVA model with the base LLM and with RaDialog finetuned vision modules...")

    model_base = (
        LLAVA_BASE_MODEL_NAME
        if model_path == _get_hf_model_path(repo_id=RADIALOG_BASELINE_LORA_ADAPTER_REPO_ID)
        else model_path
    )
    kwargs = _resolve_model_configuration(kwargs, device, device_map, precision)
    model = _load_base_LlavaLamaForCausalLM_model(model_base, model_path, **kwargs)
    model = vision.reset_mm_projector_to_fix_wrong_shape(model)
    model = _load_additional_non_lora_llava_weights(
        model, model_path, skip_vision_related_weights=False
    )

    model.config.tokenizer_padding_side = "left"  # TODO why?

    print("Merging model with vision tower weights...")
    assert model.config.mm_vision_tower == "biovil"
    model = vision.merge_llm_with_vision_tower(
        model=model,
        device=model.device,
        non_lora_trainables=_read_non_lora_trainables(model_path=model_path),
        tokenizer_length=len(
            tokenizer.load_pretrained_llava_tokenizer_with_image_support(
                model_base=LLAVA_BASE_MODEL_NAME
            )
        ),
    )

    model = _freeze_all_params(model)

    return model


########################## USECASE-SPECIFIC MODEL LOADING #####################


def load_pretrained_llava_model_for_sft_training(
    device_str: str = shared.torch_devices.cuda.value,
    precision: Precision = "16bit",
    radialog_lora_weights_path: str = RadialogLoraWeightsPath.ORIGINAL.value,
) -> peft.PeftModelForCausalLM:
    print(
        f"Adding LoRA adapters to the model for SFT training or inference from Radialog Lora Weights path: {radialog_lora_weights_path}"
    )
    model = load_baseline_llava_model_with_vision_modules(device=device_str, precision=precision)
    model = adapters.add_finetuned_lora_adapters_to_LlavaLlamaForCausalLM_model(
        model, radialog_lora_weights_path=radialog_lora_weights_path
    )
    return model


def load_pretrained_llava_model_for_ppo_training_with_lora_adapters(
    llava_model_path: str | None = RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
    device_str: str = shared.torch_devices.cuda.value,
    precision: Precision = "16bit",
    adapter_path: str | None = None,
) -> trl.models.modeling_value_head.AutoModelForCausalLMWithValueHead:
    if adapter_path is None:
        print(
            f"Adding fresh set of LoRA adapters and a fresh value head to the model for PPO training using Llava model loaded from: {llava_model_path}"
        )
    else:
        print(
            f"Adding loaded LoRA adapters and a fresh value head to the model for PPO training using Llava model and adapters loaded from: {llava_model_path} and adapter path: {adapter_path}"
        )
    model = load_baseline_llava_model_with_vision_modules(
        model_path=(
            path.Path(llava_model_path)
            if llava_model_path
            else _get_hf_model_path(repo_id=RADIALOG_BASELINE_LORA_ADAPTER_REPO_ID)
        ),
        device=device_str,
        precision=precision,
    )
    model = adapters.add_finetuned_or_fresh_lora_adapters_and_fresh_value_head_to_LlavaLlamaForCausalLM_model(
        model, radialog_lora_weights_path=adapter_path
    )
    return model


load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters = (
    load_pretrained_llava_model_for_ppo_training_with_lora_adapters  # for legacy purposes
)


def save_lora_merged_llava_model_to_local_dir(
    model_save_name: str,
    radialog_lora_weights_path: str = RadialogLoraWeightsPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
):
    """
    Use this function to save the merged model to a local directory in full precision.
    """
    model = load_baseline_llava_model_with_vision_modules(
        device=shared.torch_devices.cuda.value,
        precision="16bit",
    )
    model = adapters.add_finetuned_lora_adapters_to_LlavaLlamaForCausalLM_model(
        model, radialog_lora_weights_path=radialog_lora_weights_path
    )
    model = t.cast(peft.LoraModel, model)
    model.model.vision_tower = None
    model = model.merge_and_unload()
    path = os.path.join(MODEL_SAVING_DIR, model_save_name)
    model.get_model().save_pretrained(path)
