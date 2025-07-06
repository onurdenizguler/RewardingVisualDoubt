from pathlib import Path

import peft
from transformers import AutoModelForCausalLM

from RewardingVisualDoubt import infrastructure, shared

from . import vllm


def shortcut_load_radialog_binary_qa_model_after_ppo_training() -> (
    vllm.trl.models.modeling_value_head.AutoModelForCausalLMWithValueHead
):
    return vllm.load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters(
        device_str="cuda" if infrastructure.chech_cuda_availablity() else "cpu",
        llava_model_path=vllm.RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
        precision="4bit",
        adapter_path="/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radialog_binary_qa_ppo_training/2025-05-17/best_eval_model/adapter_model.bin",
    )


def shortcut_load_radialog_binary_qa_sft_model() -> vllm.llava.LlavaLlamaForCausalLM:
    return vllm.load_baseline_llava_model_with_vision_modules(
        model_path=Path(vllm.RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value),
        device="cuda" if infrastructure.chech_cuda_availablity() else "cpu",
        precision="4bit",
    )


def shortcut_load_the_original_radialog_model() -> peft.PeftModel:
    return vllm.load_pretrained_llava_model_for_sft_training(
        device_str=(
            shared.torch_devices.cuda.value
            if infrastructure.chech_cuda_availablity()
            else shared.torch_devices.cpu.value
        ),
        precision="4bit",
        radialog_lora_weights_path=vllm.RadialogLoraWeightsPath.ORIGINAL.value,
    )
