The vllm module provides a unified interface to load and extend the LLaVA Vision-Language Model with support for LoRA adapters and PPO-style reinforcement learning using a value head. It is designed for fine-tuning vision-language models with reward modeling and visual confidence alignment. Within scope of the project, we assume [RaDialog](https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generatio) weights to be given. 

# 🚀 Features

Load and run LLaVA models with vision encoder support.

- Attach LoRA adapters for parameter-efficient fine-tuning.
- Add a value head for reinforcement learning with PPO.
- Unified tokenizer for image-conditioned text generation.

# 📁 Submodules Overview

`vllm.py`

- Downloading models and weights from HuggingFace or local paths.
- Loading base LLaVA models with RaDialog-finetuned vision towers.
- Integrating task-specific LoRA adapters for:
  - Supervised Fine-Tuning (SFT)
  - PPO training with or without fresh adapters
- Merging LoRA adapters with base models for export.
- Freezing parameters selectively (e.g., non-LoRA layers).
- Loading additional non-LoRA weights for vision modules.
- Managing model precision (16bit, 8bit, 4bit) and device mapping.


```adapters.py```

LoRA Adapter and TRL Value Head Utilities for RaDialog's LLaVA Models

This module provides utilities to integrate Low-Rank Adaptation (LoRA) adapters into LLaVA models, including support for value heads used in reinforcement learning. It includes:

- A standard LoRA configuration tailored for RaDialog tasks.
- Functions to:
  - Load RaDialog LoRA weights in a format compatible with `PeftModel` and `AutoModelForCausalLMWithValueHead`.
  - Add pretrained or freshly initialized LoRA adapters to `LlavaLlamaForCausalLM` models.
  - Optionally include a freshly initialized or pretrained value head for RL training.
- Support for quantized models (4-bit/8-bit) using `bitsandbytes`.
- Automatic casting of vision-related modules to `bfloat16` post-LoRA integration for quantized models.

This module ensures flexible integration of task-specific adapters during fine-tuning or inference in RaDialog pipelines.



```tokenizer.py```

This module configures and loads tokenizers for LLaVA-based models, supporting both text-only and multi-modal use cases. It includes:

- Default model base: `"liuhaotian/llava-v1.5-7b"`.
- Functions to:
  - Load a text-only tokenizer and set `pad_token_id` to the `eos_token_id` for Vicuna/LLaMA compatibility.
  - Load a tokenizer with added special tokens for image patch input and optional start/end markers.
  - Modify an existing `LlamaTokenizer` to support multimodal input by appending image-related special tokens.

Useful for setting up tokenizers in both pure text generation and vision-language modeling workflows using LLaVA.

```vision.py```

This module integrates a BioViL-T-based vision tower into a LLaVA model and ensures proper handling of multimodal components. It includes:

- `merge_llm_with_vision_tower`: 
  - Loads and attaches a BioViL-T vision encoder (ResNet50 variant) to the LLaVA model.
  - Loads pretrained vision weights if provided in `non_lora_trainables`.
  - Moves the vision tower to the correct device and precision (bfloat16).
  
- `reset_mm_projector_to_fix_wrong_shape`: 
  - Rebuilds and reinitializes the multimodal projector layer to fix shape mismatches when using a BioViL vision tower.

- `_modify_model_for_image_input`: 
  - Resizes the LLM's token embeddings to match the tokenizer’s vocabulary length.

These utilities are essential for enabling visual input in LLaVA using a biomedical vision backbone like BioViL-T.

# ✅ Use Examples — Pretrained Model Loading & Saving

These functions are the main entry points for loading LLaVA models with BioViL vision and LoRA adapters, customized for various training scenarios:

---

## 🔧 `load_pretrained_llava_model_for_sft_training`
- Loads a baseline LLaVA model with vision.
- Adds **fine-tuned LoRA adapters** (e.g., from Radialog) for **Supervised Fine-Tuning (SFT)** or inference.

---

## 🧠 `load_pretrained_llava_model_for_ppo_training`
- Loads the LLaVA model with vision modules.
- Adds **LoRA adapters** and a **value head** for **PPO training**, enabling reward-based optimization.

---

## 🧪 `load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters`
- Loads a **pre-SFT'd LLaVA model** (e.g., binary QA with confidence).
- Adds **new/fresh LoRA adapters** and a **fresh value head** for further **PPO fine-tuning**.

---

## 💾 `save_lora_merged_llava_model_to_local_dir`
- Merges a LoRA-adapted LLaVA model into a full model.
- Removes the vision tower (if present) before saving.
- Saves the merged model in full precision to a local directory for deployment or checkpointing.
