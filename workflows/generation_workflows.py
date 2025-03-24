# %% Set script for interactive development and import modules
from RewardingVisualDoubt import infrastructure

infrastructure.make_ipython_reactive_to_changing_codebase()

import typing as t
import functools

import torch
from datasets import IterableDataset
from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from torch.utils.data import DataLoader

from RewardingVisualDoubt import dataset, inference, mimic_cxr, prompter, shared, vllm

STOP_STR = (
    conv_vicuna_v1.copy().sep
    if conv_vicuna_v1.copy().sep_style != SeparatorStyle.TWO
    else conv_vicuna_v1.copy().sep2
)

# %%
########### 1. TEST RIGINAL RADIALOG MODEL'S REPORT GENERATION BEHAVIOUR ###########
# # %% REPORT GENERATION USE-CASE DATASET
# # %% load the tokenizer
# tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
#     model_base=vllm.LLAVA_BASE_MODEL_NAME
# )

# # %% load the model
# model = vllm.load_pretrained_llava_model(skip_lora_adapters=False)
# NEEDS UPDATE W.R.T. new dataloader logic
# report_generation_prompted_mimic_cxr_llava_model_input_test_dataset = dataset.PromptedMimicCxrLlavaModelInputDataset(
#     mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
#     tokenizer=tokenizer,
#     prompter=prompter.build_report_generation_instruction_from_findings,
#     image_transform=dataset.biovil_image_transformer,
#     split=dataset.DatasetSplit.TRAIN,
#     supports_label=True,
#     # device=torch.device(shared.torch_devices.cuda.value),
# )

########### 2. TEST  ORIGINAL RADIALOG MODEL'S BINARY QA BEHAVIOUR ###########
# %% load the tokenizer
tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)

# %% load the model
model = vllm.load_pretrained_llava_model(skip_lora_adapters=False)
padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)
padding_tokenizer.padding_side = "left"
dataset_test = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
    split=dataset.DatasetSplit.TEST,
    tokenizer=tokenizer,
    prompter=prompter.build_binary_qa_instruction_from_disease_under_study,
)
dataloader_test = dataset.get_mimic_cxr_llava_model_input_dataloader(
    dataset=dataset_test, batch_size=1, padding_tokenizer=padding_tokenizer, num_workers=8
)

inference.generate_from_dataloader(STOP_STR, tokenizer, model, dataloader_test)


########### 3. TEST SFT-TRAINED MODEL FOR BINARY QA TASK WITH CONFIDENCE REQUEST ###########
# %%
SFT_LORA_WEIGHTS_PATH = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/training_checkpoints/best_model_epoch0_step179.pth/adapter_model.bin"
model = vllm.load_pretrained_llava_model_for_sft_training(
    device_str=shared.torch_devices.cuda.value,
    precision="4bit",
    radialog_lora_weights_path=SFT_LORA_WEIGHTS_PATH,
)
tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)

# %% ASK A FEW BINARY QA QUESTIONS EXPECTING CONFIDENCE OUTPUT

padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)
padding_tokenizer.padding_side = "left"
dataset_test = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
    split=dataset.DatasetSplit.TEST,
    tokenizer=tokenizer,
    prompter=functools.partial(
        prompter.build_binary_qa_prompt_with_response_and_confidence_for_sft, is_for_inference=True
    ),
)
dataloader_test = dataset.get_mimic_cxr_llava_model_input_dataloader(
    dataset=dataset_test, batch_size=1, padding_tokenizer=padding_tokenizer, num_workers=8
)

print("Starting inference...")
inference.generate_from_dataloader(
    STOP_STR, tokenizer, model, dataloader_test, num_batches_to_test=50
)
