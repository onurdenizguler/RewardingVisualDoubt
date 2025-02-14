
# %%
%load_ext autoreload
%autoreload 2
import tokenize

import torch
from datasets import Dataset, IterableDataset
from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from RewardingVisualDoubt import (dataset, inference, mimic_cxr, prompter,
                                  shared, vllm)

#model= vllm.load_pretrained_llava_model()

tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(model_base=vllm.LLAVA_BASE_MODEL_NAME)


# %% Create dataset via dataset generator into HF Dataset (lazy loading)
dataset_generator = dataset.create_dataset_generator_from_mimic_cxr_dataset_df(
    mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
    prompter=prompter.build_report_generation_instruction_from_findings,
    tokenizer=tokenizer,
    device=torch.device(shared.torch_devices.cuda.value),
    shuffle=False,
)

no_split_dataset = IterableDataset.from_generator(generator=dataset_generator)
for datapoint in no_split_dataset:
    print(datapoint)
    break

# %% Create dataset via dataset generator into TorchDataset (lazy loading)
prompted_mimic_cxr_llava_model_input_dataset_train = dataset.PromptedMimicCxrLlavaModelInputDataset(
    mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
    tokenizer=tokenizer,
    prompter=prompter.build_report_generation_instruction_from_findings,
    image_transform=dataset.biovil_image_transformer,
    split=dataset.DatasetSplit.TRAIN,
    device=torch.device(shared.torch_devices.cuda.value),
)
# for datapoint in prompted_mimic_cxr_llava_model_input_dataset_train:
#     print(datapoint)
#     break

print(len(prompted_mimic_cxr_llava_model_input_dataset_train))


# dataloader = DataLoader(
#     dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2
# )


