# %% Set script for interactive development and import modules
from IPython.core.getipython import get_ipython

ipython_client = get_ipython()
if ipython_client:
    ipython_client.run_line_magic(magic_name="load_ext", line="autoreload")
    ipython_client.run_line_magic(magic_name="autoreload", line="2")

import torch
from datasets import IterableDataset

from RewardingVisualDoubt import dataset, mimic_cxr, prompter, shared, vllm

# from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
# from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


# model= vllm.load_pretrained_llava_model()

tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)


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
report_generation_prompted_mimic_cxr_llava_model_input_dataset_train = (
    dataset.PromptedMimicCxrLlavaModelInputDataset(
        mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
        tokenizer=tokenizer,
        prompter=prompter.build_report_generation_instruction_from_findings,
        image_transform=dataset.biovil_image_transformer,
        split=dataset.DatasetSplit.TRAIN,
        device=torch.device(shared.torch_devices.cuda.value),
    )
)
# for datapoint in prompted_mimic_cxr_llava_model_input_dataset_train:
#     print(datapoint)
#     break

print(len(report_generation_prompted_mimic_cxr_llava_model_input_dataset_train))

# %%
binary_qa_prompted_mimic_cxr_llava_model_input_dataset_train = (
    dataset.PromptedMimicCxrLlavaModelInputDataset(
        mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
        tokenizer=tokenizer,
        prompter=prompter.build_binary_qa_instruction_from_findings_under_study,
        image_transform=dataset.biovil_image_transformer,
        split=dataset.DatasetSplit.TRAIN,
        device=torch.device(shared.torch_devices.cuda.value),
    )
)

# %%

dataloader = DataLoader(
    dataset=binary_qa_prompted_mimic_cxr_llava_model_input_dataset_train,
    batch_size=8,
    collate_fn=lambda x: collate_fn(
        x, padding_value=tokenizer.eos_token_id
    ),  # vicuna/llama does not have a padding token, use the EOS token instead
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

# %%
