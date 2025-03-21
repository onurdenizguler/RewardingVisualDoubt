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

# %% load the tokenizer
tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)

# %% load the model
model = vllm.load_pretrained_llava_model(skip_lora_adapters=False)

# %% REPORT GENERATION USE-CASE DATASET
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

# dataloader = DataLoader(
#     dataset=report_generation_prompted_mimic_cxr_llava_model_input_test_dataset,
#     batch_size=1,
#     collate_fn=lambda x: dataset.prompted_mimic_cxr_llava_model_input_collate_fn(
#         x, padding_value=tokenizer.eos_token_id
#     ),  # vicuna/llama does not have a padding token, use the EOS token instead
#     shuffle=False,
#     num_workers=8,
#     pin_memory=True,
#     drop_last=True,
#     persistent_workers=True,
# )

# # %% GENERATE A FEW REPORTS
# for idx, datapoint in enumerate(dataloader):
#     llava_model_input_dict = datapoint[0]
#     llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
#         llava_model_input_dict, torch.device(shared.torch_devices.cuda.value)
#     )
#     input_ids, images = (
#         llava_model_input_dict["text_prompt_input_ids"],
#         llava_model_input_dict["images"],
#     )

#     stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
#     pred = inference.generate_radialog_report_for_single_study(
#         model, tokenizer, input_ids, images, stopping_criteria
#     )
#     print(f"Metadata: {datapoint[3]}")
#     print(f"Prompt: {datapoint[2]}")
#     print(f"Label:", datapoint[1])
#     print(f"File_idx {idx}, ASSISTANT: ", pred)
#     if idx == 10:
#         break


# %% ASK A FEW BINARY QA QUESTIONS
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

for idx, batch in enumerate(dataloader_test):
    batch = t.cast(dataset.MimicCxrLlavaModelInputBatchDict, batch)
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, torch.device(shared.torch_devices.cuda.value)
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
    pred = inference.generate_radialog_answer_for_binary_qa_for_single_study(
        model, tokenizer, input_ids, images, stopping_criteria
    )
    print(f"\n Metadata: {batch['batch_mimic_cxr_datapoint_metadata']}")
    print(f"Prompt: {batch['batch_prompts']}")
    print(f"Label:", batch["batch_labels"])
    print(f"File_idx {idx}, ASSISTANT: ", pred)
    if idx == 10:
        break


########### TESTING SFT-TRAINED MODEL FOR BINARY QA TASK WITH CONFIDENCE REQUEST ###########
# %%
SFT_LORA_WEIGHTS_PATH = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/training_checkpoints/best_model_epoch0_step299.pth/adapter_model.bin"
model = vllm.load_pretrained_llava_model(
    skip_lora_adapters=True, device=shared.torch_devices.cuda.value, precision="4bit"
)
model = vllm.add_pretrained_RaDialog_lora_adapters_to_LlavaLlamaForCausalLM_model(
    model, radialog_lora_weights_path=SFT_LORA_WEIGHTS_PATH
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

for idx, batch in enumerate(dataloader_test):
    batch = t.cast(dataset.MimicCxrLlavaModelInputBatchDict, batch)
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, torch.device(shared.torch_devices.cuda.value)
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
    pred = inference.generate_radialog_answer_for_binary_qa_for_single_study(
        model, tokenizer, input_ids, images, stopping_criteria
    )
    print(f"\n Metadata: {batch['batch_mimic_cxr_datapoint_metadata']}")
    print(f"Prompt: {batch['batch_prompts']}")
    print(f"Label:", batch["batch_labels"])
    print(f"File_idx {idx}, ASSISTANT: ", pred)
    if idx == 10:
        break


# %%
# ####################################################################################################
# ARCHIVED CODE
# Create dataset via dataset generator into HF Dataset (lazy loading)
# dataset_generator = dataset.create_dataset_generator_from_mimic_cxr_dataset_df(
#     mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
#     prompter=prompter.build_report_generation_instruction_from_findings,
#     tokenizer=tokenizer,
#     device=torch.device(shared.torch_devices.cuda.value),
#     shuffle=False,
# )

# no_split_dataset = IterableDataset.from_generator(generator=dataset_generator)
# for datapoint in no_split_dataset:
#     print(datapoint)
#     break
