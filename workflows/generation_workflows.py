# %% Set script for interactive development and import modules
from encodings.punycode import T
from IPython.core.getipython import get_ipython

ipython_client = get_ipython()
if ipython_client:
    ipython_client.run_line_magic(magic_name="load_ext", line="autoreload")
    ipython_client.run_line_magic(magic_name="autoreload", line="2")

import torch
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
model = vllm.load_pretrained_llava_model()

# %% load the test split of the mimic-cxr dataset as a PyTorch dataset
report_generation_prompted_mimic_cxr_llava_model_input_test_dataset = dataset.PromptedMimicCxrLlavaModelInputDataset(
    mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
    tokenizer=tokenizer,
    prompter=prompter.build_report_generation_instruction_from_findings,
    image_transform=dataset.biovil_image_transformer,
    split=dataset.DatasetSplit.TRAIN,
    supports_label=True,
    # device=torch.device(shared.torch_devices.cuda.value),
)

dataloader = DataLoader(
    dataset=report_generation_prompted_mimic_cxr_llava_model_input_test_dataset,
    batch_size=1,
    collate_fn=lambda x: dataset.prompted_mimic_cxr_llava_model_input_collate_fn(
        x, padding_value=tokenizer.eos_token_id
    ),  # vicuna/llama does not have a padding token, use the EOS token instead
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

# %% generate a few reports
for idx, datapoint in enumerate(dataloader):
    llava_model_input_dict = datapoint[0]
    llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        llava_model_input_dict, torch.device(shared.torch_devices.cuda.value)
    )
    input_ids, images = (
        llava_model_input_dict["text_prompt_input_ids"],
        llava_model_input_dict["images"],
    )

    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
    pred = inference.generate_radialog_report_for_single_study(
        model, tokenizer, input_ids, images, stopping_criteria
    )
    print(f"Metadata: {datapoint[3]}")
    print(f"Prompt: {datapoint[2]}")
    print(f"Label:", datapoint[1])
    print(f"File_idx {idx}, ASSISTANT: ", pred)
    if idx == 10:
        break


# %% ask a few binary qa questions
