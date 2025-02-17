# %% Set script for interactive development and import modules
from IPython.core.getipython import get_ipython

ipython_client = get_ipython()
if ipython_client:
    ipython_client.run_line_magic(magic_name="load_ext", line="autoreload")
    ipython_client.run_line_magic(magic_name="autoreload", line="2")

import torch
from torch.utils.data import DataLoader

from RewardingVisualDoubt import dataset, mimic_cxr, prompter, shared, vllm

# from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
# from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)


# model= vllm.load_pretrained_llava_model()


# %%
binary_qa_prompted_mimic_cxr_llava_model_input_dataset_train = (
    dataset.BinaryQAPromptedMimicCxrLlavaModelInputDataset(
        balanced_binary_qa_mimic_cxr_df=mimic_cxr.create_balanced_binary_qa_mimic_cxr_dataset_df(
            mimic_cxr.create_mimic_cxr_dataset_df()
        ),
        tokenizer=tokenizer,
        prompter=prompter.build_binary_qa_instruction_from_disease_under_study,
        image_transform=dataset.biovil_image_transformer,
        split=dataset.DatasetSplit.TRAIN,
    )
)

# %%

dataloader = DataLoader(
    dataset=binary_qa_prompted_mimic_cxr_llava_model_input_dataset_train,
    batch_size=2,
    collate_fn=lambda x: dataset.prompted_mimic_cxr_llava_model_input_collate_fn(
        x, padding_value=tokenizer.eos_token_id
    ),  # vicuna/llama does not have a padding token, use the EOS token instead
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

for idx, batch in enumerate(dataloader):
    print(batch)
    break

# %%
