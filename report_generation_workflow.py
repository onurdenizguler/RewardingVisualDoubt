# %%
%load_ext autoreload
%autoreload 2
from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria

from RewardingVisualDoubt import dataset, inference, mimic_cxr, prompter, vllm

STOP_STR = conv_vicuna_v1.copy().sep if conv_vicuna_v1.copy().sep_style != SeparatorStyle.TWO else conv_vicuna_v1.copy().sep2

# %% load the model
model = vllm.load_pretrained_llava_model()
tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(model_base=vllm.LLAVA_BASE_MODEL_NAME)
# %% load the entire dataset
dataset_ = dataset.create_dataset_generator_from_mimic_cxr_dataset_df(
    mimic_cxr_df=mimic_cxr.create_mimic_cxr_dataset_df(),
    prompter=prompter.build_report_generation_instruction_from_findings,
    tokenizer=tokenizer,
    device=model.device,
)()

# %% generate a few reports
for idx, datapoint in enumerate(dataset_):
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, datapoint["text_prompt_input_ids"])
    pred = inference.generate_report_for_single_study(
        model, tokenizer, datapoint["text_prompt_input_ids"], datapoint["images"], stopping_criteria
    )
    print(f"File_idx {idx}, ASSISTANT: ", pred)
    if idx == 10:
        break