# %%
%load_ext autoreload
%autoreload 2

# %%
import inference


tokenizer, model, image_processor, context_len = inference.load_visual_language_model_Llava()

# %%
import dataset
import prompter


import create_dataset

dataset_generator = create_dataset.create_dataset_generator_from_mimic_cxr_dataset_df(
    mimic_cxr_df=dataset.create_mimic_cxr_dataset_df(),
    prompter=prompter.build_report_generation_instruction_from_findings,
    tokenizer=tokenizer,
    device=model.device,
)
my_dataset = dataset_generator()
for datapoint in my_dataset:
    print(datapoint["text_prompt_input_ids"].device)
    break


# %%
# generate a report
import torch 
from LLAVA_Biovil.llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    remap_to_uint8,
)
from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1

stop_str = prompter.conv.sep if prompter.conv.sep_style != SeparatorStyle.TWO else prompter.conv.sep2

for datapoint in my_dataset:
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, datapoint["text_prompt_input_ids"])
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=datapoint["text_prompt_input_ids"],
            images=datapoint["images"],
            do_sample=False,
            use_cache=True,
            max_new_tokens=300,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id,
        )

    pred = tokenizer.decode(output_ids[0, datapoint["text_prompt_input_ids"].shape[1] :]).strip().replace("</s>", "")
    print("ASSISTANT: ", pred)
# %%
