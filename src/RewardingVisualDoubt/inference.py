import typing as t

import torch
import transformers
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria

from RewardingVisualDoubt import dataset, shared


def generate_radialog_report_for_single_study(
    model, tokenizer, input_ids, image_tensor, stopping_criteria
) -> str:

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            use_cache=True,
            max_new_tokens=300,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id,
        )
    pred = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip().replace("</s>", "")
    return pred


def generate_radialog_answer_for_binary_qa_for_single_study(
    model, tokenizer, input_ids, image_tensor, stopping_criteria
) -> str:
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            use_cache=True,
            max_new_tokens=300,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id,
        )
    pred = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip().replace("</s>", "")
    return pred


def generate_from_dataloader(
    STOP_STR: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    dataloader_test,
    num_batches_to_test: int = 10,
):
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
        pred = generate_radialog_answer_for_binary_qa_for_single_study(
            model, tokenizer, input_ids, images, stopping_criteria
        )
        print(f"\n Metadata: {batch['batch_mimic_cxr_datapoint_metadata']}")
        print(f"Prompt: {batch['batch_prompts']}")
        print(f"Label:", batch["batch_labels"])
        print(f"File_idx {idx}, ASSISTANT: ", pred)
        if idx == num_batches_to_test:
            break
