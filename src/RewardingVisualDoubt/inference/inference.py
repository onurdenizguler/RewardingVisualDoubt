import dataclasses
import typing as t

import peft
import torch
import transformers
import trl

from RewardingVisualDoubt import dataset, shared, response


def generate_radialog_report_for_single_study(
    model, tokenizer, input_ids, image_tensor, stopping_criteria, attention_mask=None
) -> str:
    if attention_mask is not None:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.pad_token_id,
            )

    else:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
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


def batch_decode_generated_texts(
    tokenizer: transformers.PreTrainedTokenizer,
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> list[str]:
    generated_texts = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    return [text.strip().replace("</s>", "") for text in generated_texts]


def generate_from_dataloader_for_batch(
    STOP_STR: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    dataloader_test,
    num_batches_to_test: int = 10,
    **generation_kwargs,
):
    for i, batch in enumerate(dataloader_test):
        batch = t.cast(dataset.MimicCxrLlavaModelInputBatchDict, batch)
        batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
        batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
            batch_llava_model_input_dict, torch.device(shared.torch_devices.cuda.value)
        )
        input_ids, images = (
            batch_llava_model_input_dict["text_prompt_input_ids"],
            batch_llava_model_input_dict["images"],
        )
        stopping_criteria = shared.KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
        attention_mask = batch["batch_attention_mask"]
        output_ids = model.generate(
            input_ids=input_ids,
            images=images,
            do_sample=True,
            use_cache=True,
            max_new_tokens=300,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        generated_texts = batch_decode_generated_texts(tokenizer, output_ids, input_ids)
        for idx, text in enumerate(generated_texts):
            pred = text.strip().replace("</s>", "")
            print(f"\n Metadata: {batch['batch_mimic_cxr_datapoint_metadata'][idx]}")
            print(f"Prompt: {batch['batch_prompts'][idx]}")
            print(f"Label:", bool(batch["batch_labels"][idx]))
            print(f"File_idx {idx}, ASSISTANT: ", pred)
        if i + 1 == num_batches_to_test:
            break


@dataclasses.dataclass
class GeneratedReportWithConfidenceRecord:
    subject_id: int
    study_id: int
    gt_report: str
    generated_report: str
    confidence: int | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def generate_reports_and_parse_confidences_for_batch(
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    tokenizer: transformers.PreTrainedTokenizer,
    device: torch.device,
    model: trl.AutoModelForCausalLMWithValueHead | peft.PeftModelForCausalLM,
    generation_kwargs: dict,
) -> list[GeneratedReportWithConfidenceRecord]:
    input_ids, images, stopping_criteria, attention_mask, batch_metadata_list = (
        dataset.unpack_report_generation_batch_with_attention_mask_and_metadata(
            device=device,
            tokenizer=tokenizer,
            batch=batch,
        )
    )

    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            **generation_kwargs,
        )

    generated_ids = t.cast(torch.Tensor, generated_ids)

    generated_texts = batch_decode_generated_texts(
        tokenizer=tokenizer,
        output_ids=generated_ids,
        input_ids=input_ids,
    )

    confidences = response.parse_confidences(generated_texts, granular_confidence=False)
    records = []
    for text_idx, text in enumerate(generated_texts):

        records.append(
            GeneratedReportWithConfidenceRecord(
                subject_id=int(batch_metadata_list[text_idx].subject_id),
                study_id=int(batch_metadata_list[text_idx].study_id),
                gt_report=str(batch_metadata_list[text_idx].report),
                generated_report=str(text),
                confidence=confidences[text_idx],
            )
        )
    return records
