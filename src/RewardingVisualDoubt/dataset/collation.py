import typing as t

import torch
import transformers

from . import dataset

######################## COLLATE FUNCTIONS FOR DATALOADERS ########################


def pad_batch_text_sequences(
    batch: list[transformers.tokenization_utils_base.EncodedInput],
    padding_tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a batch of text sequences using the provided tokenizer.
    """
    padded_text_inputs_dict = padding_tokenizer.pad(
        encoded_inputs={"input_ids": batch},
        padding=True,
        max_length=None,
        return_tensors="pt",
    )

    text_inputs = t.cast(torch.Tensor, padded_text_inputs_dict["input_ids"])
    attention_mask = t.cast(torch.Tensor, padded_text_inputs_dict["attention_mask"])
    return text_inputs, attention_mask


def collate_text_batch_of_list_of_input_ids(
    text_input_ids: list[transformers.tokenization_utils_base.EncodedInput],
    padding_tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_inputs, attention_mask = pad_batch_text_sequences(
        batch=text_input_ids,
        padding_tokenizer=padding_tokenizer,
    )
    return text_inputs, attention_mask


def prompted_mimic_cxr_llava_model_input_collate_fn(
    batch: list[dataset.ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDict],
    padding_tokenizer: transformers.PreTrainedTokenizer,
) -> dataset.MimicCxrLlavaModelInputBatchDict:
    """
    Custom collate function to pad text sequences and stack images.
    """
    llava_model_inputs, labels, prompts, mimic_cxr_datapoint_metadatas = zip(
        *[
            (
                item["llava_model_input_dict"],
                item["label"],
                item["prompt"],
                item["mimic_cxr_datapoint_metadata"],
            )
            for item in batch
        ]
    )

    text_inputs = t.cast(
        list[transformers.tokenization_utils_base.EncodedInput],
        [
            torch.as_tensor(sample["text_prompt_input_ids"]).clone().detach()
            for sample in llava_model_inputs
        ],
    )
    images = torch.stack([torch.as_tensor(sample["images"]) for sample in llava_model_inputs])

    text_inputs, attention_mask = pad_batch_text_sequences(text_inputs, padding_tokenizer)

    batch_llava_model_inputs = dataset.LlavaModelInputDict(
        text_prompt_input_ids=text_inputs, images=images
    )
    batch_labels = (
        torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else list(labels)
    )
    batch_metadata = list(mimic_cxr_datapoint_metadatas)
    batch_prompts = list(prompts)

    return dataset.MimicCxrLlavaModelInputBatchDict(
        batch_llava_model_input_dict=batch_llava_model_inputs,
        batch_attention_mask=attention_mask,
        batch_labels=batch_labels,
        batch_prompts=batch_prompts,
        batch_mimic_cxr_datapoint_metadata=batch_metadata,
    )


def prompted_mimic_cxr_llava_model_input_collate_fn_for_sft(
    batch: list[dataset.BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT],
    padding_tokenizer: transformers.PreTrainedTokenizer,
) -> dataset.MimicCxrLlavaModelInputBatchDictForSFT:

    llava_model_inputs, labels, prompts, mimic_cxr_datapoint_metadatas, expected_output_ids = zip(
        *[
            (
                item["llava_model_input_dict"],
                item["label"],
                item["prompt"],
                item["mimic_cxr_datapoint_metadata"],
                item["expected_output_ids"],
            )
            for item in batch
        ]
    )

    text_inputs = t.cast(
        list[transformers.tokenization_utils_base.EncodedInput],
        [
            torch.as_tensor(sample["text_prompt_input_ids"]).clone().detach()
            for sample in llava_model_inputs
        ],
    )
    images = torch.stack([torch.as_tensor(sample["images"]) for sample in llava_model_inputs])

    text_inputs, attention_mask = pad_batch_text_sequences(text_inputs, padding_tokenizer)

    batch_llava_model_inputs = dataset.LlavaModelInputDict(
        text_prompt_input_ids=text_inputs, images=images
    )
    batch_labels = (
        torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else list(labels)
    )
    batch_metadata = list(mimic_cxr_datapoint_metadatas)
    batch_prompts = list(prompts)

    padded_expected_ouput_ids, _ = pad_batch_text_sequences(
        batch=t.cast(list[transformers.tokenization_utils_base.EncodedInput], expected_output_ids),
        padding_tokenizer=padding_tokenizer,
    )

    return dataset.MimicCxrLlavaModelInputBatchDictForSFT(
        batch_llava_model_input_dict=batch_llava_model_inputs,
        batch_attention_mask=attention_mask,
        batch_labels=batch_labels,
        batch_prompts=batch_prompts,
        batch_mimic_cxr_datapoint_metadata=batch_metadata,
        batch_expected_output_ids=padded_expected_ouput_ids,
    )
