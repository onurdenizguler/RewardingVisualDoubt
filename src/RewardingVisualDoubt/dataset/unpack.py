import typing as t

import torch
import transformers

from RewardingVisualDoubt import prompter, shared

from . import dataset, mimic_cxr

#########################################################################################################################
# BINARY QA BATCH UNPACKING
#########################################################################################################################


def unpack_binary_qa_batch(
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    postprocessor: t.Callable[[t.Any], list[torch.Tensor]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    shared.KeywordsStoppingCriteria,
    list[torch.Tensor],
]:

    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)
    input_ids_list = postprocessor(
        input_ids
    )  # WARNING: Is performed, as the ppo_trainer.generate() method handles padding and batching itself
    return input_ids, images, labels, stopping_criteria, input_ids_list


def unpack_binary_qa_batch_with_attention_mask(
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    shared.KeywordsStoppingCriteria,
    torch.Tensor,
]:

    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)
    attention_mask = batch["batch_attention_mask"].to(device)
    return input_ids, images, labels, stopping_criteria, attention_mask


########################################################################################################################
# REPORT GENERATION BATCH UNPACKING
########################################################################################################################


def typical_unpacking_for_report_generation(
    device: torch.device, batch: dataset.MimicCxrLlavaModelInputBatchDict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[mimic_cxr.MimicCxrDatapoint]]:
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    attention_mask = batch["batch_attention_mask"].to(device)
    batch_metadata_list = t.cast(
        list[mimic_cxr.MimicCxrDatapoint], batch["batch_mimic_cxr_datapoint_metadata"]
    )

    return input_ids, images, attention_mask, batch_metadata_list


def unpack_report_generation_batch_with_attention_mask_and_metadata(
    device: torch.device | str,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    shared.KeywordsStoppingCriteria,
    torch.Tensor,
    list[mimic_cxr.MimicCxrDatapoint],
]:
    if isinstance(device, str):
        device = torch.device(device)

    input_ids, images, attention_mask, batch_metadata_list = (
        typical_unpacking_for_report_generation(device, batch)
    )

    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)

    return input_ids, images, stopping_criteria, attention_mask, batch_metadata_list


def unpack_report_generation_batch_with_attention_mask_and_metadata_for_ppo(
    device: torch.device | str,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    postprocessor: t.Callable[[t.Any], list[torch.Tensor]],
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    shared.KeywordsStoppingCriteria,
    list[mimic_cxr.MimicCxrDatapoint],
]:
    if isinstance(device, str):
        device = torch.device(device)

    input_ids, images, _, batch_metadata_list = typical_unpacking_for_report_generation(
        device, batch
    )

    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)
    input_ids_list = postprocessor(
        input_ids
    )  # WARNING: Is performed, as the ppo_trainer.generate() method handles padding and batching itself

    return input_ids, input_ids_list, images, stopping_criteria, batch_metadata_list


def unpack_report_generation_batch_with_attention_mask_and_metadata_for_sft(
    device: torch.device | str,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[mimic_cxr.MimicCxrDatapoint],
]:
    if isinstance(device, str):
        device = torch.device(device)

    input_ids, images, attention_mask, batch_metadata_list = (
        typical_unpacking_for_report_generation(device, batch)
    )

    labels = batch["batch_expected_output_ids"].to(device)
    is_input_verified = torch.all((input_ids == labels) | (labels == -100)).item()

    if not is_input_verified:
        print("(input_ids == labels) | (labels == -100) verification failed.")

    return input_ids, images, labels, attention_mask, batch_metadata_list
