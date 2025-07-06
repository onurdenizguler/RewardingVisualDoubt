import typing as t

import torch
import transformers

from RewardingVisualDoubt import prompter, shared

from . import dataset


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


def unpack_report_generation_batch_with_attention_mask(
    device: torch.device | str,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    shared.KeywordsStoppingCriteria,
    torch.Tensor,
]:
    if isinstance(device, str):
        device = torch.device(device)

    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)
    attention_mask = batch["batch_attention_mask"].to(device)
    return input_ids, images, stopping_criteria, attention_mask
