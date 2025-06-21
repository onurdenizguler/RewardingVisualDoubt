import typing as t

import torch
import transformers


from . import dataset
from RewardingVisualDoubt import prompter, shared


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
