from typing import TypedDict

import torch
import transformers

from RewardingVisualDoubt import shared

TOKEN_INDEX_OF_THE_WORD_IMAGE = (
    1967  # 1967 is the index of the image token in the tokenizer (the word image)
)


class ReformulatedQueryAndResponseDict(TypedDict):
    query_ids: torch.Tensor
    response_ids: torch.Tensor


def remove_padding(tensor, pad_token) -> torch.Tensor:
    # TODO: better alternative is output = generation[(1 - mask).sum() :]  # remove padding
    start_idx = 0
    # while start_idx < len(tensor) and tensor[start_idx] == pad_token:
    #     start_idx += 1

    # Find the end index where padding starts again
    end_idx = len(tensor) - 1
    while end_idx >= 0 and tensor[end_idx] == pad_token:
        end_idx -= 1

    # Slice the tensor to remove padding, add 1 to end_idx to include the last non-pad token
    trimmed_tensor = tensor[start_idx : end_idx + 1]
    return trimmed_tensor


def remove_preciding_padding_from_batch_tensor(batch: torch.Tensor):
    trimmed_sequences = []
    for seq in batch:
        # Find the first occurrence of token `1`
        ones = (seq == 1).nonzero(as_tuple=True)[0]
        if len(ones) > 0:
            first_one_idx = ones[0].item()
            trimmed_seq = seq[first_one_idx:]
            trimmed_sequences.append(trimmed_seq)
        else:
            raise Exception("Error at remove_preciding_padding_from_batch_tensor")

    # If you want to pad back to the same length (optional):
    return trimmed_sequences


def remove_trailing_padding_from_prediction(
    prediction: torch.Tensor, pad_token_id: int | None
) -> list[torch.Tensor]:
    """
    Remove padding tokens from the end of the tensor
    args:
        tensor: torch.Tensor: a batch of generations by an LM with each generation having trailing padding tokens at the end
        pad_token: int
    """
    assert pad_token_id is not None, "pad_token_id must be provided"
    return [remove_padding(p, pad_token_id) for p in prediction]


def replace_image_token_with_another_token(
    prediction: torch.Tensor,
    image_token_id: int = shared.LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
) -> torch.Tensor:
    # TODO: Consider adding the special image token to tokenizer for future editions
    prediction[prediction == image_token_id] = replacement_token_id
    return prediction


def replace_image_token_with_another_token_for_list_of_tensors(
    predictions: list[torch.Tensor],
    image_token_id: int = shared.LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
) -> list[torch.Tensor]:
    return [
        replace_image_token_with_another_token(p, image_token_id, replacement_token_id)
        for p in predictions
    ]


def get_likeliest_token_from_logits(
    logits: torch.Tensor,
):
    """
    Get the most likely token from the logits
    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
    Returns:
        torch.Tensor of shape (batch_size, sequence_length)
    """
    # Get the most likely token from the logits
    return torch.argmax(logits, dim=-1)


def reformulate_query_and_response_for_binary_qa(
    query_ids: torch.Tensor, response: str, tokenizer: transformers.PreTrainedTokenizer
) -> ReformulatedQueryAndResponseDict:
    """
    Merge the verbal part of a Binary Q&A questions's response with the Binary Q&A question while
    leaving out the confidence part as the response to perform PPO on.
    We assume that whenever the response has the left curly bracket, it has the confidence part.
    Therefore we split the response by the left curly bracket and take the first part as the verbal part.
        Args:
            query: str: the question (Typically "<some-instructions> Does the image display disease?")
            response: str: the generated response (Typically 'Yes, the image displays disease. {"confidence": 4}')
    """
    if not "{" in response:
        return ReformulatedQueryAndResponseDict(
            query_ids=query_ids,
            response_ids=torch.tensor(
                tokenizer.encode(response, add_special_tokens=False),
                dtype=torch.long,
                device=query_ids.device,
            ),
        )

    verbal_part = response.split("{")[0]
    verbal_part_ids = tokenizer.encode(verbal_part, add_special_tokens=False)
    verbal_part_ids = torch.tensor(verbal_part_ids, dtype=torch.long, device=query_ids.device)
    query_ids_updated = torch.cat(
        (query_ids, verbal_part_ids),
        dim=0,
    )
    confidence_only_response = "{" + response.split("{")[1].strip()
    confidence_only_response_ids = torch.tensor(
        tokenizer.encode(confidence_only_response, add_special_tokens=False),
        dtype=torch.long,
        device=query_ids.device,
    )

    return ReformulatedQueryAndResponseDict(
        query_ids=query_ids_updated,
        response_ids=confidence_only_response_ids,
    )
