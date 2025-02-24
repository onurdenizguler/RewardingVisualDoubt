import torch

from RewardingVisualDoubt import dataset

LLAVA_IMAGE_TOKEN_INDEX = -200  # as defined by the llava repo
TOKEN_INDEX_OF_THE_WORD_IMAGE = (
    1967  # 1967 is the index of the image token in the tokenizer (the word image)
)


# def remove_padding_and_append(input_ids, new_input_ids_list, pad_token_id=0):
#     """
#     Args:
#         input_ids (torch.Tensor): [batch_size, seq_len] input tensor with padding.
#         new_input_ids_list (List[torch.Tensor]): List of tensors to append to each sequence.
#         pad_token_id (int): The token ID used for padding.

#     Returns:
#         torch.Tensor: Padded tensor after appending new input IDs.
#     """
#     batch_size, seq_len = input_ids.size()
#     processed_seqs = []

#     for i in range(batch_size):
#         # Get the sequence without padding
#         seq = input_ids[i]
#         valid_len = (seq != pad_token_id).sum()  # Count non-pad tokens
#         trimmed_seq = seq[:valid_len]

#         # Append new input_ids
#         new_ids = new_input_ids_list[i]  # Should be a tensor
#         combined_seq = torch.cat([trimmed_seq, new_ids], dim=0)

#         processed_seqs.append(combined_seq)

#     # Pad sequences to the same length
#     padded_seqs = pad_sequence(processed_seqs, batch_first=True, padding_value=pad_token_id)

#     return padded_seqs


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
    image_token_id: int = LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
) -> torch.Tensor:
    # TODO: Consider adding the special image token to tokenizer for future editions
    prediction[prediction == image_token_id] = replacement_token_id
    return prediction


def replace_image_token_with_another_token_for_list_of_tensors(
    predictions: list[torch.Tensor],
    image_token_id: int = LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
) -> list[torch.Tensor]:
    return [
        replace_image_token_with_another_token(p, image_token_id, replacement_token_id)
        for p in predictions
    ]
