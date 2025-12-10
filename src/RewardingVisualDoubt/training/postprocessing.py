import typing as t

import numpy as np
import re
import torch
import transformers

from RewardingVisualDoubt import response, shared


TOKEN_INDEX_OF_THE_WORD_IMAGE = (
    1967  # 1967 is the index of the image token in the tokenizer (the word image)
)


class ReformulatedQueryAndResponseDict(t.TypedDict):
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


# TODO: Change name to "preceding", add return typehinting
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
    by_cloning: bool = False,
) -> torch.Tensor:
    # TODO: Consider adding the special image token to tokenizer for future editions
    if by_cloning:
        prediction = prediction.clone()
    prediction[prediction == image_token_id] = replacement_token_id
    return prediction


def replace_image_token_with_another_token_for_list_of_tensors(
    predictions: list[torch.Tensor],
    image_token_id: int = shared.LLAVA_IMAGE_TOKEN_INDEX,
    replacement_token_id: int = TOKEN_INDEX_OF_THE_WORD_IMAGE,
    by_cloning: bool = False,
) -> list[torch.Tensor]:
    return [
        replace_image_token_with_another_token(p, image_token_id, replacement_token_id, by_cloning)
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


def reformulate_query_and_response(
    query_ids: torch.Tensor, response: str, tokenizer: transformers.PreTrainedTokenizer
) -> ReformulatedQueryAndResponseDict:
    """
    Merge the verbal part of a generated response with the input prompt while
    leaving out the confidence part as the response to perform PPO on.
    We assume that whenever the response has the left curly bracket, it has the confidence part.
    Therefore we split the response by the left curly bracket and take the first part as the verbal part.
        Args:
            query: str: the question (E.g. for binary q&a "<some-instructions> Does the image display disease?")
            response: str: the generated response (E.g. for binary q&a 'Yes, the image displays disease. {"confidence": 4}')
    """
    if not "{" in response or "confidence" not in response.split("{")[1]:
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


def remove_confidence_part_from_generated_responses(responses: list[str]) -> list[str]:
    confidence_stripped_generated_responses = []
    for response in responses:
        if "confidence" in response:
            if len(response.split(("confidence"))[0]) > 2:
                confidence_stripped_generated_responses.append(
                    response.split(("confidence"))[0][:-2].strip()
                )
            else:
                confidence_stripped_generated_responses.append(
                    response.split(("confidence"))[0].strip()
                )
        else:
            confidence_stripped_generated_responses.append(response.strip())
    return confidence_stripped_generated_responses


def normalize_confidence_scores(
    confidence_scores: list[int | None], granular: bool = False
) -> list[float]:
    """
    Normalize confidence scores to a range of 0 to 1.0.
    """
    if granular:
        return [
            score / shared.POSSIBLE_GRANULAR_CONFIDENCES[-1]
            for score in confidence_scores
            if score is not None
        ]
    return [
        score / shared.POSSIBLE_CONFIDENCES[-1] for score in confidence_scores if score is not None
    ]


def _replace_confidence_value_in_text(
    text: str, old_confidence_value: int, new_confidence_value: int
) -> str:

    # Match "confidence" (with optional quotes and whitespace), colon, optional space, and capture the number
    pattern = r'(["\']?confidence["\']?\s*:\s*)(\d{1,2})'

    def replace_confidence(match):
        current_val = int(match.group(2))
        if current_val == old_confidence_value:
            return f"{match.group(1)}{new_confidence_value}"
        else:
            return match.group(0)  # no change

    return re.sub(pattern, replace_confidence, text)


def _select_random_confidence(granular_confidence: bool) -> int:
    return (
        np.random.choice(shared.POSSIBLE_CONFIDENCES)
        if not granular_confidence
        else np.random.choice(shared.POSSIBLE_GRANULAR_CONFIDENCES)
    )


def overwrite_confidence(
    generated_texts: list[str],
    confidences: list[int | None],
    granular_confidence: bool,
) -> list[str]:
    """
    Overwrite the confidence of the predictions to a new value.
    Expected response format: "Yes, the patience has a disease. {"confidence": 10}"
    """
    updated_generated_texts = []
    for idx, confidence in enumerate(confidences):
        if confidence is not None:  # The generated text is guaranteed to be a valid prediction
            # change the confidence to a new value
            selected_new_confidence = _select_random_confidence(granular_confidence)
            while confidence == selected_new_confidence:
                selected_new_confidence = _select_random_confidence(granular_confidence)
            try:
                updated_generated_texts.append(
                    _replace_confidence_value_in_text(
                        generated_texts[idx], confidence, selected_new_confidence
                    )
                )
            except:
                updated_generated_texts.append(generated_texts[idx])

        else:
            updated_generated_texts.append(generated_texts[idx])

    return updated_generated_texts


def handle_random_confidence_replacement(
    tokenizer: transformers.PreTrainedTokenizer,
    generated_texts: list[str],
    generated_confidence_values: list[int | None],
    device: torch.device,
    granular_confidence: bool,
) -> tuple[list[torch.Tensor], list[str], list[int | None], list[int | None], list[bool]]:
    old_generated_confidence_values = generated_confidence_values.copy()
    generated_texts = overwrite_confidence(
        generated_texts, generated_confidence_values, granular_confidence=granular_confidence
    )
    generated_ids = []
    for text in generated_texts:
        ids = t.cast(torch.Tensor, tokenizer.encode(text, return_tensors="pt"))
        generated_ids.append(ids.squeeze(0).to(device=device))
    generated_confidence_values = response.parse_confidences(generated_texts, granular_confidence)
    is_confidence_randomly_replaced = [
        old_conf != new_conf
        for old_conf, new_conf in zip(old_generated_confidence_values, generated_confidence_values)
    ]

    return (
        generated_ids,
        generated_texts,
        generated_confidence_values,
        old_generated_confidence_values,
        is_confidence_randomly_replaced,
    )
