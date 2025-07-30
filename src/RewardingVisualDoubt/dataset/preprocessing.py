import inspect
import typing as t

import torch
import transformers

from RewardingVisualDoubt import shared

from . import mimic_cxr


def create_report_generation_prompt(
    report_generation_prompter: t.Callable[..., str],
    datapoint: mimic_cxr.MimicCxrDatapoint,
    **kwargs
) -> str:

    prompter_params = inspect.signature(report_generation_prompter).parameters
    possible_inputs = {
        "findings": mimic_cxr.convert_binary_chexpert_findings_to_string(datapoint.disease_labels),
        "possible_confidences": shared.POSSIBLE_CONFIDENCES,
        "gt_report": datapoint.report,
        **kwargs,
    }
    filtered_args = {k: v for k, v in possible_inputs.items() if k in prompter_params}

    return report_generation_prompter(**filtered_args)


def create_prompt_for_binary_qa(
    binary_qa_prompter: t.Callable[..., str],
    datapoint: mimic_cxr.MimicCxrBinaryQADatapoint,
    **kwargs
) -> str:
    prompter_params = inspect.signature(binary_qa_prompter).parameters
    possible_inputs = {
        "chexpert_finding_str": datapoint.disease.value,
        "occurrence_of_disease": datapoint.label == mimic_cxr.ChexpertLabel.POSITIVE,
        "possible_confidences": shared.POSSIBLE_CONFIDENCES,
        **kwargs,
    }
    filtered_args = {k: v for k, v in possible_inputs.items() if k in prompter_params}

    return binary_qa_prompter(**filtered_args)


def create_fact_checking_prompt_for_generated_sentence_against_gt_report(
    fact_checking_prompter: t.Callable[..., str], gt_report: str, generated_sentence: str
) -> str:
    prompter_params = inspect.signature(fact_checking_prompter).parameters
    possible_inputs = {
        "gt_report": gt_report,
        "generated_sentence": generated_sentence,
    }
    filtered_args = {k: v for k, v in possible_inputs.items() if k in prompter_params}

    return fact_checking_prompter(**filtered_args)


def tokenize_text_input(
    text_input: str, tokenizer: transformers.PreTrainedTokenizer, do_unsqueeze: bool = False
) -> torch.Tensor:
    input_ids = t.cast(
        torch.Tensor,
        shared.tokenizer_image_token(
            text_input, tokenizer, shared.LLAVA_IMAGE_TOKEN_INDEX, return_tensors="pt"
        ),
    )
    return input_ids if not do_unsqueeze else input_ids.unsqueeze(0)


def replace_confidence_score_tokens_with_value(
    input_ids: torch.Tensor,
) -> torch.Tensor:
    reversed_ids = input_ids.flip(0)
    COLON_TOKEN_ID = 1115

    try:
        # Find index of the first COLON_TOKEN_ID occuring in the reversed ids
        colon_idx = (reversed_ids == COLON_TOKEN_ID).nonzero(as_tuple=True)[0]
        if colon_idx.numel() > 0:
            colon_idx = colon_idx[0]

            # Replace tokens from index 2 to (colon_idx - 1), ensuring valid range
            if colon_idx > 2:
                reversed_ids[2 : colon_idx - 1] = -100  # IGNORE_INDEX VALUE

        # Reverse back to original order
        output_ids = reversed_ids.flip(0)
    except Exception as e:
        print("Error in _replace_confidence_score_tokens_with_value", e)
        output_ids = input_ids
    return output_ids
