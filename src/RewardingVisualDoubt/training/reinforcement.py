from typing import TypedDict

import numpy as np
import re


from RewardingVisualDoubt import shared


class GameLogs(TypedDict):
    queries: list[str]
    responses: list[str]
    ppo_target_responses: list[str]
    is_answer_correct: list[bool]
    scores: list[float]
    confidences: list[int | None]
    confidences_after_replacement: list[int | None]
    is_confidence_randomly_replaced: list[bool]


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


def _select_random_confidence() -> int:
    return np.random.choice(shared.POSSIBLE_CONFIDENCES)


def overwrite_confidence(
    generated_texts: list[str],
    confidences: list[int | None],
) -> list[str]:
    """
    Overwrite the confidence of the predictions to a new value.
    Expected response format: "Yes, the patience has a disease. {"confidence": 10}"
    """
    updated_generated_texts = []
    for idx, confidence in enumerate(confidences):
        if confidence is not None:  # The generated text is guaranteed to be a valid prediction
            # change the confidence to a new value
            selected_new_confidence = _select_random_confidence()
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
