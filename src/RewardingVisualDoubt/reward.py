import math
import typing as t
from RewardingVisualDoubt import shared

SCALE = 5.0
MAX_REWARD = -0.0010005003335835344
MIN_REWARD = -6.907755278982137 / 2
WRONG_FORMAT_PENALTY = -SCALE * 3.0

############### Reward functions for Binary Q&A Training ###############


def default_reward_function(
    confidence: int | None, is_answer_correct: bool | None, granular_confidence: bool = False
) -> float:
    if confidence is None or is_answer_correct is None:
        return WRONG_FORMAT_PENALTY

    upper_confidence_boundary = (
        shared.POSSIBLE_GRANULAR_CONFIDENCES[-1]
        if granular_confidence
        else shared.POSSIBLE_CONFIDENCES[-1]
    )
    normalized_confidence = min(0.999, max(0.001, confidence / upper_confidence_boundary))

    if is_answer_correct:
        score = math.log(normalized_confidence)
    else:
        score = math.log(1 - normalized_confidence)

    # TODO change formula to 2*(score-MIN_REWARD)/(MAX_REWARD-MIN_REWARD) - 1 for cases when MAX_REWARD is not close to 0
    norm_score = (score - MIN_REWARD) / (
        MAX_REWARD - MIN_REWARD
    )  # Will normalize the score to [-1, 1] when max reward is close to 0, hacky formula
    if is_answer_correct:
        norm_score += 0.25
    return float(SCALE * norm_score)


def reward_extreme_confidence(confidence: int, **kwargs) -> float:
    if confidence is None:
        return WRONG_FORMAT_PENALTY

    if confidence > 7:
        score = 5.0
    else:
        score = 0.5
    return score


def reward_teachers_pet_behaviour(
    confidence: int, is_answer_correct: bool | None, **kwargs
) -> float:
    """
    Train a student to be a 'teacher's pet' by rewarding high confidence in correct answers and low confidence in incorrect ones.
    """
    if confidence is None or is_answer_correct is None:
        return WRONG_FORMAT_PENALTY

    if is_answer_correct:
        if confidence > 6:
            score = 5.0
        else:
            score = 0.5
    else:
        if confidence < 4:
            score = 5.0
        else:
            score = 0.5

    return score


def generated_answer_and_confidence_to_reward(
    generated_answer_label: bool | None,
    generated_confidence_value: int | None,
    ground_truth_label: bool,
    granular_confidence: bool,
    reward_function: t.Callable = default_reward_function,
) -> float:

    is_answer_correct = (
        generated_answer_label == ground_truth_label if generated_answer_label is not None else None
    )
    reward = reward_function(
        confidence=generated_confidence_value,
        is_answer_correct=is_answer_correct,
        granular_confidence=granular_confidence,
    )
    return reward
