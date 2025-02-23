# TEMPLATE PUNISHMENTS
# check if the confidence is an int
# check if the confidence is between 0 and 10
# check if the confidence and reward key exist

# CONFIDENCE ESTIMATE PUNISHMENTS


import math


MAX_REWARD = -0.0010005003335835344
MIN_REWARD = -6.907755278982137 / 2
SCALE = 5.0
WRONG_FORMAT_PENALTY = -SCALE * 3.0


def reward_function(confidence: int | None, is_answer_correct: bool) -> float:
    if confidence is None:
        return WRONG_FORMAT_PENALTY

    normalized_confidence = min(0.999, max(0.001, confidence / 10))

    if is_answer_correct:
        score = math.log(normalized_confidence)
    else:
        score = math.log(1 - normalized_confidence)

    norm_score = (score - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
    if is_answer_correct:
        norm_score += 0.25
    return float(SCALE * norm_score)


def generated_answer_and_confidence_to_reward(
    generated_answer_label: bool | None,
    generated_confidence_value: int | None,
    ground_truth_label: bool,
) -> float:

    is_correct = generated_answer_label == ground_truth_label
    reward = reward_function(generated_confidence_value, is_correct)
    return reward
