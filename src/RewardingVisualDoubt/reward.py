import dataclasses
import math
import typing as t

from RewardingVisualDoubt import shared


def normalize_confidence(confidence: int, granular_confidence: bool, clipped: bool = True) -> float:
    upper_confidence_boundary = (
        shared.POSSIBLE_GRANULAR_CONFIDENCES[-1]
        if granular_confidence
        else shared.POSSIBLE_CONFIDENCES[-1]
    )
    if clipped:
        normalized_confidence = min(0.999, max(0.001, confidence / upper_confidence_boundary))
    else:
        normalized_confidence = confidence / upper_confidence_boundary
    return normalized_confidence


############### Reward functions for Binary Q&A Training ###############

SCALE = 5.0
MAX_REWARD = -0.0010005003335835344
MIN_REWARD = -6.907755278982137 / 2
WRONG_FORMAT_PENALTY = -SCALE * 3.0


def default_reward_function(
    confidence: int | None, is_answer_correct: bool | None, granular_confidence: bool = False
) -> float:
    if confidence is None or is_answer_correct is None:
        return WRONG_FORMAT_PENALTY

    normalized_confidence = normalize_confidence(confidence, granular_confidence)

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


#########################################################################################################
# Reward functions for report generation
##########################################################################################################


Scaling = t.Literal["shifted", "centered", "tanh", "logistic"]

WRONG_FORMAT_PENALTY_REPORT_GENERATION: float = -3.0  # returned when inputs are None or invalid


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    scaling: Scaling = "shifted"
    eps: float = 1e-6  # clipping constant (avoid log(0))
    scale: float = 1.0  # final multiplicative scale
    squash_scale: t.Optional[float] = (
        None  # if None, default uses 1 / log(2) so the inflection is at the coin-flip point (Reward = -log(2)) for smooth squashes ("tanh" / "logistic") else, the effective slope is 1/squash_scale.
    )

    def __str__(self) -> str:
        return f"RewardConfig(scaling={self.scaling}, eps={self.eps}, scale={self.scale}, squash_scale={self.squash_scale})"


def raw_log_likelihood_reward(eps: float, p_hat: float, p_star: float) -> float:
    p_hat_clipped = min(max(p_hat, eps), 1.0 - eps)  # clip to avoid log(0)
    # R(p*, p^) = p* log p^ + (1-p*) log(1-p^)
    raw_reward = p_star * math.log(p_hat_clipped) + (1.0 - p_star) * math.log(1.0 - p_hat_clipped)
    return raw_reward


def normalize_and_scale_reward(
    R: float, R0: float, R_min: float, R_max: float, config: RewardConfig
) -> float:

    match config.scaling:
        case "shifted":
            # Linear min–max: R_min -> -1, R_max -> +1 (coin-flip tends to be > 0)
            shaped = 2 * (R - R_min) / (R_max - R_min) - 1
        case "centered":
            # Piecewise linear with coin-flip at 0, lower branch maps [R_min, R0] -> [-1, 0], upper branch maps [R0, 0] -> [ 0, 1]
            if R <= R0:
                shaped = (R - R_min) / (-R0 - R_min) - 1.0
            else:
                shaped = (R - R0) / (-R0)
        case "tanh" | "logistic":
            # Smooth, monotone squashes centred at the coin-flip point.
            # We first shift so that coin-flip is at 0, then apply a squash.

            x = R - R0  # now x=0 at coin-flip (R = R0)

            if config.squash_scale is None:
                # default slope so that the characteristic scale matches log(2)
                alpha = 1.0 / math.log(2.0)
            else:
                alpha = 1.0 / float(config.squash_scale)

            z = alpha * x
            if config.scaling == "tanh":
                shaped = math.tanh(z)  # in (-1, 1)
            else:
                shaped = 2.0 / (1.0 + math.exp(-z)) - 1.0  # in (-1, 1)
    return shaped


def scaled_normalized_log_likelihood_reward(
    confidence: int | None,
    accuracy: float | None,
    granular_confidence: bool = False,
    config: RewardConfig = RewardConfig(),
) -> float:
    """
    Configurable reward for Bernoulli targets using log-likelihood shaped into [-1, 1],
    then scaled by `config.scale`.

    config : RewardConfig
        - scaling:   "shifted"  -> linear min-max to [-1, 1]
                      "centered" -> piecewise linear s.t. R_min -> -1, coin-flip -> 0, best -> +1
                      "tanh"     -> smooth tanh centred at coin-flip, in (-1, 1)
                      "logistic" -> smooth logistic centred at coin-flip, in (-1, 1)
        - eps:       small clip to keep logs finite
        - scale:     final multiplicative scale
        - squash_scale: optional slope control for smooth variants
    """

    if confidence is None or accuracy is None:
        return WRONG_FORMAT_PENALTY_REPORT_GENERATION * config.scale

    normalized_confidence = normalize_confidence(confidence, granular_confidence, clipped=False)
    p_hat = normalized_confidence
    p_star = accuracy

    raw_reward = raw_log_likelihood_reward(config.eps, p_hat, p_star)

    R_min = math.log(config.eps)  # worst-case (after clipping)
    R0 = -math.log(2.0)  # coin-flip (uninformative)
    R_max = 0  # math.log(1 - config.eps)  # best case (perfect prediction)

    shaped = normalize_and_scale_reward(raw_reward, R0, R_min, R_max, config)

    return float(config.scale * shaped)


def get_max_and_min_reward(
    reward_function: t.Callable,
    granular_confidence: bool = False,
    config: RewardConfig = RewardConfig(),
) -> tuple[float, float]:
    """
    Get the maximum possible reward for a given reward function.
    """
    if granular_confidence:
        max_confidence = shared.POSSIBLE_GRANULAR_CONFIDENCES[-1]
    else:
        max_confidence = shared.POSSIBLE_CONFIDENCES[-1]

    # Assuming perfect accuracy (1.0)
    max_reward = reward_function(max_confidence, 1.0, granular_confidence, config)

    # Assuming worst-case accuracy (0.0)
    min_reward = reward_function(max_confidence, 0.0, granular_confidence, config)

    return max_reward, min_reward


########################################################################
# Quadratic, distance-based reward
########################################################################


@dataclasses.dataclass(frozen=True)
class QuadraticBlendRewardConfig:
    beta: float = 3.0  # makes it steeper of a reward decline to get away from the peak reward.
    scale: float = 1.0  # final multiplicative scalar applied to the shaped reward
    r_max: float = 0.0  # reward at the peak (p_hat == p_star)

    def __str__(self) -> str:
        return (
            f"QuadraticBlendRewardConfig(beta={self.beta}, scale={self.scale}, r_max={self.r_max})"
        )


def _quadratic_blend_penalty(distance: float, beta: float) -> float:
    """
    distance in [0, 1]; returns a penalty in [0, 1] with:
      penalty(0) = 0, penalty(1) = 1,
      penalty(d) = d^2 * (1 + beta * d^2) / (1 + beta).
    """
    d2 = distance * distance
    return (d2 * (1.0 + beta * d2)) / (1.0 + beta)


def quadratic_blend_reward_from_probs(
    p_hat: float,
    p_star: float,
    config: QuadraticBlendRewardConfig,
) -> float:
    """
    Distance-only, equal-peak reward. Returns: R = scale * ( r_max - penalty(distance) )
    where penalty is in [0,1] and distance is defined as |p_hat - p_star|.
    """

    p_hat = min(max(p_hat, 0.0), 1.0)
    p_star = min(max(p_star, 0.0), 1.0)

    d = abs(p_hat - p_star)
    penalty = _quadratic_blend_penalty(d, config.beta)
    shaped = config.r_max - penalty
    return float(config.scale * shaped)


def scaled_quadratic_blend_distance_reward(
    confidence: int | None,
    accuracy: float | None,
    granular_confidence: bool,
    config: QuadraticBlendRewardConfig,
) -> float:
    if confidence is None or accuracy is None:
        return WRONG_FORMAT_PENALTY_REPORT_GENERATION * config.scale

    p_hat = normalize_confidence(confidence, granular_confidence, clipped=False)
    p_star = float(accuracy)
    return quadratic_blend_reward_from_probs(p_hat, p_star, config)
