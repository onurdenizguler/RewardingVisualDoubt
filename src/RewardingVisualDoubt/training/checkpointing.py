import datetime
import dataclasses
import os
import pathlib as path
import typing as t

import numpy as np
import peft
import trl


from RewardingVisualDoubt import shared, reward, evaluation

from . import parameters


@dataclasses.dataclass
class PatienceTracker:
    value: int = 0


@dataclasses.dataclass
class TrainingSuccessKPIs:
    mean_score_train: list[float]
    conf_distribution_kl_eval: list[float]
    mean_score_eval: list[float]
    ece_eval: list[float]
    heuristic_aggregated_score_eval: list[float]
    weighted_mean_of_std_of_accuracies: list[float]


def init_training_success_kpis() -> TrainingSuccessKPIs:
    return TrainingSuccessKPIs(
        mean_score_train=[],
        conf_distribution_kl_eval=[],
        mean_score_eval=[],
        ece_eval=[],
        heuristic_aggregated_score_eval=[],
        weighted_mean_of_std_of_accuracies=[],
    )


def save_best_eval_lora_adapters_to_dir(
    model: peft.PeftModelForCausalLM,
    epoch: int,
    step: int,
    out_dir: str | path.Path,
    name_of_fine_tuning: str,
):
    save_dir = os.path.join(
        out_dir,
        name_of_fine_tuning,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        "best_eval_model_epoch_{}_step_{}".format(epoch + 1, step + 1),
    )
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)


def save_best_eval_lora_adapters_and_value_head_to_dir(
    ppo_trainer: trl.PPOTrainer,
    epoch: int,
    step: int,
    out_dir: str | path.Path,
    name_of_fine_tuning: str,
    reward_config: reward.RewardConfig | None = None,
):
    if not reward_config:
        middle_dir_name = "no_reward_config_specified"
    else:
        middle_dir_name = str(reward_config)
    save_dir = os.path.join(
        out_dir,
        name_of_fine_tuning,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        middle_dir_name,
        "best_eval_model_epoch_{}_step_{}".format(epoch + 1, step + 1),
    )
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.save_pretrained(save_dir)


def save_training_checkpoint_lora_adapters_and_value_head_to_dir(
    ppo_trainer: trl.PPOTrainer,
    epoch: int,
    step: int,
    out_dir: str | path.Path,
    name_of_fine_tuning: str,
    reward_config: reward.RewardConfig | None = None,
):

    if not reward_config:
        middle_dir_name = "no_reward_config_specified"
    else:
        middle_dir_name = str(reward_config)
    save_dir = os.path.join(
        out_dir,
        name_of_fine_tuning,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        middle_dir_name,
        "checkpoint_epoch_{}_step_{}".format(epoch + 1, step + 1),
    )
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.save_pretrained(save_dir)


def calculate_relative_improvement(
    current_score: float,
    all_scores: list[float],
) -> float:
    return (current_score - max(all_scores)) / (abs(max(all_scores)) + 1e-12)


def calculate_slope(y: np.ndarray) -> float:
    """Least-squares slope of y vs. [0, 1, …, len(y)-1]."""
    x = np.arange(len(y))
    x_mean, y_mean = x.mean(), y.mean()
    num = np.dot(x - x_mean, y - y_mean)
    den = np.dot(x - x_mean, x - x_mean)
    return 0.0 if den == 0 else num / den


def report_generation_ppo_decision_to_break(
    step: int,
    patience: PatienceTracker,
    heuristic_aggregated_scores: list[float],
    hyperparameters: parameters.ReportGenerationPPOHyperparameters,
) -> bool:

    window = 10  # look at the last 10 scores
    min_slope = 0.0  # treat <= 0 as "flat" / no improvement
    slope_of_latest_scores = calculate_slope(
        np.array(
            heuristic_aggregated_scores[-window:]
            if window < len(heuristic_aggregated_scores)
            else heuristic_aggregated_scores
        )
    )
    decision_to_break: bool = False
    reason: str = ""

    if slope_of_latest_scores >= min_slope:
        patience.value = 0
    else:
        patience.value += 1
        if (
            hyperparameters.early_stopping_patience
            and patience.value >= hyperparameters.early_stopping_patience
        ):
            decision_to_break = True
            reason = "lack of relative improvement after early_stopping_patience"

    if hyperparameters.max_steps and step >= hyperparameters.max_steps:
        decision_to_break = True
        reason = "max_steps"

    if decision_to_break:
        print(
            f"Early stopping at step {step} due to: {reason} "
            f"best_score={max(heuristic_aggregated_scores):.4f}, current_score={heuristic_aggregated_scores[-1]:.4f}, "
            # f"relative_improvement={relative_improvement:.4f}"
        )
        return True

    return False
