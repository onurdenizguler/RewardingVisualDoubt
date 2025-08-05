import datetime
import dataclasses
import os
import pathlib as path
import typing as t

import peft
import trl


from RewardingVisualDoubt import evaluation

from . import parameters


@dataclasses.dataclass
class PatienceTracker:
    value: int = 0


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
):
    save_dir = os.path.join(
        out_dir,
        name_of_fine_tuning,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        "best_eval_model_epoch_{}_step_{}".format(epoch + 1, step + 1),
    )
    os.makedirs(save_dir, exist_ok=True)
    ppo_trainer.save_pretrained(save_dir)


def report_generation_ppo_decision_to_break(
    step: int,
    patience: PatienceTracker,
    reward_ece_and_distribution_score: evaluation.RewardECEAndDistributionScore,
    best_reward_ece_and_distribution_kl_eval_aggregated_score: float,
    hyperparameters: parameters.ReportGenerationPPOHyperparameters,
    heuristics_fn: t.Callable[[evaluation.RewardECEAndDistributionScore], float],
) -> bool:

    if (
        reward_ece_and_distribution_score.conf_distribution_kl_divergence is float("inf")
        or reward_ece_and_distribution_score.ece is 1.0
    ):
        reward_ece_and_distribution_kl_eval_aggregated_score = (
            best_reward_ece_and_distribution_kl_eval_aggregated_score * 0.001
        )
    else:
        reward_ece_and_distribution_kl_eval_aggregated_score = heuristics_fn(
            reward_ece_and_distribution_score
        )

    relative_improvement = (
        reward_ece_and_distribution_kl_eval_aggregated_score
        - best_reward_ece_and_distribution_kl_eval_aggregated_score
    ) / (abs(best_reward_ece_and_distribution_kl_eval_aggregated_score) + 1e-12)

    if (
        reward_ece_and_distribution_kl_eval_aggregated_score
        > best_reward_ece_and_distribution_kl_eval_aggregated_score
        and relative_improvement >= hyperparameters.early_stopping_min_improvement
    ):
        best_reward_ece_and_distribution_kl_eval_aggregated_score = (
            reward_ece_and_distribution_kl_eval_aggregated_score
        )
        patience.value = 0
    else:
        patience.value += 1
        if (
            hyperparameters.early_stopping_patience
            and patience.value >= hyperparameters.early_stopping_patience
        ):
            print(
                f"Early stopping at step {step}: "
                f"best_score={best_reward_ece_and_distribution_kl_eval_aggregated_score:.4f}, current_score={reward_ece_and_distribution_kl_eval_aggregated_score:.4f}, "
                f"relative_improvement={relative_improvement:.4f}"
            )
            return True

    if hyperparameters.max_steps and step >= hyperparameters.max_steps:
        return True

    return False
