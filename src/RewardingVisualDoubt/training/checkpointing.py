import datetime
import os
import pathlib as path

import peft
import trl


from RewardingVisualDoubt import evaluation

from . import parameters


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
    patience: int,
    eval_conf_distribution_kl: float,
    ece_eval: float,
    best_ece_and_distribution_score: float,
    hyperparameters: parameters.ReportGenerationPPOHyperparameters,
) -> bool:

    if eval_conf_distribution_kl is 0 or ece_eval is None:
        current_ece_and_distribution_score = best_ece_and_distribution_score * 0.001
    else:
        current_ece_and_distribution_score = evaluation.reward_ece_and_distribution_score_heuristic(
            ece=ece_eval,
            conf_distribution_kl_divergence=eval_conf_distribution_kl,
            avg_reward=0.0,  # Not used in this heuristic
        )

    relative_improvement = (
        current_ece_and_distribution_score - best_ece_and_distribution_score
    ) / (abs(best_ece_and_distribution_score) + 1e-12)

    if (
        current_ece_and_distribution_score > best_ece_and_distribution_score
        and relative_improvement >= hyperparameters.early_stopping_min_improvement
    ):
        best_ece_and_distribution_score = current_ece_and_distribution_score
        patience = 0
    else:
        patience += 1
        if (
            hyperparameters.early_stopping_patience
            and patience >= hyperparameters.early_stopping_patience
        ):
            print(
                f"Early stopping at step {step}: "
                f"best_score={best_ece_and_distribution_score:.4f}, current_score={current_ece_and_distribution_score:.4f}, "
                f"relative_improvement={relative_improvement:.4f}"
            )
            return True

    if hyperparameters.max_steps and step >= hyperparameters.max_steps:
        return True

    return False
