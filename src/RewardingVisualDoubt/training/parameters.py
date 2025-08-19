import dataclasses
import importlib
import typing as t
from pathlib import Path

import yaml

from RewardingVisualDoubt import reward


@dataclasses.dataclass(frozen=True)
class Parameters:

    def print_param_values(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            print(f"{field.name} = {value}")


@dataclasses.dataclass(frozen=True)
class TrainingMetaParameters(Parameters):
    name_of_fine_tuning: str
    llava_model_path: str
    adapter_path: str | None
    out_dir: Path
    perform_validation_before_starting_training: bool
    n_training_batches_to_skip: int
    num_batches_to_evaluate: int
    save_training_model_every_n_checkpoints: int
    plot_confidence_calibration_for_training_batches_every_n_batch: int


@dataclasses.dataclass(frozen=True)
class ReportGenerationPPOHyperparameters(Parameters):
    num_epochs: int
    steps_until_checkpoint: int
    gradient_accumulation_steps: int
    batch_size: int
    eval_batch_size: int
    mini_batch_size: int
    learning_rate: float
    chance_to_change_confidence: float
    reward_function: t.Callable
    reward_config: reward.RewardConfig
    reward_ece_and_distribution_score_heuristic: t.Callable
    granular_confidence: bool = False
    max_steps: t.Optional[int] = None
    early_stopping_patience: t.Optional[int] = None
    early_stopping_min_improvement: float = 0.0
    selected_train_datapoints_json: str | None = None
    selected_eval_datapopoints_json: str | None = None


def load_callable(path: str) -> t.Any:
    """
    Given a dotted path like "pkg.mod.func", import and return the function.
    """
    mod_path, func_name = path.rsplit(".", 1)
    module = importlib.import_module(mod_path)
    return getattr(module, func_name)


def load_default_configs(
    path: str,
) -> tuple[TrainingMetaParameters, ReportGenerationPPOHyperparameters]:
    data = yaml.safe_load(open(path, "r"))

    tm = TrainingMetaParameters(
        **{
            **data["training_metaparameters"],
            "out_dir": Path(data["training_metaparameters"]["out_dir"]),
        }
    )

    # 3) Resolve reward_function string to callable
    ppo_dict = data["report_generation_ppo_hyperparameters"].copy()
    rf = ppo_dict["reward_function"]
    hf = ppo_dict["reward_ece_and_distribution_score_heuristic"]
    if rf == "RewardingVisualDoubt.reward.scaled_quadratic_blend_distance_reward":
        ppo_dict["reward_config"] = reward.QuadraticBlendRewardConfig(**ppo_dict["reward_config"])
    if rf == "RewardingVisualDoubt.reward.scaled_normalized_log_likelihood_reward":
        ppo_dict["reward_config"] = reward.RewardConfig(**ppo_dict["reward_config"])
    if isinstance(rf, str):
        ppo_dict["reward_function"] = load_callable(rf)
    if isinstance(hf, str):
        ppo_dict["reward_ece_and_distribution_score_heuristic"] = load_callable(hf)

    ppo = ReportGenerationPPOHyperparameters(**ppo_dict)
    return tm, ppo
