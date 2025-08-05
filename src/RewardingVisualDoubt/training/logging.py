import dataclasses
import datetime
import typing as t

import numpy as np
import torch
import transformers
import wandb

from RewardingVisualDoubt import dataset, evaluation, shared

from . import llava_ppo, parameters, postprocessing


class GameLogs(t.TypedDict):
    queries: list[str]
    responses: list[str]
    ppo_target_responses: list[str]
    is_answer_correct: list[bool]
    scores: list[float]
    confidences: list[int | None]
    confidences_after_replacement: list[int | None]
    is_confidence_randomly_replaced: list[bool]


class GameLogsForReportGeneration(t.TypedDict):
    queries: list[str]
    responses: list[str]
    ppo_target_responses: list[str]
    accuracies: list[float | None]
    scores: list[float]
    confidences: list[int | None]
    confidences_after_replacement: list[int | None]
    is_confidence_randomly_replaced: list[bool]


@dataclasses.dataclass
class ReportGenerationRunFinalMetrics:
    mean_score_train_at_last_checkpoint: float

    last_mean_score_eval: float
    last_ece_eval: float
    last_conf_distribution_kl_eval: float
    last_ece_and_conf_distribution_kl_eval: float

    best_mean_score_eval: float
    best_ece_eval: float
    best_conf_distribution_kl_eval: float
    best_ece_and_conf_distribution_kl_eval: float


@dataclasses.dataclass
class ReportGenerationPostPPOOptimizationAssetsBatch:
    scores: list[torch.FloatTensor]
    green_scores: list[float | None]
    generated_confidence_values: list[int | None]
    generated_confidence_values_after_replacement: list[int | None]
    is_confidence_randomly_replaced: list[bool]
    generated_texts: list[str]
    ppo_responses: list[torch.LongTensor]
    stats: dict[str, t.Any]


GameLogType = t.Literal["GameLogs", "GameLogsForReportGeneration"]


def create_empty_game_logs() -> GameLogs:
    return GameLogs(
        queries=[],
        responses=[],
        ppo_target_responses=[],
        is_answer_correct=[],
        scores=[],
        confidences=[],
        confidences_after_replacement=[],
        is_confidence_randomly_replaced=[],
    )


def create_empty_game_logs_for_report_generation() -> GameLogsForReportGeneration:
    return GameLogsForReportGeneration(
        queries=[],
        responses=[],
        ppo_target_responses=[],
        accuracies=[],
        scores=[],
        confidences=[],
        confidences_after_replacement=[],
        is_confidence_randomly_replaced=[],
    )


##########################################################################################
# BINARY QA LOGGING METHODS
##########################################################################################


def _handle_accumulating_game_logs_for_binary_qa(
    accumulating_game_logs: GameLogs,
    queries: list[str],
    responses: list[str],
    ppo_target_responses: list[str],
    is_answer_correct: list[bool],
    scores: list[torch.FloatTensor],
    confidences: list[int | None],
    old_generated_confidence_values: list[int | None],
    is_confidence_randomly_replaced: list[bool],
) -> GameLogs:

    accumulating_game_logs["queries"].extend(queries)
    accumulating_game_logs["responses"].extend(responses)
    accumulating_game_logs["ppo_target_responses"].extend(ppo_target_responses)
    accumulating_game_logs["is_answer_correct"].extend(is_answer_correct)
    accumulating_game_logs["scores"].extend([score.item() for score in scores])
    accumulating_game_logs["confidences"].extend(old_generated_confidence_values)
    accumulating_game_logs["confidences_after_replacement"].extend(confidences)
    accumulating_game_logs["is_confidence_randomly_replaced"].extend(
        is_confidence_randomly_replaced
    )

    truncated_accumulating_game_logs: GameLogs = {
        "queries": [],
        "responses": [],
        "ppo_target_responses": [],
        "is_answer_correct": [],
        "scores": [],
        "confidences": [],
        "confidences_after_replacement": [],
        "is_confidence_randomly_replaced": [],
    }
    # If more than 500 rows, remove the extra rows
    if len(accumulating_game_logs["queries"]) > 500:
        for key in accumulating_game_logs.keys():
            truncated_accumulating_game_logs[key] = accumulating_game_logs[key][-500:]
    else:
        truncated_accumulating_game_logs = accumulating_game_logs

    return truncated_accumulating_game_logs


def log_custom_metrics_for_binary_qa(
    tokenizer: transformers.PreTrainedTokenizer,
    accumulating_game_logs: GameLogs,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    generated_texts: list[str],
    generated_confidence_values: list[int | None],
    old_generated_confidence_values: list[int | None],
    is_confidence_randomly_replaced: list[bool],
    generated_answer_labels: list[bool | None],
    scores: list[torch.FloatTensor],
    ppo_responses: list[torch.LongTensor],
    stats: dict[str, t.Any],
    queries_with_gt_labels: list[str],
    log_calibration_plot: bool,
) -> list[list[str]]:

    truncated_accumulating_game_logs = _handle_accumulating_game_logs_for_binary_qa(
        accumulating_game_logs,
        queries=queries_with_gt_labels,
        responses=generated_texts,
        ppo_target_responses=tokenizer.batch_decode(ppo_responses),
        is_answer_correct=[
            (gt_label is not None) and (gt_label == predicted_label)
            for gt_label, predicted_label in zip(generated_answer_labels, labels.bool().tolist())
        ],
        scores=scores,
        confidences=generated_confidence_values,
        old_generated_confidence_values=old_generated_confidence_values,
        is_confidence_randomly_replaced=is_confidence_randomly_replaced,
    )
    batch_size = input_ids.shape[0]
    table_rows = [
        list(r)
        for r in zip(
            accumulating_game_logs["queries"][-batch_size:],
            accumulating_game_logs["responses"][-batch_size:],
            accumulating_game_logs["ppo_target_responses"][-batch_size:],
            accumulating_game_logs["is_answer_correct"][-batch_size:],
            accumulating_game_logs["confidences"][-batch_size:],
            accumulating_game_logs["confidences_after_replacement"][-batch_size:],
            accumulating_game_logs["is_confidence_randomly_replaced"][-batch_size:],
            accumulating_game_logs["scores"][-batch_size:],
        )
    ]

    if log_calibration_plot:
        try:
            stats["confidence_calibration_last_500_samples"] = wandb.Image(
                evaluation.plot_calibration_curve(
                    confidences=truncated_accumulating_game_logs["confidences"],
                    accuracies=truncated_accumulating_game_logs["is_answer_correct"],
                )
            )
            stats["confidence_calibration_all_samples"] = wandb.Image(
                evaluation.plot_calibration_curve(
                    confidences=accumulating_game_logs["confidences"],
                    accuracies=accumulating_game_logs["is_answer_correct"],
                )
            )

        except:
            pass

    bins_100 = evaluation.binify_accuracies(
        confidences=truncated_accumulating_game_logs["confidences"][-100:],
        accuracies=truncated_accumulating_game_logs["is_answer_correct"][-100:],
    )
    bins_500 = evaluation.binify_accuracies(
        confidences=truncated_accumulating_game_logs["confidences"],
        accuracies=truncated_accumulating_game_logs["is_answer_correct"],
    )
    if bins_100:
        counts_100, avg_acc_100 = bins_100
        calibration_table_rows_last_100 = [
            [i, avg_acc_100[i], counts_100[i]] for i in range(shared.POSSIBLE_CONFIDENCES[-1] + 1)
        ]
        stats["confidence_calibration_table_last_100_samples"] = wandb.Table(
            columns=["confidence_bin", "accuracy", "count"],
            rows=calibration_table_rows_last_100,
        )
        stats["ECE_LAST_100"] = evaluation.compute_ece(avg_acc_100, counts_100)
    if bins_500:
        counts_500, avg_acc_500 = bins_500
        calibration_table_rows_last_500 = [
            [i, avg_acc_500[i], counts_500[i]]
            for i in list(range(shared.POSSIBLE_CONFIDENCES[-1] + 1))
        ]
        stats["confidence_calibration_table_last_500_samples"] = wandb.Table(
            columns=["confidence_bin", "accuracy", "count"],
            rows=calibration_table_rows_last_500,
        )
        stats["ECE_LAST_500"] = evaluation.compute_ece(avg_acc_500, counts_500)

    stats["ratio_of_changed_confidences"] = accumulating_game_logs[
        "is_confidence_randomly_replaced"
    ].count(True) / len(accumulating_game_logs["is_confidence_randomly_replaced"])

    return table_rows


##########################################################################################
# REPORT GENERATION LOGGING METHODS
##########################################################################################


def _handle_accumulating_game_logs_for_report_generation(
    accumulating_game_logs: GameLogsForReportGeneration,
    queries: list[str],
    responses: list[str],
    ppo_target_responses: list[str],
    accuracies: list[float | None],
    scores: list[float],
    confidences: list[int | None],
    confidences_after_replacement: list[int | None],
    is_confidence_randomly_replaced: list[bool],
) -> GameLogsForReportGeneration:

    for k, v in locals().items():
        if k == "accumulating_game_logs":
            continue
        else:
            accumulating_game_logs[k].extend(v)

    truncated_accumulating_game_logs = accumulating_game_logs
    if len(accumulating_game_logs["queries"]) > 500:
        for key in accumulating_game_logs.keys():
            truncated_accumulating_game_logs[key] = accumulating_game_logs[key][-500:]
    else:
        truncated_accumulating_game_logs = accumulating_game_logs

    return truncated_accumulating_game_logs


def _prepare_table_and_plots_for_report_generation(
    batch_size: int,
    accumulating_game_logs: GameLogsForReportGeneration,
    truncated_accumulating_game_logs: GameLogsForReportGeneration,
    post_optimization_assets: ReportGenerationPostPPOOptimizationAssetsBatch,
    log_calibration_plot: bool,
):

    column_names = [
        "query",
        "response",
        "ppo_target_response",
        "is_answer_correct",
        "confidence",
        "confidence_after_replacement",
        "is_confidence_randomly_replaced",
        "reward",
    ]

    table_rows = [
        list(r)
        for r in zip(
            accumulating_game_logs["queries"][-batch_size:],
            accumulating_game_logs["responses"][-batch_size:],
            accumulating_game_logs["ppo_target_responses"][-batch_size:],
            accumulating_game_logs["accuracies"][-batch_size:],
            accumulating_game_logs["confidences"][-batch_size:],
            accumulating_game_logs["confidences_after_replacement"][-batch_size:],
            accumulating_game_logs["is_confidence_randomly_replaced"][-batch_size:],
            accumulating_game_logs["scores"][-batch_size:],
        )
    ]

    if log_calibration_plot:
        try:
            post_optimization_assets.stats["confidence_calibration_last_500_samples"] = wandb.Image(
                evaluation.plot_calibration_curve(
                    confidences=truncated_accumulating_game_logs["confidences"],
                    accuracies=truncated_accumulating_game_logs["accuracies"],
                )
            )
            post_optimization_assets.stats["confidence_calibration_all_samples"] = wandb.Image(
                evaluation.plot_calibration_curve(
                    confidences=accumulating_game_logs["confidences"],
                    accuracies=accumulating_game_logs["accuracies"],
                )
            )

        except:
            pass

    return table_rows, column_names


def calculate_kpi_metrics(
    generated_confidence_values_list, scores_list, green_scores_list, hyperparameters
):
    bins = evaluation.binify_accuracies(
        confidences=generated_confidence_values_list,
        accuracies=green_scores_list,
    )
    ece = 1.0  # Set to maximum value to be able to calculate ece_and_distribution_score in the worst case that no bins are returned
    if bins:
        counts, avg_acc = bins
        ece = evaluation.compute_ece(avg_acc, counts)

    conf_distribution_kl = evaluation.compute_confidence_distribution_metric(
        postprocessing.normalize_confidence_scores(
            generated_confidence_values_list,
            hyperparameters.granular_confidence,
        ),
        num_bins=11 if not hyperparameters.granular_confidence else 101,
    )
    mean_score = float(np.mean(scores_list))
    std_score = float(np.std(scores_list))
    return ece, conf_distribution_kl, mean_score, std_score


def log_eval_metrics_for_report_generation_ppo(
    generated_confidence_values_list: list[int | None],
    scores_list: list[float],
    green_scores_list: list[float | None],
    hyperparameters: parameters.ReportGenerationPPOHyperparameters,
) -> t.Tuple[
    float,
    float,
    float,
    float,
]:
    ece, conf_distribution_kl, mean_score, std_score = calculate_kpi_metrics(
        generated_confidence_values_list, scores_list, green_scores_list, hyperparameters
    )

    try:
        wandb.log(
            {
                "val_conf_calib": wandb.Image(
                    evaluation.plot_calibration_curve(
                        confidences=generated_confidence_values_list,
                        accuracies=green_scores_list,
                    )
                )
            }
        )
    except:
        pass

    wandb.log({"mean_score_eval": mean_score})
    wandb.log({"std_score_eval": std_score})
    wandb.log({"ece_eval": ece})
    wandb.log({"conf_distribution_kl_eval": conf_distribution_kl})

    return ece, conf_distribution_kl, mean_score, mean_score


def log_train_metrics_for_report_generation_ppo(
    post_optimization_assets: ReportGenerationPostPPOOptimizationAssetsBatch,
    device,
    tokenizer: transformers.LlamaTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    hyperparameters: parameters.ReportGenerationPPOHyperparameters,
    accumulating_game_logs: GameLogsForReportGeneration | None = None,
    log_calibration_plot: bool = False,
) -> None:

    input_ids, input_ids_list, images, stopping_criteria, batch_metadata_list = (
        dataset.unpack_report_generation_batch_with_attention_mask_and_metadata_for_ppo(
            device=device,
            tokenizer=tokenizer,
            batch=batch,
            postprocessor=postprocessing.remove_preciding_padding_from_batch_tensor,
        )
    )

    ######## 5.9 Create a report and log the training stats #########
    input_texts_list: list[str] = tokenizer.batch_decode(
        postprocessing.replace_image_token_with_another_token_for_list_of_tensors(input_ids_list)
    )
    queries_with_gt_info: list[str] = []
    # queries_with_gt_information = #build strings with patience info, gt findings, etc [
    #     f"(GT Label: {str(labels.bool().tolist()[idx])}) - {input_prompt}"
    #     for idx, input_prompt in enumerate(input_texts_list)
    # ]

    table_rows = []
    if accumulating_game_logs:

        truncated_accumulating_game_logs = _handle_accumulating_game_logs_for_report_generation(
            accumulating_game_logs,
            queries=queries_with_gt_info,
            responses=post_optimization_assets.generated_texts,
            ppo_target_responses=tokenizer.batch_decode(post_optimization_assets.ppo_responses),
            accuracies=[accuracy for accuracy in post_optimization_assets.green_scores],
            scores=[float(score.item()) for score in post_optimization_assets.scores],
            confidences=post_optimization_assets.generated_confidence_values,
            confidences_after_replacement=post_optimization_assets.generated_confidence_values_after_replacement,
            is_confidence_randomly_replaced=post_optimization_assets.is_confidence_randomly_replaced,
        )

        batch_size = input_ids.shape[0]
        table_rows, column_names = _prepare_table_and_plots_for_report_generation(
            batch_size=batch_size,
            accumulating_game_logs=accumulating_game_logs,
            truncated_accumulating_game_logs=truncated_accumulating_game_logs,
            post_optimization_assets=post_optimization_assets,
            log_calibration_plot=log_calibration_plot,
        )

        ece_100, conf_distribution_kl_100, mean_score_100, std_score_100 = calculate_kpi_metrics(
            generated_confidence_values_list=truncated_accumulating_game_logs["confidences"][-100:],
            scores_list=truncated_accumulating_game_logs["scores"][-100:],
            green_scores_list=truncated_accumulating_game_logs["accuracies"][-100:],
            hyperparameters=hyperparameters,
        )

        ece_500, conf_distribution_kl_500, mean_score_500, std_score_500 = calculate_kpi_metrics(
            generated_confidence_values_list=truncated_accumulating_game_logs["confidences"][-500:],
            scores_list=truncated_accumulating_game_logs["scores"][-500:],
            green_scores_list=truncated_accumulating_game_logs["accuracies"][-500:],
            hyperparameters=hyperparameters,
        )

        wandb.log({"mean_score_train_last_100": mean_score_100})
        wandb.log({"std_score_train_last_100": std_score_100})
        wandb.log({"ece_train_last_100": ece_100})
        wandb.log({"conf_distribution_kl_train_last_100": conf_distribution_kl_100})

        wandb.log({"mean_score_train_last_500": mean_score_500})
        wandb.log({"std_score_train_last_500": std_score_500})
        wandb.log({"ece_train_last_500": ece_500})
        wandb.log({"conf_distribution_kl_train_last_500": conf_distribution_kl_500})

        post_optimization_assets.stats["ratio_of_changed_confidences"] = accumulating_game_logs[
            "is_confidence_randomly_replaced"
        ].count(True) / len(accumulating_game_logs["is_confidence_randomly_replaced"])

        ppo_trainer.log_stats(
            stats=post_optimization_assets.stats,
            table_rows=table_rows,
            column_names=column_names,
            rewards=post_optimization_assets.scores,
        )


############################################################################################################
# WANDB CONFIGURATION METHODS
############################################################################################################


def get_wandb_parameters_for_sft(
    learning_rate: float,
    prompter: t.Callable,
    num_epochs: int,
    steps_until_checkpoint: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    num_batches_to_evaluate: int,
    n_training_batches_to_skip: int,
) -> t.Dict[str, t.Any]:
    return {
        "id": f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{format(learning_rate, '.0e')}",
        "config": {
            "date_of_training": datetime.datetime.now().strftime("%Y-%m-%d"),
            "learning_rate": learning_rate,
            "prompter": prompter.__name__,
            "num_epochs": num_epochs,
            "steps_until_checkpoint": steps_until_checkpoint,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_batches_to_evaluate": num_batches_to_evaluate,
            "n_training_batches_to_skip": n_training_batches_to_skip,
        },
    }


def get_wandb_parameters_for_report_generation_ppo(
    metaparameters: parameters.TrainingMetaParameters,
    hyperparameters: parameters.ReportGenerationPPOHyperparameters,
    prompter: t.Callable,
) -> t.Dict[str, t.Any]:

    return {
        "id": f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{format(hyperparameters.learning_rate, '.0e')}_{hyperparameters.chance_to_change_confidence}_{hyperparameters.reward_config.scaling}",
        "config": {
            "date_of_training": datetime.datetime.now().strftime("%Y-%m-%d"),
            "learning_rate": hyperparameters.learning_rate,
            "chance_to_change_confidence": hyperparameters.chance_to_change_confidence,
            "reward_function": hyperparameters.reward_function.__name__,
            "reward_scaling_method": hyperparameters.reward_config.scaling,
            "reward_eps": hyperparameters.reward_config.eps,
            "reward_scale": hyperparameters.reward_config.scale,
            "reward_squash_scale": hyperparameters.reward_config.squash_scale,
            "granular_confidence": str(hyperparameters.granular_confidence),
            "prompter": prompter.__name__,
            "starting_llava_model_path": metaparameters.llava_model_path,
            "starting_adapter_path": metaparameters.adapter_path,
            "num_epochs": hyperparameters.num_epochs,
            "perform_validation_before_starting_training": str(
                metaparameters.perform_validation_before_starting_training
            ),
            "steps_until_checkpoint": hyperparameters.steps_until_checkpoint,
            "batch_size": hyperparameters.batch_size,
            "mini_batch_size": hyperparameters.mini_batch_size,
            "gradient_accumulation_steps": hyperparameters.gradient_accumulation_steps,
            "num_batches_to_evaluate": metaparameters.num_batches_to_evaluate,
            "n_training_batches_to_skip": metaparameters.n_training_batches_to_skip,
        },
    }
