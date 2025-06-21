import typing as t

import torch
import transformers
import wandb

from RewardingVisualDoubt import evaluation, shared

from . import reinforcement


def _handle_accumulating_game_logs_for_binary_qa(
    accumulating_game_logs: reinforcement.GameLogs,
    queries: list[str],
    responses: list[str],
    ppo_target_responses: list[str],
    is_answer_correct: list[bool],
    scores: list[torch.FloatTensor],
    confidences: list[int | None],
    old_generated_confidence_values: list[int | None],
    is_confidence_randomly_replaced: list[bool],
) -> reinforcement.GameLogs:

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

    truncated_accumulating_game_logs: reinforcement.GameLogs = {
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
    accumulating_game_logs: reinforcement.GameLogs,
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

    try:
        stats["confidence_calibration_last_500_samples"] = wandb.Image(
            evaluation.plot_calibration_curve(
                confidences=truncated_accumulating_game_logs["confidences"],
                is_answer_correct=truncated_accumulating_game_logs["is_answer_correct"],
            )
        )
        stats["confidence_calibration_all_samples"] = wandb.Image(
            evaluation.plot_calibration_curve(
                confidences=accumulating_game_logs["confidences"],
                is_answer_correct=accumulating_game_logs["is_answer_correct"],
            )
        )

    except:
        pass

    bins_100 = evaluation.binify_accuracies(
        confidences=truncated_accumulating_game_logs["confidences"][-100:],
        is_answer_correct=truncated_accumulating_game_logs["is_answer_correct"][-100:],
    )
    bins_500 = evaluation.binify_accuracies(
        confidences=truncated_accumulating_game_logs["confidences"],
        is_answer_correct=truncated_accumulating_game_logs["is_answer_correct"],
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
