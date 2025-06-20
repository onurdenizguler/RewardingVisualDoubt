import functools
import random
import typing as t

import torch
import transformers
import trl
import wandb

from RewardingVisualDoubt import dataset, evaluation, prompter, response, reward, shared

from . import llava_ppo, reinforcement, postprocessing


############### BINARY Q&A STEPS ###############


def radialog_binary_qa_ppo_evaluation_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_eval: dict,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    reward_function: t.Callable,
) -> tuple[
    list[float],
    list[int | None],
    list[bool],
]:

    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)
    input_ids_list = postprocessing.remove_preciding_padding_from_batch_tensor(
        input_ids
    )  # WARNING: Is performed, as the ppo_trainer.generate() method handles padding and batching itself

    ######### 5.2 Generate the binary q&a answer and remove trailing padding tokens #########
    model.eval()
    model.gradient_checkpointing_disable()
    generated_ids = ppo_trainer.generate(
        query_tensor=input_ids_list,  # ppo_trainer.generate() method admits list of tensors, handles padding and batching itself
        images=images,
        return_prompt=False,
        batch_size=input_ids.shape[0],
        use_cache=True,  # => not compatible with gradient checkpointing, that's why we disable it here.
        stopping_criteria=[stopping_criteria],
        **generation_kwargs_eval,
    )

    ######### 5.3 Parse the responses and compute the scores #########
    generated_texts = tokenizer.batch_decode(generated_ids)
    generated_confidence_values = response.parse_confidences(generated_texts)
    generated_answer_labels = response.parse_binary_labels(generated_texts)

    scores = [
        reward.generated_answer_and_confidence_to_reward(
            generated_answer_label,
            generated_confidence_value,
            ground_truth_label,
            reward_function=reward_function,
        )
        for generated_answer_label, generated_confidence_value, ground_truth_label in zip(
            generated_answer_labels, generated_confidence_values, labels.bool().tolist()
        )
    ]

    is_answer_correct = [
        (gt_label == predicted_label) and (gt_label is not None)
        for gt_label, predicted_label in zip(generated_answer_labels, labels.bool().tolist())
    ]
    return scores, generated_confidence_values, is_answer_correct


def radialog_binary_qa_ppo_training_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_ppo: dict,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    reward_function: t.Callable,
    accumulating_game_logs: reinforcement.GameLogs | None = None,
    chance_to_change_confidence: float = 0.5,  # Default value: every 2nd batch
) -> list[torch.FloatTensor]:

    ######### 5.1 Unpack the batch #########
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = shared.KeywordsStoppingCriteria([prompter.STOP_STR], tokenizer, input_ids)
    input_ids_list = postprocessing.remove_preciding_padding_from_batch_tensor(
        input_ids
    )  # WARNING: Is performed, as the ppo_trainer.generate() method handles padding and batching itself

    ######### 5.2 Generate the binary q&a answer and remove trailing padding tokens #########
    model.eval()
    model.gradient_checkpointing_disable()
    generated_ids = ppo_trainer.generate(
        query_tensor=input_ids_list,  # ppo_trainer.generate() method admits list of tensors, handles padding and batching itself
        images=images,
        return_prompt=False,
        batch_size=input_ids.shape[0],
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        **generation_kwargs_ppo,
    )

    ######### 5.3 Parse the responses and compute the scores #########
    generated_texts = tokenizer.batch_decode(generated_ids)
    generated_confidence_values = response.parse_confidences(generated_texts)
    old_generated_confidence_values = generated_confidence_values.copy()
    is_confidence_randomly_replaced = [False] * len(generated_confidence_values)

    if random.random() < chance_to_change_confidence:  # Replace with random confidence
        (
            generated_ids,
            generated_texts,
            generated_confidence_values,
            old_generated_confidence_values,
            is_confidence_randomly_replaced,
        ) = _handle_random_confidence_replacement(
            tokenizer,
            generated_texts,
            generated_confidence_values,
            device=device,
        )

    generated_answer_labels = response.parse_binary_labels(generated_texts)

    scores = [
        reward.generated_answer_and_confidence_to_reward(
            generated_answer_label,
            generated_confidence_value,
            ground_truth_label,
            reward_function=reward_function,
        )
        for generated_answer_label, generated_confidence_value, ground_truth_label in zip(
            generated_answer_labels, generated_confidence_values, labels.bool().tolist()
        )
    ]

    scores = t.cast(
        list[torch.FloatTensor],
        [torch.tensor(s).to(device) for s in scores],
    )

    ######### 5.7 Take a PPO optimization step #########
    reformulate_query_and_response_fn = functools.partial(
        postprocessing.reformulate_query_and_response_for_binary_qa, tokenizer=tokenizer
    )
    reformulated_query_and_responses: list[postprocessing.ReformulatedQueryAndResponseDict] = [
        reformulate_query_and_response_fn(query_ids=cur_input_ids, response=cur_response_text)
        for cur_input_ids, cur_response_text in zip(input_ids_list, generated_texts)
    ]
    model.train()
    model.gradient_checkpointing_enable()
    ppo_queries = t.cast(
        list[torch.LongTensor],
        [
            reformulated_query_and_response["query_ids"]
            for reformulated_query_and_response in reformulated_query_and_responses
        ],
    )
    ppo_responses = t.cast(
        list[torch.LongTensor],
        [
            reformulated_query_and_response["response_ids"]
            for reformulated_query_and_response in reformulated_query_and_responses
        ],
    )
    stats = ppo_trainer.multimodal_step(
        queries=ppo_queries,
        responses=ppo_responses,
        scores=scores,
        images=images,
    )

    ######### 5.8 Create a report and log the training stats #########
    input_texts_list = tokenizer.batch_decode(
        postprocessing.replace_image_token_with_another_token_for_list_of_tensors(input_ids_list)
    )
    queries_with_gt_labels = [
        f"(GT Label: {str(labels.bool().tolist()[idx])}) - {input_prompt}"
        for idx, input_prompt in enumerate(input_texts_list)
    ]

    table_rows = []
    if accumulating_game_logs:
        truncated_accumulating_game_logs = _handle_accumulating_game_logs(
            accumulating_game_logs,
            queries=queries_with_gt_labels,
            responses=generated_texts,
            ppo_target_responses=tokenizer.batch_decode(ppo_responses),
            is_answer_correct=[
                (gt_label is not None) and (gt_label == predicted_label)
                for gt_label, predicted_label in zip(
                    generated_answer_labels, labels.bool().tolist()
                )
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

        stats["ratio_of_changed_confidences"] = accumulating_game_logs[
            "is_confidence_randomly_replaced"
        ].count(True) / len(accumulating_game_logs["is_confidence_randomly_replaced"])

    ppo_trainer.log_stats(
        stats=stats,
        table_rows=table_rows,
        column_names=[
            "query",
            "response",
            "ppo_target_response",
            "is_answer_correct",
            "confidence",
            "confidence_after_replacement",
            "is_confidence_randomly_replaced",
            "reward",
        ],
        rewards=scores,
    )

    return scores


def _handle_random_confidence_replacement(
    tokenizer: transformers.PreTrainedTokenizer,
    generated_texts: list[str],
    generated_confidence_values: list[int | None],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[str], list[int | None], list[int | None], list[bool]]:
    old_generated_confidence_values = generated_confidence_values.copy()
    generated_texts = reinforcement.overwrite_confidence(
        generated_texts, generated_confidence_values
    )
    generated_ids = []
    for text in generated_texts:
        ids = t.cast(torch.Tensor, tokenizer.encode(text, return_tensors="pt"))
        generated_ids.append(ids.squeeze(0).to(device=device))
    generated_confidence_values = response.parse_confidences(generated_texts)
    is_confidence_randomly_replaced = [
        old_conf != new_conf
        for old_conf, new_conf in zip(old_generated_confidence_values, generated_confidence_values)
    ]

    return (
        generated_ids,
        generated_texts,
        generated_confidence_values,
        old_generated_confidence_values,
        is_confidence_randomly_replaced,
    )


def _handle_accumulating_game_logs(
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
