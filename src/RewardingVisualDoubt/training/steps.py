import functools
import random
import typing as t

import torch
import transformers
import trl

from RewardingVisualDoubt import dataset, response, reward

from . import llava_ppo, reinforcement, postprocessing, logging


############### BINARY Q&A STEPS ###############


def radialog_binary_qa_ppo_evaluation_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_eval: dict,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    reward_function: t.Callable,
    granular_confidence: bool = False,
) -> tuple[
    list[float],
    list[int | None],
    list[bool],
]:

    ######### 5.1 Unpack the batch #########

    input_ids, images, labels, stopping_criteria, input_ids_list = dataset.unpack_binary_qa_batch(
        device=device,
        tokenizer=tokenizer,
        batch=batch,
        postprocessor=postprocessing.remove_preciding_padding_from_batch_tensor,
    )

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
    generated_confidence_values = response.parse_confidences(generated_texts, granular_confidence)
    generated_answer_labels = response.parse_binary_labels(generated_texts)

    scores = [
        reward.generated_answer_and_confidence_to_reward(
            generated_answer_label,
            generated_confidence_value,
            ground_truth_label,
            granular_confidence=granular_confidence,
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
    granular_confidence: bool = False,
) -> list[torch.FloatTensor]:

    ######### 5.1 Unpack the batch #########

    input_ids, images, labels, stopping_criteria, input_ids_list = dataset.unpack_binary_qa_batch(
        device=device,
        tokenizer=tokenizer,
        batch=batch,
        postprocessor=postprocessing.remove_preciding_padding_from_batch_tensor,
    )

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
    generated_confidence_values = response.parse_confidences(generated_texts, granular_confidence)
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
            granular_confidence=granular_confidence,
        )

    generated_answer_labels = response.parse_binary_labels(generated_texts)

    scores = [
        reward.generated_answer_and_confidence_to_reward(
            generated_answer_label,
            generated_confidence_value,
            ground_truth_label,
            granular_confidence,
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
        table_rows = logging.log_custom_metrics_for_binary_qa(
            tokenizer,
            accumulating_game_logs,
            input_ids,
            labels,
            generated_texts,
            generated_confidence_values,
            old_generated_confidence_values,
            is_confidence_randomly_replaced,
            generated_answer_labels,
            scores,
            ppo_responses,
            stats,
            queries_with_gt_labels,
        )

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
    granular_confidence: bool,
) -> tuple[list[torch.Tensor], list[str], list[int | None], list[int | None], list[bool]]:
    old_generated_confidence_values = generated_confidence_values.copy()
    generated_texts = reinforcement.overwrite_confidence(
        generated_texts, generated_confidence_values, granular_confidence=granular_confidence
    )
    generated_ids = []
    for text in generated_texts:
        ids = t.cast(torch.Tensor, tokenizer.encode(text, return_tensors="pt"))
        generated_ids.append(ids.squeeze(0).to(device=device))
    generated_confidence_values = response.parse_confidences(generated_texts, granular_confidence)
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
