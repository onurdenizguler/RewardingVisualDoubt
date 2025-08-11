import dataclasses
import functools
import random
import typing as t

import torch
import transformers
import trl

from RewardingVisualDoubt import dataset, green, response, reward, shared

from . import llava_ppo, logging, postprocessing

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

    ######### 5.2 Generate the binary q&a answer #########
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
    accumulating_game_logs: logging.GameLogs | None = None,
    chance_to_change_confidence: float = 0.5,  # Default value: every 2nd batch
    granular_confidence: bool = False,
    log_calibration_plot: bool = False,
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
        ) = postprocessing.handle_random_confidence_replacement(
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
        postprocessing.reformulate_query_and_response, tokenizer=tokenizer
    )
    reformulated_query_and_responses: list[postprocessing.ReformulatedQueryAndResponseDict] = [
        reformulate_query_and_response_fn(query_ids=cur_input_ids, response=cur_response_text)
        for cur_input_ids, cur_response_text in zip(input_ids_list, generated_texts)
    ]

    ppo_queries, ppo_responses = (
        prepare_ppo_queries_and_responses_from_reformulated_query_and_responses(
            reformulated_query_and_responses=reformulated_query_and_responses
        )
    )

    model.train()
    model.gradient_checkpointing_enable()
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
            log_calibration_plot,
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


####################################################################################################################
# REPORT GENERATION STEPS
####################################################################################################################


def generate_reports(
    model: trl.AutoModelForCausalLMWithValueHead,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    input_ids_list: list[torch.Tensor],
    input_ids: torch.Tensor,
    images: torch.Tensor,
    stopping_criteria: shared.KeywordsStoppingCriteria,
    generation_kwargs: dict,
) -> list[torch.Tensor]:

    model.eval()
    model.gradient_checkpointing_disable()
    with torch.inference_mode():
        generated_ids = ppo_trainer.generate(
            query_tensor=input_ids_list,  # ppo_trainer.generate() method admits list of tensors, handles padding and batching itself
            images=images,
            return_prompt=False,
            batch_size=input_ids.shape[0],
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            **generation_kwargs,
        )
    return generated_ids


def radialog_report_generation_ppo_evaluation_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_eval: dict,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    reward_function: t.Callable,
    reward_config: reward.RewardConfig,
    granular_confidence: bool = False,
) -> tuple[list[float], list[int | None], list[float | None]]:

    ######### 5.1 Unpack the batch #########

    input_ids, input_ids_list, images, stopping_criteria, batch_metadata_list = (
        dataset.unpack_report_generation_batch_with_attention_mask_and_metadata_for_ppo(
            device=device,
            tokenizer=tokenizer,
            batch=batch,
            postprocessor=postprocessing.remove_preciding_padding_from_batch_tensor,
        )
    )

    ######### 5.2 Generate the reports #########
    generated_ids = generate_reports(
        model,
        ppo_trainer,
        input_ids_list,
        input_ids,
        images,
        stopping_criteria,
        generation_kwargs_eval,
    )

    ######### 5.3 Parse the responses and create post-generation assets #########
    (
        generated_texts,
        confidence_stripped_generated_texts,
        generated_confidence_values,
        gt_reports_list,
    ) = create_post_report_generation_assets(
        generated_ids=generated_ids,
        batch_metadata_list=batch_metadata_list,
        tokenizer=tokenizer,
        granular_confidence=granular_confidence,
    )

    ########## 5.5 Fetch GREEN scores for the generated reports #########
    green_scores = green.get_green_score_for_batch_of_generated_reports(
        generated_reports=confidence_stripped_generated_texts, gt_reports=gt_reports_list
    )

    ########## 5.6 Compute the scores (rewards) #########

    scores = [
        reward_function(
            confidence=confidence,
            accuracy=accuracy,
            granular_confidence=False,
            config=reward_config,
        )
        for confidence, accuracy in zip(generated_confidence_values, green_scores)
    ]

    return scores, generated_confidence_values, green_scores


def radialog_report_generation_ppo_training_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_ppo: dict,
    ppo_trainer: llava_ppo.MultimodalPPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    reward_function: t.Callable[..., float],
    reward_config: reward.RewardConfig,
    random_number_generator: random.Random,
    chance_to_change_confidence: float = 0.5,  # Default value: every 2nd batch
    granular_confidence: bool = False,
) -> logging.ReportGenerationPostPPOOptimizationAssetsBatch:

    ######### 5.1 Unpack the batch #########

    input_ids, input_ids_list, images, stopping_criteria, batch_metadata_list = (
        dataset.unpack_report_generation_batch_with_attention_mask_and_metadata_for_ppo(
            device=device,
            tokenizer=tokenizer,
            batch=batch,
            postprocessor=postprocessing.remove_preciding_padding_from_batch_tensor,
        )
    )

    ######### 5.2 Generate the reports #########
    generated_ids = generate_reports(
        model,
        ppo_trainer,
        input_ids_list,
        input_ids,
        images,
        stopping_criteria,
        generation_kwargs_ppo,
    )

    ######### 5.3 Parse the responses and create post-generation assets #########
    (
        generated_texts,
        confidence_stripped_generated_texts,
        generated_confidence_values,
        gt_reports_list,
    ) = create_post_report_generation_assets(
        generated_ids=generated_ids,
        batch_metadata_list=batch_metadata_list,
        tokenizer=tokenizer,
        granular_confidence=granular_confidence,
    )

    generated_confidence_values_after_replacement = generated_confidence_values.copy()
    is_confidence_randomly_replaced = [False] * len(generated_confidence_values)

    ############# 5.4 Handle random confidence replacement #########

    if (
        random_number_generator.random() < chance_to_change_confidence
    ):  # Replace with random confidence
        (
            generated_ids,
            generated_texts,
            generated_confidence_values_after_replacement,
            generated_confidence_values,
            is_confidence_randomly_replaced,
        ) = postprocessing.handle_random_confidence_replacement(
            tokenizer,
            generated_texts,
            generated_confidence_values,
            device=device,
            granular_confidence=granular_confidence,
        )

    ########## 5.5 Fetch GREEN scores for the generated reports #########
    green_scores = green.get_green_score_for_batch_of_generated_reports(
        generated_reports=confidence_stripped_generated_texts, gt_reports=gt_reports_list
    )

    ########## 5.6 Compute the scores (rewards) #########

    scores = [
        reward_function(
            confidence=confidence,
            accuracy=accuracy,
            granular_confidence=granular_confidence,
            config=reward_config,
        )
        for confidence, accuracy in zip(generated_confidence_values_after_replacement, green_scores)
    ]

    scores = t.cast(
        list[torch.FloatTensor],
        [torch.tensor(s).to(device) for s in scores],
    )

    ######### 5.7 Reformulate the query and response such that the generated confidence part is the only PPO target #########

    reformulate_query_and_response_fn = functools.partial(
        postprocessing.reformulate_query_and_response, tokenizer=tokenizer
    )
    reformulated_query_and_responses: list[postprocessing.ReformulatedQueryAndResponseDict] = [
        reformulate_query_and_response_fn(query_ids=cur_input_ids, response=cur_response_text)
        for cur_input_ids, cur_response_text in zip(input_ids_list, generated_texts)
    ]

    ########## 5.8 Take a PPO optimization step #########
    ppo_queries, ppo_responses = (
        prepare_ppo_queries_and_responses_from_reformulated_query_and_responses(
            reformulated_query_and_responses=reformulated_query_and_responses
        )
    )
    model.train()
    model.gradient_checkpointing_enable()
    stats = ppo_trainer.multimodal_step(
        queries=ppo_queries,
        responses=ppo_responses,
        scores=scores,
        images=images,
    )

    post_optimization_assets = create_post_ppo_optimization_assets(
        generated_confidence_values=generated_confidence_values,
        generated_confidence_values_after_replacement=generated_confidence_values_after_replacement,
        is_confidence_randomly_replaced=is_confidence_randomly_replaced,
        generated_texts=generated_texts,
        ppo_responses=ppo_responses,
        stats=stats,
        scores=scores,
        green_scores=green_scores,
    )
    return post_optimization_assets


#####################################################################################################################
# HELPER FUNCTIONS
#####################################################################################################################


def prepare_ppo_queries_and_responses_from_reformulated_query_and_responses(
    reformulated_query_and_responses: list[postprocessing.ReformulatedQueryAndResponseDict],
) -> tuple[list[torch.LongTensor], list[torch.LongTensor]]:

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
    return ppo_queries, ppo_responses


def create_post_report_generation_assets(
    generated_ids: list[torch.Tensor],
    batch_metadata_list: list[dataset.MimicCxrDatapoint],
    tokenizer: transformers.PreTrainedTokenizer,
    granular_confidence: bool,
) -> tuple[
    list[str],
    list[str],
    list[int | None],
    list[str],
]:
    generated_texts = tokenizer.batch_decode(generated_ids)
    generated_confidence_values = response.parse_confidences(generated_texts, granular_confidence)
    confidence_stripped_generated_texts = (
        postprocessing.remove_confidence_part_from_generated_responses(generated_texts)
    )
    gt_reports_list = [metadata.report for metadata in batch_metadata_list]
    return (
        generated_texts,
        confidence_stripped_generated_texts,
        generated_confidence_values,
        gt_reports_list,
    )


def create_post_ppo_optimization_assets(
    generated_confidence_values: list[int | None],
    generated_confidence_values_after_replacement: list[int | None],
    is_confidence_randomly_replaced: list[bool],
    generated_texts: list[str],
    ppo_responses: list[torch.LongTensor],
    stats: dict,
    scores: list[torch.FloatTensor],
    green_scores: list[float | None],
) -> logging.ReportGenerationPostPPOOptimizationAssetsBatch:

    return logging.ReportGenerationPostPPOOptimizationAssetsBatch(
        scores=scores,
        green_scores=green_scores,
        generated_confidence_values=generated_confidence_values,
        generated_confidence_values_after_replacement=generated_confidence_values_after_replacement,
        is_confidence_randomly_replaced=is_confidence_randomly_replaced,
        generated_texts=generated_texts,
        ppo_responses=ppo_responses,
        stats=stats,
    )
