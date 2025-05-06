import dataclasses
import datetime
import functools
import itertools
import os
import pathlib as path
import random
import typing as t

import accelerate
import numpy as np
import torch
import transformers
import trl
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from tqdm import tqdm

import wandb
from RewardingVisualDoubt import (
    dataset,
    evaluation,
    prompter,
    response,
    reward,
    shared,
    training,
    vllm,
)


SELECTED_STEPS_UNTIL_CHECKPOINT = 50
SELECTED_NUM_BATCHES_TO_EVALUATE = 20
SELECTED_LEARNING_RATE = 1e-5
SELECTED_CHANCE_TO_CHANGE_CONFIDENCE = 0.4
SELECTED_ADAPTER_PATH = None
SELECTED_BATCH_SIZE = 8
SELECTED_NUM_EPOCHS = 1
SELECTED_NUM_TRAINING_BATCHES_TO_SKIP = 1  # Do not set this to be much higher than a 100
SELECTED_NUM_GRADIENT_ACCUMULATION_STEPS = 2
SELECTED_NUM_MINI_BATCHES = 4
SELECTED_OUTPUT_DIR = path.Path("/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models")
SELECTED_FINETUNING_NAME = "radialog_binary_qa_ppo_training"
SELECTED_REWARD_FUNCTION = reward.default_reward_function


STOP_STR = prompter.Seperator.END_OF_SEQUENCE_SEPERATOR.value


def radialog_binary_qa_ppo_evaluation_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_eval: dict,
    ppo_trainer: training.MultimodalPPOTrainer,
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
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
    input_ids_list = training.remove_preciding_padding_from_batch_tensor(
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


def _handle_random_confidence_replacement(
    tokenizer: transformers.PreTrainedTokenizer,
    generated_texts: list[str],
    generated_confidence_values: list[int | None],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[str], list[int | None], list[int | None], list[bool]]:
    old_generated_confidence_values = generated_confidence_values.copy()
    generated_texts = training.overwrite_confidence(generated_texts, generated_confidence_values)
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
    accumulating_game_logs: training.GameLogs,
    queries: list[str],
    responses: list[str],
    is_answer_correct: list[bool],
    scores: list[torch.FloatTensor],
    confidences: list[int | None],
    old_generated_confidence_values: list[int | None],
    is_confidence_randomly_replaced: list[bool],
) -> training.GameLogs:

    accumulating_game_logs["queries"].extend(queries)
    accumulating_game_logs["responses"].extend(responses)
    accumulating_game_logs["is_answer_correct"].extend(is_answer_correct)
    accumulating_game_logs["scores"].extend([score.item() for score in scores])
    accumulating_game_logs["confidences"].extend(old_generated_confidence_values)
    accumulating_game_logs["confidences_after_replacement"].extend(confidences)
    accumulating_game_logs["is_confidence_randomly_replaced"].extend(
        is_confidence_randomly_replaced
    )

    truncated_accumulating_game_logs: training.GameLogs = {
        "queries": [],
        "responses": [],
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


def radialog_binary_qa_ppo_training_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_ppo: dict,
    ppo_trainer: training.MultimodalPPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
    reward_function: t.Callable,
    accumulating_game_logs: training.GameLogs | None = None,
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
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
    input_ids_list = training.remove_preciding_padding_from_batch_tensor(
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
    model.train()
    model.gradient_checkpointing_enable()
    stats = ppo_trainer.multimodal_step(
        queries=t.cast(list[torch.LongTensor], input_ids_list),
        responses=t.cast(list[torch.LongTensor], generated_ids),
        scores=scores,
        images=images,
    )

    ######### 5.8 Create a report and log the training stats #########
    batch_report = {}
    queries_with_gt_labels = [
        f"(GT Label: {str(labels.bool().tolist()[idx])}) - {input_prompt}"
        for idx, input_prompt in enumerate(
            tokenizer.batch_decode(
                training.replace_image_token_with_another_token_for_list_of_tensors(input_ids_list)
            )
        )
    ]
    if accumulating_game_logs:
        truncated_accumulating_game_logs = _handle_accumulating_game_logs(
            accumulating_game_logs,
            queries=queries_with_gt_labels,
            responses=generated_texts,
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
        table_rows = [
            list(r)
            for r in zip(
                truncated_accumulating_game_logs["queries"],
                truncated_accumulating_game_logs["responses"],
                truncated_accumulating_game_logs["is_answer_correct"],
                truncated_accumulating_game_logs["confidences"],
                truncated_accumulating_game_logs["confidences_after_replacement"],
                truncated_accumulating_game_logs["is_confidence_randomly_replaced"],
                truncated_accumulating_game_logs["scores"],
            )
        ]
        stats["accumulating_game_logs"] = wandb.Table(
            columns=[
                "query",
                "response",
                "is_answer_correct",
                "confidence",
                "confidence_after_replacement",
                "is_confidence_randomly_replaced",
                "reward",
            ],
            rows=table_rows,
        )
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

    batch_report["query"] = queries_with_gt_labels
    batch_report["response"] = generated_texts

    ppo_trainer.log_stats(stats=stats, batch=batch_report, rewards=scores)

    return scores


def train(
    name_of_fine_tuning: str = SELECTED_FINETUNING_NAME,
    num_epochs: int = SELECTED_NUM_EPOCHS,
    steps_until_checkpoint: int = SELECTED_STEPS_UNTIL_CHECKPOINT,
    batch_size: int = SELECTED_BATCH_SIZE,
    num_batches_to_evaluate: int = SELECTED_NUM_BATCHES_TO_EVALUATE,
    n_training_batches_to_skip: int = SELECTED_NUM_TRAINING_BATCHES_TO_SKIP,
    gradient_accumulation_steps: int = SELECTED_NUM_GRADIENT_ACCUMULATION_STEPS,
    mini_batch_size: int = SELECTED_NUM_MINI_BATCHES,
    learning_rate: float = SELECTED_LEARNING_RATE,
    chance_to_change_confidence: float = SELECTED_CHANCE_TO_CHANGE_CONFIDENCE,
    reward_function: t.Callable = SELECTED_REWARD_FUNCTION,
    adapter_path: str | None = SELECTED_ADAPTER_PATH,
    out_dir: path.Path = SELECTED_OUTPUT_DIR,
):

    ######################################## 0. Define the environment ########################################

    device_str = (
        shared.torch_devices.cuda.value
        if torch.cuda.is_available()
        else shared.torch_devices.cpu.value
    )
    device = torch.device(device_str)

    ######################################## 1. Load the model and tokenizer ########################################

    model = vllm.load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters(
        device_str=device_str,
        llava_model_path=vllm.RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
        precision="4bit",
        adapter_path=adapter_path,
    )

    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer.padding_side = "left"

    tokenizer.padding_side = "left"
    model.config.padding_side = "left"
    model.config.tokenizer_padding_side = "left"

    ######################################## 2. Load the datasets and the dataloaders ########################################

    print("Loading the datasets and the dataloaders...")
    prompter_ = functools.partial(
        prompter.build_binary_qa_prompt_with_response_and_confidence_for_sft, is_for_inference=True
    )
    dataset_train = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.TRAIN,
        tokenizer=tokenizer,
        prompter=prompter_,
    )
    dataset_eval = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.VALIDATION,
        tokenizer=tokenizer,
        prompter=prompter_,
    )

    dataloader_train = dataset.get_mimic_cxr_llava_model_input_dataloader(
        dataset=dataset_train,
        batch_size=batch_size,
        padding_tokenizer=padding_tokenizer,
        num_workers=8,
    )

    dataloader_eval = dataset.get_mimic_cxr_llava_model_input_dataloader(
        dataset=dataset_eval,
        batch_size=2 * batch_size,
        padding_tokenizer=padding_tokenizer,
        num_workers=8,
    )

    eval_batch_iterator = iter(dataloader_eval)

    ######################################## 3. Define the PPO and generation configurations ########################################

    ppo_config = trl.PPOConfig(
        learning_rate=learning_rate,
        task_name="gpt",
        ppo_epochs=1,
        batch_size=batch_size,
        # backward_batch_size=MINI_BATCH_SIZE,  # Default value from TRL library is 1, gets overwritten anyways at __init__ time
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        tracker_project_name="radialog_binary_qa_ppo_training",
        project_kwargs=dataclasses.asdict(
            accelerate.utils.ProjectConfiguration(
                project_dir="radialog_binary_qa_ppo_training", logging_dir="logs"
            )
        ),
        remove_unused_columns=False,
        kl_penalty="kl",
        init_kl_coef=0.05,
    )

    generation_kwargs_ppo = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "temperature": 1.0,  # DONT BE CREATIVE
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.pad_token_id,  # most decoder models don't have a padding token - use EOS token instead (for this tokenizer it was already set to eos_token_id)
        "max_new_tokens": 50,  # let's not be chatty, we need only a few tokens to generate confidence but also let us not limit the response too much
        "eos_token_id": tokenizer.eos_token_id,  # (instead of ppo_terminators list)
    }

    generation_kwargs_eval = {
        "top_p": 0.9,
        "temperature": 0.6,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 256,
        "eos_token_id": tokenizer.eos_token_id,
    }

    ######################################## 4. Get trainer and set training aspirations ########################################

    ppo_trainer = t.cast(
        training.MultimodalPPOTrainer,
        training.MultimodalPPOTrainer(
            model=model,
            config=ppo_config,
            tokenizer=tokenizer,
        ),
    )

    ######################################## 5. Train the model ########################################

    for epoch in range(num_epochs):

        best_eval_score = -100
        train_scores_until_checkpoint = []
        accumulating_game_logs: training.GameLogs = {
            "queries": [],
            "responses": [],
            "is_answer_correct": [],
            "scores": [],
            "confidences": [],
            "confidences_after_replacement": [],
            "is_confidence_randomly_replaced": [],
        }
        iterator_train = itertools.islice(iter(dataloader_train), n_training_batches_to_skip, None)
        for step in tqdm(
            range(len(dataloader_train) - n_training_batches_to_skip),
            desc="Taking training steps...",
        ):

            batch: dataset.MimicCxrLlavaModelInputBatchDict = next(iterator_train)

            ######### 5.1 - 5.8 Perform a training step #########
            scores = radialog_binary_qa_ppo_training_step(
                model,
                device,
                tokenizer,
                generation_kwargs_ppo,
                ppo_trainer,
                batch,
                reward_function,
                accumulating_game_logs,
                chance_to_change_confidence,
            )

            train_scores_until_checkpoint += [s.item() for s in scores]

            ######### 5.9 Checkpoint the model if checkpoint step is reached #########
            if (step + 1) % steps_until_checkpoint == 0:

                mean_train_score = np.mean(train_scores_until_checkpoint)

                print(
                    f"Arrived at checkpoint {step + 1}. Average training score: {mean_train_score}"
                )
                print("Saving the model checkpoint...")
                # Create dir if it does not exist
                save_dir = os.path.join(
                    out_dir,
                    name_of_fine_tuning,
                    datetime.datetime.now().strftime("%Y-%m-%d"),
                    f"checkpoint-{step + 1}",
                )
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(
                    save_dir,
                )

                model.eval()
                eval_batch_scores_list = []
                eval_batch_generated_confidence_values_list = []
                eval_batch_is_answer_correct_list = []
                for _ in tqdm(
                    range(num_batches_to_evaluate),
                    desc=f"Running evaluation at checkpoint {step + 1}",
                ):
                    try:
                        eval_batch = next(eval_batch_iterator)
                    except StopIteration:
                        eval_batch_iterator = iter(dataloader_eval)
                        eval_batch = next(eval_batch_iterator)
                    scores, generated_confidence_values, is_answer_correct = (
                        radialog_binary_qa_ppo_evaluation_step(
                            model,
                            device,
                            tokenizer,
                            generation_kwargs_eval,
                            ppo_trainer,
                            eval_batch,
                            reward.default_reward_function,
                        )
                    )
                    eval_batch_scores_list.extend(scores)
                    eval_batch_generated_confidence_values_list.extend(generated_confidence_values)
                    eval_batch_is_answer_correct_list.extend(is_answer_correct)

                mean_eval_score = np.mean(eval_batch_scores_list)
                mean_std_score = np.std(eval_batch_scores_list)
                wandb.log({"mean_score_training": mean_train_score})
                wandb.log({"mean_score_evaluation": mean_eval_score})
                wandb.log({"std_score_evaluation": mean_std_score})

                wandb.log(
                    {
                        "val_conf_calib": wandb.Image(
                            evaluation.plot_calibration_curve(
                                confidences=eval_batch_generated_confidence_values_list,
                                is_answer_correct=eval_batch_is_answer_correct_list,
                            )
                        )
                    }
                )

                # Save the best performing model
                if mean_eval_score > best_eval_score:
                    save_dir = os.path.join(
                        out_dir,
                        name_of_fine_tuning,
                        datetime.datetime.now().strftime("%Y-%m-%d"),
                        "best_eval_model",
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    best_eval_score = mean_eval_score


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "da3cb086bbc110c16cbc5ba4c284a19b0b461710"
    train()
