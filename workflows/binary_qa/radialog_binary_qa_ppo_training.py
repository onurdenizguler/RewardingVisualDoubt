import dataclasses
import datetime
import functools
import itertools
import os
import pathlib as path
import typing as t

import accelerate
import numpy as np
import torch
import trl
import wandb
from tqdm import tqdm

from RewardingVisualDoubt import dataset, evaluation, prompter, reward, shared, training, vllm

SELECTED_STEPS_UNTIL_CHECKPOINT = 50
SELECTED_LEARNING_RATE = 1e-5
SELECTED_PERFORM_VALIDATION_BEFORE_STARTING_TRAINING = True
SELECTED_REWARD_SCALE = reward.SCALE
SELECTED_CHANCE_TO_CHANGE_CONFIDENCE = 0.4
SELECTED_ADAPTER_PATH = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radialog_binary_qa_ppo_training/2025-06-21_b/checkpoint-150/adapter_model.bin"
SELECTED_GRANULAR_CONFIDENCE = True

SELECTED_BATCH_SIZE = 12
SELECTED_NUM_BATCHES_TO_EVALUATE = 0  # Set to zero to evaulate on the whole validation set
SELECTED_NUM_GRADIENT_ACCUMULATION_STEPS = 3
SELECTED_MINI_BATCH_SIZE = 4
SELECTED_NUM_EPOCHS = 1
SELECTED_NUM_TRAINING_BATCHES_TO_SKIP = 1  # Do not set this to be much higher than a 100
SELECTED_OUTPUT_DIR = path.Path("/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models")
SELECTED_FINETUNING_NAME = "radialog_binary_qa_ppo_training"
SELECTED_REWARD_FUNCTION = reward.default_reward_function


def train(
    name_of_fine_tuning: str = SELECTED_FINETUNING_NAME,
    num_epochs: int = SELECTED_NUM_EPOCHS,
    steps_until_checkpoint: int = SELECTED_STEPS_UNTIL_CHECKPOINT,
    batch_size: int = SELECTED_BATCH_SIZE,
    num_batches_to_evaluate: int = SELECTED_NUM_BATCHES_TO_EVALUATE,
    n_training_batches_to_skip: int = SELECTED_NUM_TRAINING_BATCHES_TO_SKIP,
    gradient_accumulation_steps: int = SELECTED_NUM_GRADIENT_ACCUMULATION_STEPS,
    mini_batch_size: int = SELECTED_MINI_BATCH_SIZE,
    learning_rate: float = SELECTED_LEARNING_RATE,
    chance_to_change_confidence: float = SELECTED_CHANCE_TO_CHANGE_CONFIDENCE,
    reward_function: t.Callable = SELECTED_REWARD_FUNCTION,
    adapter_path: str | None = SELECTED_ADAPTER_PATH,
    out_dir: path.Path = SELECTED_OUTPUT_DIR,
    perform_validation_before_starting_training: bool = SELECTED_PERFORM_VALIDATION_BEFORE_STARTING_TRAINING,
    granular_confidence: bool = SELECTED_GRANULAR_CONFIDENCE,
):

    ######################################## 0. Define the environment ########################################

    device_str = (
        shared.torch_devices.cuda.value
        if torch.cuda.is_available()
        else shared.torch_devices.cpu.value
    )
    device = torch.device(device_str)

    ######################################## 1. Load the model and tokenizer ########################################

    selected_llava_model_path = (
        vllm.RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value
    )
    model = vllm.load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters(
        device_str=device_str,
        llava_model_path=selected_llava_model_path,
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
    selected_prompter_fn = prompter.build_binary_qa_prompt_with_response_and_confidence_for_sft
    prompter_ = functools.partial(
        selected_prompter_fn,
        is_for_inference=True,
        granular_confidence=granular_confidence,
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
        tracker_project_name=name_of_fine_tuning,
        tracker_kwargs={
            "wandb": {
                "id": f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{format(SELECTED_LEARNING_RATE, '.0e')}_{SELECTED_CHANCE_TO_CHANGE_CONFIDENCE}_{SELECTED_REWARD_SCALE}",
                "config": {
                    "date_of_training": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "learning_rate": learning_rate,
                    "chance_to_change_confidence": chance_to_change_confidence,
                    "reward_function": reward_function.__name__,
                    "reward_scale": reward.SCALE,
                    "prompter": selected_prompter_fn.__name__,
                    "granular_confidence": str(granular_confidence),
                    "starting_llava_model_path": selected_llava_model_path,
                    "starting_adapter_path": adapter_path,
                    "num_epochs": num_epochs,
                    "perform_validation_before_starting_training": str(
                        perform_validation_before_starting_training
                    ),
                    "steps_until_checkpoint": steps_until_checkpoint,
                    "batch_size": batch_size,
                    "mini_batch_size": mini_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "num_batches_to_evaluate": num_batches_to_evaluate,
                    "n_training_batches_to_skip": n_training_batches_to_skip,
                },
            }
        },
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
        "temperature": 1.0,  # Just take the logits as they are
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.pad_token_id,  # most decoder models don't have a padding token - use EOS token instead (for this tokenizer it was already set to eos_token_id)
        "max_new_tokens": 50,  # let's not be chatty, we need only a few tokens to generate confidence but also let us not limit the response too much
        "eos_token_id": tokenizer.eos_token_id,  # (instead of ppo_terminators list)
    }

    generation_kwargs_eval = {
        "top_p": 1.0,  # Let us limit the sampling a bit
        "temperature": 1.0,  # Decrease the randomness
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
            "ppo_target_responses": [],
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
            if not perform_validation_before_starting_training:

                batch: dataset.MimicCxrLlavaModelInputBatchDict = next(iterator_train)

                ######### 5.1 - 5.8 Perform a training step #########
                scores = training.radialog_binary_qa_ppo_training_step(
                    model,
                    device,
                    tokenizer,
                    generation_kwargs_ppo,
                    ppo_trainer,
                    batch,
                    reward_function,
                    accumulating_game_logs,
                    chance_to_change_confidence,
                    granular_confidence,
                    log_calibration_plot=(step % int(steps_until_checkpoint / 4))
                    == 0,  # log at every quarter of the checkpoint interval
                )

                train_scores_until_checkpoint += [s.item() for s in scores]

            ######### 5.9 Checkpoint the model if checkpoint step is reached #########
            if (
                step + 1
            ) % steps_until_checkpoint == 0 or perform_validation_before_starting_training:

                if not perform_validation_before_starting_training:
                    mean_train_score = np.mean(train_scores_until_checkpoint)

                    print(
                        f"Arrived at checkpoint {step + 1}. Average training score: {mean_train_score}"
                    )
                    if (step + 1) % (3 * steps_until_checkpoint) == 0:
                        print("Saving the model checkpoint...")
                        # Create dir if it does not exist
                        save_dir = os.path.join(
                            out_dir,
                            name_of_fine_tuning,
                            datetime.datetime.now().strftime("%Y-%m-%d"),
                            f"checkpoint-{step + 1}",
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        ppo_trainer.save_pretrained(
                            save_dir,
                        )
                    wandb.log({"mean_score_training": mean_train_score})

                model.eval()
                eval_batch_scores_list = []
                eval_batch_generated_confidence_values_list = []
                eval_batch_is_answer_correct_list = []
                for _ in tqdm(
                    range(
                        num_batches_to_evaluate
                        if num_batches_to_evaluate > 0
                        else len(dataloader_eval)
                    ),
                    desc=f"Running evaluation at checkpoint {step + 1}",
                ):
                    try:
                        eval_batch = next(eval_batch_iterator)
                    except StopIteration:
                        eval_batch_iterator = iter(dataloader_eval)
                        eval_batch = next(eval_batch_iterator)
                    scores, generated_confidence_values, is_answer_correct = (
                        training.radialog_binary_qa_ppo_evaluation_step(
                            model,
                            device,
                            tokenizer,
                            generation_kwargs_eval,
                            ppo_trainer,
                            eval_batch,
                            reward.default_reward_function,
                            granular_confidence,
                        )
                    )
                    eval_batch_scores_list.extend(scores)
                    eval_batch_generated_confidence_values_list.extend(generated_confidence_values)
                    eval_batch_is_answer_correct_list.extend(is_answer_correct)

                mean_eval_score = np.mean(eval_batch_scores_list)
                mean_std_score = np.std(eval_batch_scores_list)
                wandb.log({"mean_score_evaluation": mean_eval_score})
                wandb.log({"std_score_evaluation": mean_std_score})
                try:
                    wandb.log(
                        {
                            "val_conf_calib": wandb.Image(
                                evaluation.plot_calibration_curve(
                                    confidences=eval_batch_generated_confidence_values_list,
                                    accuracies=eval_batch_is_answer_correct_list,
                                )
                            )
                        }
                    )
                except:
                    pass

                # Save the best performing model
                if not perform_validation_before_starting_training:
                    if mean_eval_score > best_eval_score:
                        training.save_best_eval_lora_adapters_and_value_head_to_dir(
                            ppo_trainer,
                            epoch,
                            step,
                            out_dir,
                            name_of_fine_tuning,
                        )
                        best_eval_score = mean_eval_score

                perform_validation_before_starting_training = False


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "da3cb086bbc110c16cbc5ba4c284a19b0b461710"
    train()
