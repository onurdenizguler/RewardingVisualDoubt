import dataclasses
import datetime
import functools
import itertools
import os
import time
import typing as t

import accelerate
import numpy as np
import trl
import wandb
from tqdm import tqdm

from RewardingVisualDoubt import (
    dataset,
    green,
    prompter,
    shared,
    training,
    vllm,
)


def train(
    metaparameters: training.parameters.TrainingMetaParameters,
    hyperparameters: training.parameters.ReportGenerationPPOHyperparameters,
) -> training.ReportGenerationRunFinalMetrics:

    ######################################## 1. Load the model and tokenizer ########################################

    device, device_str = shared.get_device_and_device_str()
    model = vllm.load_pretrained_llava_model_for_ppo_training_with_lora_adapters(
        device_str=device_str,
        llava_model_path=metaparameters.llava_model_path,
        precision="4bit",
        adapter_path=metaparameters.adapter_path,
    )
    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )

    model.config.padding_side = "left"

    ######################################## 2. Load the datasets and the dataloaders ########################################

    print("Loading the datasets and the dataloaders...")
    selected_prompter_fn = (
        prompter.build_report_generation_prompt_with_response_and_confidence_for_sft
    )
    prompter_ = functools.partial(
        selected_prompter_fn,
        is_for_inference=True,
    )

    dataset_train = dataset.get_report_generation_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.TRAIN, tokenizer=tokenizer, prompter=prompter_
    )

    dataset_eval = dataset.get_report_generation_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.VALIDATION,
        tokenizer=tokenizer,
        prompter=prompter_,
    )

    dataloader_train = dataset.get_mimic_cxr_llava_model_input_dataloader(
        dataset_train,
        batch_size=hyperparameters.batch_size,
        padding_tokenizer=vllm.load_pretrained_llava_tokenizer_with_image_support(
            for_use_in_padding=True
        ),
        num_workers=8,
    )

    dataloader_eval = dataset.get_mimic_cxr_llava_model_input_dataloader(
        dataset_eval,
        batch_size=2 * hyperparameters.batch_size,
        padding_tokenizer=vllm.load_pretrained_llava_tokenizer_with_image_support(
            for_use_in_padding=True
        ),
        num_workers=8,
    )

    eval_batch_iterator = iter(dataloader_eval)
    iterator_train = itertools.islice(
        iter(dataloader_train), metaparameters.n_training_batches_to_skip, None
    )

    ######################################## 3. Define the PPO and generation configurations ########################################

    ppo_config = trl.PPOConfig(
        learning_rate=hyperparameters.learning_rate,
        task_name="gpt",
        ppo_epochs=1,
        batch_size=hyperparameters.batch_size,
        # backward_batch_size=MINI_BATCH_SIZE,  # Default value from TRL library is 1, gets overwritten anyways at __init__ time
        mini_batch_size=hyperparameters.mini_batch_size,
        gradient_accumulation_steps=hyperparameters.gradient_accumulation_steps,
        log_with="wandb",
        tracker_project_name=metaparameters.name_of_fine_tuning,
        tracker_kwargs={
            "wandb": training.get_wandb_parameters_for_report_generation_ppo(
                metaparameters, hyperparameters, selected_prompter_fn
            ),
        },
        project_kwargs=dataclasses.asdict(
            accelerate.utils.ProjectConfiguration(
                project_dir=metaparameters.name_of_fine_tuning, logging_dir="logs"
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
        "max_new_tokens": 300,  # let's not be chatty, we need only a few tokens to generate confidence but also let us not limit the response too much
        "eos_token_id": tokenizer.eos_token_id,  # (instead of ppo_terminators list)
    }

    generation_kwargs_eval = {
        "top_p": 1.0,  # Let us limit the sampling a bit
        "temperature": 1.0,  # Decrease the randomness
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 300,
        "eos_token_id": tokenizer.eos_token_id,
    }

    ########################################## 4. Get trainer and set training aspirations and setup the llama.cpp environment ########################################

    ppo_trainer = t.cast(
        training.MultimodalPPOTrainer,
        training.MultimodalPPOTrainer(
            model=model,
            config=ppo_config,
            tokenizer=tokenizer,
        ),
    )

    print("Starting the llama server...")
    green.start_llama_server(quantization=green.Quantization.Q4_K_M)
    while not green.is_server_alive():
        time.sleep(1)

    ######################################## 5. Train the model ######################################################################################################

    best_score_eval: float = -float("inf")
    mean_score_train: float = -float("inf")
    conf_distribution_kl_eval: float = float("inf")
    ece_eval: float = 1.0
    best_ece_eval: float = 1.0
    best_conf_distribution_kl_eval: float = float("inf")
    best_ece_and_conf_distribution_kl_eval: float = -float("inf")
    patience: int = 0
    scores_until_checkpoint_train: list[float] = []
    accumulating_game_logs = training.create_empty_game_logs()
    perform_validation_before_starting_training = (
        metaparameters.perform_validation_before_starting_training
    )

    for epoch in range(hyperparameters.num_epochs):

        for step in tqdm(
            range(len(dataloader_train) - metaparameters.n_training_batches_to_skip),
            desc=f"Taking training steps... Epoch {epoch+1}/{hyperparameters.num_epochs}",
        ):
            if not perform_validation_before_starting_training:

                batch: dataset.MimicCxrLlavaModelInputBatchDict = next(iterator_train)

                ######### 5.1 - 5.9 Perform a training step #########
                scores, generated_confidence_values, green_scores = (
                    training.radialog_report_generation_ppo_training_step(
                        model,
                        device,
                        tokenizer,
                        generation_kwargs_ppo,
                        ppo_trainer,
                        batch,
                        hyperparameters.reward_function,
                        hyperparameters.reward_config,
                        accumulating_game_logs,
                        chance_to_change_confidence=hyperparameters.chance_to_change_confidence,
                        granular_confidence=hyperparameters.granular_confidence,
                        log_calibration_plot=(
                            step % int(hyperparameters.steps_until_checkpoint / 8)
                        )
                        == 0,  # log at every eighth of the checkpoint interval
                    )
                )

                scores_until_checkpoint_train += [s.item() for s in scores]

            ######### 5.10 Checkpoint the model if checkpoint step is reached #########
            perform_eval = (
                (step + 1) % hyperparameters.steps_until_checkpoint == 0
                or perform_validation_before_starting_training
            )
            if perform_eval:

                if not perform_validation_before_starting_training:

                    mean_train_score = float(np.mean(scores_until_checkpoint_train))
                    print(
                        f"Arrived at checkpoint {step + 1}. Average training score: {mean_train_score}"
                    )
                    if (step + 1) % (3 * hyperparameters.steps_until_checkpoint) == 0:
                        print("Saving the model checkpoint...")
                        # Create dir if it does not exist
                        save_dir = os.path.join(
                            metaparameters.out_dir,
                            metaparameters.name_of_fine_tuning,
                            datetime.datetime.now().strftime("%Y-%m-%d"),
                            f"checkpoint-{step + 1}",
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        ppo_trainer.save_pretrained(
                            save_dir,
                        )
                    wandb.log({"mean_score_training": mean_train_score})

                model.eval()
                eval_batch_scores_list: list[float] = []
                eval_batch_generated_confidence_values_list: list[int | None] = []
                eval_batch_green_scores_list: list[float | None] = []
                for _ in tqdm(
                    range(
                        metaparameters.num_batches_to_evaluate
                        if metaparameters.num_batches_to_evaluate > 0
                        else len(dataloader_eval)
                    ),
                    desc=f"Running evaluation at checkpoint {step + 1}",
                ):
                    try:
                        eval_batch = next(eval_batch_iterator)
                    except StopIteration:
                        eval_batch_iterator = iter(dataloader_eval)
                        eval_batch = next(eval_batch_iterator)

                    scores, generated_confidence_values, green_scores = (
                        training.radialog_report_generation_ppo_evaluation_step(
                            model=model,
                            device=device,
                            tokenizer=tokenizer,
                            generation_kwargs_eval=generation_kwargs_eval,
                            ppo_trainer=ppo_trainer,
                            batch=eval_batch,
                            reward_function=hyperparameters.reward_function,
                            reward_config=hyperparameters.reward_config,
                            granular_confidence=hyperparameters.granular_confidence,
                        )
                    )
                    eval_batch_scores_list.extend(scores)
                    eval_batch_generated_confidence_values_list.extend(generated_confidence_values)
                    eval_batch_green_scores_list.extend(green_scores)

                ece_eval, eval_conf_distribution_kl, mean_eval_score, _ = (
                    training.log_eval_metrics_for_report_generation_ppo(
                        eval_batch_generated_confidence_values_list,
                        eval_batch_scores_list,
                        eval_batch_green_scores_list,
                        hyperparameters,
                    )
                )

                decision_to_break = training.report_generation_ppo_decision_to_break(
                    step=step,
                    patience=patience,
                    eval_conf_distribution_kl=eval_conf_distribution_kl,
                    ece_eval=ece_eval,
                    best_ece_and_distribution_score=best_ece_and_conf_distribution_kl_eval,
                    hyperparameters=hyperparameters,
                )

                if not perform_validation_before_starting_training:
                    if mean_eval_score > best_eval_score:
                        training.save_best_eval_lora_adapters_and_value_head_to_dir(
                            ppo_trainer,
                            epoch,
                            step,
                            metaparameters.out_dir,
                            metaparameters.name_of_fine_tuning,
                        )
                        best_eval_score = mean_eval_score

                perform_validation_before_starting_training = False

                if decision_to_break:
                    break

        if decision_to_break:
            break

    return training.ReportGenerationRunFinalMetrics(
        mean_score_train_at_last_checkpoint=0,
        last_mean_score_eval=0,
        last_ece_eval=0,
        last_conf_distribution_kl_eval=0,
        last_ece_and_conf_distribution_kl_eval=0,
        best_mean_score_eval=0,
        best_ece_eval=0,
        best_conf_distribution_kl_eval=0,
        best_ece_and_conf_distribution_kl_eval=0,
    )
