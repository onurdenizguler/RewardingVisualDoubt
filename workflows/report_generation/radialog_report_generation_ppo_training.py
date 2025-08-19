import functools
import itertools
import random
import sys
import time
import typing as t

import numpy as np
import trl
import wandb
from tqdm import tqdm

from RewardingVisualDoubt import (
    dataset,
    evaluation,
    green,
    prompter,
    reward,
    shared,
    training,
    vllm,
)


def train(
    metaparameters: training.parameters.TrainingMetaParameters,
    hyperparameters: training.parameters.ReportGenerationPPOHyperparameters,
) -> training.ReportGenerationRunFinalMetrics:

    ######################################## 1. Load the RaDialog model and tokenizer, start the RadLLamaGREEN llama.cpp server ########################################

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

    print("Starting the llama server...")
    if not green.is_server_alive():
        green.start_llama_server(quantization=green.Quantization.Q4_K_M)
    while not green.is_server_alive():
        time.sleep(1)

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
        batch_size=hyperparameters.eval_batch_size,
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
        "top_k": 0.0,  # No top-k sampling
        "top_p": 1.0,  # Let us limit the sampling a bit
        "temperature": 1.0,  # Decrease the randomness
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 300,
        "eos_token_id": tokenizer.eos_token_id,
    }

    ########################################## 4. Get trainer and initialize the wandb tracker ########################################

    ppo_trainer = t.cast(
        training.MultimodalPPOTrainer,
        training.MultimodalPPOTrainer(
            model=model,
            config=ppo_config,
            tokenizer=tokenizer,
        ),
    )

    wandb.init(
        project=metaparameters.name_of_fine_tuning,
        dir="/home/guests/deniz_gueler/repos/RewardingVisualDoubt/logs/radialog_report_generation_ppo_training",
        **training.get_wandb_parameters_for_report_generation_ppo(
            metaparameters, hyperparameters, selected_prompter_fn
        ),
    )

    ######################################## 5. Train the model ######################################################################################################

    kpis = training.init_training_success_kpis()
    patience = training.PatienceTracker()
    scores_until_checkpoint_train: list[float] = []
    accumulating_game_logs = training.create_empty_game_logs_for_report_generation()
    perform_validation_before_starting_training = (
        metaparameters.perform_validation_before_starting_training
    )
    heuristic_fn: t.Callable[
        [evaluation.RewardECEAndDistributionScore, int, tuple[float, float]], float
    ] = hyperparameters.reward_ece_and_distribution_score_heuristic
    random_number_generator = random.Random(53355335)

    for epoch in range(hyperparameters.num_epochs):

        for step in tqdm(
            range(len(dataloader_train) - metaparameters.n_training_batches_to_skip),
            desc=f"Taking training steps... Epoch {epoch+1}/{hyperparameters.num_epochs}",
            file=sys.stderr,
            dynamic_ncols=True,
        ):
            if not perform_validation_before_starting_training:

                batch: dataset.MimicCxrLlavaModelInputBatchDict = next(iterator_train)

                ######### 5.1 - 5.9 Perform a training step #########
                post_optimization_assets = training.radialog_report_generation_ppo_training_step(
                    model,
                    device,
                    tokenizer,
                    generation_kwargs_ppo,
                    ppo_trainer,
                    batch,
                    hyperparameters.reward_function,
                    hyperparameters.reward_config,
                    random_number_generator,
                    chance_to_change_confidence=hyperparameters.chance_to_change_confidence,
                    granular_confidence=hyperparameters.granular_confidence,
                )

                scores_until_checkpoint_train += [s.item() for s in post_optimization_assets.scores]
                training.log_train_metrics_for_report_generation_ppo(
                    step=step + 1,
                    post_optimization_assets=post_optimization_assets,
                    device=device,
                    tokenizer=tokenizer,
                    batch=batch,
                    hyperparameters=hyperparameters,
                    accumulating_game_logs=accumulating_game_logs,
                    log_calibration_plot=(
                        (step + 1)
                        % metaparameters.plot_confidence_calibration_for_training_batches_every_n_batch
                    )
                    == 0,  # log at every (1/n) of the checkpoint interval
                )

            ######### 5.10 Checkpoint the model if checkpoint step is reached #########
            perform_eval = (
                (step + 1) % hyperparameters.steps_until_checkpoint == 0
                or perform_validation_before_starting_training
            )
            if perform_eval:

                if not perform_validation_before_starting_training:

                    kpis.mean_score_train.append(float(np.mean(scores_until_checkpoint_train)))
                    print(
                        f"Arrived at checkpoint {step + 1}. Average training score: {kpis.mean_score_train[-1]}"
                    )
                    if (step + 1) % (
                        hyperparameters.steps_until_checkpoint
                        * metaparameters.save_training_model_every_n_checkpoints
                    ) == 0:
                        print("Saving the model checkpoint...")
                        training.save_training_checkpoint_lora_adapters_and_value_head_to_dir(
                            ppo_trainer,
                            epoch,
                            step,
                            metaparameters.out_dir,
                            metaparameters.name_of_fine_tuning,
                            hyperparameters.reward_config,
                        )
                    wandb.log(
                        {"mean_score_training_checkpoint": kpis.mean_score_train}, step=step + 1
                    )
                    scores_until_checkpoint_train = []

                model.eval()
                batch_scores_list_eval: list[float] = []
                batch_generated_confidence_values_list_eval: list[int | None] = []
                batch_green_scores_list_eval: list[float | None] = []
                for _ in tqdm(
                    range(
                        metaparameters.num_batches_to_evaluate
                        if metaparameters.num_batches_to_evaluate > 0
                        else len(dataloader_eval)
                    ),
                    desc=f"Running evaluation at checkpoint {step + 1}",
                    file=sys.stderr,
                    dynamic_ncols=True,
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
                    batch_scores_list_eval.extend(scores)
                    batch_generated_confidence_values_list_eval.extend(generated_confidence_values)
                    batch_green_scores_list_eval.extend(green_scores)

                (
                    weighted_mean_of_std_of_accuracies,
                    ece_eval,
                    conf_distribution_kl_eval,
                    mean_score_eval,
                    heuristic_aggregated_score_eval,
                ) = training.log_eval_metrics_for_report_generation_ppo(
                    step + 1,
                    batch_generated_confidence_values_list_eval,
                    batch_scores_list_eval,
                    batch_green_scores_list_eval,
                    hyperparameters,
                )

                kpis.ece_eval.append(ece_eval)
                kpis.conf_distribution_kl_eval.append(conf_distribution_kl_eval)
                kpis.mean_score_eval.append(mean_score_eval)
                kpis.heuristic_aggregated_score_eval.append(heuristic_aggregated_score_eval)
                kpis.weighted_mean_of_std_of_accuracies.append(weighted_mean_of_std_of_accuracies)

                decision_to_break = training.report_generation_ppo_decision_to_break(
                    step=step + 1,
                    patience=patience,
                    heuristic_aggregated_scores=kpis.heuristic_aggregated_score_eval,
                    hyperparameters=hyperparameters,
                )

                if not perform_validation_before_starting_training:
                    if kpis.heuristic_aggregated_score_eval[-1] >= max(
                        kpis.heuristic_aggregated_score_eval
                    ):
                        print("Saving best eval model yet...")
                        training.save_best_eval_lora_adapters_and_value_head_to_dir(
                            ppo_trainer,
                            epoch,
                            step,
                            metaparameters.out_dir,
                            metaparameters.name_of_fine_tuning,
                            reward_config=hyperparameters.reward_config,
                        )

                perform_validation_before_starting_training = False

                if decision_to_break:
                    break

        if decision_to_break:
            break

    wandb.finish()

    return training.ReportGenerationRunFinalMetrics(
        mean_score_train_at_last_checkpoint=kpis.mean_score_train[-1],
        last_mean_score_eval=kpis.mean_score_eval[-1],
        last_ece_eval=kpis.ece_eval[-1],
        last_conf_distribution_kl_eval=kpis.conf_distribution_kl_eval[-1],
        last_ece_and_conf_distribution_kl_eval=kpis.heuristic_aggregated_score_eval[-1],
        last_weighted_mean_of_std_of_accuracies=kpis.weighted_mean_of_std_of_accuracies[-1],
        best_mean_score_eval=max(kpis.mean_score_eval),
        best_ece_eval=max(kpis.ece_eval),
        best_conf_distribution_kl_eval=max(kpis.conf_distribution_kl_eval),
        best_ece_and_conf_distribution_kl_eval=max(kpis.heuristic_aggregated_score_eval),
        best_weighted_mean_of_std_of_accuracies=max(kpis.weighted_mean_of_std_of_accuracies),
    )


if __name__ == "__main__":
    import argparse, pickle
    from pathlib import Path

    TRIAL_DIR = Path(
        "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/logs/radialog_report_generation_ppo_training/trials"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", required=True, help="Path to pickle with metaparameters & hyperparameters"
    )
    args = parser.parse_args()

    with open(args.spec, "rb") as f:
        payload = pickle.load(f)

    metaparameters: training.TrainingMetaParameters = payload["metaparameters"]
    hyperparameters: training.ReportGenerationPPOHyperparameters = payload["hyperparameters"]

    # Run training (new process → fresh CUDA context)
    result = train(metaparameters, hyperparameters)

    run_hash = training.create_hash_out_of_parameters(metaparameters, hyperparameters)
    pkl_dir = TRIAL_DIR / "final_metrics"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_dir / f"final_metrics_{run_hash}.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
