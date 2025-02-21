# %% Set script for interactive development and import modules
from RewardingVisualDoubt import infrastructure

infrastructure.make_ipython_reactive_to_changing_codebase()

import pathlib as path

import torch
from torch.utils.data import DataLoader

# from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from RewardingVisualDoubt import dataset, mimic_cxr, prompter, shared, vllm

DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_DIR = path.Path("output")


# %%
def train(out_dir: path.Path = DEFAULT_OUTPUT_DIR, batch_size: int = DEFAULT_BATCH_SIZE):

    ######################################## 0. Define the environment ########################################

    # TODO: Arg parsing etc
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)

    # parameters = {
    #     "experiment_name": out_dir,
    #     "model_dir": model_dir,
    #     "tokenizer_dir": tokenizer_dir,
    #     "lr": lr,
    #     "epochs": epochs,
    #     "episode_length": episode_length,
    #     "batchsize": batchsize,
    # }
    # with open(os.path.join(out_dir, "parameters.json"), "w") as outfile:
    #     json.dump(parameters, outfile)

    device_str = (
        shared.torch_devices.cuda.value
        if torch.cuda.is_available()
        else shared.torch_devices.cpu.value
    )
    device = torch.device(device_str)

    ######################################## 1. Load the model and tokenizer ########################################

    model = vllm.load_pretrained_llava_model_for_ppo_training(device_str=device_str)
    model_ref = vllm.load_pretrained_llava_model_for_ppo_training(device_str=device_str)

    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer.padding_side = "left"

    # TODO do i need a tokenizer dir? # tokenizer = load_tokenizer(tokenizer_dir)
    # TODO: need padding from the left???
    #### model.config.tokenizer_padding_side = "left"  # RaDialog loading logic handles it alreay
    #### model.padding_side='left' - PAUL DOES IT
    #### model.pad_token_id = tokenizer.eos_token_id - PAUL DOES IT

    ######################################## 2. Load the datasets and the dataloaders ########################################

    dataset_train = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.TRAIN,
        tokenizer=tokenizer,
        prompter=prompter.build_binary_qa_instruction_from_disease_under_study,
    )
    dataset_eval = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.VALIDATION,
        tokenizer=tokenizer,
        prompter=prompter.build_binary_qa_instruction_from_disease_under_study,
    )

    dataloader_train = dataset.get_mimic_cxr_llava_model_input_dataloader(
        dataset=dataset_train,
        batch_size=batch_size,
        padding_tokenizer=padding_tokenizer,
        # num_workers=4, # Let Torch decide.
    )

    dataloader_eval = dataset.get_mimic_cxr_llava_model_input_dataloader(
        dataset=dataset_eval,
        batch_size=2 * batch_size,
        padding_tokenizer=padding_tokenizer,
        # num_workers=4, # Let Torch decide.
    )

    ######################################## 3. Define the PPO and generation configurations ########################################

    config = PPOConfig(
        learning_rate=lr,
        task_name="gpt",
        batch_size=batchsize,
        mini_batch_size=int(batchsize / 2),
        log_with=log_with,
        project_kwargs={"logging_dir": out_dir},
        remove_unused_columns=False,
        optimize_device_cache=True,
        init_kl_coef=0.05,
    )

    prediction_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids("ĠConfidence"),
    ]

    ppo_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs_prediction = {
        "max_new_tokens": 256,
        "eos_token_id": prediction_terminators,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    generation_kwargs_ppo = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id,  # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32,
        "eos_token_id": ppo_terminators,
        "max_new_tokens": 500,
    }

    eval_terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generation_kwargs_eval = {
        "max_new_tokens": 256,
        "eos_token_id": eval_terminators,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
    }

    ######################################## 4. Get trainer and set training aspirations ########################################

    ppo_trainer = PPOTrainerNoCache(
        model=model,
        config=config,
        dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # For random exploration
    confidences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    confidences = [str(c) for c in confidences]
    confidences_tokens = tokenizer.convert_tokens_to_ids(confidences)
    chance_to_change_confidence = 0
    steps_to_reach_zero = len(ppo_trainer.dataloader)
    reduce_per_step = chance_to_change_confidence / steps_to_reach_zero

    best_reward = -100
    best_reward_epoch = -1

    ######################################## 5. Train the model ########################################

    for epoch in range(epochs):

        rewards_epoch = []
        for idx, batch in enumerate(ppo_trainer.dataloader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_candidates = batch["gt_candidates"]
            questions = batch["question"]
            is_multiple_choice = batch["is_multiple_choice"]

            model.eval()

            prediction = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs_prediction
            )
            prediction = [remove_padding(p, tokenizer.pad_token_id) for p in prediction]

            # Generate confidence
            model.train()
            response_tensors = ppo_trainer.generate(
                prediction, return_prompt=False, **generation_kwargs_ppo
            )

            # Create prediction + confidence output
            total_tensor = [torch.cat((p, c), 0) for p, c in zip(prediction, response_tensors)]
            answer_only_tensor = [
                total_tensor[i][len(input_ids[i]) :] for i in range(len(input_ids))
            ]

            # For random exploration
            if np.random.random() < chance_to_change_confidence:
                answer_only_tensor = [
                    change_confidence(a, confidences_tokens, np.random.choice(confidences_tokens))
                    for a in answer_only_tensor
                ]

            responses_decoded = tokenizer.batch_decode(answer_only_tensor, skip_special_tokens=True)

            # Parse prediction and confidence
            results = [
                response_to_QAResult(question, response, gt, is_mc)
                for question, response, gt, is_mc in zip(
                    questions, responses_decoded, gt_candidates, is_multiple_choice
                )
            ]

            # Compute rewards
            rewards = [QAResult_to_reward(r) for r in results]
            rewards_epoch += rewards
            rewards = [torch.tensor(r).to(device) for r in rewards]

            # Create log data
            batch["response"] = responses_decoded
            batch["query"] = batch["question"]

            try:
                if is_unsloth:
                    FastLanguageModel.for_training(model.pretrained_model)
                stats = ppo_trainer.step(prediction, response_tensors, rewards)
            except IndexError:
                print(f"INDEX ERROR detected and ignored {idx}")

            ppo_trainer.log_stats(
                stats, batch, rewards, columns_to_log=["query", "response", "answer"]
            )

            # For random exploration
            chance_to_change_confidence -= reduce_per_step
            chance_to_change_confidence = max(0, chance_to_change_confidence)

        avg_reward = np.mean(rewards_epoch)

        print(f"Finished epoch {epoch}. Average reward: {avg_reward}")
        ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned"))

        # Evaluate model after each epoch
        if is_unsloth:
            FastLanguageModel.for_inference(model.pretrained_model)
        else:
            model.eval()
        mean_reward, std_reward = evaluate_model(
            model, dataloader_validation, tokenizer, generation_kwargs_eval, device
        )
        if log_with == "wandb":
            wandb.log({"mean_reward_evaluation": mean_reward})
            wandb.log({"std_reward_evaluation": std_reward})
            wandb.log({"exploration_prob": chance_to_change_confidence})

        # Save the best performing model
        mean_reward = avg_reward
        if mean_reward > best_reward:
            ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned_best"))
            best_reward = mean_reward
            best_reward_epoch = epoch

    print("Finished Training!")
    print(f"Best avg reward {best_reward} in epoch {best_reward_epoch}")
    return
