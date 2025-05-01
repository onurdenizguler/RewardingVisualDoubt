# %% Set script for interactive development and import modules
from RewardingVisualDoubt import infrastructure

infrastructure.make_ipython_reactive_to_changing_codebase()
infrastructure.supress_known_warnings()

import dataclasses
import os
import pathlib as path
import typing as t
import functools

import accelerate
import numpy as np
import torch
import transformers
import trl
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
import wandb

from RewardingVisualDoubt import dataset, prompter, response, reward, shared, evaluation
from RewardingVisualDoubt import training as training
from RewardingVisualDoubt import vllm

DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_DIR = path.Path("output")
STEPS_UNTIL_CHECKPOINT = 50
NUM_BATCHES_TO_EVALUATE = 15

STOP_STR = prompter.Seperator.END_OF_SEQUENCE_SEPERATOR.value


@dataclasses.dataclass
class Hyperparameters:
    num_epochs: int
    batch_size: int
    episode_length: int
    learning_rate: float
    reward_scaler: float


def calculate_mean_and_std_rewards(eval_batch_rewards_list):
    pass


def radialog_binary_qa_ppo_evaluation_step(model, batch):
    pass


def _handle_accumulating_game_logs(
    accumulating_game_logs: training.GameLogs,
    queries: list[str],
    responses: list[str],
    is_answer_correct: list[bool],
    scores: list[torch.FloatTensor],
    confidences: list[int | None],
) -> training.GameLogs:

    accumulating_game_logs["queries"].extend(queries)
    accumulating_game_logs["responses"].extend(responses)
    accumulating_game_logs["is_answer_correct"].extend(is_answer_correct)
    accumulating_game_logs["scores"].extend([score.item() for score in scores])
    accumulating_game_logs["confidences"].extend(confidences)

    truncated_accumulating_game_logs: training.GameLogs = {
        "queries": [],
        "responses": [],
        "is_answer_correct": [],
        "scores": [],
        "confidences": [],
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
):

    ######### 5.1 Unpack the batch #########
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    attention_mask = batch["batch_attention_mask"].to(
        device
    )  # WARNING: Unused, as the ppo_trainer.generate() method handles padding and batching itself
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)
    input_ids_list = training.remove_preciding_padding_from_batch_tensor(
        input_ids
    )  # WARNING: Is performed, as the ppo_trainer.generate() method handles padding and batching itself

    ######### 5.2 Generate the binary q&a answer and remove trailing padding tokens #########
    # model.eval()
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
    generated_answer_labels = response.parse_binary_labels(generated_texts)
    generated_confidence_values = response.parse_confidences(generated_texts)

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
    model.pretrained_model.enable_input_require_grads()
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
        )
        table_rows = [
            list(r)
            for r in zip(
                truncated_accumulating_game_logs["queries"],
                truncated_accumulating_game_logs["responses"],
                truncated_accumulating_game_logs["is_answer_correct"],
                truncated_accumulating_game_logs["confidences"],
                truncated_accumulating_game_logs["scores"],
            )
        ]
        stats["accumulating_game_logs"] = wandb.Table(
            columns=["query", "response", "is_answer_correct", "confidence", "reward"],
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


# %%
def train(
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    mini_batch_size: int,
    learning_rate: float,
    out_dir: path.Path,
):

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

    ######################################## 0. Define the environment ########################################

    device_str = (
        shared.torch_devices.cuda.value
        if torch.cuda.is_available()
        else shared.torch_devices.cpu.value
    )
    device = torch.device(device_str)

    ######################################## 1. Load the model and tokenizer ########################################

    print("Loading the model and tokenizer...")
    model = vllm.load_pretrained_llava_model_for_ppo_training(
        device_str=device_str,
        precision="4bit",
        radialog_lora_weights_path=vllm.RadialogLoraWeightsPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
    )
    print("Model loaded.")
    model.pretrained_model.print_trainable_parameters()

    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer.padding_side = "left"

    # TODO: not sure if needed but just to be safe for now (remove when sure)
    tokenizer.padding_side = "left"
    model.config.padding_side = "left"
    model.config.tokenizer_padding_side = "left"
    # model.pad_token_id = tokenizer.eos_token_id

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
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        project_kwargs=dataclasses.asdict(
            accelerate.utils.ProjectConfiguration(
                project_dir="radialog_binary_qa_ppo_training", logging_dir="logs"
            )
        ),
        remove_unused_columns=False,
        kl_penalty="kl",
        # optimize_device_cache=True,
        init_kl_coef=0.05,
    )

    generation_kwargs_ppo = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "temperature": 1.0,  # DONT BE CREATIVE
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.pad_token_id,  # most decoder models don't have a padding token - use EOS token instead (for this tokenizer it was already set to eos_token_id)
        "max_new_tokens": 50,  # let's not be chatty, we need a few tokens to generate confidence but also let us not limit the response too much
        "eos_token_id": tokenizer.eos_token_id,  # (instead of ppo_terminators list)
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

        best_reward = -100
        best_reward_epoch = -1
        rewards_until_checkpoint = []

        for step, batch in enumerate(dataloader_train):

            batch: dataset.MimicCxrLlavaModelInputBatchDict = batch

            ######### 5.1 - 5.8 Perform a training step #########
            rewards, batch_report = radialog_binary_qa_ppo_training_step(
                model=model,
                device=device,
                tokenizer=tokenizer,
                generation_kwargs_ppo=generation_kwargs_ppo,
                ppo_trainer=ppo_trainer,
                batch=batch,
            )
            rewards_until_checkpoint += [r.item() for r in rewards]

            ######### 5.9 Checkpoint the model if checkpoint step is reached #########
            if (step + 1) % STEPS_UNTIL_CHECKPOINT == 0:

                avg_reward = np.mean(rewards_until_checkpoint)

                print(f"Arrived at checkpoint {step + 1}. Average reward: {avg_reward}")
                print("Saving the model checkpoint...")
                ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned"))

                print(f"Running evaluation at checkpoint {step + 1}")
                model.eval()
                eval_batch_rewards_list = []
                for _ in range(NUM_BATCHES_TO_EVALUATE):
                    try:
                        eval_batch = next(eval_batch_iterator)
                    except StopIteration:
                        eval_batch_iterator = iter(dataloader_eval)
                        eval_batch = next(eval_batch_iterator)

                    eval_batch_rewards = radialog_binary_qa_ppo_evaluation_step(
                        model, eval_batch
                    )  # TODO implement
                    eval_batch_rewards_list.append(eval_batch_rewards)

                # mean_reward, std_reward = calculate_mean_and_std_rewards(eval_batch_rewards_list) # TODO implement

                # if log_with == "wandb":
                #     wandb.log({"mean_reward_evaluation": mean_reward})
                #     wandb.log({"std_reward_evaluation": std_reward})
                #     wandb.log({"exploration_prob": chance_to_change_confidence})

                # Save the best performing model
                mean_reward = avg_reward
                if mean_reward > best_reward:
                    ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned_best"))
                    best_reward = mean_reward
                    best_reward_epoch = epoch

    # print("Finished Training!")
    # print(f"Best avg reward {best_reward} in epoch {best_reward_epoch}")
    return


# get the single value from the single dim tensor
def get_value_from_tensor(tensor):
    return tensor.item()
