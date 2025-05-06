# %% Set script for interactive development and import modules
from RewardingVisualDoubt import infrastructure

infrastructure.make_ipython_reactive_to_changing_codebase()
infrastructure.supress_known_warnings()

import dataclasses
import functools
import os
import pathlib as path
import typing as t

import accelerate
import numpy as np
import torch
import transformers
import trl
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria

from RewardingVisualDoubt import dataset, prompter, response, reward, shared
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


def radialog_binary_qa_ppo_training_step(
    model: trl.AutoModelForCausalLMWithValueHead,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_kwargs_ppo: dict,
    ppo_trainer: trl.PPOTrainer,
    batch: dataset.MimicCxrLlavaModelInputBatchDict,
):

    ######### 5.1 Unpack the batch #########
    batch: dataset.MimicCxrLlavaModelInputBatchDict = batch
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    attention_mask = batch["batch_attention_mask"].to(device)
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)

    ######### 5.2 Generate the binary q&a answer and remove trailing padding tokens #########
    model.eval()
    prompt_and_generated_answers_ids = model.generate(
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        do_sample=False,
        use_cache=True,
        max_new_tokens=300,
        stopping_criteria=[stopping_criteria],
        pad_token_id=tokenizer.pad_token_id,
    )

    prompt_and_generated_answers_ids = training.remove_trailing_padding_from_prediction(
        prompt_and_generated_answers_ids, tokenizer.pad_token_id
    )

    ######### 5.3 Append confidence request to the generated answers #########
    prompt_and_generated_answers_with_confidence_requests_ids = []
    for item in prompt_and_generated_answers_ids:
        confidence_request_input_ids = (
            tokenizer(
                prompter.build_post_generation_user_confidence_request(),
                return_tensors="pt",
            )
            .input_ids.to(device)
            .squeeze(0)
        )[
            1:
        ]  # drop start of sequence token
        prompt_and_generated_answers_with_confidence_requests_ids.append(
            torch.cat((item, confidence_request_input_ids), 0)
        )

    ######### 5.4 Generate the confidences #########
    model.train()
    generated_confidences_ids = ppo_trainer.generate(
        prompt_and_generated_answers_with_confidence_requests_ids,  # ppo_trainer.generate() method admits list of tensors, handles padding and batching itself
        images=images,
        return_prompt=False,
        **generation_kwargs_ppo,
    )

    ######### 5.5 Arrange all generations into useful variables #########
    complete_conversation_ids = [
        torch.cat((p, c), 0)
        for p, c in zip(
            prompt_and_generated_answers_with_confidence_requests_ids,
            generated_confidences_ids,
        )
    ]
    generated_answer_only_ids = [
        prompt_and_generated_answers_ids[i][len(input_ids[i]) :] for i in range(len(input_ids))
    ]
    prompt_and_generated_answers_with_confidence_requests_ids = (
        training.replace_image_token_with_another_token_for_list_of_tensors(
            prompt_and_generated_answers_with_confidence_requests_ids
        )
    )
    generated_answers_texts = tokenizer.batch_decode(
        generated_answer_only_ids,
        skip_special_tokens=True,
    )
    generated_confidences_texts = tokenizer.batch_decode(
        generated_confidences_ids,
        skip_special_tokens=True,
    )

    ######### 5.6 Parse the responses and compute the rewards #########
    generated_answer_labels = response.parse_binary_labels(generated_answers_texts)
    generated_confidence_values = response.parse_confidences(generated_confidences_texts)

    rewards = [
        reward.generated_answer_and_confidence_to_reward(
            generated_answer_label, generated_confidence_value, ground_truth_label
        )
        for generated_answer_label, generated_confidence_value, ground_truth_label in zip(
            generated_answer_labels, generated_confidence_values, labels.bool().tolist()
        )
    ]

    rewards = t.cast(
        list[torch.FloatTensor],
        [torch.tensor(r).to(device) for r in rewards],
    )

    ######### 5.7 Take a PPO optimization step #########
    stats = ppo_trainer.step(
        queries=t.cast(
            list[torch.LongTensor],
            prompt_and_generated_answers_with_confidence_requests_ids,
        ),
        responses=t.cast(list[torch.LongTensor], generated_answer_only_ids),
        scores=rewards,
    )

    ######### 5.8 Create a report and log the training stats #########
    batch_report = {}
    # batch_report["complete_conversation_text"] = tokenizer.batch_decode(
    #     training.replace_image_token_with_another_token_for_list_of_tensors(
    #         complete_conversation_ids
    #     ),
    #     skip_special_tokens=True,
    # )
    batch_report["query"] = tokenizer.batch_decode(
        prompt_and_generated_answers_with_confidence_requests_ids, skip_special_tokens=True
    )
    batch_report["response"] = generated_confidence_values
    # batch_report["generated_answer_labels"] = generated_answer_labels
    batch_report["generated_answers_texts"] = generated_answers_texts
    batch_report["generated_confidences_texts"] = generated_confidences_texts

    ppo_trainer.log_stats(
        stats=stats, batch=batch_report, rewards=rewards
    )  # The game logs will not be logged because the batch does not contain the keys 'query' and 'response'

    return rewards, batch_report


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

    model = vllm.load_pretrained_llava_model_for_ppo_training(
        device_str=device_str,
        precision="4bit",
        radialog_lora_weights_path=vllm.RadialogLoraWeightsPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
    )

    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    padding_tokenizer.padding_side = "left"

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
        trl.PPOTrainer,
        trl.PPOTrainer(
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
