# %% Set script for interactive development and import modules
from IPython.core.getipython import get_ipython

ipython_client = get_ipython()
if ipython_client:
    ipython_client.run_line_magic(magic_name="load_ext", line="autoreload")
    ipython_client.run_line_magic(magic_name="autoreload", line="2")

import pathlib as path

import torch
from torch.utils.data import DataLoader
# from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from RewardingVisualDoubt import dataset, mimic_cxr, prompter, shared, vllm

DEFAULT_BATCH_SIZE = 8
DEFAULT_OUTPUT_DIR = path.Path("output")

# %% load tokenizer
tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)


# %%
def train(out_dir: path.Path = DEFAULT_OUTPUT_DIR, batch_size: int = DEFAULT_BATCH_SIZE):
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

    device = (
        torch.device(shared.torch_devices.cuda.value)
        if torch.cuda.is_available()
        else torch.device(shared.torch_devices.cpu.value)
    )

    model = vllm.load_pretrained_llava_model()
    model = vllm.add_value_head_to_LlavaLlamaForCausalLM_model(model)
    model.to(device)

    model_ref = vllm.load_pretrained_llava_model()
    model_ref = vllm.add_value_head_to_LlavaLlamaForCausalLM_model(model)
    model_ref.to(device)

    # tokenizer = load_tokenizer(tokenizer_dir) # TODO do i need a tokenizer dir?
    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )
    # tokenizer.pad_token = tokenizer.eos_token # TODO do i need to set this? i'm doing it in collate fn

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

    config = PPOConfig(
        learning_rate=lr,
        task_name="gpt",
        batch_size=batch_size,
        mini_batch_size=batch_size,
        log_with="wandb",
        project_kwargs={"logging_dir": out_dir},
        # cliprange=0.1,
        # cliprange_value=0.1
        # adap_kl_ctrl=False,
        init_kl_coef=0.0001,
        target=1,
        # horizon=32,
        # gamma=0.8
    )

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=model_ref,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        # optimizer=optimizer
    )

    optimizer_sst = AdamW(model.parameters(), lr=5e-5)

    generation_kwargs = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": model.config.eos_token_id,  # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32,
    }

    generation_kwargs_first_part = {
        "max_new_tokens": 1,
    }

    best_reward = -100
    best_reward_epoch = -1

    for epoch in range(epochs):
        model.train()

        rewards_epoch = []
        sst_losses = []
        for idx, batch in enumerate(ppo_trainer.dataloader):
            inputs = batch["inputs"]
            # inputs = [i for i in inputs]
            label_tokens = tokenizer(
                batch["label"], return_tensors="pt", padding=True
            ).input_ids.to(device)

            supervised_input = torch.cat((inputs, label_tokens[:, 1].reshape(-1, 1)), dim=-1)

            encoding = {}
            encoding["input_ids"] = supervised_input
            encoding["token_type_ids"] = torch.zeros_like(supervised_input)
            encoding["attention_mask"] = torch.ones_like(supervised_input)

            encoding = {k: v.to(device) for k, v in encoding.items()}

            outputs = model(**encoding, labels=encoding["input_ids"])
            loss = outputs[1]
            loss.backward()
            optimizer_sst.step()

            sst_losses.append(loss.item())

            # generate the prediction without confidence
            model.eval()
            prediction = model.generate(inputs, **generation_kwargs_first_part)
            prediction = [p for p in prediction]

            # generate confidence
            model.train()
            response_tensors = ppo_trainer.generate(
                prediction, return_prompt=False, **generation_kwargs
            )

            # Create prediction + confidence output
            total_tensor = [
                torch.cat((p[-1].unsqueeze(0), c), 0) for p, c in zip(prediction, response_tensors)
            ]
            outputs = tokenizer.batch_decode(total_tensor, skip_special_tokens=True)

            # Compute rewards
            rewards = [reward_function(o, l) for o, l in zip(outputs, batch["label"])]
            rewards_epoch += rewards
            rewards = [torch.tensor(r).to(device) for r in rewards]

            # Create log data
            expected_prediction = [get_expected_label(o) for o in batch["num_ones"]]
            batch["query"] = expected_prediction
            batch["response"] = outputs

            try:
                # stats = ppo_trainer.step(inputs, response_tensors, rewards)
                stats = ppo_trainer.step(prediction, response_tensors, rewards)
            except IndexError:
                print(f"INDEX ERROR detected and ignored {idx}")

                # for i, o in zip(batch["sample"], response_tensors):
                #     print(f"{i} : {o}")

            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])

        avg_reward = np.mean(rewards_epoch)

        mean_sst_loss = np.mean(np.array(sst_losses))
        print(f"SST Losses: {mean_sst_loss}")
        wandb.log({"mean_sst_train": mean_sst_loss})

        # scheduler.step(avg_reward)
        # scheduler.step()
        print(f"Finished epoch {epoch}. Average reward: {avg_reward}")
        ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned"))
        mean_reward = evaluate_model(model, dataset_eval, tokenizer, device)
        wandb.log({"mean_reward_evaluation": mean_reward})
        print(f"Evaluation reward: {mean_reward}")

        if mean_reward > best_reward:
            ppo_trainer.save_pretrained(os.path.join(out_dir, "model_finetuned_best"))
            best_reward = mean_reward
            best_reward_epoch = epoch

    print("Finished Training!")
    print(f"Best avg reward {best_reward} in epoch {best_reward_epoch}")
    return
