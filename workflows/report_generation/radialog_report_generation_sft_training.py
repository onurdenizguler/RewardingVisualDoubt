from RewardingVisualDoubt import infrastructure

infrastructure.make_ipython_reactive_to_changing_codebase()


import functools
import itertools
import os
import pathlib as path

import torch
import transformers
import wandb
from tqdm import tqdm

from RewardingVisualDoubt import dataset, prompter, shared, training, vllm

SELECTED_STEPS_UNTIL_CHECKPOINT = 50
SELECTED_LEARNING_RATE = 5e-5

SELECTED_BATCH_SIZE = 12
SELECTED_NUM_BATCHES_TO_EVALUATE = 7  # Set to zero to evaulate on the whole validation set
SELECTED_NUM_GRADIENT_ACCUMULATION_STEPS = 3
SELECTED_NUM_EPOCHS = 1
SELECTED_NUM_TRAINING_BATCHES_TO_SKIP = 0  # Do not set this to be much higher than a 100
SELECTED_OUTPUT_DIR = path.Path("/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models")
SELECTED_FINETUNING_NAME = "radialog_report_generation_sft_training"


def train(
    batch_size: int = SELECTED_BATCH_SIZE,
    n_training_batches_to_skip: int = SELECTED_NUM_TRAINING_BATCHES_TO_SKIP,
    learning_rate: float = SELECTED_LEARNING_RATE,
    num_epochs: int = SELECTED_NUM_EPOCHS,
    steps_until_checkpoint: int = SELECTED_STEPS_UNTIL_CHECKPOINT,
    gradient_accumulation_steps: int = SELECTED_NUM_GRADIENT_ACCUMULATION_STEPS,
    num_batches_to_evaluate: int = SELECTED_NUM_BATCHES_TO_EVALUATE,
    name_of_fine_tuning: str = SELECTED_FINETUNING_NAME,
    out_dir: path.Path = SELECTED_OUTPUT_DIR,
):
    ######################################## 1. Load the model and tokenizer ########################################
    device, device_str = shared.get_device_and_device_str()
    model = vllm.load_pretrained_llava_model_for_sft_training(
        device_str=device_str, precision="4bit"
    )
    tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
        model_base=vllm.LLAVA_BASE_MODEL_NAME
    )

    ######################################## 2. Load the datasets and the dataloaders ########################################

    print("Loading the datasets and the dataloaders...")
    selected_prompter_fn = (
        prompter.build_report_generation_prompt_with_response_and_confidence_for_sft
    )
    prompter_ = functools.partial(
        selected_prompter_fn,
        is_for_inference=False,
    )

    dataset_train = (
        dataset.get_report_generation_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
            split=dataset.DatasetSplit.TRAIN,
            tokenizer=tokenizer,
            prompter=prompter_,
        )
    )
    dataset_eval = (
        dataset.get_report_generation_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
            split=dataset.DatasetSplit.VALIDATION,
            tokenizer=tokenizer,
            prompter=prompter_,
        )
    )

    dataloader_train = dataset.get_mimic_cxr_llava_model_input_dataloader_for_sft(
        dataset=dataset_train,
        batch_size=batch_size,
        padding_tokenizer=vllm.load_pretrained_llava_tokenizer_with_image_support(),
        num_workers=8,
    )

    dataloader_eval = dataset.get_mimic_cxr_llava_model_input_dataloader_for_sft(
        dataset=dataset_eval,
        batch_size=2 * batch_size,
        padding_tokenizer=vllm.load_pretrained_llava_tokenizer_with_image_support(),
        num_workers=8,
    )

    eval_batch_iterator = iter(dataloader_eval)
    iterator_train = itertools.islice(iter(dataloader_train), n_training_batches_to_skip, None)

    ######################################## 3. Train the model ########################################

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler: transformers.optimization.LambdaLR = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * num_epochs,
    )

    best_val_loss = float("inf")
    wandb.init(
        project="report-gen-confidence-gen-sft",
        **training.get_wandb_parameters_for_sft(
            learning_rate=learning_rate,
            prompter=selected_prompter_fn,
            num_epochs=num_epochs,
            steps_until_checkpoint=steps_until_checkpoint,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_size=batch_size,
            num_batches_to_evaluate=num_batches_to_evaluate,
            n_training_batches_to_skip=n_training_batches_to_skip,
        ),
    )
    logging_steps = gradient_accumulation_steps

    model.train()
    for epoch in range(num_epochs):
        for step in tqdm(
            range(len(dataloader_train) - n_training_batches_to_skip),
            desc=f"Taking training steps... Epoch {epoch+1}/{num_epochs}",
        ):
            batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT = next(iterator_train)
            loss = training.radialog_report_generation_sft_training_step(
                model=model,
                device=device,
                tokenizer=tokenizer,
                batch=batch,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                current_step=step,
            )

            if step % logging_steps == 0:
                wandb.log(
                    {"train_loss": loss * gradient_accumulation_steps},
                    step=step + epoch * len(dataloader_train),
                )

            if (step + 1) % steps_until_checkpoint == 0:
                print(f"Arrived at checkpoint {step + 1}. Starting validation.")

                model.eval()
                val_losses = []

                for _ in tqdm(
                    range(
                        num_batches_to_evaluate
                        if num_batches_to_evaluate > 0
                        else len(dataloader_eval)
                    ),
                    desc=f"Running evaluation at checkpoint {step + 1}",
                ):
                    try:
                        eval_batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT = next(
                            eval_batch_iterator
                        )
                    except StopIteration:
                        eval_batch_iterator = iter(dataloader_eval)
                        eval_batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT = next(
                            eval_batch_iterator
                        )
                    val_loss = training.radialog_report_generation_sft_evaluation_step(
                        batch=eval_batch,
                        model=model,
                        device=device,
                        tokenizer=tokenizer,
                    )
                    val_losses.append(val_loss)

                avg_val_loss = sum(val_losses) / len(val_losses)
                wandb.log({"val_loss": avg_val_loss}, step=step + epoch * len(dataloader_train))

                # ---- Checkpoint Saving ----
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    training.save_best_eval_lora_adapters_to_dir(
                        model, epoch, step, out_dir, name_of_fine_tuning
                    )
                    print(f"🔥 Saved best eval LoRA adapter.")

                model.train()  # Resume training


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "REDUCTED"
    train()
