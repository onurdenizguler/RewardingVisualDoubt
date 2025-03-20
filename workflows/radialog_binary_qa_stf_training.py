# %% Set script for interactive development and import modules
from RewardingVisualDoubt import infrastructure, training

infrastructure.make_ipython_reactive_to_changing_codebase()
infrastructure.supress_known_warnings()

import os
import pathlib as path

import torch
import transformers
import wandb
from tqdm import tqdm

from RewardingVisualDoubt import dataset, prompter, response, reward, shared
from RewardingVisualDoubt import training as training
from RewardingVisualDoubt import vllm

os.environ["WANDB_API_KEY"] = "da3cb086bbc110c16cbc5ba4c284a19b0b461710"

######################################## 1. Load the model and tokenizer ########################################

device_str = (
    shared.torch_devices.cuda.value if torch.cuda.is_available() else shared.torch_devices.cpu.value
)
device = torch.device(device_str)

model = vllm.load_pretrained_llava_model_for_sft_training(device_str=device_str, precision="4bit")

tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)
padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)
padding_tokenizer.padding_side = "left"
padding_tokenizer.pad_token = padding_tokenizer.bos_token


######################################## 2. Load the datasets and the dataloaders ########################################

DEFAULT_BATCH_SIZE = 4
DEFAULT_OUTPUT_DIR = path.Path("output")


print("Loading the datasets and the dataloaders...")
dataset_train = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
    split=dataset.DatasetSplit.TRAIN,
    tokenizer=tokenizer,
    prompter=prompter.build_binary_qa_prompt_with_response_and_confidence_for_sft,
)
dataset_eval = dataset.get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
    split=dataset.DatasetSplit.VALIDATION,
    tokenizer=tokenizer,
    prompter=prompter.build_binary_qa_prompt_with_response_and_confidence_for_sft,
)

padding_tokenizer.pad_token = padding_tokenizer.bos_token  # TODO how about this?

dataloader_train = dataset.get_mimic_cxr_llava_model_input_dataloader_for_sft(
    dataset=dataset_train,
    batch_size=DEFAULT_BATCH_SIZE,
    padding_tokenizer=padding_tokenizer,
    num_workers=8,  # Let Torch decide.
    simplified_batch=False,
)

dataloader_eval = dataset.get_mimic_cxr_llava_model_input_dataloader_for_sft(
    dataset=dataset_eval,
    batch_size=DEFAULT_BATCH_SIZE,
    padding_tokenizer=padding_tokenizer,
    num_workers=8,  # Let Torch decide.
    simplified_batch=False,
)

eval_batch_iterator = iter(dataloader_eval)


######################################## 3. Train the model ########################################

# A BATCH OF 4 SAMPLES TAKES 5sec to take a training step
NUM_EPOCHS = 1
CHECKPOINT_DIR = "training_checkpoints"
GRAD_ACCUM_STEPS = 4  # every 16 sample, take a step
LOGGING_STEPS = GRAD_ACCUM_STEPS * 2  # at every second gradient step
STEPS_UNTIL_CHECKPOINT = 60  # equivalent roughly to Every 30 minutes
NUM_BATCHES_TO_EVALUATE = 60
LR = 5e-5

# ---- Optimizer & Scheduler ----
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = transformers.get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train) * NUM_EPOCHS,
)

model.train()
best_val_loss = float("inf")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
wandb.init(project="radialog-confidence-score-sft")


for epoch in range(NUM_EPOCHS):
    loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for step, batch in enumerate(loop):
        ######### 3.1 Unpack the batch #########
        batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT = batch
        batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
        batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
            batch_llava_model_input_dict, device
        )
        input_ids, images = (
            batch_llava_model_input_dict["text_prompt_input_ids"],
            batch_llava_model_input_dict["images"],
        )
        labels = batch["batch_expected_output_ids"].to(device)
        attention_mask = batch["batch_attention_mask"].to(device)

        is_input_verified = torch.all((input_ids == labels) | (labels == -100)).item()

        if not is_input_verified:
            print("(input_ids == labels) | (labels == -100) verification failed.")

        ######### 3.2 Forward pass and backward pass #########
        outputs = model(
            input_ids=input_ids, images=images, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # ---- Logging ----
        if step % LOGGING_STEPS == 0:
            wandb.log({"train_loss": loss.item() * GRAD_ACCUM_STEPS})

        ######### 3.3 Validation #########

        if (step + 1) % STEPS_UNTIL_CHECKPOINT == 0:
            print(f"Arrived at checkpoint {step + 1}. Starting validation.")
            model.eval()
            val_losses = []
            val_progress = tqdm(
                range(NUM_BATCHES_TO_EVALUATE), desc="Validation", position=1, leave=False
            )
            for _ in val_progress:
                try:
                    eval_batch = next(eval_batch_iterator)
                except StopIteration:
                    eval_batch_iterator = iter(dataloader_eval)
                    eval_batch = next(eval_batch_iterator)

                with torch.no_grad():
                    batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT = eval_batch
                    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
                    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
                        batch_llava_model_input_dict, device
                    )
                    input_ids, images = (
                        batch_llava_model_input_dict["text_prompt_input_ids"],
                        batch_llava_model_input_dict["images"],
                    )
                    labels = batch["batch_expected_output_ids"].to(device)
                    attention_mask = batch["batch_attention_mask"].to(device)

                    val_outputs = model(
                        input_ids=input_ids,
                        images=images,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    val_loss = val_outputs.loss.item()
                    val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"val_loss": avg_val_loss})

            # ---- Checkpoint Saving ----
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, f"best_model_epoch{epoch}_step{step}.pth"
                )
                model.save_pretrained(checkpoint_path)
                print(f"🔥 Saved best LoRA adapter at {checkpoint_path}")

            model.train()  # Resume training

wandb.finish()
checkpoint_path = os.path.join(CHECKPOINT_DIR, f"llava_lora_final.pth")
model.save_pretrained(checkpoint_path)
