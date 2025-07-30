import peft
import torch
import transformers

from RewardingVisualDoubt import dataset


def radialog_report_generation_sft_training_step(
    model: peft.PeftModelForCausalLM,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: transformers.optimization.LambdaLR,
    gradient_accumulation_steps: int,
    current_step: int,
) -> float:

    input_ids, images, labels, attention_mask, batch_metadata_list = (
        dataset.unpack_report_generation_batch_with_attention_mask_and_metadata_for_sft(
            device=device,
            tokenizer=tokenizer,
            batch=batch,
        )
    )

    outputs = model(
        input_ids=input_ids, images=images, attention_mask=attention_mask, labels=labels
    )

    loss: torch.Tensor = outputs.loss / gradient_accumulation_steps
    loss.backward()

    if (current_step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return loss.item()


def radialog_report_generation_sft_evaluation_step(
    batch: dataset.MimicCxrLlavaModelInputBatchDictForSFT,
    model: peft.PeftModelForCausalLM,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
) -> float:

    with torch.no_grad():
        input_ids, images, labels, attention_mask, batch_metadata_list = (
            dataset.unpack_report_generation_batch_with_attention_mask_and_metadata_for_sft(
                device=device,
                tokenizer=tokenizer,
                batch=batch,
            )
        )

        val_outputs = model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            labels=labels,
        )
        return val_outputs.loss.item()
