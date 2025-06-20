import typing as t
import torch
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from tqdm import tqdm
from RewardingVisualDoubt import (
    dataset,
    prompter,
    response,
    shared,
    training,
    vllm,
)
import json
import functools
from pathlib import Path

# RESULTS_OUTPUT_PATH = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/data/radialog_original_binary_qa_results_train.jsonl"
RESULTS_OUTPUT_PATH = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/data/radialog_confidence_ppo_2025-05-07_binary_qa_results_train.jsonl"


def writeback_results(
    input_prompts: t.List[str],
    answers: t.List[str],
    gt_labels: list[bool],
    predictions: list[bool | None],
    is_answer_correct: list[bool],
):

    with open(RESULTS_OUTPUT_PATH, "a") as f:

        for i in range(len(input_prompts)):
            result = {
                "gt_label": gt_labels[i],
                "prediction": predictions[i],
                "is_answer_correct": is_answer_correct[i],
                "input_prompt": input_prompts[i],
                "answer": answers[i],
            }
            f.write(json.dumps(result) + "\n")


STOP_STR = prompter.Seperator.END_OF_SEQUENCE_SEPERATOR.value

BATCH_SIZE = 16

device_str = (
    shared.torch_devices.cuda.value if torch.cuda.is_available() else shared.torch_devices.cpu.value
)
device = torch.device(device_str)


# Load the original RaDialog model

# model = vllm.load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters(
#     device_str=device_str,
#     llava_model_path=str(
#         vllm._get_hf_model_path(repo_id=vllm.RADIALOG_BASELINE_LORA_ADAPTER_REPO_ID)
#     ),
#     precision="16bit",
#     adapter_path=vllm.RadialogLoraWeightsPath.ORIGINAL.value,
# )

# Load the STT RaDialog model

model = vllm.load_pretrained_llava_model_for_ppo_training_with_fresh_lora_adapters(
    device_str=device_str,
    llava_model_path=vllm.RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value,
    precision="4bit",
    adapter_path="/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radialog_binary_qa_ppo_training/2025-05-07/checkpoint-1100/adapter_model.bin",
)
# Load the STT RaDialog model without fresh lora adapters
# model = vllm.load_baseline_llava_model_with_vision_modules(
#     model_path=Path(vllm.RadialogMergedLlavaModelPath.BINARY_QA_WITH_CONFIDENCE_SFT.value),
#     device=device_str,
#     precision="16bit",
# )

model.config.padding_side = "left"
model.config.tokenizer_padding_side = "left"

tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)
padding_tokenizer = vllm.load_pretrained_llava_tokenizer_with_image_support(
    model_base=vllm.LLAVA_BASE_MODEL_NAME
)
padding_tokenizer.padding_side = "left"

tokenizer.padding_side = "left"


print("Loading the datasets and the dataloaders...")
# prompter_ = prompter.build_binary_qa_instruction_from_disease_under_study # If using the original RaDialog model, use this prompter.
prompter_ = functools.partial(
    prompter.build_binary_qa_prompt_with_response_and_confidence_for_sft, is_for_inference=True
)  # if using the RaDialog model after SFT for confidence generation, use this prompter.

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
    batch_size=BATCH_SIZE,
    padding_tokenizer=padding_tokenizer,
    num_workers=8,
)

dataloader_eval = dataset.get_mimic_cxr_llava_model_input_dataloader(
    dataset=dataset_eval,
    batch_size=BATCH_SIZE,
    padding_tokenizer=padding_tokenizer,
    num_workers=8,
)

eval_batch_iterator = iter(dataloader_eval)
iterator_train = iter(dataloader_train)

model.eval()

for step in tqdm(
    range(len(dataloader_train)),
    desc="Evaulating on the traing set",
):
    batch: dataset.MimicCxrLlavaModelInputBatchDict = next(iterator_train)
    batch_llava_model_input_dict = batch["batch_llava_model_input_dict"]
    batch_llava_model_input_dict = dataset.move_llava_model_input_dict_to_device(
        batch_llava_model_input_dict, device
    )
    input_ids, images = (
        batch_llava_model_input_dict["text_prompt_input_ids"],
        batch_llava_model_input_dict["images"],
    )
    labels = t.cast(torch.Tensor, batch["batch_labels"]).to(device)
    stopping_criteria = KeywordsStoppingCriteria([STOP_STR], tokenizer, input_ids)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            images=images,
            do_sample=True,
            use_cache=True,
            max_new_tokens=300,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_ids = training.replace_image_token_with_another_token(generated_ids)

    generated_texts = tokenizer.batch_decode(generated_ids)
    input_prompts = [text.split("ASSISTANT:")[0] for text in generated_texts]
    generated_texts = [text.split("ASSISTANT:")[1] for text in generated_texts]
    generated_answer_labels = response.parse_binary_labels(generated_texts)

    gt_labels = labels.bool().tolist()
    is_answer_correct = [
        (gt_label == predicted_label) and (gt_label is not None)
        for gt_label, predicted_label in zip(generated_answer_labels, labels.bool().tolist())
    ]
    writeback_results(
        input_prompts=input_prompts,
        answers=generated_texts,
        gt_labels=gt_labels,
        predictions=generated_answer_labels,
        is_answer_correct=is_answer_correct,
    )
