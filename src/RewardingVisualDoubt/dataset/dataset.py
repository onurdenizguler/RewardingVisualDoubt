import enum
import inspect
import typing as T
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from utils import create_chest_xray_transform_for_inference, remap_to_uint8

from RewardingVisualDoubt import shared

from . import mimic_cxr, domain, sampling

biovil_image_transformer = create_chest_xray_transform_for_inference(512, center_crop_size=448)
bfloat16_dtype: torch.dtype = torch.bfloat16


# TODO ?Should i always cast images to dtype=torch.bfloat16!!!! (The logic currently is quite convoluted and images get cast only while being moved to device)

BinaryQANumSamplesPerDisease = {
    domain.DatasetSplit.TRAIN: 3000,
    domain.DatasetSplit.VALIDATION: 50,
    domain.DatasetSplit.TEST: 50,
}


######################## MODEL INPUTS AND BATCHES ########################


#### Datapoint level dataclasses


class LlavaModelInputDict(T.TypedDict):
    text_prompt_input_ids: torch.Tensor
    images: torch.Tensor  # Image tensor of dtype 16


@dataclass
class LlavaModelInput:
    text_prompt_input_ids: torch.Tensor  # Tokenized text
    images: torch.Tensor  # Image tensor

    def as_dict(self) -> LlavaModelInputDict:
        return T.cast(LlavaModelInputDict, asdict(self))


class PromptedMimicCxrLlavaModelInputDatapointDict(T.TypedDict):
    llava_model_input_dict: LlavaModelInputDict
    label: bool | None
    prompt: str
    mimic_cxr_datapoint_metadata: mimic_cxr.MimicCxrDatapoint


class BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict(T.TypedDict):
    llava_model_input_dict: LlavaModelInputDict
    label: bool | None
    prompt: str
    mimic_cxr_datapoint_metadata: mimic_cxr.MimicCxrBinaryQADatapoint


class BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT(
    BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict
):
    expected_output_ids: torch.Tensor


#### Batch level dataclasses


class MimicCxrLlavaModelInputBatchDict(T.TypedDict):
    batch_llava_model_input_dict: LlavaModelInputDict
    batch_attention_mask: torch.Tensor
    batch_labels: torch.Tensor | list[bool] | list[None]
    batch_prompts: list[str]
    batch_mimic_cxr_datapoint_metadata: list[mimic_cxr.MimicCxrBinaryQADatapoint]


class MimicCxrLlavaModelInputBatchDictForSFT(MimicCxrLlavaModelInputBatchDict):
    batch_expected_output_ids: torch.Tensor


######################## DATA INTERFACING ########################


def _load_image(
    image_path: Path,
) -> Image.Image:
    """Load a single image"""
    image = Image.open(image_path)
    image = remap_to_uint8(np.array(image))
    image = Image.fromarray(image).convert("L")
    return image


######################## TRAINING/INFERENCE TIME DEVICE HELPERS ########################


def move_llava_model_input_to_device(
    model_input: LlavaModelInput, device: torch.device
) -> LlavaModelInput:
    model_input.images = model_input.images.to(device, dtype=torch.bfloat16)
    model_input.text_prompt_input_ids = model_input.text_prompt_input_ids.to(device)
    return model_input


def move_llava_model_input_dict_to_device(
    model_input_dict: LlavaModelInputDict, device: torch.device
) -> LlavaModelInputDict:
    model_input_dict["images"] = model_input_dict["images"].to(device, dtype=torch.bfloat16)
    model_input_dict["text_prompt_input_ids"] = model_input_dict["text_prompt_input_ids"].to(device)
    return model_input_dict


######################## DATA PREPERATION ########################


def _create_prompt(
    prompter: T.Callable[[str], str],
    disease_labels: T.List[mimic_cxr.ChexpertFinding],
) -> str:
    findings_string = mimic_cxr.convert_binary_chexpert_findings_to_string(disease_labels)
    text_input = prompter(findings_string)
    return text_input


def _create_prompt_for_binary_qa(
    binary_qa_prompter: T.Callable[..., str],
    datapoint: mimic_cxr.MimicCxrBinaryQADatapoint,
    **kwargs
) -> str:
    prompter_params = inspect.signature(binary_qa_prompter).parameters
    possible_inputs = {
        "chexpert_finding_str": datapoint.disease.value,
        "occurrence_of_disease": datapoint.label == mimic_cxr.ChexpertLabel.POSITIVE,
        "possible_confidences": shared.POSSIBLE_CONFIDENCES,
        **kwargs,
    }
    filtered_args = {k: v for k, v in possible_inputs.items() if k in prompter_params}

    return binary_qa_prompter(**filtered_args)


def _create_llava_model_input_from_mimic_cxr_datapoint(
    datapoint: mimic_cxr.MimicCxrDatapoint | mimic_cxr.MimicCxrBinaryQADatapoint,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: T.Callable[[str], str],
    image_transform: T.Callable[[Image.Image], torch.Tensor] | None = biovil_image_transformer,
) -> LlavaModelInput:
    """Convert a study and prompt into model input format"""

    # Handle image
    image = _load_image(datapoint.img_path)
    if image_transform:
        image = image_transform(image)  # .unsqueeze(0)
    else:
        image = torch.tensor(np.array(image))  # .unsqueeze(0)

    if isinstance(datapoint, mimic_cxr.MimicCxrBinaryQADatapoint):
        text_input = _create_prompt_for_binary_qa(prompter, datapoint)
    elif isinstance(datapoint, mimic_cxr.MimicCxrDatapoint):
        text_input = _create_prompt(prompter, datapoint.disease_labels)
    else:
        exception_message = (
            "datapoint must be an instance of either MimicCxrDatapoint or MimicCxrBinaryQADatapoint"
        )
        raise ValueError(exception_message)

    # get findings from the datapoint
    # Handle text input
    input_ids = tokenizer_image_token(
        text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )  # .unsqueeze(0)

    return LlavaModelInput(
        text_prompt_input_ids=input_ids,
        images=image,
    )


def _create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(
    row: pd.Series,
) -> mimic_cxr.MimicCxrDatapoint:
    mimic_cxr_datapoint = mimic_cxr.MimicCxrDatapoint(
        subject_id=row["subject_id"],
        study_id=row["study_id"],
        disease_labels=[
            mimic_cxr.ChexpertFinding(finding)
            for finding, value in row.iloc[2:16].items()
            if value == 1
        ],
        img_path=row["img_path"],
    )
    return mimic_cxr_datapoint


def _create_mimic_cxr_binary_qa_datapoint_from_mimic_cxr_dataset_df_row(
    row: pd.Series,
) -> mimic_cxr.MimicCxrBinaryQADatapoint:
    mimic_cxr_datapoint = mimic_cxr.MimicCxrBinaryQADatapoint(
        subject_id=row["subject_id"],
        study_id=row["study_id"],
        img_path=row["img_path"],
        disease=mimic_cxr.ChexpertFinding(row["disease"]),
        label=mimic_cxr.ChexpertLabel(row["label"]),
    )
    return mimic_cxr_datapoint


def _replace_confidence_score_tokens_with_value(
    input_ids: torch.Tensor,
) -> torch.Tensor:
    reversed_ids = input_ids.flip(0)
    COLON_TOKEN_ID = 1115

    try:
        # Find index of the first COLON_TOKEN_ID occuring in the reversed ids
        colon_idx = (reversed_ids == COLON_TOKEN_ID).nonzero(as_tuple=True)[0]
        if colon_idx.numel() > 0:
            colon_idx = colon_idx[0]

            # Replace tokens from index 2 to (colon_idx - 1), ensuring valid range
            if colon_idx > 2:
                reversed_ids[2 : colon_idx - 1] = -100  # IGNORE_INDEX VALUE

        # Reverse back to original order
        output_ids = reversed_ids.flip(0)
    except Exception as e:
        print("Error in _replace_confidence_score_tokens_with_value", e)
        output_ids = input_ids
    return output_ids


######################## TORCH DATASETS ########################


@dataclass
class PromptedMimicCxrLlavaModelInputDataset(TorchDataset):
    """
    mimic_cxr_df: pandas DataFrame containing MIMIC-CXR metadata (including `img_path`, `subject_id`, `study_id`, `split`).
    """

    mimic_cxr_df: pd.DataFrame
    tokenizer: transformers.PreTrainedTokenizer
    prompter: T.Callable[[str], str]
    image_transform: T.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer
    split: domain.DatasetSplit = domain.DatasetSplit.TRAIN
    supports_label: bool = False

    def __post_init__(self):
        self.df = self.mimic_cxr_df[self.mimic_cxr_df["split"] == self.split.value]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> PromptedMimicCxrLlavaModelInputDatapointDict:

        row = self.df.iloc[idx]
        mimic_cxr_datapoint = _create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(row)
        prompt = _create_prompt(self.prompter, mimic_cxr_datapoint.disease_labels)
        llava_model_input = _create_llava_model_input_from_mimic_cxr_datapoint(
            mimic_cxr_datapoint, self.tokenizer, self.prompter, self.image_transform
        )

        label = True if self.supports_label else None  # TODO fetch label from actual gt

        return PromptedMimicCxrLlavaModelInputDatapointDict(
            llava_model_input_dict=llava_model_input.as_dict(),
            label=label,
            prompt=prompt,
            mimic_cxr_datapoint_metadata=mimic_cxr_datapoint,
        )


@dataclass
class BinaryQAPromptedMimicCxrLlavaModelInputDataset(TorchDataset):
    """
    mimic_cxr_df: pandas DataFrame containing MIMIC-CXR metadata (including `img_path`, `subject_id`, `study_id`, `split`).
    This dataset is used for training a model to answer binary questions about the presence of a specific disease in a chest x-ray image.
    Each datapoint in this dataset contains a chest x-ray image, a prompt asking about one of the 14 possible CheXpert findings, and a label indicating the presence or absence of the finding in the image.
    """

    balanced_binary_qa_mimic_cxr_df: pd.DataFrame
    tokenizer: transformers.PreTrainedTokenizer
    prompter: T.Callable[[str], str]
    image_transform: T.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer
    split: domain.DatasetSplit = domain.DatasetSplit.TRAIN
    supports_label: bool = False

    def __post_init__(self):
        self.df = self.balanced_binary_qa_mimic_cxr_df[
            self.balanced_binary_qa_mimic_cxr_df["split"] == self.split.value
        ]

    def __len__(self):
        return len(self.df)

    def _prepare_binary_qa_prompted_mimic_cxr_llave_model_input_datapoint_dict(
        self, idx: int
    ) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict:
        row = self.df.iloc[idx]
        mimic_cxr_binary_qa_datapoint = (
            _create_mimic_cxr_binary_qa_datapoint_from_mimic_cxr_dataset_df_row(row)
        )
        prompt = _create_prompt_for_binary_qa(self.prompter, mimic_cxr_binary_qa_datapoint)
        llava_model_input = _create_llava_model_input_from_mimic_cxr_datapoint(
            mimic_cxr_binary_qa_datapoint, self.tokenizer, self.prompter, self.image_transform
        )
        label = mimic_cxr_binary_qa_datapoint.label

        return BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict(
            llava_model_input_dict=llava_model_input.as_dict(),
            label=bool(label.value),
            prompt=prompt,
            mimic_cxr_datapoint_metadata=mimic_cxr_binary_qa_datapoint,
        )

    def __getitem__(self, idx: int) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict:
        return self._prepare_binary_qa_prompted_mimic_cxr_llave_model_input_datapoint_dict(idx)


@dataclass
class BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT(
    BinaryQAPromptedMimicCxrLlavaModelInputDataset
):
    """
    This dataset is used for training a model to answer binary questions about the presence of a specific disease in a chest x-ray image and provide the confidence score right after the answer.
    """

    def __getitem__(self, idx: int) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT:
        datapoint_dict = (
            self._prepare_binary_qa_prompted_mimic_cxr_llave_model_input_datapoint_dict(idx)
        )
        expected_output_ids = _replace_confidence_score_tokens_with_value(
            datapoint_dict["llava_model_input_dict"]["text_prompt_input_ids"],
        )
        return BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT(
            **datapoint_dict, expected_output_ids=expected_output_ids
        )


def get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
    split: domain.DatasetSplit,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: T.Callable[..., str],
) -> BinaryQAPromptedMimicCxrLlavaModelInputDataset:
    return BinaryQAPromptedMimicCxrLlavaModelInputDataset(
        balanced_binary_qa_mimic_cxr_df=sampling.create_balanced_binary_qa_mimic_cxr_dataset_df(
            mimic_cxr.create_mimic_cxr_dataset_df(),
            split=split,
            num_samples_per_disease=BinaryQANumSamplesPerDisease[split],
        ),
        tokenizer=tokenizer,
        prompter=prompter,
        image_transform=biovil_image_transformer,
        split=split,
    )


def get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
    split: domain.DatasetSplit,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: T.Callable[..., str],
) -> BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT:
    return BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT(
        balanced_binary_qa_mimic_cxr_df=sampling.create_balanced_binary_qa_mimic_cxr_dataset_df(
            mimic_cxr.create_mimic_cxr_dataset_df(),
            split=split,
            num_samples_per_disease=BinaryQANumSamplesPerDisease[split],
        ),
        tokenizer=tokenizer,
        prompter=prompter,
        image_transform=biovil_image_transformer,
        split=split,
    )


######################## COLLATE FUNCTIONS FOR DATALOADERS ########################


def prompted_mimic_cxr_llava_model_input_collate_fn(
    batch: list[PromptedMimicCxrLlavaModelInputDatapointDict],
    padding_tokenizer: transformers.PreTrainedTokenizer,
) -> MimicCxrLlavaModelInputBatchDict:
    """
    Custom collate function to pad text sequences and stack images.
    """
    llava_model_inputs, labels, prompts, mimic_cxr_datapoint_metadatas = zip(
        *[
            (
                item["llava_model_input_dict"],
                item["label"],
                item["prompt"],
                item["mimic_cxr_datapoint_metadata"],
            )
            for item in batch
        ]
    )

    text_inputs = T.cast(
        list[transformers.tokenization_utils_base.EncodedInput],
        [
            torch.as_tensor(sample["text_prompt_input_ids"]).clone().detach()
            for sample in llava_model_inputs
        ],
    )
    images = torch.stack([torch.as_tensor(sample["images"]) for sample in llava_model_inputs])

    padded_text_inputs_dict = padding_tokenizer.pad(
        encoded_inputs={"input_ids": text_inputs},
        padding=True,
        max_length=None,
        return_tensors="pt",
    )
    text_inputs = T.cast(torch.Tensor, padded_text_inputs_dict["input_ids"])
    attention_mask = padded_text_inputs_dict["attention_mask"]

    batch_llava_model_inputs = LlavaModelInputDict(text_prompt_input_ids=text_inputs, images=images)
    batch_labels = (
        torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else list(labels)
    )
    batch_metadata = list(mimic_cxr_datapoint_metadatas)
    batch_prompts = list(prompts)

    return MimicCxrLlavaModelInputBatchDict(
        batch_llava_model_input_dict=batch_llava_model_inputs,
        batch_attention_mask=attention_mask,
        batch_labels=batch_labels,
        batch_prompts=batch_prompts,
        batch_mimic_cxr_datapoint_metadata=batch_metadata,
    )


def prompted_mimic_cxr_llava_model_input_collate_fn_for_sft(
    batch: list[BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT],
    padding_tokenizer: transformers.PreTrainedTokenizer,
) -> MimicCxrLlavaModelInputBatchDictForSFT:

    llava_model_inputs, labels, prompts, mimic_cxr_datapoint_metadatas, expected_output_ids = zip(
        *[
            (
                item["llava_model_input_dict"],
                item["label"],
                item["prompt"],
                item["mimic_cxr_datapoint_metadata"],
                item["expected_output_ids"],
            )
            for item in batch
        ]
    )

    text_inputs = [
        torch.as_tensor(sample["text_prompt_input_ids"]).clone().detach()
        for sample in llava_model_inputs
    ]
    images = torch.stack([torch.as_tensor(sample["images"]) for sample in llava_model_inputs])

    padded_text_inputs_dict = padding_tokenizer.pad(
        encoded_inputs={"input_ids": text_inputs},
        padding=True,
        max_length=None,
        return_tensors="pt",
    )
    text_inputs = padded_text_inputs_dict["input_ids"]
    attention_mask = padded_text_inputs_dict["attention_mask"]

    batch_llava_model_inputs = LlavaModelInputDict(text_prompt_input_ids=text_inputs, images=images)
    batch_labels = (
        torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else list(labels)
    )
    batch_metadata = list(mimic_cxr_datapoint_metadatas)
    batch_prompts = list(prompts)

    padded_expected_ouput_ids = padding_tokenizer.pad(
        encoded_inputs={"input_ids": expected_output_ids},
        padding=True,
        max_length=None,
        return_tensors="pt",
    )
    padded_expected_ouput_ids = padded_expected_ouput_ids["input_ids"]

    return MimicCxrLlavaModelInputBatchDictForSFT(
        batch_llava_model_input_dict=batch_llava_model_inputs,
        batch_attention_mask=attention_mask,
        batch_labels=batch_labels,
        batch_prompts=batch_prompts,
        batch_mimic_cxr_datapoint_metadata=batch_metadata,
        batch_expected_output_ids=padded_expected_ouput_ids,
    )


######################## GET DATALOADERS ########################


def get_mimic_cxr_llava_model_input_dataloader(
    dataset: (
        PromptedMimicCxrLlavaModelInputDataset | BinaryQAPromptedMimicCxrLlavaModelInputDataset
    ),
    batch_size: int,
    padding_tokenizer: transformers.PreTrainedTokenizer,
    num_workers: T.Optional[int] = None,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=lambda x: prompted_mimic_cxr_llava_model_input_collate_fn(x, padding_tokenizer),
        shuffle=False,
        num_workers=num_workers if num_workers else 0,  # TODO let torch decide!
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def get_mimic_cxr_llava_model_input_dataloader_for_sft(
    dataset: BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT,
    batch_size: int,
    padding_tokenizer: transformers.PreTrainedTokenizer,
    num_workers: T.Optional[int] = None,
) -> DataLoader:

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=lambda x: prompted_mimic_cxr_llava_model_input_collate_fn_for_sft(
            x, padding_tokenizer
        ),
        shuffle=False,
        num_workers=num_workers if num_workers else 0,  # TODO let torch decide!
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
