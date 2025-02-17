import enum
import random
import typing as T
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub.lfs import TypedDict
import numpy as np
import pandas as pd
import torch
from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from transformers import PreTrainedTokenizer
from utils import create_chest_xray_transform_for_inference, remap_to_uint8

from RewardingVisualDoubt import mimic_cxr

biovil_image_transformer = create_chest_xray_transform_for_inference(512, center_crop_size=448)
bfloat16_dtype: torch.dtype = torch.bfloat16


class DatasetSplit(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class LlavaModelInputDict(T.TypedDict):
    text_prompt_input_ids: torch.Tensor
    images: torch.Tensor  # Image tensor of dtype 16


# model domain
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


# data interface coupled with loading conversion logic
def _load_image(
    image_path: Path,
) -> Image.Image:
    """Load a single image"""
    image = Image.open(image_path)
    image = remap_to_uint8(np.array(image))
    image = Image.fromarray(image).convert("L")
    return image


def _create_prompt(
    prompter: T.Callable[[str], str],
    disease_labels: T.List[mimic_cxr.ChexpertFinding],
) -> str:
    findings_string = mimic_cxr.convert_binary_chexpert_findings_to_string(disease_labels)
    text_input = prompter(findings_string)
    return text_input


def _create_prompt_for_binary_qa(
    binary_qa_prompter: T.Callable[[str], str], disease: mimic_cxr.ChexpertFinding
) -> str:
    return binary_qa_prompter(disease.value)


# training/inference time logic
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


# model repository function coupled with datapoint and prompter
def _create_llava_model_input_from_mimic_cxr_datapoint(
    datapoint: mimic_cxr.MimicCxrDatapoint | mimic_cxr.MimicCxrBinaryQADatapoint,
    tokenizer: PreTrainedTokenizer,
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
        text_input = _create_prompt_for_binary_qa(prompter, datapoint.disease)
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


@dataclass
class PromptedMimicCxrLlavaModelInputDataset(TorchDataset):
    """
    mimic_cxr_df: pandas DataFrame containing MIMIC-CXR metadata (including `img_path`, `subject_id`, `study_id`, `split`).
    """

    mimic_cxr_df: pd.DataFrame
    tokenizer: PreTrainedTokenizer
    prompter: T.Callable[[str], str]
    image_transform: T.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer
    split: DatasetSplit = DatasetSplit.TRAIN
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
    Each datapoint in this dataset contains a chest x-ray image, a prompt asking about one of the 13 possible CheXpert diseases, and a label indicating the presence or absence of the finding in the image.
    """

    balanced_binary_qa_mimic_cxr_df: pd.DataFrame
    tokenizer: PreTrainedTokenizer
    prompter: T.Callable[[str], str]
    image_transform: T.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer
    split: DatasetSplit = DatasetSplit.TRAIN
    supports_label: bool = False

    def __post_init__(self):
        self.df = self.balanced_binary_qa_mimic_cxr_df[
            self.balanced_binary_qa_mimic_cxr_df["split"] == self.split.value
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict:

        row = self.df.iloc[idx]
        mimic_cxr_binary_qa_datapoint = (
            _create_mimic_cxr_binary_qa_datapoint_from_mimic_cxr_dataset_df_row(row)
        )
        prompt = _create_prompt_for_binary_qa(self.prompter, mimic_cxr_binary_qa_datapoint.disease)
        llava_model_input = _create_llava_model_input_from_mimic_cxr_datapoint(
            mimic_cxr_binary_qa_datapoint, self.tokenizer, self.prompter, self.image_transform
        )

        label = mimic_cxr_binary_qa_datapoint.label
        # TODO tokenize label?

        return BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict(
            llava_model_input_dict=llava_model_input.as_dict(),
            label=bool(label.value),
            prompt=prompt,
            mimic_cxr_datapoint_metadata=mimic_cxr_binary_qa_datapoint,
        )


# TODO: Prompt creation logic for random binary question asking + label assignment


def prompted_mimic_cxr_llava_model_input_collate_fn(
    batch: list[PromptedMimicCxrLlavaModelInputDatapointDict], padding_value
) -> tuple[LlavaModelInputDict, torch.Tensor | list, list[str], list[mimic_cxr.MimicCxrDatapoint]]:
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

    text_inputs = [
        torch.as_tensor(sample["text_prompt_input_ids"]).clone().detach()
        for sample in llava_model_inputs
    ]

    images = torch.stack(
        [torch.as_tensor(sample["images"]) for sample in llava_model_inputs]
    )  # Stack images

    # Pad text sequences to the same length
    text_inputs = torch_pad_sequence(text_inputs, batch_first=True, padding_value=padding_value)

    # Reconstruct llava_model_input with padded sequences
    batch_llava_model_inputs = LlavaModelInputDict(text_prompt_input_ids=text_inputs, images=images)

    # Collect labels as a tensor (or list if they are None)
    batch_labels = (
        torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else list(labels)
    )

    # Metadata remains unchanged as a list
    batch_metadata = list(mimic_cxr_datapoint_metadatas)
    batch_prompts = list(prompts)

    return batch_llava_model_inputs, batch_labels, batch_prompts, batch_metadata


# Probably retire this function
def create_dataset_generator_from_mimic_cxr_dataset_df(
    mimic_cxr_df: pd.DataFrame,
    prompter: T.Callable[[str], str],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    image_transform: T.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer,
    shuffle: bool = False,
    # TODO add split support
) -> T.Callable[[], T.Iterator[T.Dict[str, torch.Tensor]]]:
    """Returns a generator function that yields model inputs"""
    indices = list(range(len(mimic_cxr_df)))

    def dataset_generator() -> T.Iterator[T.Dict[str, torch.Tensor]]:
        if shuffle:
            random.shuffle(indices, random.seed(42))

        for idx in indices:
            row = mimic_cxr_df.iloc[idx]
            mimic_cxr_datapoint = _create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(row)
            yield asdict(
                move_llava_model_input_to_device(
                    _create_llava_model_input_from_mimic_cxr_datapoint(
                        mimic_cxr_datapoint, tokenizer, prompter, image_transform
                    ),
                    device=device,
                )
            )

    return dataset_generator
