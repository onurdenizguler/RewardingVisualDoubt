import enum
import random
import typing as T
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose
from transformers import PreTrainedTokenizer
from utils import create_chest_xray_transform_for_inference, remap_to_uint8

from RewardingVisualDoubt import mimic_cxr, shared

biovil_image_transformer = create_chest_xray_transform_for_inference(512, center_crop_size=448)


class DatasetSplit(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# model domain
@dataclass
class LlavaModelInput:
    text_prompt_input_ids: torch.Tensor  # Tokenized text
    images: torch.Tensor  # Image tensor


# data interface coupled with loading conversion logic
def _load_image(
    image_path: Path,
) -> Image.Image:
    """Load a single image"""
    image = Image.open(image_path)
    image = remap_to_uint8(np.array(image))
    image = Image.fromarray(image).convert("L")
    return image


# training/inference time logic
def move_llava_model_input_to_device(
    model_input: LlavaModelInput, device: torch.device
) -> LlavaModelInput:
    model_input.images = model_input.images.to(device, dtype=torch.bfloat16)
    model_input.text_prompt_input_ids = model_input.text_prompt_input_ids.to(device)
    return model_input


# model repository function coupled with datapoint and prompter
def _create_llava_model_input_from_mimic_cxr_datapoint(
    datapoint: mimic_cxr.MimicCxrDatapoint,
    tokenizer: PreTrainedTokenizer,
    prompter: T.Callable[[str], str],
    image_transform: T.Callable[[Image.Image], torch.Tensor] | None = biovil_image_transformer,
) -> LlavaModelInput:
    """Convert a study and prompt into model input format"""

    # Handle image
    image = _load_image(datapoint.img_path)
    if image_transform:
        image = image_transform(image).unsqueeze(0)
    else:
        image = torch.tensor(np.array(image)).unsqueeze(0)

    findings_string = mimic_cxr.convert_binary_chexpert_findings_to_string(datapoint.disease_labels)
    text_input = prompter(findings_string)

    # get findings from the datapoint
    # Handle text input
    input_ids = tokenizer_image_token(
        text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)

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
    device: torch.device = torch.device(shared.torch_devices.cpu.value)

    def __post_init__(self):
        """This runs after dataclass initializes the fields"""
        self.df = self.mimic_cxr_df[self.mimic_cxr_df["split"] == self.split.value]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        mimic_cxr_datapoint = _create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(row)
        llava_model_input = _create_llava_model_input_from_mimic_cxr_datapoint(
            mimic_cxr_datapoint, self.tokenizer, self.prompter, self.image_transform
        )

        return asdict(move_llava_model_input_to_device(llava_model_input, self.device))


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
