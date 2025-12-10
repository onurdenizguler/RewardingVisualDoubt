import enum
import typing as t

import torch
from LLAVA_Biovil import llava

###### IMPORTS FROM THE RADIALOG/LLAVA REPOSITORY ######
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from utils import create_chest_xray_transform_for_inference, remap_to_uint8


class torch_devices(enum.Enum):
    cuda = "cuda"
    cpu = "cpu"


def get_device_and_device_str() -> tuple[torch.device, str]:
    device_str = torch_devices.cuda.value if torch.cuda.is_available() else torch_devices.cpu.value
    device = torch.device(device_str)
    return device, device_str


def round_green_scores_to_nearest_float(green_scores: list[float | None]) -> list[float | None]:
    rounded_scores = []
    for score in green_scores:
        rounded_scores.append(round(score, 1) if score else score)
    return rounded_scores


POSSIBLE_CONFIDENCES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
POSSIBLE_GRANULAR_CONFIDENCES = list(range(0, 101))  # 0-100 inclusive
LLAVA_IMAGE_TOKEN_INDEX = -200  # as defined by the llava repo
IGNORE_INDEX = -100
