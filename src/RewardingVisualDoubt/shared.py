# %%
import enum
import typing as t

###### IMPORTS FROM THE RADIALOG/LLAVA REPOSITORY ######
from LLAVA_Biovil.llava.mm_utils import tokenizer_image_token
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria
from utils import remap_to_uint8, create_chest_xray_transform_for_inference
from LLAVA_Biovil import llava


class torch_devices(enum.Enum):
    cuda = "cuda"
    cpu = "cpu"


POSSIBLE_CONFIDENCES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
POSSIBLE_GRANULAR_CONFIDENCES = list(range(0, 101))  # 0-100 inclusive
LLAVA_IMAGE_TOKEN_INDEX = -200  # as defined by the llava repo
IGNORE_INDEX = -100
