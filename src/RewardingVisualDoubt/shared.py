# %%
import enum
import typing as t
from LLAVA_Biovil.llava.mm_utils import KeywordsStoppingCriteria


class torch_devices(enum.Enum):
    cuda = "cuda"
    cpu = "cpu"


POSSIBLE_CONFIDENCES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LLAVA_IMAGE_TOKEN_INDEX = -200  # as defined by the llava repo
IGNORE_INDEX = -100
