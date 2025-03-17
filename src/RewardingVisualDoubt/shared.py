# %%
import enum
import typing as t


class torch_devices(enum.Enum):
    cuda = "cuda"
    cpu = "cpu"


POSSIBLE_CONFIDENCES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
