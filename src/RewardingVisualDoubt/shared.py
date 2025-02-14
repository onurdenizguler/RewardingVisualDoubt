# %%
import enum
import typing as t


class torch_devices(enum.Enum):
    cuda = "cuda"
    cpu = "cpu"
