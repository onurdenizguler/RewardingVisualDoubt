import enum


class Quantization(enum.Enum):
    f16 = "f16"
    Q4_K_M = "Q4_K_M"


DEFAULT_RADLLAMA_GGUF_DIR = (
    "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/models/radllama_gguf"
)
DEFAULT_RADLLAMA_MODEL_REPO = "mradermacher/GREEN-RadLlama2-7b-GGUF"


def create_radllama_model_filename(quantization: Quantization) -> str:
    return f"GREEN-RadLlama2-7b.{quantization.value}.gguf"
