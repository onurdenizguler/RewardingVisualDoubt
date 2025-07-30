from pathlib import Path

import huggingface_hub

from .shared import (DEFAULT_RADLLAMA_GGUF_DIR, DEFAULT_RADLLAMA_MODEL_REPO,
                     Quantization, create_radllama_model_filename)


def download_radllama_gguf(
    destination_dir: str = DEFAULT_RADLLAMA_GGUF_DIR,
    model_repo: str = DEFAULT_RADLLAMA_MODEL_REPO,
    quantization: Quantization = Quantization.Q4_K_M,
) -> Path:
    """
    Downloads the GGUF model file for GREEN-RadLlama2 from HuggingFace.

    Parameters:
    - destination_dir (str): Local directory to save the model.
    - quantization (str): Quantization level. Default is "Q4_K_M".
                          Other valid options may include "Q5_K_M", "Q8_0", etc.

    Returns:
    - Path to the downloaded GGUF file.
    """
    filename = create_radllama_model_filename(quantization)

    output_path = huggingface_hub.hf_hub_download(
        repo_id=model_repo,
        filename=filename,
        local_dir=destination_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Model downloaded to: {output_path}")
    return Path(output_path)
