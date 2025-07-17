from pathlib import Path

import numpy as np
from PIL import Image
from RewardingVisualDoubt import shared


######################## DATA INTERFACING ########################


def load_image(
    image_path: Path,
) -> Image.Image:
    """Load a single image"""
    image = Image.open(image_path)
    image = shared.remap_to_uint8(np.array(image))
    image = Image.fromarray(image).convert("L")
    return image
