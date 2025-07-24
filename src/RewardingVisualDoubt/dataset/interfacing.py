import json
import typing as t
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


def append_records_to_json_file(
    records: list[dict],
    output_file: str,
) -> None:
    """Append records to a JSON file, creating it if it doesn't exist."""

    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_file_path.exists():
        with open(output_file_path, "w") as f:
            pass
    with open(output_file_path, "a") as f:
        for r in records:
            json.dump(r, f)
            f.write("\n")
        return


def read_records_from_json_file(
    input_file: str,
) -> list[dict]:
    """Read records from a JSON file."""
    records = []
    with open(input_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
    return records
