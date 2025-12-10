import json
import re
from typing import List, Optional


def parse_binary_labels(generated_answers: List[str]) -> List[bool | None]:
    """
    Parses binary labels from generated answers.
    Returns True for 'yes' and False for 'no'.
    Returns None for ambiguous cases.
    """
    labels = []
    for answer in generated_answers:
        clean_answer = answer.strip().lower()
        if "yes" in clean_answer:
            labels.append(True)
        elif "no" in clean_answer:
            labels.append(False)
        else:
            labels.append(None)  # For ambiguous cases
    return labels


def parse_confidences(
    generated_confidences: List[str], granular_confidence: bool
) -> List[Optional[int]]:
    """
    Extracts the confidence score (0-10 for regular confidence or 0-100 for granular confidence) from the generated confidences.
    Returns None if extraction fails or value is invalid.
    """
    confidences = []
    LOWER_CONFIDENCE_BOUND = 0
    UPPER_CONFIDENCE_BOUND = (
        10 if not granular_confidence else 100
    )  # TODO infer bounds from shared.py
    for conf in generated_confidences:
        # Clean and search for confidence key-value pair, allowing single quotes and unquoted keys
        clean_conf = conf.replace("\n", "").strip()
        pattern = r'["\']?confidence["\']?\s*:\s*(\d+(?:\.\d+)?)'
        if len(re.findall(pattern, clean_conf, re.IGNORECASE)) > 1:
            # If there are multiple matches, we cannot determine which one to use
            confidences.append(None)
            continue
        match = re.search(pattern, clean_conf, re.IGNORECASE)
        if match:
            try:
                value = int(
                    match.group(1)
                )  # TODO: Ensure it's an integer instead of converting it to an int anyways
                if LOWER_CONFIDENCE_BOUND <= value <= UPPER_CONFIDENCE_BOUND:
                    confidences.append(value)
                else:
                    confidences.append(None)
            except ValueError:
                confidences.append(None)
        else:
            confidences.append(None)
    return confidences
