import json
import re
from typing import List, Optional


def parse_binary_labels(generated_answers: List[str]) -> List[bool]:
    """
    Parses binary labels from generated answers.
    Returns True for 'yes' and False for 'no'.
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


def parse_confidences(generated_confidences: List[str]) -> List[Optional[int]]:
    """
    Extracts the confidence score (0-10) from the generated confidences.
    Returns None if extraction fails or value is invalid.
    """
    confidences = []
    for conf in generated_confidences:
        # Clean and search for confidence key-value pair, allowing single quotes and unquoted keys
        clean_conf = conf.replace("\n", "").strip()
        match = re.search(
            r'["\']?confidence["\']?\s*:\s*(\d+(?:\.\d+)?)', clean_conf, re.IGNORECASE
        )
        if match:
            try:
                value = int(match.group(1))  # Ensure it's an integer
                if 0 <= value <= 10:
                    confidences.append(value)
                else:
                    confidences.append(None)
            except ValueError:
                confidences.append(None)
        else:
            confidences.append(None)
    return confidences
