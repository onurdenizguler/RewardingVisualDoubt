import typing as t

import torch


# Whitebox confidence calibration evaluation tools.
def extract_yes_no_probs_from_logits(logits: torch.Tensor) -> t.Tuple[t.List[float], t.List[float]]:
    last_token_logits = logits[:, -1]
    probs = torch.softmax(last_token_logits, dim=-1)

    yes_token_id = 3869
    no_token_id = 1939

    yes_probs = probs[:, yes_token_id]
    no_probs = probs[:, no_token_id]

    # Normalize the yes and no probabilities
    denominator = yes_probs + no_probs + 1e-8  # avoid division by zero
    normalized_yes = (yes_probs / denominator).tolist()
    normalized_no = (no_probs / denominator).tolist()
    return normalized_yes, normalized_no
