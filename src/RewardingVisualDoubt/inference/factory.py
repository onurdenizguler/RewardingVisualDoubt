import transformers


def get_typical_report_generation_kwargs(tokenizer: transformers.PreTrainedTokenizer) -> dict:

    return {
        "top_k": 0.0,  # No top-k sampling
        "top_p": 1.0,  # Let us limit the sampling a bit
        "temperature": 1.0,  # Decrease the randomness
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 300,
        "eos_token_id": tokenizer.eos_token_id,
    }
