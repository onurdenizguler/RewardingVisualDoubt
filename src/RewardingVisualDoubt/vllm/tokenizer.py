import transformers
from LLAVA_Biovil.llava.constants import (DEFAULT_IM_END_TOKEN,
                                          DEFAULT_IM_START_TOKEN,
                                          DEFAULT_IMAGE_PATCH_TOKEN)

LLAVA_BASE_MODEL_NAME = "liuhaotian/llava-v1.5-7b"


def _modify_tokenizer_for_image_input(
    tokenizer: transformers.LlamaTokenizer,
    mm_use_im_start_end: bool = False,
    mm_use_im_patch_token: bool = False,
) -> transformers.LlamaTokenizer:

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    return tokenizer


def load_pretrained_text_only_llava_tokenizer(
    model_base: str = LLAVA_BASE_MODEL_NAME,
) -> transformers.LlamaTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_base, use_fast=False)
    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )  # vicuna/llama does not have a padding token, use the EOS token instead
    assert isinstance(tokenizer, transformers.LlamaTokenizer)
    return tokenizer


def load_pretrained_llava_tokenizer_with_image_support(
    model_base: str = LLAVA_BASE_MODEL_NAME, for_use_in_padding: bool = False
) -> transformers.LlamaTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_base, use_fast=False
    )  # vicuna/llama does not have a padding token, use the EOS token instead
    assert isinstance(tokenizer, transformers.LlamaTokenizer)
    tokenizer = _modify_tokenizer_for_image_input(tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if for_use_in_padding:
        tokenizer.padding_side = "left"
    return tokenizer
