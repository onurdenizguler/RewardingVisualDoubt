import typing as t
from pathlib import Path

import transformers
from huggingface_hub import snapshot_download
from LLAVA_Biovil.biovil_t.model import ImageModel
from LLAVA_Biovil.llava import LlavaLlamaForCausalLM
from LLAVA_Biovil.llava.model.builder import \
    load_pretrained_model as llava_load_pretrained_model

LLAVA_BASE_MODEL_NAME = "liuhaotian/llava-v1.5-7b"
LLAVA_LORA_ADAPTER = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
FINETUNED_LLAVA_REPO_ID = "ChantalPellegrini/RaDialog-interactive-radiology-report-generation"


def _get_hf_model_path(repo_id) -> Path:
    return Path(snapshot_download(repo_id=repo_id, revision="main"))


def load_visual_language_model_Llava() -> (
    tuple[transformers.PreTrainedTokenizer, LlavaLlamaForCausalLM, ImageModel, int]
):

    tokenizer, model, image_processor, context_len = llava_load_pretrained_model(
        model_path=_get_hf_model_path(repo_id=FINETUNED_LLAVA_REPO_ID),
        model_base=LLAVA_BASE_MODEL_NAME,
        model_name=LLAVA_LORA_ADAPTER,
        load_8bit=False,
        load_4bit=False,
    )

    model.config.tokenizer_padding_side = "left"

    return tokenizer, model, image_processor, context_len
