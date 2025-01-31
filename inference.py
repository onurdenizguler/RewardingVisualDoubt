import typing as t
from pathlib import Path

import transformers
from huggingface_hub import snapshot_download
from LLAVA_Biovil.biovil_t.model import ImageModel
from LLAVA_Biovil.llava import LlavaLlamaForCausalLM
from LLAVA_Biovil.llava.model.builder import (
    load_pretrained_model as llava_load_pretrained_model,
)

# import from hugginfface transfomers types: PreTrainedTokenizer
from transformers import PreTrainedTokenizer

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


def _tokenize_input_dialogue():
    # use llava tokenizer
    # mm_utils->tokenizer_image_token
    pass


def _compose_input_dialogue():
    # use llava conversation conv_vicuna_v1
    pass


def _transform_xray_image_for_inference():
    # use utils->create_chest_xray_transform_for_inference
    pass


def _map_binary_findings_to_cp_class_names():
    # utils->init_chexpert_predictor
    pass


# Do not load chexpert for now!
def get_findings_for_study():
    findings = _map_binary_findings_to_cp_class_names()
    pass


def generate_report_for_single_study(model, xray_image, dialogue_history, binary_findings):
    # tokenize input dialogue
    # compose input dialogue
    # transform xray image for inference
    # get findings for study
    # generate report

    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=image_tensor,
    #         do_sample=False,
    #         use_cache=True,
    #         max_new_tokens=300,
    #         stopping_criteria=[stopping_criteria],
    #         pad_token_id=tokenizer.pad_token_id,
    #     )

    # pred = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip().replace("</s>", "")
    # convert_binary_findings_to_text()
    pass


def binary_qa_for_single_study(model):

    pass
