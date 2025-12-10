import typing as t
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from RewardingVisualDoubt import shared

from . import domain, interfacing, mimic_cxr, preprocessing, sampling

biovil_image_transformer = shared.create_chest_xray_transform_for_inference(
    512, center_crop_size=448
)
bfloat16_dtype: torch.dtype = torch.bfloat16


BinaryQANumSamplesPerDisease = {
    domain.DatasetSplit.TRAIN: 3000,
    domain.DatasetSplit.VALIDATION: 50,
    domain.DatasetSplit.TEST: 50,
}


######################## MODEL INPUTS AND BATCHES ########################


#### Datapoint level dataclasses


class LlavaModelInputDict(t.TypedDict):
    text_prompt_input_ids: torch.Tensor
    images: torch.Tensor  # Image tensor of dtype 16


@dataclass
class LlavaModelInput:
    text_prompt_input_ids: torch.Tensor  # Tokenized text
    images: torch.Tensor  # Image tensor

    def as_dict(self) -> LlavaModelInputDict:
        return t.cast(LlavaModelInputDict, asdict(self))


class ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDict(t.TypedDict):
    llava_model_input_dict: LlavaModelInputDict
    label: bool | None
    prompt: str
    mimic_cxr_datapoint_metadata: mimic_cxr.MimicCxrDatapoint


class ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDictForSFT(
    ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDict
):
    expected_output_ids: torch.Tensor


class BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict(t.TypedDict):
    llava_model_input_dict: LlavaModelInputDict
    label: bool | None
    prompt: str
    mimic_cxr_datapoint_metadata: mimic_cxr.MimicCxrBinaryQADatapoint


class BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT(
    BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict
):
    expected_output_ids: torch.Tensor


#### Batch level dataclasses


class MimicCxrLlavaModelInputBatchDict(t.TypedDict):
    batch_llava_model_input_dict: LlavaModelInputDict
    batch_attention_mask: torch.Tensor
    batch_labels: torch.Tensor | list[bool] | list[None]
    batch_prompts: list[str]
    batch_mimic_cxr_datapoint_metadata: list[
        mimic_cxr.MimicCxrBinaryQADatapoint | mimic_cxr.MimicCxrDatapoint
    ]


class MimicCxrLlavaModelInputBatchDictForSFT(MimicCxrLlavaModelInputBatchDict):
    batch_expected_output_ids: torch.Tensor


######################## TRAINING/INFERENCE TIME DEVICE HELPERS ########################


def move_llava_model_input_to_device(
    model_input: LlavaModelInput, device: torch.device
) -> LlavaModelInput:
    model_input.images = model_input.images.to(device, dtype=torch.bfloat16)
    model_input.text_prompt_input_ids = model_input.text_prompt_input_ids.to(device)
    return model_input


def move_llava_model_input_dict_to_device(
    model_input_dict: LlavaModelInputDict, device: torch.device
) -> LlavaModelInputDict:
    model_input_dict["images"] = model_input_dict["images"].to(device, dtype=torch.bfloat16)
    model_input_dict["text_prompt_input_ids"] = model_input_dict["text_prompt_input_ids"].to(device)
    return model_input_dict


######################## DATA PREPERATION ########################


def _create_llava_model_input_from_mimic_cxr_datapoint(
    datapoint: mimic_cxr.MimicCxrDatapoint | mimic_cxr.MimicCxrBinaryQADatapoint,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: t.Callable[[str], str],
    image_transform: t.Callable[[Image.Image], torch.Tensor] | None = biovil_image_transformer,
) -> LlavaModelInput:
    """Convert a study and prompt into model input format"""

    image = interfacing.load_image(datapoint.img_path)
    if image_transform:
        image = image_transform(image)
    else:
        image = torch.tensor(np.array(image))

    if isinstance(datapoint, mimic_cxr.MimicCxrBinaryQADatapoint):
        text_input = preprocessing.create_prompt_for_binary_qa(prompter, datapoint)
    elif isinstance(datapoint, mimic_cxr.MimicCxrDatapoint):
        text_input = preprocessing.create_report_generation_prompt(prompter, datapoint)
    else:
        exception_message = (
            "datapoint must be an instance of either MimicCxrDatapoint or MimicCxrBinaryQADatapoint"
        )
        raise ValueError(exception_message)

    input_ids = preprocessing.tokenize_text_input(text_input, tokenizer)

    return LlavaModelInput(
        text_prompt_input_ids=input_ids,
        images=image,
    )


######################## TORCH DATASETS ########################

############################################################
# Report Generation Datasets
############################################################


@dataclass
class ReportGenerationPromptedMimicCxrLlavaModelInputDataset(TorchDataset):
    """
    mimic_cxr_df: pandas DataFrame containing MIMIC-CXR metadata (including `img_path`, `subject_id`, `study_id`, `split`).
    """

    mimic_cxr_df: pd.DataFrame
    tokenizer: transformers.PreTrainedTokenizer
    prompter: t.Callable[[str], str]
    image_transform: t.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer
    split: domain.DatasetSplit = domain.DatasetSplit.TRAIN

    def __post_init__(self):
        self.df = self.mimic_cxr_df[self.mimic_cxr_df["split"] == self.split.value]

    def __len__(self):
        return len(self.df)

    def _prepare_report_generation_prompted_mimic_cxr_llave_model_input_datapoint_dict(
        self, idx: int
    ) -> ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDict:
        row = self.df.iloc[idx]
        mimic_cxr_datapoint = mimic_cxr.create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(
            row
        )
        prompt = preprocessing.create_report_generation_prompt(self.prompter, mimic_cxr_datapoint)
        llava_model_input = _create_llava_model_input_from_mimic_cxr_datapoint(
            mimic_cxr_datapoint, self.tokenizer, self.prompter, self.image_transform
        )
        label = None

        return ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDict(
            llava_model_input_dict=llava_model_input.as_dict(),
            label=label,
            prompt=prompt,
            mimic_cxr_datapoint_metadata=mimic_cxr_datapoint,
        )

    def __getitem__(self, idx: int) -> ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDict:
        return self._prepare_report_generation_prompted_mimic_cxr_llave_model_input_datapoint_dict(
            idx
        )


@dataclass
class ReportGenerationPromptedMimicCxrLlavaModelInputDatasetForSFT(
    ReportGenerationPromptedMimicCxrLlavaModelInputDataset
):
    """
    This dataset is used for training a model to generate reports given a chest x-ray image and provide the confidence score right after the generated report.
    """

    def __getitem__(
        self, idx: int
    ) -> ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDictForSFT:
        datapoint_dict = (
            self._prepare_report_generation_prompted_mimic_cxr_llave_model_input_datapoint_dict(idx)
        )
        expected_output_ids = preprocessing.replace_confidence_score_tokens_with_value(
            datapoint_dict["llava_model_input_dict"]["text_prompt_input_ids"],
        )
        return ReportGenerationPromptedMimicCxrLlavaModelInputDatapointDictForSFT(
            **datapoint_dict, expected_output_ids=expected_output_ids
        )


def get_report_generation_prompted_mimic_cxr_llava_model_input_dataset(
    split: domain.DatasetSplit,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: t.Callable[..., str],
) -> ReportGenerationPromptedMimicCxrLlavaModelInputDataset:
    return ReportGenerationPromptedMimicCxrLlavaModelInputDataset(
        mimic_cxr_df=sampling.remove_samples_with_missing_reports(
            mimic_cxr.create_mimic_cxr_dataset_df()
        ),
        tokenizer=tokenizer,
        prompter=prompter,
        image_transform=biovil_image_transformer,
        split=split,
    )


def get_report_generation_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
    split: domain.DatasetSplit,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: t.Callable[..., str],
) -> ReportGenerationPromptedMimicCxrLlavaModelInputDatasetForSFT:
    return ReportGenerationPromptedMimicCxrLlavaModelInputDatasetForSFT(
        mimic_cxr_df=sampling.remove_samples_with_missing_reports(
            mimic_cxr.create_mimic_cxr_dataset_df()
        ),
        tokenizer=tokenizer,
        prompter=prompter,
        image_transform=biovil_image_transformer,
        split=split,
    )


############################################################
# Binary QA Datasets
############################################################


@dataclass
class BinaryQAPromptedMimicCxrLlavaModelInputDataset(TorchDataset):
    """
    mimic_cxr_df: pandas DataFrame containing MIMIC-CXR metadata (including `img_path`, `subject_id`, `study_id`, `split`).
    This dataset is used for training a model to answer binary questions about the presence of a specific disease in a chest x-ray image.
    Each datapoint in this dataset contains a chest x-ray image, a prompt asking about one of the 14 possible CheXpert findings, and a label indicating the presence or absence of the finding in the image.
    """

    balanced_binary_qa_mimic_cxr_df: pd.DataFrame
    tokenizer: transformers.PreTrainedTokenizer
    prompter: t.Callable[[str], str]
    image_transform: t.Callable[[Image.Image], torch.Tensor] = biovil_image_transformer
    split: domain.DatasetSplit = domain.DatasetSplit.TRAIN
    supports_label: bool = False

    def __post_init__(self):
        self.df = self.balanced_binary_qa_mimic_cxr_df[
            self.balanced_binary_qa_mimic_cxr_df["split"] == self.split.value
        ]

    def __len__(self):
        return len(self.df)

    def _prepare_binary_qa_prompted_mimic_cxr_llave_model_input_datapoint_dict(
        self, idx: int
    ) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict:
        row = self.df.iloc[idx]
        mimic_cxr_binary_qa_datapoint = (
            mimic_cxr.create_mimic_cxr_binary_qa_datapoint_from_mimic_cxr_dataset_df_row(row)
        )
        prompt = preprocessing.create_prompt_for_binary_qa(
            self.prompter, mimic_cxr_binary_qa_datapoint
        )
        llava_model_input = _create_llava_model_input_from_mimic_cxr_datapoint(
            mimic_cxr_binary_qa_datapoint, self.tokenizer, self.prompter, self.image_transform
        )
        label = mimic_cxr_binary_qa_datapoint.label

        return BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict(
            llava_model_input_dict=llava_model_input.as_dict(),
            label=bool(label.value),
            prompt=prompt,
            mimic_cxr_datapoint_metadata=mimic_cxr_binary_qa_datapoint,
        )

    def __getitem__(self, idx: int) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDict:
        return self._prepare_binary_qa_prompted_mimic_cxr_llave_model_input_datapoint_dict(idx)


@dataclass
class BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT(
    BinaryQAPromptedMimicCxrLlavaModelInputDataset
):
    """
    This dataset is used for training a model to answer binary questions about the presence of a specific disease in a chest x-ray image and provide the confidence score right after the answer.
    """

    def __getitem__(self, idx: int) -> BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT:
        datapoint_dict = (
            self._prepare_binary_qa_prompted_mimic_cxr_llave_model_input_datapoint_dict(idx)
        )
        expected_output_ids = preprocessing.replace_confidence_score_tokens_with_value(
            datapoint_dict["llava_model_input_dict"]["text_prompt_input_ids"],
        )
        return BinaryQAPromptedMimicCxrLlavaModelInputDatapointDictForSFT(
            **datapoint_dict, expected_output_ids=expected_output_ids
        )


def get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset(
    split: domain.DatasetSplit,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: t.Callable[..., str],
) -> BinaryQAPromptedMimicCxrLlavaModelInputDataset:
    return BinaryQAPromptedMimicCxrLlavaModelInputDataset(
        balanced_binary_qa_mimic_cxr_df=sampling.create_balanced_binary_qa_mimic_cxr_dataset_df(
            mimic_cxr.create_mimic_cxr_dataset_df(),
            split=split,
            num_samples_per_disease=BinaryQANumSamplesPerDisease[split],
        ),
        tokenizer=tokenizer,
        prompter=prompter,
        image_transform=biovil_image_transformer,
        split=split,
    )


def get_binary_qa_prompted_mimic_cxr_llava_model_input_dataset_for_sft(
    split: domain.DatasetSplit,
    tokenizer: transformers.PreTrainedTokenizer,
    prompter: t.Callable[..., str],
) -> BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT:
    return BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT(
        balanced_binary_qa_mimic_cxr_df=sampling.create_balanced_binary_qa_mimic_cxr_dataset_df(
            mimic_cxr.create_mimic_cxr_dataset_df(),
            split=split,
            num_samples_per_disease=BinaryQANumSamplesPerDisease[split],
        ),
        tokenizer=tokenizer,
        prompter=prompter,
        image_transform=biovil_image_transformer,
        split=split,
    )


######################## BATCH DATASETS ########################

# generated_sentences_per_ground_truth_list: list[
#     mimic_cxr.GeneratedSentencesAgainstGroundTurthReportDatapoint
# ] = [
#     mimic_cxr.GeneratedSentencesAgainstGroundTurthReportDatapoint(
#         groundtruth_report=SAMPLE_GT_REPORT,
#         generated_sentences=SAMPLE_GENERATED_SENTENCES,
#     )
# ]


def create_generated_sentence_against_gt_report_fact_checking_list_of_input_ids(
    gt_reports: list[str],
    generated_sentences_list: list[list[str]],
    prompter: t.Callable,
    tokenizer: transformers.PreTrainedTokenizer,
) -> list[transformers.tokenization_utils_base.EncodedInput]:
    texts = []
    for gt_report, generated_sentences in zip(gt_reports, generated_sentences_list):
        for generated_sentence in generated_sentences:
            prompt = (
                preprocessing.create_fact_checking_prompt_for_generated_sentence_against_gt_report(
                    fact_checking_prompter=prompter,
                    gt_report=gt_report,
                    generated_sentence=generated_sentence,
                )
            )
            texts.append(prompt)
    text_input_ids = t.cast(
        list[transformers.tokenization_utils_base.EncodedInput],
        [
            preprocessing.tokenize_text_input(
                text_input=text, tokenizer=tokenizer, do_unsqueeze=False
            )
            for text in texts
        ],
    )
    return text_input_ids
