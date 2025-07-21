import typing as t

import transformers
from torch.utils.data import DataLoader

from . import collation, dataset

######################## GET DATALOADERS ########################


def get_mimic_cxr_llava_model_input_dataloader(
    dataset: (
        dataset.ReportGenerationPromptedMimicCxrLlavaModelInputDataset
        | dataset.BinaryQAPromptedMimicCxrLlavaModelInputDataset
    ),
    batch_size: int,
    padding_tokenizer: transformers.PreTrainedTokenizer,
    num_workers: t.Optional[int] = None,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collation.prompted_mimic_cxr_llava_model_input_collate_fn(
            x, padding_tokenizer
        ),
        shuffle=False,
        num_workers=num_workers if num_workers else 0,  # TODO let torch decide!
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def get_mimic_cxr_llava_model_input_dataloader_for_sft(
    dataset: dataset.BinaryQAPromptedMimicCxrLlavaModelInputDatasetForSFT,
    batch_size: int,
    padding_tokenizer: transformers.PreTrainedTokenizer,
    num_workers: t.Optional[int] = None,
) -> DataLoader:

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collation.prompted_mimic_cxr_llava_model_input_collate_fn_for_sft(
            x, padding_tokenizer
        ),
        shuffle=False,
        num_workers=num_workers if num_workers else 0,  # TODO let torch decide!
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
