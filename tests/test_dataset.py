from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from transformers import PreTrainedTokenizer

from RewardingVisualDoubt import dataset

############ FIXTURES ############


def create_mock_image():
    return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))


def create_sample_df():
    return pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "study_id": ["study1", "study2"],
            "split": ["train", "validation"],
            "img_path": ["path1.jpg", "path2.jpg"],
            "Atelectasis": [1, 0],
            "Cardiomegaly": [0, 1],
            "Consolidation": [0, 0],
            "Edema": [0, 0],
            "Enlarged Cardiomediastinum": [0, 0],
            "Fracture": [0, 0],
            "Lung Lesion": [0, 0],
            "Lung Opacity": [0, 0],
            "No Finding": [0, 0],
            "Pleural Effusion": [0, 0],
            "Pleural Other": [0, 0],
            "Pneumonia": [0, 0],
            "Pneumothorax": [0, 0],
            "Support Devices": [0, 0],
        }
    )


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.encode.return_value = torch.tensor([1, 2, 3])
    return tokenizer


@pytest.fixture
def mock_prompter():
    return lambda x: f"Prompt: {x}"


@pytest.fixture
def mock_image_transform():
    return lambda x: torch.zeros(1, 3, 224, 224)


############ TEST CASES ############


def test_dataset_initialization():
    df = create_sample_df()
    dataset_ = dataset.PromptedMimicCxrLlavaModelInputDataset(
        mimic_cxr_df=df,
        tokenizer=Mock(spec=PreTrainedTokenizer),
        prompter=lambda x: x,
        split=dataset.DatasetSplit.TRAIN,
    )
    assert len(dataset_) == 1  # Only one training sample in mock data
    assert len(dataset_.df) == 1
    assert dataset_.df.iloc[0]["subject_id"] == "s1"


@patch("PIL.Image.open")
def test_load_image(mock_open):
    mock_open.return_value = create_mock_image()
    image = dataset._load_image(Path("dummy.jpg"))
    assert isinstance(image, Image.Image)
    assert image.mode == "L"  # Check if grayscale


def test_move_to_device():
    model_input = dataset.LlavaModelInput(
        text_prompt_input_ids=torch.tensor([1, 2, 3]), images=torch.zeros(1, 3, 224, 224)
    )
    device = torch.device("cpu")
    moved_input = dataset.move_llava_model_input_to_device(model_input, device)
    assert moved_input.text_prompt_input_ids.device == device
    assert moved_input.images.device == device
    assert moved_input.images.dtype == torch.bfloat16


def test_create_mimic_cxr_datapoint():
    df = create_sample_df()
    row = df.iloc[0]
    datapoint = dataset._create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(row)
    assert datapoint.subject_id == "s1"
    assert datapoint.study_id == "study1"
    assert len(datapoint.disease_labels) == 1  # Only Atelectasis is 1
    assert str(datapoint.disease_labels[0]) == "Atelectasis"


def test_dataset_generator():
    df = create_sample_df()
    generator_fn = dataset.create_dataset_generator_from_mimic_cxr_dataset_df(
        mimic_cxr_df=df,
        prompter=lambda x: x,
        tokenizer=Mock(spec=PreTrainedTokenizer),
        device=torch.device("cpu"),
        image_transform=lambda x: torch.zeros(1, 3, 224, 224),
        shuffle=True,
    )

    generator = generator_fn()
    first_item = next(generator)
    assert isinstance(first_item, dict)
    assert "text_prompt_input_ids" in first_item
    assert "images" in first_item


@pytest.mark.parametrize(
    "split,expected_length",
    [
        (dataset.DatasetSplit.TRAIN, 1),
        (dataset.DatasetSplit.VALIDATION, 1),
        (dataset.DatasetSplit.TEST, 0),
    ],
)
def test_dataset_splits(split, expected_length):
    df = create_sample_df()
    dataset_ = dataset.PromptedMimicCxrLlavaModelInputDataset(
        mimic_cxr_df=df, tokenizer=Mock(spec=PreTrainedTokenizer), prompter=lambda x: x, split=split
    )
    assert len(dataset_) == expected_length


def test_dataset_getitem():
    df = create_sample_df()
    dataset_ = dataset.PromptedMimicCxrLlavaModelInputDataset(
        mimic_cxr_df=df,
        tokenizer=Mock(spec=PreTrainedTokenizer),
        prompter=lambda x: x,
        image_transform=lambda x: torch.zeros(1, 3, 224, 224),
        split=dataset.DatasetSplit.TRAIN,
    )

    with patch("PIL.Image.open", return_value=create_mock_image()):
        item = dataset_[0]
        assert isinstance(item, dict)
        assert "text_prompt_input_ids" in item
        assert "images" in item
        assert isinstance(item["images"], torch.Tensor)
        assert isinstance(item["text_prompt_input_ids"], torch.Tensor)
