from RewardingVisualDoubt import dataset
import transformers
import typing as t
import json
from torch.utils.data import Subset


REPORT_GENERATION_IN_DISTRIBUTION_TEST_SET_JSON = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/workflows/report_generation/selected_datapoints/988_test_datapoints_balanced_difficulty_sampled_from_validation_set_idx505-2120.json"


def get_in_distribution_report_generation_test_set(
    tokenizer: transformers.PreTrainedTokenizer, prompter_: t.Callable
) -> dataset.ReportGenerationPromptedMimicCxrLlavaModelInputDataset:
    dataset_ = dataset.get_report_generation_prompted_mimic_cxr_llava_model_input_dataset(
        split=dataset.DatasetSplit.VALIDATION, tokenizer=tokenizer, prompter=prompter_
    )

    with open(REPORT_GENERATION_IN_DISTRIBUTION_TEST_SET_JSON) as f:
        sampled_idx = json.load(f)
    dataset_test = Subset(dataset_, sampled_idx)
    dataset_test = t.cast(
        dataset.ReportGenerationPromptedMimicCxrLlavaModelInputDataset, dataset_test
    )
    return dataset_test
