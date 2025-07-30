import enum
import os

import pandas as pd

from . import domain, mimic_cxr

################# REPORTY GENERATION SAMPLING METHODS #################


def remove_samples_with_missing_reports(
    df: pd.DataFrame,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df[~df["findings"].isna()]
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


################# BINARY Q&A SAMPLING METHODS #################


class SamplingMethod(enum.Enum):
    UPSAMPLE = "upsample"
    NONE = "none"


def _sample_for_disease(
    n_samples_per_disease: int,
    disease: str,
    pos_samples: pd.DataFrame,
    neg_samples: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    half_n = n_samples_per_disease // 2
    n_pos = min(len(pos_samples), half_n)
    n_neg = min(
        len(neg_samples), n_samples_per_disease - n_pos
    )  # Fill remaining slots with negatives

    sampled_pos = pos_samples.sample(n=n_pos, random_state=42) if n_pos > 0 else pd.DataFrame()
    sampled_neg = neg_samples.sample(n=n_neg, random_state=42) if n_neg > 0 else pd.DataFrame()
    combined_disease_sample = pd.concat([sampled_pos, sampled_neg])
    if len(combined_disease_sample) < n_samples_per_disease:
        print(
            f"{disease}: only {len(combined_disease_sample)} samples available (pos={n_pos}, neg={n_neg})"
        )
    return combined_disease_sample


def _upsample_for_disease(
    n_samples_per_disease: int,
    disease: str,
    pos_samples: pd.DataFrame,
    neg_samples: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    half_n = n_samples_per_disease // 2
    pos_needed = half_n - len(pos_samples)
    neg_needed = half_n - len(neg_samples)
    if pos_needed > 0:
        print(f"Upsampling {disease} positive samples from {len(pos_samples)} to {half_n}")
        pos_samples = pd.concat(
            [
                pos_samples,
                pos_samples.sample(n=pos_needed, replace=True, random_state=random_state),
            ]
        )
    else:
        pos_samples = pos_samples.sample(n=half_n, random_state=random_state)

    if neg_needed > 0:
        print(f"Upsampling {disease} negative samples from {len(neg_samples)} to {half_n}")
        neg_samples = pd.concat(
            [
                neg_samples,
                neg_samples.sample(n=neg_needed, replace=True, random_state=random_state),
            ]
        )
    else:
        neg_samples = neg_samples.sample(n=half_n, random_state=random_state)
    combined_disease_sample = pd.concat([pos_samples, neg_samples])
    return combined_disease_sample


def sample_balanced_per_disease(
    df_melt: pd.DataFrame,
    n_samples_per_disease,
    sampling_method: SamplingMethod,
    shuffle=True,
    random_state=42,
):
    """
    Create a dataframe with n samples per disease.
    Args:
        df_melt (pd.DataFrame): The melted dataframe with binary labels for each disease.
        n_samples_per_disease (int): The number of samples per disease.
        sampling_method (SamplingMethod): The sampling method to use. Can be either "upsample" or "none".
        1. "upsample": This method is primarily to be used for the training split which has an abundance of samples and upsampling is often not necessary for n = 3000 or lower.
        2. "none": Sample without replacement.
        shuffle (bool): Whether to shuffle the dataframe after sampling.
        random_state (int): The random state to use for reproducibility.
    Returns:
        pd.DataFrame: A dataframe with n samples per disease.
    This method is primarily to be used for the training split which has an abudance of samples and upsampling is often not necessary for n = 3000 or lower.
    """

    sampled_df_list = []

    for disease in df_melt["disease"].unique():

        disease_df = df_melt[df_melt["disease"] == disease]
        pos_samples = disease_df[disease_df["label"] == True]
        neg_samples = disease_df[disease_df["label"] == False]

        sampling_fn = (
            _sample_for_disease if sampling_method == SamplingMethod.NONE else _upsample_for_disease
        )
        combined_disease_sample = sampling_fn(
            n_samples_per_disease, disease, pos_samples, neg_samples, random_state=random_state
        )
        sampled_df_list.append(combined_disease_sample)

    balanced_df = pd.concat(sampled_df_list).reset_index(drop=True)
    if shuffle:
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced_df


def create_balanced_binary_qa_mimic_cxr_dataset_df(
    mimic_cxr_df: pd.DataFrame,
    split: domain.DatasetSplit,
    num_samples_per_disease: int,
) -> pd.DataFrame:

    name_of_the_df = f"balanced_binary_qa_mimic_cxr_df_{split.value}_{num_samples_per_disease}_samples_per_disease"
    name_of_the_cache_file = f"{name_of_the_df}.pkl"

    if os.path.exists(os.path.join(mimic_cxr.CACHE_DIR, name_of_the_cache_file)):
        print(f"Loading {name_of_the_df} from cache")
        return pd.read_pickle(os.path.join(mimic_cxr.CACHE_DIR, name_of_the_cache_file))

    mimic_cxr_df = mimic_cxr_df.copy()[mimic_cxr_df["split"] == split.value]
    df_melt = mimic_cxr.melt_mimic_cxr_df_into_binary_unambiguous_labels(mimic_cxr_df)

    balanced_df = sample_balanced_per_disease(
        df_melt,
        n_samples_per_disease=num_samples_per_disease,
        sampling_method=(
            SamplingMethod.UPSAMPLE if split == domain.DatasetSplit.TRAIN else SamplingMethod.NONE
        ),
        shuffle=True,
        random_state=42,
    )

    print(f"Saving {name_of_the_df} to cache")
    balanced_df.to_pickle(os.path.join(mimic_cxr.CACHE_DIR, name_of_the_cache_file))

    return balanced_df
