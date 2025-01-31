# %%
import os

import pandas as pd

# from datasets import ClassLabel, Dataset, Features, Image, Sequence

import mimic_cxr


def _aggregate_and_sort_split_and_findings_information(
    split_df: pd.DataFrame, findings_df: pd.DataFrame
) -> pd.DataFrame:
    merged_df = pd.merge(findings_df, split_df, on=["subject_id", "study_id"], how="inner")
    merged_df = merged_df.sort_values(["subject_id", "study_id"])
    # Make sure the column names match the ChexpertFinding types
    for df_finding_col_name, chexpert_finding_name in zip(
        merged_df.columns[2:15].tolist(), [finding.value for finding in mimic_cxr.ChexpertFinding]
    ):
        assert (
            df_finding_col_name == chexpert_finding_name
        ), "Column names do not match ChexpertFinding types"
    return merged_df


def _resolve_img_path(mimic_cxr_df: pd.DataFrame) -> pd.DataFrame:
    mimic_cxr_df = mimic_cxr_df.copy()
    mimic_cxr_df["img_path"] = mimic_cxr_df.apply(
        lambda row: os.path.join(
            mimic_cxr.MIMIC_CXR_DATASET_ROOT_DIR,
            mimic_cxr.MimicCxrRelativeImgPath(
                subject_id=row["subject_id"], study_id=row["study_id"], dicom_id=row["dicom_id"]
            ).relative_img_path,
        ),
        axis=1,
    )
    return mimic_cxr_df


def create_mimic_cxr_dataset_df(
    mimic_cxr_split_csv_path: str = mimic_cxr.MIMIC_CXR_SPLIT_CSV_PATH,
    mimic_cxr_findings_csv_path: str = mimic_cxr.MIMIC_CXR_FINDINGS_CSV_PATH,
):
    """
    Create a Hugging Face dataset from the X-ray images and their labels
    Using an optimized directory traversal approach
    """

    split_df = pd.read_csv(mimic_cxr_split_csv_path)
    findings_df = pd.read_csv(mimic_cxr_findings_csv_path)
    findings_and_split_df = _aggregate_and_sort_split_and_findings_information(
        split_df, findings_df
    )
    findings_and_split_df = _resolve_img_path(findings_and_split_df)

    return findings_and_split_df


# %%
create_mimic_cxr_dataset_df().head()

# %%
