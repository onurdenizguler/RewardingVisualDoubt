from datasets import Dataset, Image, Features, ClassLabel, Sequence
import pandas as pd
import os
from pathlib import Path
import numpy as np
import dataclasses


MIMIC_CXR_FILES_DIR_NAME_PATTERN = "p1[0-9]"
MIMIC_CXR_PATIENT_DIR_NAME_PATTERN = "p*"
MIMIC_CXR_STUDY_DIR_NAME_PATTERN = "s*"
MIMIC_CXR_DATASET_ROOT_DIR = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files"
MIMIC_CXR_SPLIT_CSV_PATH = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
MIMIC_CXR_FINDINGS_CSV_PATH = (
    "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
)


@dataclasses.dataclass
class MimicCxrRelativeImgPath:
    subject_id: str
    study_id: str
    dicom_id: str

    @property
    def relative_img_path(self) -> str:
        subject_prefix = str(self.subject_id)[:2]
        return os.path.join(
            f"p{subject_prefix}",
            f"p{self.subject_id}",
            f"s{self.study_id}",
            f"{self.dicom_id}.jpg",
        )


def _aggregate_and_sort_split_and_findings_information(
    split_df: pd.DataFrame, findings_df: pd.DataFrame
) -> pd.DataFrame:
    merged_df = pd.merge(findings_df, split_df, on=["subject_id", "study_id"], how="inner")
    merged_df = merged_df.sort_values(["subject_id", "study_id"])
    return merged_df


def _resolve_img_path(mimic_cxr_df: pd.DataFrame) -> pd.DataFrame:
    mimic_cxr_df = mimic_cxr_df.copy()
    mimic_cxr_df["img_path"] = mimic_cxr_df.apply(
        lambda row: os.path.join(
            MIMIC_CXR_DATASET_ROOT_DIR,
            MimicCxrRelativeImgPath(
                subject_id=row["subject_id"], study_id=row["study_id"], dicom_id=row["dicom_id"]
            ).relative_img_path,
        ),
        axis=1,
    )
    return mimic_cxr_df


def create_mimic_cxr_dataset_df(
    mimic_cxr_split_csv_path: str = MIMIC_CXR_SPLIT_CSV_PATH,
    mimic_cxr_findings_csv_path: str = MIMIC_CXR_FINDINGS_CSV_PATH,
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
