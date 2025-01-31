import dataclasses
import enum
import os
import typing as T
from pathlib import Path

import pandas as pd

MIMIC_CXR_FILES_DIR_NAME_PATTERN = "p1[0-9]"
MIMIC_CXR_PATIENT_DIR_NAME_PATTERN = "p*"
MIMIC_CXR_STUDY_DIR_NAME_PATTERN = "s*"
MIMIC_CXR_DATASET_ROOT_DIR = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files"
MIMIC_CXR_SPLIT_CSV_PATH = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
MIMIC_CXR_FINDINGS_CSV_PATH = (
    "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
)


class ChexpertFinding(enum.Enum):
    ATELECTASIS = "Atelectasis"
    CARDIOMEGALY = "Cardiomegaly"
    CONSOLIDATION = "Consolidation"
    EDEMA = "Edema"
    ENLARGED_CARDIOMEDIASTINUM = "Enlarged Cardiomediastinum"
    FRACTURE = "Fracture"
    LUNG_LESION = "Lung Lesion"
    LUNG_OPACITY = "Lung Opacity"
    NO_FINDING = "No Finding"
    PLEURAL_EFFUSION = "Pleural Effusion"
    PLEURAL_OTHER = "Pleural Other"
    PNEUMONIA = "Pneumonia"
    PNEUMOTHORAX = "Pneumothorax"
    SUPPORT_DEVICES = "Support Devices"


@dataclasses.dataclass
class MimicCxrDatapoint:
    subject_id: int
    study_id: int
    disease_labels: T.List[ChexpertFinding]
    img_path: Path


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
    # Make sure the column names match the ChexpertFinding types
    for df_finding_col_name, chexpert_finding_name in zip(
        merged_df.columns[2:15].tolist(), [finding.value for finding in ChexpertFinding]
    ):
        assert (
            df_finding_col_name == chexpert_finding_name
        ), "Column names do not match ChexpertFinding types"
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


def convert_binary_chexpert_findings_to_string(chexpert_finding_list: list[ChexpertFinding]) -> str:
    return ", ".join([finding.value for finding in chexpert_finding_list]).lower().strip()


def create_mimic_cxr_dataset_df(
    mimic_cxr_split_csv_path: str = MIMIC_CXR_SPLIT_CSV_PATH,
    mimic_cxr_findings_csv_path: str = MIMIC_CXR_FINDINGS_CSV_PATH,
) -> pd.DataFrame:
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
