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
MIMIC_CXR_REPORTS_ROOT_DIR = "/home/data/DIVA/mimic/mimic-cxr-reports/files"
MIMIC_CXR_SPLIT_CSV_PATH = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
MIMIC_CXR_FINDINGS_CSV_PATH = (
    "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
)
MIMIC_CXR_SECTIONED_CSV_PATH = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic_cxr_sectioned.csv"
CACHE_DIR = "/home/guests/deniz_gueler/repos/RewardingVisualDoubt/.cache"


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


class ChexpertLabel(enum.Enum):
    POSITIVE = 1.0
    NEGATIVE = 0.0
    UNCERTAIN = -1.0


@dataclasses.dataclass
class MimicCxrDatapoint:
    subject_id: int
    study_id: int
    disease_labels: T.List[ChexpertFinding]
    img_path: Path
    report: str


@dataclasses.dataclass
class MimicCxrBinaryQADatapoint:
    subject_id: int
    study_id: int
    img_path: Path
    disease: ChexpertFinding
    label: ChexpertLabel


@dataclasses.dataclass
class MimicCxrReportGenerationDatapoint:
    subject_id: int
    study_id: int
    disease_labels: T.List[ChexpertFinding]
    img_path: Path
    report: str


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


@dataclasses.dataclass
class MimicCxrRelativeReportPath:
    subject_id: str
    study_id: str
    dicom_id: str

    @property
    def relative_img_path(self) -> str:
        subject_prefix = str(self.subject_id)[:2]
        return os.path.join(
            f"p{subject_prefix}",
            f"p{self.subject_id}",
            f"s{self.study_id}.txt",
        )


def create_mimic_cxr_datapoint_from_mimic_cxr_dataset_df_row(
    row: pd.Series,
) -> MimicCxrDatapoint:
    mimic_cxr_datapoint = MimicCxrDatapoint(
        subject_id=row["subject_id"],
        study_id=row["study_id"],
        disease_labels=[
            ChexpertFinding(finding) for finding, value in row.iloc[2:16].items() if value == 1
        ],
        img_path=row["img_path"],
        report=row["findings"],
    )
    return mimic_cxr_datapoint


def create_mimic_cxr_binary_qa_datapoint_from_mimic_cxr_dataset_df_row(
    row: pd.Series,
) -> MimicCxrBinaryQADatapoint:
    mimic_cxr_datapoint = MimicCxrBinaryQADatapoint(
        subject_id=row["subject_id"],
        study_id=row["study_id"],
        img_path=row["img_path"],
        disease=ChexpertFinding(row["disease"]),
        label=ChexpertLabel(row["label"]),
    )
    return mimic_cxr_datapoint


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


def resolve_report_path(mimic_cxr_df: pd.DataFrame) -> pd.DataFrame:
    mimic_cxr_df = mimic_cxr_df.copy()
    mimic_cxr_df["report_path"] = mimic_cxr_df.apply(
        lambda row: os.path.join(
            MIMIC_CXR_REPORTS_ROOT_DIR,
            MimicCxrRelativeReportPath(
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
    mimic_cxr_sectioned_csv_path: str = MIMIC_CXR_SECTIONED_CSV_PATH,
) -> pd.DataFrame:

    # check if df pickle exists in cache dir
    if os.path.exists(os.path.join(CACHE_DIR, "mimic_cxr_df.pkl")):
        print("Loading mimic_cxr_df from cache")
        return pd.read_pickle(os.path.join(CACHE_DIR, "mimic_cxr_df.pkl"))

    split_df = pd.read_csv(mimic_cxr_split_csv_path)
    findings_df = pd.read_csv(mimic_cxr_findings_csv_path)
    findings_and_split_df = _aggregate_and_sort_split_and_findings_information(
        split_df, findings_df
    )
    findings_and_split_df = _resolve_img_path(findings_and_split_df)

    # add the freetext "findings" context of the reports
    sectioned_df = pd.read_csv(mimic_cxr_sectioned_csv_path).reset_index()
    findings_and_split_df["study_id_str"] = findings_and_split_df["study_id"].astype(str)
    sectioned_df["study_id_str"] = sectioned_df["index"].apply(lambda x: x[1:])

    findings_and_split_df_with_reports = pd.merge(
        findings_and_split_df,
        sectioned_df[["study_id_str", "dicom_id", "findings"]],
        left_on=["study_id_str", "dicom_id"],
        right_on=["study_id_str", "dicom_id"],  # remove the first character
        how="left",
    )

    # save df to cache dir
    print("Saving mimic_cxr_df to cache")
    findings_and_split_df_with_reports.to_pickle(os.path.join(CACHE_DIR, "mimic_cxr_df.pkl"))

    return findings_and_split_df_with_reports


def melt_mimic_cxr_df_into_binary_unambiguous_labels(mimic_cxr_df: pd.DataFrame) -> pd.DataFrame:
    """Melt the mimic_cxr_df into a long format with binary labels for each disease.
    Each row corresponds to one disease label for a study.
    The ambiguous labels (-1.0) are filtered out, and only confident labels (1.0 and 0.0) are kept.
    The labels are converted to booleans (1.0 becomes True, 0.0 becomes False).
    Args:
        mimic_cxr_df (pd.DataFrame): The original mimic_cxr_df with findings and split information.
    Returns:
        pd.DataFrame: A melted dataframe with binary labels for each disease.
    """

    disease_cols = [disease.value for disease in ChexpertFinding]
    disease_cols.pop(disease_cols.index("No Finding"))

    # Melt the original dataframe so that each row corresponds to one disease label for a study
    df_melt = mimic_cxr_df.melt(
        id_vars=["subject_id", "study_id", "dicom_id", "split", "img_path"],
        value_vars=disease_cols,
        var_name="disease",
        value_name="label",
    )

    # Filter out uncertain (-1.0) and missing values; keep only confident labels (1.0 and 0.0)
    df_melt = df_melt[
        df_melt["label"].isin([ChexpertLabel.POSITIVE.value, ChexpertLabel.NEGATIVE.value])
    ].copy()

    # Convert labels to booleans (1.0 becomes True, 0.0 becomes False)
    df_melt["label"] = df_melt["label"].astype(bool)

    return df_melt
