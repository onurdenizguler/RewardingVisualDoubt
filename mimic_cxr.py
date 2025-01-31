import dataclasses
import enum
import typing as T
from pathlib import Path
import os

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


def convert_binary_chexpert_findings_to_string(chexpert_finding_list: list[ChexpertFinding]) -> str:
    return ", ".join([finding.value for finding in chexpert_finding_list]).lower().strip()
