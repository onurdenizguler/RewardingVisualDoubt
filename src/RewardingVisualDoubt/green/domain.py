import dataclasses
import enum


@dataclasses.dataclass
class GreenDefaultGenerationParameters:
    max_length: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False


class GreenModelNames(enum.Enum):
    RADLLAMA2 = "StanfordAIMI/GREEN-RadLlama2-7b"
    RADPHI2 = "StanfordAIMI/GREEN-RadPhi2"


class GreenCategories(enum.Enum):
    SIGNIFICANT = "Clinically Significant Errors"
    INSIGNIFICANT = "Clinically Insignificant Errors"
    MATCHED_FINDINDS = "Matched Findings"


class GreenSubCategories(enum.Enum):
    A = "(a) False report of a finding in the candidate"
    B = "(b) Missing a finding present in the reference"
    C = "(c) Misidentification of a finding's anatomic location/position"
    D = "(d) Misassessment of the severity of a finding"
    E = "(e) Mentioning a comparison that isn't in the reference"
    F = "(f) Omitting a comparison detailing a change from a prior study"
