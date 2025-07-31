import dataclasses
import enum


class GreenDefaultGenerationParameters(enum.Enum):
    max_length = 2048
    temperature = 0.0
    top_p = 1.0
    top_k = 0
    do_sample = False


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
