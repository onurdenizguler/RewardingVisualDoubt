from .batching import *
from .domain import *
from .evaluate.evaluate import *
from .llama_client import *
from .llama_server import *
from .postprocessing import *
from .prompter import *
from .shared import *
from .pipeline import *

EXAMPLE_REFERENCE_REPORT = "The cardiac and mediastinal silhouettes are stable.  Hilar contours are stable. There is persistent blunting of the right costophrenic angle. There is mild increased interstitial markings bilaterally suggesting interstitial edema. Left mid lung atelectasis is linear. No pneumothorax is seen."
EXAMPLE_CANDIDATE_REPORT = "The lungs are well-expanded with persistent mild pulmonary edema and vascular congestion. No focal consolidation, pleural effusion, or pneumothorax. Mild cardiomegaly is unchanged. Mediastinal contour and hila are unremarkable. Visualized upper abdomen is unremarkable."
EXAMPLE_PROMPT = """
Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.

 Process Overview: You will be presented with:

 1. The criteria for making a judgment.
 2. The reference radiology report.
 3. The candidate radiology report.
 4. The desired format for your assessment.

 1. Criteria for Judgment:

    For each candidate report, determine:

 The count of clinically significant errors.
 The count of clinically insignificant errors.

 Errors can fall into one of these categories:

 a) False report of a finding in the candidate.
 b) Missing a finding present in the reference.
 c) Misidentification of a finding's anatomic location/position.
 d) Misassessment of the severity of a finding.
 e) Mentioning a comparison that isn't in the reference.
 f) Omitting a comparison detailing a change from a prior study.
 Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.

 2. Reference Report:
    The cardiac and mediastinal silhouettes are stable.  Hilar contours are stable. There is persistent blunting of the right costophrenic angle. There is mild increased interstitial markings bilaterally suggesting interstitial edema. Left mid lung atelectasis is linear. No pneumothorax is seen.

 3. Candidate Report:
    The lungs are well-expanded with persistent mild pulmonary edema and vascular congestion. No focal consolidation, pleural effusion, or pneumothorax. Mild cardiomegaly is unchanged. Mediastinal contour and hila are unremarkable. Visualized upper abdomen is unremarkable.

 4. Reporting Your Assessment:

 Follow this specific format for your output, even if no errors are found:
 ```
    [Explanation]:
    <Explanation>

    [Clinically Significant Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
    ....
    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

 [Clinically Insignificant Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
 ....
    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

 [Matched Findings]:
    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>
    ```
"""
