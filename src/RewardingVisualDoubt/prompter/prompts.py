DEFAULT_RADIALOG_SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant acting as an experienced radiologist. "
    "The assistant gives professional, detailed, and polite answers to the user's questions."
)

FACT_CHECKING_INITIAL_INSTRUCTION = (
    "You are to act as a radiologist and check a claim your medical student has written"
    "against the ground truth report. Carefully read the ground truth report and then"
    "read the student's sentence to see if it is correct given the ground truth report. "
    "If the student's sentence is correct, return 'True'. If it is incorrect, return 'False'. "
    "If the student's sentence is not related to the ground truth report, return 'Unrelated'. "
    "Report: {gt_report} \n"
    "Student's sentence: {generated_sentence} "
)

REPORT_GENERATION_INITIAL_INSTRUCTION = (
    "<image>. Predicted Findings: {findings}. You are to act as a radiologist and write"
    "the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. "
    "Write in the style of a radiologist, write one fluent text without enumeration, "
    "be concise and don't provide explanations or reasons."
)

BINARY_QA_INITIAL_INSTRUCTION = (
    "<image>. You are to act as a radiologist and answer the following question: "
    "Is the following disease visible in the given X-ray image: {chexpert_finding_str}? "
)

BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION = (
    "<image>. You are to act as a radiologist and answer a single question. "
    "After you respond, please provide your self evaluation of your confidence. "
    "Provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. "
    "A value close to 0 means you think there is a high probability that the answer is wrong. "
    'Your confidence is to be reported in a JSON dictionary of the following format: {"confidence": int}. '
)


BINARY_QA_INITIAL_INSTRUCTION_WITH_GRANULAR_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION = (
    "<image>. You are to act as a radiologist and answer a single question. "
    "After you respond, please provide your self evaluation of your confidence. "
    "Provide a confidence between 0 and 100, of how sure you are the answer is correct. "
    "A value close to 0 means you think there is a high probability that the answer is wrong. "
    'Your confidence is to be reported in a JSON dictionary of the following format: {"confidence": int}. '
)

BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST = (
    BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION
    + "Is the following disease visible in the given X-ray image: {chexpert_finding_str}, and how confident are you? "
)


POST_GENERATION_USER_CONFIDENCE_REQUEST = (
    "Now evaluate your own response. How confident are you in your answer? "
    "Provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. "
    "A value close to 0 means you think there is a high probability that the answer is wrong. "
    "The closer the value is to 10, the higher you think is the probability that the answer is correct. "
)

POST_GENERATION_ASSISTANT_CONFIDENCE_COMPLIANCE = (
    "When asked how confident I am about a response, I consistently provide it in a JSON object, adhering to my policy."
    "The confidence JSON follows this structure: {'confidence': int}."
    "Here's my confidence JSON about my last response: "
)


BINARY_QA_ASSISTANT_RESPONSE_WITH_CONFIDENCE = (
    'Yes, the disease is visible in the X-ray image. {{"confidence": {confidence_score} }}'
)

POST_GENERATION_USER_CONFIDENCE_REQUEST_OPTIONS = [
    (
        "Now evaluate your own response. How confident are you in your answer? "
        "Provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. "
        "A value close to 0 means you think there is a high probability that the answer is wrong. "
        "The closer the value is to 10, the higher you think is the probability that the answer is correct. "
    ),
    (
        "Reflect on your response. How certain are you about its accuracy?"
        "Rate your confidence on a scale from 0 to 10, where 0 means very low confidence and 10 means absolute certainty."
        "A score near 0 indicates you believe the answer is likely incorrect, while a score near 10 suggests strong confidence in its correctness."
    ),
    (
        "Assess your answer. How sure are you that it's right?"
        "Give a confidence rating between 0 and 10, with 0 being complete doubt and 10 being total certainty."
        "Lower numbers suggest high uncertainty, while higher numbers reflect greater confidence in the accuracy of the response."
    ),
    (
        "Take a moment to evaluate your reply. How confident do you feel about it?"
        "Select a number from 0 to 10, where 0 means you're almost certain the answer is wrong, and 10 means you're fully confident it's correct."
    ),
    (
        "Re-examine your response. How likely do you think it is that your answer is correct?"
        "Choose a confidence level from 0 (very unsure) to 10 (completely sure)."
        "The closer the number is to 10, the more confident you are in the answer's accuracy."
    ),
    (
        "Review your answer. How much confidence do you have in its correctness?"
        "Assign a score between 0 and 10, where 0 signals high doubt and 10 reflects total certainty."
        "A lower score indicates a greater chance of error, while a higher score shows stronger confidence."
    ),
    (
        "Consider your answer carefully. How sure are you that it’s accurate?"
        "Rate your confidence from 0 (not confident at all) to 10 (completely confident)."
        "Numbers near 0 imply low certainty, while numbers near 10 indicate high confidence in the answer."
    ),
    (
        "Evaluate your response. How strongly do you believe it is correct?"
        "Use a scale from 0 to 10, with 0 representing complete uncertainty and 10 representing full confidence."
        "The lower the number, the less confident you are; the higher the number, the more certain you feel."
    ),
    (
        "Look over your answer. How much trust do you place in its accuracy?"
        "Pick a number from 0 to 10, where 0 means you believe it’s probably wrong and 10 means you’re certain it’s right."
        "A score closer to 10 shows higher confidence, while one near 0 shows lower confidence."
    ),
    (
        "Think critically about your response. How confident are you in its validity?"
        "Select a confidence score between 0 (no confidence) and 10 (full confidence)."
        "A value near 0 suggests strong doubt, while a value near 10 suggests strong certainty."
    ),
    (
        "Analyze your answer. To what extent do you believe it’s correct?"
        "Rate your confidence from 0 to 10, where 0 reflects low certainty and 10 reflects high certainty."
        "Lower numbers show more doubt, while higher numbers show more confidence in your answer."
    ),
]


BINARY_QA_USER_QUESTION_OPTIONS = [
    ("Is there evidence of {finding} in the image?"),
    ("Is there any {finding}?"),
    ("Is any {finding} evident in the report?"),
    ("Is there any indication of {finding} in the report?"),
    ("Does the report mention  {finding}?"),
    ("Does the patient have {finding}?"),
    ("Is there any sign of  {finding} in the report?"),
]


BINARY_QA_POSTIVE_ASSISTANT_RESPONSE_WITH_CONFIDENCE_OPTIONS = [
    ('Yes, the image shows {finding}. {{"confidence": {confidence_score}}}'),
    ('Yes, the patient has {finding}. {{"confidence": {confidence_score}}}'),
    ('Yes, there is evidence of that in the image. {{"confidence": {confidence_score} }}'),
    ('Yes, indications of {finding} are present. {{"confidence": {confidence_score}}}'),
    ('Yes, the image findings align with {finding}. {{"confidence": {confidence_score}}}'),
    ('Yes, the image suggests {finding}. {{"confidence": {confidence_score}}}'),
    ('Yes, {finding} is visible. {{"confidence": {confidence_score}}}'),
    ('Yes, the observed characteristics match {finding}. {{"confidence": {confidence_score}}}'),
    ('Yes, the image presents signs of {finding}. {{"confidence": {confidence_score}}}'),
    ('Yes, the diagnostic markers indicate {finding}. {{"confidence": {confidence_score}}}'),
]

BINARY_QA_NEGATIVE_ASSISTANT_RESPONSE_WITH_CONFIDENCE_OPTIONS = [
    ('No, the image shows no {finding}. {{"confidence": {confidence_score}}}'),
    ('No, there is no evidence of that in the image. {{"confidence": {confidence_score}}}'),
    ('No, the patient does not have {finding}. {{"confidence": {confidence_score}}}'),
    ('No, the image does not show {finding}. {{"confidence": {confidence_score}}}'),
    ('No, there are no indications of {finding}. {{"confidence": {confidence_score}}}'),
    ('No, {finding} is not observed in the image. {{"confidence": {confidence_score}}}'),
    ('No, the image does not indicate {finding}. {{"confidence": {confidence_score}}}'),
    ('No, there is no detectable evidence of {finding}. {{"confidence": {confidence_score}}}'),
    ('No, the image does not provide signs of {finding}. {{"confidence": {confidence_score}}}'),
]
