GREEN_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['from'] == 'human' %}\n"
    "{{ '<|user|>\n' + message['value'] + eos_token }}\n"
    "{% elif message['from'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['value'] + eos_token }}\n"
    "{% elif message['from'] == 'gpt' %}\n"
    "{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


def make_prompt(text1, text2, max_len=300):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = (
        f"Objective: Evaluate the accuracy of a candidate radiology report in comparison "
        f"to a reference radiology report composed by expert radiologists.\n\n "
        f"Process Overview: You will be presented with:\n\n "
        f"1. The criteria for making a judgment.\n "
        f"2. The reference radiology report.\n "
        f"3. The candidate radiology report.\n "
        f"4. The desired format for your assessment.\n\n "
        f"1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n "
        f"The count of clinically significant errors.\n "
        f"The count of clinically insignificant errors.\n\n "
        f"Errors can fall into one of these categories:\n\n "
        f"a) False report of a finding in the candidate.\n "
        f"b) Missing a finding present in the reference.\n "
        f"c) Misidentification of a finding's anatomic location/position.\n "
        f"d) Misassessment of the severity of a finding.\n "
        f"e) Mentioning a comparison that isn't in the reference.\n "
        f"f) Omitting a comparison detailing a change from a prior study.\n "
        f"Note: Concentrate on the clinical findings rather than the report's writing style. "
        f"Evaluate only the findings that appear in both reports.\n\n "
        f"2. Reference Report:\n    {text1}\n\n "
        f"3. Candidate Report:\n    {text2}\n\n "
        f"4. Reporting Your Assessment:\n\n "
        f"Follow this specific format for your output, even if no errors are found:\n "
        f"```\n    [Explanation]:\n    <Explanation>\n\n    "
        f"[Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; "
        f"<Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n "
        f"[Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n "
        f"....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n "
        f"[Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n "
    )
    return prompt
