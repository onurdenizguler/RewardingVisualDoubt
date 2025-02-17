import textwrap

import conversation

REPORT_GENERATION_INITIAL_INSTRUCTION = textwrap.dedent(
    """  
 <image>. Predicted Findings: {findings}. You are to act as a radiologist and write 
 the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. 
 Write in the style of a radiologist, write one fluent text without enumeration, 
 be concise and don't provide explanations or reasons.
 """
)

BINARY_QA_INITIAL_INSTRUCTION = textwrap.dedent(
    """  
 <image>. You are to act as a radiologist and answer the following question: 
 Is the following disease visible in the given X-ray image: {chexpert_finding_str}?)
 """
)


def build_report_generation_instruction_from_findings(findings: str) -> str:
    conv = conversation.conv_vicuna_v1.copy()
    conv.append_message("USER", REPORT_GENERATION_INITIAL_INSTRUCTION.format(findings=findings))
    conv.append_message("ASSISTANT", None)
    text_input = conv.get_prompt()
    return text_input


def build_binary_qa_instruction_from_disease_under_study(
    chexpert_finding_str: str,
) -> str:
    conv = conversation.conv_vicuna_v1.copy()
    conv.append_message(
        "USER",
        BINARY_QA_INITIAL_INSTRUCTION.format(chexpert_finding_str=chexpert_finding_str),
    )
    conv.append_message("ASSISTANT", None)
    text_input = conv.get_prompt()
    return text_input
