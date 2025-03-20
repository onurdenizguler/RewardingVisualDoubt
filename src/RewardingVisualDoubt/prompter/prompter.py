# %%
import dataclasses
import enum
import random
import re
import typing as T

from RewardingVisualDoubt.prompter import prompts

DEFAULT_RADIALOG_SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant acting as an experienced radiologist. "
    "The assistant gives professional, detailed, and polite answers to the user's questions."
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


class Seperator(enum.Enum):
    BLANK_SPACE_SEPERATOR = " "
    END_OF_SEQUENCE_SEPERATOR = "</s>"


class Role(enum.Enum):
    """A class that keeps all conversation roles."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"


@dataclasses.dataclass
class Message:
    role: Role
    text: str | None


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: list[Role]
    messages: list[Message]
    seperator_1: Seperator
    seperator_2: Seperator


def _extract_variables_from_formattable_string(format_string):
    # This pattern matches both {var} and {"var"} patterns
    pattern = r"{([^{}]+)}|{\"([^{}\"]+)\"}"

    matches = re.findall(pattern, format_string)

    # Process the matches to extract variable names

    variables = []
    for match in matches:
        # Each match is a tuple with one empty string and one variable name
        var_name = match[0] or match[1]
        variables.append(var_name)

    return variables


def _sample_response_template(options: list[str]):
    index = random.randint(0, len(options) - 1)
    response_template = options[index]
    return response_template


def _add_message_to_conversation(conversation: Conversation, message: Message) -> Conversation:
    conversation.messages.append(message)
    return conversation


def _convert_conversation_into_prompt(conversation: Conversation) -> str:
    messages = conversation.messages

    seps = [conversation.seperator_1.value, conversation.seperator_2.value]
    prompt = ""
    if conversation.system != "":
        prompt = conversation.system + seps[0]
    for i, message in enumerate(messages):
        if message.text:
            prompt += (
                message.role.value + ": " + message.text + seps[0]
            )  # seps[i % 2] # TODO do not EOS the sequences for now (PoC phase)
        else:
            prompt += message.role.value + ":"

    return prompt


def _get_vicuna_conversation() -> Conversation:
    return Conversation(
        system=DEFAULT_RADIALOG_SYSTEM_MESSAGE,
        roles=[Role.USER, Role.ASSISTANT],
        messages=[],
        seperator_1=Seperator.BLANK_SPACE_SEPERATOR,
        seperator_2=Seperator.END_OF_SEQUENCE_SEPERATOR,
    )


def _get_vicuna_conversatio_without_system_message() -> Conversation:
    return Conversation(
        system="",
        roles=[Role.USER, Role.ASSISTANT],
        messages=[],
        seperator_1=Seperator.BLANK_SPACE_SEPERATOR,
        seperator_2=Seperator.END_OF_SEQUENCE_SEPERATOR,
    )


def build_report_generation_instruction_from_findings(findings: str) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=REPORT_GENERATION_INITIAL_INSTRUCTION.format(findings=findings),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
    )
    return _convert_conversation_into_prompt(conversation)


def build_binary_qa_instruction_from_disease_under_study(
    chexpert_finding_str: str,
) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=BINARY_QA_INITIAL_INSTRUCTION.format(chexpert_finding_str=chexpert_finding_str),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
    )
    return _convert_conversation_into_prompt(conversation)


def build_binary_qa_instruction_from_disease_under_study_with_confidence_example(
    chexpert_finding_str: str,
) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.USER, text="Please provide me your response template."),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.ASSISTANT,
            text=(
                "I will provide my responses in the following format:"
                "<response> confidence_score: <score>"
            ),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=BINARY_QA_INITIAL_INSTRUCTION.format(chexpert_finding_str=chexpert_finding_str),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.ASSISTANT,
            text=None,
        ),
    )
    return _convert_conversation_into_prompt(conversation)


def build_binary_qa_instruction_from_disease_under_study_with_confidence_request(
    chexpert_finding_str: str,
) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST.format(
                chexpert_finding_str=chexpert_finding_str
            ),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
    )
    return _convert_conversation_into_prompt(conversation)


def build_binary_qa_prompt_with_response_and_confidence_for_sft(
    chexpert_finding_str: str,
    occurrence_of_disease: bool,
    possible_confidences: list[int],
    return_conversation: bool = False,
) -> str:

    conversation = _get_vicuna_conversation()

    question_template = _sample_response_template(options=prompts.BINARY_QA_USER_QUESTION_OPTIONS)
    question = question_template.format(finding=chexpert_finding_str)
    instruction = (
        BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION + question
    )

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=instruction,
        ),
    )

    response_template = _sample_response_template(
        options=(
            prompts.BINARY_QA_POSTIVE_ASSISTANT_RESPONSE_WITH_CONFIDENCE_OPTIONS
            if occurrence_of_disease
            else prompts.BINARY_QA_NEGATIVE_ASSISTANT_RESPONSE_WITH_CONFIDENCE_OPTIONS
        )
    )

    formattable_variables = _extract_variables_from_formattable_string(response_template)
    if "finding" in formattable_variables:
        assistant_response = response_template.format(
            finding=chexpert_finding_str, confidence_score=random.choice(possible_confidences)
        )
    else:
        assistant_response = response_template.format(
            confidence_score=random.choice(possible_confidences)
        )

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=assistant_response),
    )
    return _convert_conversation_into_prompt(conversation)


def build_binary_qa_prompt_with_response_and_confidence_for_inference(
    chexpert_finding_str: str,
) -> str:

    conversation = _get_vicuna_conversation()

    question_template = _sample_response_template(options=prompts.BINARY_QA_USER_QUESTION_OPTIONS)
    question = question_template.format(finding=chexpert_finding_str)
    instruction = (
        BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION + question
    )

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=instruction,
        ),
    )

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
    )
    return _convert_conversation_into_prompt(conversation)


def build_post_generation_user_confidence_request() -> str:
    conversation = _get_vicuna_conversatio_without_system_message()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=POST_GENERATION_USER_CONFIDENCE_REQUEST,
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=POST_GENERATION_ASSISTANT_CONFIDENCE_COMPLIANCE),
    )
    return _convert_conversation_into_prompt(conversation)
