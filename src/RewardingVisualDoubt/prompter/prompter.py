# %%
import dataclasses
import enum
import random
import re
import typing as T

from RewardingVisualDoubt.prompter import prompts


class Seperator(enum.Enum):
    BLANK_SPACE_SEPERATOR = " "
    END_OF_SEQUENCE_SEPERATOR = "</s>"


STOP_STR = Seperator.END_OF_SEQUENCE_SEPERATOR.value


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


############################################## HELPER FUNCTIONS ##############################################


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


############################################## CONVERSATION LOGIC ##############################################


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
            prompt += message.role.value + ": " + message.text + seps[i % 2]
        else:
            prompt += message.role.value + ":"

    return prompt


def _get_vicuna_conversation() -> Conversation:
    return Conversation(
        system=prompts.DEFAULT_RADIALOG_SYSTEM_MESSAGE,
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


############################################## PROMPT BUILDING ##############################################

#### LLM-as-judge requests


def build_fact_checking_instruction_for_generated_sentence_against_gt_report(
    generated_sentence: str, gt_report: str
) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=prompts.FACT_CHECKING_INITIAL_INSTRUCTION.format(
                generated_sentence=generated_sentence, gt_report=gt_report
            ),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
    )
    return _convert_conversation_into_prompt(conversation)


#### Report generation requests


def build_report_generation_instruction_from_findings(findings: str) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=prompts.REPORT_GENERATION_INITIAL_INSTRUCTION.format(findings=findings),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
    )
    return _convert_conversation_into_prompt(conversation)


def build_report_generation_prompt_with_response_and_confidence_for_sft(
    findings: str,
    possible_confidences: list[int],
    gt_report: str,
    is_for_inference: bool = False,
) -> str:

    conversation = _get_vicuna_conversation()

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=prompts.REPORT_GENERATION_INITIAL_INSTRUCTION.format(findings=findings),
        ),
    )

    if is_for_inference:
        assistant_response = None
    else:
        response_template = gt_report + ' {{"confidence": {confidence_score}}}'
        assistant_response = response_template.format(
            confidence_score=random.choice(possible_confidences)
        )

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=assistant_response),
    )
    return _convert_conversation_into_prompt(conversation)


#### Binary q&a requests


def build_binary_qa_instruction_from_disease_under_study(
    chexpert_finding_str: str,
) -> str:
    conversation = _get_vicuna_conversation()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=prompts.BINARY_QA_INITIAL_INSTRUCTION.format(
                chexpert_finding_str=chexpert_finding_str
            ),
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(role=Role.ASSISTANT, text=None),
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
            text=prompts.BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST.format(
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
    is_for_inference: bool = False,
    granular_confidence: bool = False,
) -> str:

    conversation = _get_vicuna_conversation()

    question_template = _sample_response_template(options=prompts.BINARY_QA_USER_QUESTION_OPTIONS)
    question = question_template.format(finding=chexpert_finding_str)
    instruction = (
        prompts.BINARY_QA_INITIAL_INSTRUCTION_WITH_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION
        if not granular_confidence
        else prompts.BINARY_QA_INITIAL_INSTRUCTION_WITH_GRANULAR_CONFIDENCE_REQUEST_WITHOUT_THE_QUESTION
        + question
    )

    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=instruction,
        ),
    )

    if is_for_inference:
        assistant_response = None
    else:
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


############################################## DEPRECATED PROMPT BUILDING ##############################################


def build_post_generation_user_confidence_request() -> str:
    conversation = _get_vicuna_conversatio_without_system_message()
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.USER,
            text=prompts.POST_GENERATION_USER_CONFIDENCE_REQUEST,
        ),
    )
    _add_message_to_conversation(
        conversation=conversation,
        message=Message(
            role=Role.ASSISTANT, text=prompts.POST_GENERATION_ASSISTANT_CONFIDENCE_COMPLIANCE
        ),
    )
    return _convert_conversation_into_prompt(conversation)
