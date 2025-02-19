# %%
import dataclasses
import enum
import typing as T

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


def _add_message_to_conversation(conversation: Conversation, message: Message) -> Conversation:
    conversation.messages.append(message)
    return conversation


def _convert_conversation_into_prompt(conversation: Conversation) -> str:
    messages = conversation.messages

    seps = [conversation.seperator_1.value, conversation.seperator_2.value]
    prompt = conversation.system + seps[0]
    for i, message in enumerate(messages):
        if message.text:
            prompt += message.role.value + ": " + message.text + seps[i % 2]
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
    return _convert_conversation_into_prompt(conversation)


# %%
build_binary_qa_instruction_from_disease_under_study_with_confidence_example("Pneumothorax")
