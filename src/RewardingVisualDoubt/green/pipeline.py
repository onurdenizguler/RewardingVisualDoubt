from . import domain, llama_client, postprocessing, prompter
from .evaluate import evaluate


def _create_instruction_and_token_length_for_a_single_report(
    generated_report: str,
    gt_report: str,
) -> tuple[str, int]:
    instruction = prompter.make_prompt(
        reference_report=gt_report,
        candidate_report=generated_report,
    )
    chat_style_instruction = llama_client.sync_apply_chat_template(
        messages=[{"role": "user", "content": instruction}]
    )
    token_len = len(llama_client.sync_tokenize(chat_style_instruction))
    return chat_style_instruction, token_len


def _get_response_for_single_query(
    instruction: str,
    token_length: int,
) -> llama_client.CompletionResponse:
    response = llama_client.sync_fetch_completion(
        prompt=instruction,
        n_predict=domain.GreenDefaultGenerationParameters.max_length.value - token_length,
    )
    return response


def create_instructions_and_token_lengths(
    generated_reports_list: list[str],
    gt_reports_list: list[str],
) -> tuple[list[str], list[int]]:

    instructions: list[str] = []
    token_lengths: list[int] = []

    for generated_report, gt_report in zip(generated_reports_list, gt_reports_list):

        chat_style_instruction, token_len = (
            _create_instruction_and_token_length_for_a_single_report(
                generated_report=generated_report,
                gt_report=gt_report,
            )
        )

        instructions.append(chat_style_instruction)
        token_lengths.append(token_len)

    return instructions, token_lengths


def query_llama_cpp_with_instructions(
    instructions: list[str],
    token_lengths: list[int],
) -> list[llama_client.CompletionResponse]:
    responses = []
    for instruction, token_length in zip(instructions, token_lengths):
        response = _get_response_for_single_query(
            instruction=instruction,
            token_length=token_length,
        )
        responses.append(response)
    return responses


def get_green_score_for_batch_of_generated_reports(
    gt_reports: list[str],
    generated_reports: list[str],
) -> list[float | None]:

    instructions, token_lengths = create_instructions_and_token_lengths(
        generated_reports_list=generated_reports,
        gt_reports_list=gt_reports,
    )
    responses = query_llama_cpp_with_instructions(
        instructions=instructions,
        token_lengths=token_lengths,
    )
    green_analyses = [postprocessing.clean_responses(response.content) for response in responses]
    green_scores = [evaluate.compute_green(analysis) for analysis in green_analyses]
    return green_scores


def get_green_score_for_single_generated_report(
    gt_report: str,
    generated_report: str,
) -> float | None:

    instruction, token_length = _create_instruction_and_token_length_for_a_single_report(
        generated_report=generated_report,
        gt_report=gt_report,
    )

    response = _get_response_for_single_query(
        instruction=instruction,
        token_length=token_length,
    )

    green_analysis = postprocessing.clean_responses(response.content)
    return evaluate.compute_green(green_analysis)
