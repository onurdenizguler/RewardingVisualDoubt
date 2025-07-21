import re

import numpy as np
import pandas as pd

from ..domain import GreenCategories, GreenSubCategories
from .clustering import compute_largest_cluster
from .parsing import parse_error_counts, parse_error_sentences


def process_results(
    responses: list[str],
    references: list[str],
    predictions: list[str],
    compute_summary_stats: bool = True,
):
    green_scores: list[float | None] = [compute_green(response) for response in responses]
    error_counts = pd.DataFrame(
        [compute_error_count(response) for response in responses],
        columns=[subcategory.value for subcategory in GreenSubCategories] + ["Matched Findings"],
    )

    results_df = pd.DataFrame(
        {
            "reference": references,
            "predictions": predictions,
            "green_analysis": responses,
            "green_score": green_scores,
            **error_counts,
        }
    )
    mean, std, summary = None, None, None

    if compute_summary_stats:
        mean, std, summary = compute_summary(responses, green_scores)

    return mean, std, green_scores, summary, results_df


def compute_green(response: str) -> float | None:
    sig_present, sig_errors = parse_error_counts(response, GreenCategories.SIGNIFICANT)
    matched_findings, _ = parse_error_counts(response, GreenCategories.MATCHED_FINDINDS)

    if matched_findings == 0:
        return 0

    if sig_present is None or matched_findings is None:
        return None

    assert sig_errors is not None
    return matched_findings / (matched_findings + sum(sig_errors))


def compute_error_count(response: str):
    _, sig_errors = parse_error_counts(response, GreenCategories.SIGNIFICANT)
    matched_findings, _ = parse_error_counts(response, GreenCategories.MATCHED_FINDINDS)
    assert sig_errors is not None
    return sig_errors + [matched_findings]


def compute_summary(responses: list[str], green_scores: list[float | None]):
    print("Computing summary ...")
    representative_sentences = get_representative_sentences(responses)
    accuracies = compute_accuracy(responses)
    mean = np.mean(green_scores)
    std = np.std(green_scores)

    summary = f"\n-------------{'MODEL NAME'}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
    for idx, sub_category in enumerate(GreenSubCategories):
        accuracy = accuracies[sub_category.value]
        sentences = representative_sentences[sub_category.value]
        summary += f"{sub_category.value}: {accuracy}. \n {sentences} \n\n"
    summary += "----------------------------------\n"

    return mean, std, summary


def compute_accuracy(responses: list[str]) -> dict[str, float]:
    counts = []
    for response in responses:
        _, sig_errors = parse_error_counts(response, GreenCategories.SIGNIFICANT)
        counts.append(sig_errors)

    counts = np.array(counts)

    dict_acc = {}
    for i, sub_category in enumerate(GreenSubCategories):
        error_counts = counts[:, i]
        accuracy = np.mean(error_counts == 0)
        dict_acc[sub_category.value] = accuracy

    return dict_acc


def get_representative_sentences(responses: list[str]) -> dict[str, list[str]]:
    list_sentences = []
    for response in responses:
        sentences = parse_error_sentences(response, GreenCategories.SIGNIFICANT)
        list_sentences.append(sentences)

    dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

    result_sentences_dict = {}

    for sub_category in GreenSubCategories:
        sentences = dict_sentences[sub_category.value]
        sentences = [i for i in sentences if i.strip() != ""]
        _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
        result_sentences_dict[sub_category.value] = sentences_of_largest_cluster

    return result_sentences_dict


def flatten_values_lists_of_list_dicts_to_dict(item):
    result = {}
    for i in item:
        if isinstance(i, list):
            i = i[0]
        for key, lists in i.items():
            if key not in result:
                result[key] = []
            result[key].extend(lists)

    return result
