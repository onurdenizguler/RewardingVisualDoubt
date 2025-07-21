import re

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..domain import GreenCategories, GreenSubCategories


def parse_error_counts(
    text: str, category_: GreenCategories, for_reward=False
) -> tuple[int, list[int]] | tuple[None, None]:
    category = category_.value

    pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
    category_text = re.search(pattern, text, re.DOTALL)

    sum_counts = 0
    sub_counts = [0 for i in range(6)]

    if not category_text:
        if for_reward:
            return None, None
        return sum_counts, sub_counts
    if category_text.group(1).startswith("No"):
        return sum_counts, sub_counts

    if category == "Matched Findings":
        counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
        if len(counts) > 0:
            sum_counts = int(counts[0])
        return sum_counts, sub_counts
    else:
        sub_categories = [s.value.split(" ", 1)[0] + " " for s in GreenSubCategories]
        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            sub_categories = [f"({i})" + " " for i in range(1, len(GreenSubCategories) + 1)]

        for position, sub_category in enumerate(sub_categories):
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                    if len(count) > 0:
                        sub_counts[position] = int(count[0])
        return sum(sub_counts), sub_counts


def parse_error_sentences(response, category_: GreenCategories) -> dict[str, list[str]] | list[str]:

    category = category_.value
    pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
    category_text = re.search(pattern, response, re.DOTALL)
    sub_category_dict_sentences = {}
    for sub_category in GreenSubCategories:
        sub_category_dict_sentences[sub_category.value] = []

    if not category_text or category_text.group(1).startswith("No"):
        return sub_category_dict_sentences

    if category == "Matched Findings":
        return category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")

    matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

    if len(matches) == 0:
        matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
        sub_categories = [f"({i})" + " " for i in range(1, len(GreenSubCategories) + 1)]
    else:
        sub_categories = [s.value for s in GreenSubCategories]
    for position, sub_category in enumerate(sub_categories):
        for match in range(len(matches)):
            if matches[match].startswith(sub_category):
                sentences_list = matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                sub_category_dict_sentences[sub_categories[position]] = sentences_list

    return sub_category_dict_sentences
