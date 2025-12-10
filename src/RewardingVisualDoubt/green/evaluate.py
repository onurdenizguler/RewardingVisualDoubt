import re

from .domain import GreenCategories, GreenSubCategories


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


def compute_green(response: str) -> float | None:
    sig_present, sig_errors = parse_error_counts(response, GreenCategories.SIGNIFICANT)
    matched_findings, _ = parse_error_counts(response, GreenCategories.MATCHED_FINDINDS)

    if matched_findings == 0:
        return 0

    if sig_present is None or matched_findings is None:
        return None

    assert sig_errors is not None
    return matched_findings / (matched_findings + sum(sig_errors))
