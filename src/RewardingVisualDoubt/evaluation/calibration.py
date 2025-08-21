import dataclasses
import typing as t

import math
import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class RewardECEAndDistributionScore:
    ece: float
    conf_distribution_kl_divergence: float
    avg_reward: float


def std_score_normalized_within_bin_dispersion(
    bin_means: list[float],
    bin_stds: list[float],
    bin_counts: list[int] | None = None,
    eps=1e-12,
    equal_weight=False,
):
    bin_means_ = np.asarray(bin_means, dtype=float)
    bin_stds_ = np.asarray(bin_stds, dtype=float)
    B = len(bin_means_)
    if bin_counts is None:
        bin_counts_ = np.ones(B, dtype=float)
    else:
        bin_counts_ = np.asarray(bin_counts, dtype=float)

    if equal_weight:
        w = np.ones(B) / B
    else:
        w = bin_counts_ / (bin_counts_.sum() + eps)

    # per-bin theoretical max std given mean
    sigma_max = np.sqrt(np.clip(bin_means_ * (1.0 - bin_means_), 0.0, 0.25))
    d = np.zeros(B)
    mask = sigma_max > 0
    d[mask] = bin_stds_[mask] / (sigma_max[mask] + eps)
    d = np.clip(d, 0.0, 1.0)

    # final metric in [0,1]
    return float(np.sum(w * d))


def reward_ece_and_distribution_score_heuristic(
    reward_ece_and_distribution_score: RewardECEAndDistributionScore,
    n_bins: int,
    r_max_and_r_min: tuple[float, float],
    w_ece: float = 0.33,
    w_kl: float = 0.34,
    w_reward: float = 0.33,
    **kwargs,
) -> float:
    """Returns a scalar in [0, 1]; higher = better."""
    r_max, r_min = r_max_and_r_min
    ece_norm = reward_ece_and_distribution_score.ece  # already 0–1
    kl_norm = min(
        reward_ece_and_distribution_score.conf_distribution_kl_divergence / math.log(n_bins), 1.0
    )  # 0–1
    r_norm = max(
        min((reward_ece_and_distribution_score.avg_reward - r_min) / (r_max - r_min), 1.0), 0
    )  # 0–1

    score = (
        w_ece * (1.0 - ece_norm)  # small ECE → big score
        + w_kl * (1.0 - kl_norm)  # small KL  → big score
        + w_reward * r_norm  # large reward → big score
    )
    return score / (w_ece + w_kl + w_reward)  # keeps final range [0, 1]


def accuracy_std_reward_ece_and_distribution_score_heuristic(
    reward_ece_and_distribution_score: RewardECEAndDistributionScore,
    n_bins: int,
    r_max_and_r_min: tuple[float, float],
    accuracy_std: float,
    w_std: float = 0.25,
    w_ece: float = 0.25,
    w_kl: float = 0.25,
    w_reward: float = 0.25,
    **kwargs,
) -> float:
    """Returns a scalar in [0, 1]; higher = better."""
    r_max, r_min = r_max_and_r_min
    ece_norm = reward_ece_and_distribution_score.ece  # already 0–1
    kl_norm = min(
        reward_ece_and_distribution_score.conf_distribution_kl_divergence / math.log(n_bins), 1.0
    )  # 0–1
    r_norm = max(
        min((reward_ece_and_distribution_score.avg_reward - r_min) / (r_max - r_min), 1.0), 0
    )  # 0–1

    score = (
        w_std * (1.0 - accuracy_std)
        + w_ece * (1.0 - ece_norm)  # small ECE → big score
        + w_kl * (1.0 - kl_norm)  # small KL  → big score
        + w_reward * r_norm  # large reward → big score
    )
    return score / (w_std + w_ece + w_kl + w_reward)  # keeps final range [0, 1]


def compute_ece(avg_acc: list[float], counts: list[int]):
    ece = 0.0
    for i, (acc, count) in enumerate(zip(avg_acc, counts)):
        if count == 0:
            continue  # skip empty bins
        conf = i / (len(avg_acc) - 1)  # assuming bins like 0.0, 0.1, ..., 1.0
        ece += (count / sum(counts)) * abs(acc - conf)
    return ece


def binify_accuracies(
    confidences: list[int | None],
    accuracies: list[bool] | list[float | None],
) -> None | tuple[list[int], list[float]]:

    filtered = [(c, a) for c, a in zip(confidences, accuracies) if c is not None and a is not None]
    if not filtered:
        return None
    confidences_clean, accuracies_clean = t.cast(tuple[list[int], list[bool]], zip(*filtered))
    # Initialize bins
    bin_acc = {i: [] for i in range(11)}
    for c, a in zip(confidences_clean, accuracies_clean):
        if c > 10:
            # round to the nearest integer if confidence is greater than 10
            # Give a bit more chance to binning into 0 or 100 as they do not get much data if we round starting from 5 or 95
            if c > 92:
                c = 100
            if c < 8:
                c = 0
            c = round(c / 10)
        bin_acc[c].append(a)

    counts = [len(bin_acc[i]) for i in range(11)]
    avg_acc = [sum(bin_acc[i]) / len(bin_acc[i]) if bin_acc[i] else 0.0 for i in range(11)]
    return counts, avg_acc


def binify_accuracies_with_std(
    confidences: list[int | None],
    accuracies: list[bool] | list[float | None],
) -> None | tuple[list[int], list[float], list[float]]:
    filtered = [(c, a) for c, a in zip(confidences, accuracies) if c is not None and a is not None]
    if not filtered:
        return None
    confidences_clean, accuracies_clean = t.cast(tuple[list[int], list[float]], zip(*filtered))

    bin_acc = {i: [] for i in range(11)}
    for c, a in zip(confidences_clean, accuracies_clean):
        if c > 10:
            if c > 92:
                c = 100
            if c < 8:
                c = 0
            c = round(c / 10)
        bin_acc[c].append(a)

    counts = [len(bin_acc[i]) for i in range(11)]
    avg_acc = t.cast(list[float], [np.mean(bin_acc[i]) if bin_acc[i] else 0.0 for i in range(11)])
    std_acc = t.cast(list[float], [np.std(bin_acc[i]) if bin_acc[i] else 0.0 for i in range(11)])

    return counts, avg_acc, std_acc


def plot_calibration_curve(
    confidences: list[None | int],
    accuracies: list[bool] | list[float | None],
    plot_std: bool = False,
    ece: float | None = None,
    model_name: str | None = None,
):
    """
    Generate a confidence calibration plot (reliability diagram).

    Parameters:
        confidences (List[Optional[int | None]]): List of confidence scores (0–10 or 0-100 if granular), may contain None.
        is_answer_correct (List[bool | None]): List of booleans indicating prediction correctness.

    Returns:
        matplotlib.figure.Figure: The resulting plot as a matplotlib Figure.
    """

    if plot_std:
        results = binify_accuracies_with_std(confidences, accuracies)
        if results:
            (counts, avg_acc, std_acc) = results
    else:
        results = binify_accuracies(confidences, accuracies)
        if results:
            counts, avg_acc = results

    if not results:
        return

    if plot_std:

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            range(11),
            avg_acc,
            yerr=std_acc,
            fmt="o-",
            capsize=4,
            color="C0",  # main line color (matplotlib default blue)
            ecolor="red",  # error bar color
            elinewidth=1,
            alpha=0.6,  # make it faint
            label="Model Accuracy ± Std",
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(11), avg_acc, marker="o", label="Model Accuracy")

    title = f"Confidence Calibration Plot (Overall Accuracy: {sum(a for a in accuracies if a)/len(accuracies):.2f}, n={sum(counts)})"
    if model_name:
        title += "\n" + f"{model_name}"
    if ece:
        title += "\n ECE: " + f"{ece:.2f}"
    ax.plot([0, 10], [0.0, 1.0], "k--", label="Perfect Calibration")
    ax.set_xticks(range(11))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence Level (0-10)")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    for i, (acc, count) in enumerate(zip(avg_acc, counts)):
        ax.text(i, acc + 0.03, f"n={count}", ha="center", fontsize=8)

    # Annotate sample sizes
    for i, (acc, count) in enumerate(zip(avg_acc, counts)):
        ax.text(i, acc + 0.03, f"n={count}", ha="center", fontsize=8)

    plt.close(fig)  # Prevent automatic display
    return fig


def compute_confidence_distribution_metric(
    confidences: list[float], num_bins: int = 11, eps: float = 1e-8
) -> float:
    """
    A simple metric to quantify how far the model's predicted confidences
    are from a uniform distribution (higher = more collapsed).
    Here we use KL-divergence from uniform:
        D_KL( P || U ) = sum P[i] * log(P[i] / (1/num_bins))
    """

    counts, _ = np.histogram(confidences, bins=num_bins, range=(0.0, 1.0), density=False)
    total = counts.sum()
    if total == 0:
        return math.log(num_bins)  # no data → treat as worst case (only one bin is filled)

    P = counts / total
    U = np.ones_like(P) / num_bins

    kl = float(np.sum(P * np.log((P + eps) / (U + eps))))
    return kl
