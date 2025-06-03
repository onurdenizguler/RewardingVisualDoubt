import matplotlib.pyplot as plt


def plot_calibration_curve(confidences: list[None | int], is_answer_correct: list[bool]):
    """
    Generate a confidence calibration plot (reliability diagram).

    Parameters:
        confidences (List[Optional[int | None]]): List of confidence scores (0–10), may contain None.
        is_answer_correct (List[bool | None]): List of booleans indicating prediction correctness.

    Returns:
        matplotlib.figure.Figure: The resulting plot as a matplotlib Figure.
    """
    filtered = [(c, a) for c, a in zip(confidences, is_answer_correct) if c is not None]
    if not filtered:
        return None

    confidences_clean, accuracies_clean = zip(*filtered)

    # Initialize bins
    bin_acc = {i: [] for i in range(11)}
    for c, a in zip(confidences_clean, accuracies_clean):
        bin_acc[c].append(a)

    # Compute average accuracy and counts per bin
    avg_acc = [sum(bin_acc[i]) / len(bin_acc[i]) if bin_acc[i] else 0.0 for i in range(11)]
    counts = [len(bin_acc[i]) for i in range(11)]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(11), avg_acc, marker="o", label="Model Accuracy")
    ax.plot([0, 10], [0.0, 1.0], "k--", label="Perfect Calibration")

    ax.set_xticks(range(11))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence Level (0–10)")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(
        f"Confidence Calibration Plot (Overall Accuracy: {sum(is_answer_correct)/len(is_answer_correct)})"
    )
    ax.grid(True)
    ax.legend()

    # Annotate sample sizes
    for i, (acc, count) in enumerate(zip(avg_acc, counts)):
        ax.text(i, acc + 0.03, f"n={count}", ha="center", fontsize=8)

    plt.close(fig)  # Prevent automatic display
    return fig
