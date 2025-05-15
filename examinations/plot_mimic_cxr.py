from RewardingVisualDoubt import dataset
from RewardingVisualDoubt.dataset import mimic_cxr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_num_datapoints_per_disease(df: pd.DataFrame, split: dataset.DatasetSplit | None = None):
    """
    Plot the number of datapoints per disease in the MIMIC-CXR dataset.
    Args:
        df (pd.DataFrame): DataFrame containing the melt MIMIC-CXR dataset with columns 'disease' and 'label'.
    """
    # Create the plot
    plt.figure(figsize=(12, 6))

    if split:
        df = df[df["split"] == split.value].copy()

    sns.countplot(data=df, x="disease", hue="label", order=sorted(df["disease"].unique()))

    # Customize the plot
    plt.xticks(rotation=45)
    plt.title(
        f"Distribution of Disease Labels (True/False) per Disease (Split: {split.value if split else 'All'})"
    )
    plt.xlabel("Disease")
    plt.ylabel("Count")
    plt.legend(title="Label", labels=["False", "True"])
    plt.tight_layout()

    # Display the plot
    plt.show()


def create_binary_qa_dataset_report(df: pd.DataFrame):
    """
    Args:
        df (pd.DataFrame): DataFrame containing the melt MIMIC-CXR dataset with columns 'disease' and 'label'.
    """

    report = {}
    splits = df["split"].unique().tolist()
    if len(splits) > 1:
        splits.append("all")

    for split in splits:
        if split == "all":
            subset = df
        else:
            subset = df[df["split"] == split]

        disease_data = {}
        for disease in mimic_cxr.ChexpertFinding:
            if disease == mimic_cxr.ChexpertFinding.NO_FINDING:
                continue
            disease_subset = subset[subset["disease"] == disease.value]
            total = len(disease_subset)

            true_count = disease_subset["label"].sum()
            false_count = total - true_count

            percent_true = (true_count / total * 100) if total > 0 else 0
            percent_false = (false_count / total * 100) if total > 0 else 0

            disease_data[disease.value] = {
                "count": total,
                "true_count": true_count,
                "false_count": false_count,
                "percent_true": round(percent_true, 2),
                "percent_false": round(percent_false, 2),
            }

        report[split] = pd.DataFrame.from_dict(disease_data, orient="index")

    return report
