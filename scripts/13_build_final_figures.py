"""Build the final portfolio figures from the completed Waterbirds experiments."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPARISON_DIR = PROJECT_ROOT / "results" / "metrics" / "comparison"

METHOD_SUMMARY_PATH = COMPARISON_DIR / "method_summary.csv"

DELTA_SUMMARY_PATH = COMPARISON_DIR / "paired_delta_summary.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "portfolio"

COPY_SOURCES = {
    (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "dataset"
        / "training_group_distribution.png"
    ): "01_training_group_distribution.png",
    (
        PROJECT_ROOT / "reports" / "figures" / "dataset" / "group_examples.png"
    ): "02_group_examples.png",
    (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "xai"
        / "gradcam_erm_vs_group_balanced.png"
    ): "06_gradcam_comparison.png",
}

METHOD_LABELS = {
    "ERM": "ERM",
    "GroupBalancedERM": "Group-balanced",
}

ACCURACY_METRICS = [
    ("overall_accuracy", "Overall"),
    ("worst_group_accuracy", "Worst group"),
    ("aligned_accuracy", "Aligned"),
    ("conflicting_accuracy", "Conflicting"),
]

DELTA_METRICS = [
    ("overall_accuracy", "Overall"),
    ("worst_group_accuracy", "Worst group"),
    ("aligned_accuracy", "Aligned"),
    ("conflicting_accuracy", "Conflicting"),
    ("shortcut_gap", "Shortcut gap"),
]


def require_file(path: Path) -> None:
    """Raise a clear error when a required experiment artifact is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def copy_existing_figures() -> None:
    """Copy the already-validated dataset and Grad-CAM figures."""
    for source, output_name in COPY_SOURCES.items():
        require_file(source)
        destination = OUTPUT_DIR / output_name
        shutil.copy2(source, destination)
        print(f"Saved: {destination}")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load final method summaries and paired seed deltas."""
    require_file(METHOD_SUMMARY_PATH)
    require_file(DELTA_SUMMARY_PATH)

    method_summary = pd.read_csv(METHOD_SUMMARY_PATH)
    delta_summary = pd.read_csv(DELTA_SUMMARY_PATH)

    return method_summary, delta_summary


def build_accuracy_comparison(
    method_summary: pd.DataFrame,
) -> None:
    """Plot final mean ± std accuracy metrics for both methods."""
    methods = ["ERM", "GroupBalancedERM"]
    x = np.arange(len(ACCURACY_METRICS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for method_index, method in enumerate(methods):
        means = []
        stds = []

        for metric, _ in ACCURACY_METRICS:
            row = method_summary[
                (method_summary["method"] == method)
                & (method_summary["metric"] == metric)
            ]

            if len(row) != 1:
                raise RuntimeError(f"Expected one summary row for {method}/{metric}.")

            means.append(100.0 * float(row.iloc[0]["mean"]))
            stds.append(100.0 * float(row.iloc[0]["std"]))

        offset = (method_index - (len(methods) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=METHOD_LABELS[method],
        )

        for bar, value in zip(bars, means, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Waterbirds Test Performance: ERM vs Group-Balanced ERM")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(
        x,
        [label for _, label in ACCURACY_METRICS],
    )
    ax.set_ylim(65, 102)
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "03_accuracy_comparison.png"
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def build_shortcut_gap(
    method_summary: pd.DataFrame,
) -> None:
    """Plot the final shortcut gap, where lower is better."""
    methods = ["ERM", "GroupBalancedERM"]
    means = []
    stds = []

    for method in methods:
        row = method_summary[
            (method_summary["method"] == method)
            & (method_summary["metric"] == "shortcut_gap")
        ]

        if len(row) != 1:
            raise RuntimeError(f"Expected one shortcut-gap row for {method}.")

        means.append(100.0 * float(row.iloc[0]["mean"]))
        stds.append(100.0 * float(row.iloc[0]["std"]))

    labels = [METHOD_LABELS[method] for method in methods]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=5,
    )

    for bar, value in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{value:.2f} pp",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Shortcut Gap on the Held-Out Test Set")
    ax.set_ylabel("Aligned accuracy − conflicting accuracy (pp)")
    ax.set_ylim(
        0,
        max(means) + max(stds) + 4,
    )
    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.text(
        0.5,
        0.01,
        "Lower is better: a smaller gap indicates less performance dependence "
        "on whether bird type and background are aligned.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.045, 1.0, 1.0])

    output_path = OUTPUT_DIR / "04_shortcut_gap.png"
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def build_paired_delta_plot(
    delta_summary: pd.DataFrame,
) -> None:
    """Plot balanced-minus-ERM paired deltas across the three matched seeds."""
    values = []
    errors = []
    labels = []

    for metric, label in DELTA_METRICS:
        row = delta_summary[delta_summary["metric"] == metric]

        if len(row) != 1:
            raise RuntimeError(f"Expected one paired-delta row for {metric}.")

        values.append(100.0 * float(row.iloc[0]["mean_delta"]))
        errors.append(100.0 * float(row.iloc[0]["std_delta"]))
        labels.append(label)

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(
        values,
        y,
        xerr=errors,
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_yticks(
        y,
        labels,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Paired delta: Group-balanced − ERM (percentage points)")
    ax.set_title("Effect of Group-Balanced Training Across Matched Seeds")
    ax.grid(
        axis="x",
        alpha=0.2,
    )

    for index, value in enumerate(values):
        offset = 10 if value >= 0 else -10
        alignment = "left" if value >= 0 else "right"

        ax.annotate(
            f"{value:+.2f} pp",
            xy=(value, index),
            xytext=(offset, -12),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "05_paired_deltas.png"
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Build all final portfolio figures."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    method_summary, delta_summary = load_results()

    copy_existing_figures()
    build_accuracy_comparison(method_summary)
    build_shortcut_gap(method_summary)
    build_paired_delta_plot(delta_summary)

    print("\nFinal portfolio figures completed.")


if __name__ == "__main__":
    main()
