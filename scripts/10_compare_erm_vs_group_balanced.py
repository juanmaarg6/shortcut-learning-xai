"""Compare final ERM and group-balanced Waterbirds test results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERM_PATH = PROJECT_ROOT / "results" / "metrics" / "erm" / "test_seed_summary.csv"

BALANCED_PATH = (
    PROJECT_ROOT / "results" / "metrics" / "group_balanced" / "test_seed_summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics" / "comparison"

METRICS = [
    "overall_accuracy",
    "worst_group_accuracy",
    "group_0_accuracy",
    "group_1_accuracy",
    "group_2_accuracy",
    "group_3_accuracy",
    "aligned_accuracy",
    "conflicting_accuracy",
    "shortcut_gap",
    "loss",
]


def validate_seed_alignment(
    erm: pd.DataFrame,
    balanced: pd.DataFrame,
) -> list[int]:
    """Ensure both methods contain exactly the same experimental seeds."""
    erm_seeds = sorted(erm["seed"].astype(int).tolist())
    balanced_seeds = sorted(balanced["seed"].astype(int).tolist())

    if erm_seeds != balanced_seeds:
        raise ValueError(
            "ERM and group-balanced results do not contain the same seeds. "
            f"ERM={erm_seeds}, balanced={balanced_seeds}"
        )

    return erm_seeds


def build_method_summary(
    erm: pd.DataFrame,
    balanced: pd.DataFrame,
) -> pd.DataFrame:
    """Build long-form mean/std summaries for both methods."""
    rows = []

    for method_name, frame in [
        ("ERM", erm),
        ("GroupBalancedERM", balanced),
    ]:
        for metric in METRICS:
            values = frame[metric]

            rows.append(
                {
                    "method": method_name,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "n_seeds": int(values.notna().sum()),
                }
            )

    return pd.DataFrame(rows)


def build_paired_deltas(
    erm: pd.DataFrame,
    balanced: pd.DataFrame,
) -> pd.DataFrame:
    """Compute balanced-minus-ERM deltas for each matched seed."""
    merged = erm.merge(
        balanced,
        on="seed",
        how="inner",
        suffixes=("_erm", "_balanced"),
        validate="one_to_one",
    )

    rows = []

    for row in merged.itertuples(index=False):
        seed = int(row.seed)

        for metric in METRICS:
            erm_value = float(getattr(row, f"{metric}_erm"))
            balanced_value = float(getattr(row, f"{metric}_balanced"))

            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "erm": erm_value,
                    "group_balanced": balanced_value,
                    "delta_balanced_minus_erm": (balanced_value - erm_value),
                }
            )

    return pd.DataFrame(rows)


def build_delta_summary(
    paired_deltas: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate paired seed deltas with mean and sample standard deviation."""
    return (
        paired_deltas.groupby("metric", sort=False)["delta_balanced_minus_erm"]
        .agg(
            mean_delta="mean",
            std_delta="std",
            n_seeds="count",
        )
        .reset_index()
    )


def print_comparison(
    method_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> None:
    """Print the most relevant ERM-vs-balanced comparison metrics."""
    focus_metrics = [
        "overall_accuracy",
        "worst_group_accuracy",
        "aligned_accuracy",
        "conflicting_accuracy",
        "shortcut_gap",
    ]

    print("ERM vs Group-Balanced ERM")
    print("=" * 96)

    for metric in focus_metrics:
        erm_row = method_summary[
            (method_summary["method"] == "ERM") & (method_summary["metric"] == metric)
        ].iloc[0]

        balanced_row = method_summary[
            (method_summary["method"] == "GroupBalancedERM")
            & (method_summary["metric"] == metric)
        ].iloc[0]

        delta_row = delta_summary[delta_summary["metric"] == metric].iloc[0]

        print(f"\n{metric}")
        print(
            f"  ERM:            "
            f"{100.0 * erm_row['mean']:6.2f}% "
            f"± {100.0 * erm_row['std']:5.2f} pp"
        )
        print(
            f"  Group-balanced: "
            f"{100.0 * balanced_row['mean']:6.2f}% "
            f"± {100.0 * balanced_row['std']:5.2f} pp"
        )
        print(
            f"  Paired delta:   "
            f"{100.0 * delta_row['mean_delta']:+6.2f} "
            f"± {100.0 * delta_row['std_delta']:5.2f} pp"
        )


def main() -> None:
    """Load both final test summaries and save their comparison."""
    if not ERM_PATH.is_file():
        raise FileNotFoundError(f"ERM test summary not found: {ERM_PATH}")

    if not BALANCED_PATH.is_file():
        raise FileNotFoundError(
            f"Group-balanced test summary not found: {BALANCED_PATH}"
        )

    erm = pd.read_csv(ERM_PATH)
    balanced = pd.read_csv(BALANCED_PATH)

    seeds = validate_seed_alignment(
        erm=erm,
        balanced=balanced,
    )

    method_summary = build_method_summary(
        erm=erm,
        balanced=balanced,
    )

    paired_deltas = build_paired_deltas(
        erm=erm,
        balanced=balanced,
    )

    delta_summary = build_delta_summary(
        paired_deltas=paired_deltas,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    method_summary_path = OUTPUT_DIR / "method_summary.csv"
    paired_deltas_path = OUTPUT_DIR / "paired_seed_deltas.csv"
    delta_summary_path = OUTPUT_DIR / "paired_delta_summary.csv"

    method_summary.to_csv(
        method_summary_path,
        index=False,
    )
    paired_deltas.to_csv(
        paired_deltas_path,
        index=False,
    )
    delta_summary.to_csv(
        delta_summary_path,
        index=False,
    )

    print(f"Seeds: {seeds}\n")
    print_comparison(
        method_summary=method_summary,
        delta_summary=delta_summary,
    )

    print("\nSaved")
    print("=" * 96)
    print(f"Method summary:       {method_summary_path}")
    print(f"Per-seed deltas:      {paired_deltas_path}")
    print(f"Paired delta summary: {delta_summary_path}")


if __name__ == "__main__":
    main()
