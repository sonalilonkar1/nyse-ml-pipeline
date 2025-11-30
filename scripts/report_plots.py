#!/usr/bin/env python
"""
Generate aggregate RMSE plots and Markdown summary from metrics artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FIGS_DIR = REPORTS_DIR / "figs"
SUMMARY_MD = REPORTS_DIR / "model_report.md"
PLOT_PATH = FIGS_DIR / "rmse_comparison.png"


def load_metrics() -> pd.DataFrame:
    records = []
    for file in ARTIFACTS_DIR.glob("metrics_*.json"):
        with file.open() as fh:
            data = json.load(fh)
        data["metrics_file"] = file.name
        records.append(data)

    if not records:
        raise FileNotFoundError("No metrics_*.json files found in artifacts/. Run training first.")

    df = pd.DataFrame(records)
    return df


def create_plot(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5), dpi=200)
    x_labels = summary["window"].astype(str) + "-" + summary["model"]
    plt.plot(x_labels, summary["test_rmse"], marker="o", label="test_rmse")
    plt.plot(x_labels, summary["eval_rmse"], marker="o", label="eval_rmse")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.legend()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.close()


def write_markdown(summary: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# Model Comparison", "", summary.to_markdown(index=False), "", f"![RMSE comparison]({PLOT_PATH.relative_to(ROOT)})", ""]
    SUMMARY_MD.write_text("\n".join(md))


def main() -> None:
    df = load_metrics()
    summary = (
        df.groupby(["model", "window"])[["test_rmse", "eval_rmse"]]
        .mean()
        .reset_index()
        .sort_values("eval_rmse")
    )
    create_plot(summary)
    write_markdown(summary)
    print(f"Saved plot to {PLOT_PATH} and markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()

