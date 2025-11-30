#!/usr/bin/env python
"""
Aggregate per-run metric JSON files into a CSV summary and print window averages.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
SUMMARY_CSV = REPORTS_DIR / "metrics_summary.csv"


def collect_metrics() -> pd.DataFrame:
    records = []
    for metrics_file in ARTIFACTS_DIR.glob("metrics_window_*_fold_*.json"):
        with metrics_file.open() as fh:
            metrics = json.load(fh)
        metrics["metrics_file"] = metrics_file.name
        records.append(metrics)

    if not records:
        raise FileNotFoundError("No metrics JSON files found in artifacts/. Run training first.")

    df = pd.DataFrame(records).sort_values(["window", "fold"]).reset_index(drop=True)
    return df


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_metrics()
    df.to_csv(SUMMARY_CSV, index=False)

    grouped = (
        df.groupby("window")[["test_rmse", "test_r2", "eval_rmse", "eval_r2"]]
        .mean()
        .round(4)
    )

    print("Per-window mean metrics:")
    print(grouped)
    print(f"\nSaved detailed metrics to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

