from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_config: dict,
    experiment_name: str,
    results_dir: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = results_dir / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_config["name"]
    model_dir = experiment_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(model_dir / "results.csv", index=False)
    summary_df.to_csv(model_dir / "results_summary.csv", index=False)

    with open(model_dir / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["model", "window"])
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )
