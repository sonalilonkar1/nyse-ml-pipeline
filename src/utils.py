from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_cv_summary(
    summary_df: pd.DataFrame,
    model_config: dict,
    experiment_dir: Path,
) -> None:
    """Save CV summary with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = experiment_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(cv_dir / f"summary_{timestamp}.csv", index=False)

    with open(cv_dir / f"config_{timestamp}.yaml", "w") as f:
        yaml.dump(model_config, f)


def save_train_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_config: dict,
    experiment_dir: Path,
) -> None:
    """Save training results and summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir = experiment_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(train_dir / f"results_{timestamp}.csv", index=False)
    summary_df.to_csv(train_dir / f"summary_{timestamp}.csv", index=False)

    with open(train_dir / f"config_{timestamp}.yaml", "w") as f:
        yaml.dump(model_config, f)


def save_eval_results(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    experiment_dir: Path,
) -> None:
    """Save evaluation results and summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = experiment_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(eval_dir / f"results_{timestamp}.csv", index=False)
    summary_df.to_csv(eval_dir / f"summary_{timestamp}.csv", index=False)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across folds/windows."""
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
