import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import get_model

logger = logging.getLogger(__name__)

N_FOLDS = 5
DATA_DIR = Path("data/normalized")


def load_fold_data(window: int, fold: int) -> tuple:
    """Load train/test data for a specific fold."""
    fold_dir = DATA_DIR / f"window_{window}" / f"fold_{fold}"
    X_train = pd.read_csv(fold_dir / "X_train.csv")
    y_train = pd.read_csv(fold_dir / "y_train.csv").squeeze()
    X_test = pd.read_csv(fold_dir / "X_test.csv")
    y_test = pd.read_csv(fold_dir / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def load_full_train_data(window: int) -> tuple:
    """Load full training data for a window."""
    window_dir = DATA_DIR / f"window_{window}"
    X_train = pd.read_csv(window_dir / "X_train.csv")
    y_train = pd.read_csv(window_dir / "y_train.csv").squeeze()
    return X_train, y_train


def load_eval_data(window: int) -> tuple:
    """Load evaluation data for a window."""
    window_dir = DATA_DIR / f"window_{window}"
    X_eval = pd.read_csv(window_dir / "X_eval.csv")
    y_eval = pd.read_csv(window_dir / "y_eval.csv").squeeze()
    dates_eval = pd.read_csv(window_dir / "dates_eval.csv")
    return X_eval, y_eval, dates_eval


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


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


def run_cross_validation(
    model_config: dict,
    windows: list[int] = [2, 3, 4, 5],
) -> pd.DataFrame:
    """Run k-fold CV across all windows, return summary only."""
    results = []
    model_name = model_config["name"]
    model_params = model_config["params"]

    for window in windows:
        for fold in range(N_FOLDS):
            logger.debug(f"CV {model_name} | {window=} | {fold=}")

            X_train, y_train, X_test, y_test = load_fold_data(window, fold)

            model = get_model(model_name, model_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            results.append(
                {
                    "model": model_name,
                    "window": window,
                    "fold": fold,
                    **metrics,
                }
            )

    results_df = pd.DataFrame(results)
    return summarize_results(results_df)


def run_training(
    model_config: dict,
    windows: list[int] = [2, 3, 4, 5],
    experiment_dir: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Train on full training set for each window, save models."""
    results = []
    model_paths = []
    model_name = model_config["name"]
    model_params = model_config["params"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_dir = experiment_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        logger.debug(f"Training {model_name} | {window=}")

        X_train, y_train = load_full_train_data(window)

        model = get_model(model_name, model_params)
        model.fit(X_train, y_train)

        # Save model
        model_path = train_dir / f"model_window_{window}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        model_paths.append(model_path)
        logger.debug(f"Saved model to {model_path}")

        # Evaluate on training set for sanity check
        y_pred = model.predict(X_train)
        metrics = compute_metrics(y_train, y_pred)

        results.append(
            {
                "model": model_name,
                "window": window,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    summary_df = results_df.copy()  # No aggregation needed, one row per window
    return results_df, summary_df, model_paths


def get_latest_model_path(train_dir: Path, window: int) -> Path:
    """Find most recent model for a given window."""
    pattern = f"model_window_{window}_*.pkl"
    model_files = sorted(train_dir.glob(pattern))
    if not model_files:
        raise FileNotFoundError(f"No model found for window {window} in {train_dir}")
    return model_files[-1]  # Most recent by timestamp


def run_evaluation(
    model_config: dict,
    windows: list[int] = [2, 3, 4, 5],
    experiment_dir: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trained models and evaluate on eval set."""
    results = []
    model_name = model_config["name"]
    train_dir = experiment_dir / "train"

    for window in windows:
        logger.debug(f"Evaluating {model_name} | {window=}")

        model_path = get_latest_model_path(train_dir, window)
        model = joblib.load(model_path)
        logger.debug(f"Loaded model from {model_path}")

        X_eval, y_eval, dates_eval = load_eval_data(window)

        y_pred = model.predict(X_eval)
        metrics = compute_metrics(y_eval, y_pred)

        results.append(
            {
                "model": model_name,
                "window": window,
                "model_path": str(model_path),
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    summary_df = results_df.drop(columns=["model_path"])
    return results_df, summary_df
