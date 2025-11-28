import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


N_FOLDS = 5
DATA_DIR = Path("data/normalized")


def load_fold_data(window: int, fold: int) -> tuple:
    fold_dir = DATA_DIR / f"window_{window}" / f"fold_{fold}"
    X_train = pd.read_csv(fold_dir / "X_train.csv")
    y_train = pd.read_csv(fold_dir / "y_train.csv").squeeze()
    X_test = pd.read_csv(fold_dir / "X_test.csv")
    y_test = pd.read_csv(fold_dir / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def run_cross_validation(
    model_config: dict, windows: list[int] = [2, 3, 4, 5]
) -> pd.DataFrame:
    results = []
    model_name = model_config["name"]
    model_params = model_config["params"]

    for window in windows:
        for fold in range(N_FOLDS):
            logger.info(f"Training {model_name} | {window=} | {fold=}")

            X_train, y_train, X_test, y_test = load_fold_data(window, fold)
            model = get_model(model_name, model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_pred, y_test)
            results.append(
                {
                    "model": model_name,
                    "window": window,
                    "fold": fold,
                    **metrics,
                }
            )

    return pd.DataFrame(results)
