#!/usr/bin/env python
"""
Train non-linear regressors (RandomForest, GradientBoosting, MLP) on normalized data.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "data" / "normalized"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


def _ensure_dirs() -> None:
    for directory in (MODELS_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_data(window: int, fold: int) -> Tuple[pd.DataFrame, ...]:
    base = NORMALIZED_DIR / f"window_{window}"
    fold_dir = base / f"fold_{fold}"

    X_train = pd.read_csv(fold_dir / "X_train.csv")
    y_train = pd.read_csv(fold_dir / "y_train.csv").values.ravel()
    X_test = pd.read_csv(fold_dir / "X_test.csv")
    y_test = pd.read_csv(fold_dir / "y_test.csv").values.ravel()
    X_eval = pd.read_csv(base / "X_eval.csv")
    y_eval = pd.read_csv(base / "y_eval.csv").values.ravel()

    return X_train, y_train, X_test, y_test, X_eval, y_eval


def compute_metrics(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds))
    return {"rmse": rmse, "r2": r2}


def parse_params(param_json: str) -> Dict:
    if not param_json:
        return {}
    try:
        return json.loads(param_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --params: {param_json}") from exc


def build_model(name: str, params: Dict):
    models = {
        "rf": RandomForestRegressor,
        "gbr": GradientBoostingRegressor,
        "mlp": MLPRegressor,
    }
    return models[name](**params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tree/NN regressors on normalized data.")
    parser.add_argument("--window", type=int, required=True, help="Window size (2-5).")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4).")
    parser.add_argument(
        "--model",
        choices=["rf", "gbr", "mlp"],
        required=True,
        help="Model type: rf (RandomForest), gbr (GradientBoosting), mlp (MLPRegressor).",
    )
    parser.add_argument(
        "--params",
        default="{}",
        help='JSON string of sklearn estimator kwargs (e.g. \'{"n_estimators":500}\').',
    )
    args = parser.parse_args()

    params = parse_params(args.params)
    _ensure_dirs()

    X_train, y_train, X_test, y_test, X_eval, y_eval = load_data(args.window, args.fold)
    model = build_model(args.model, params)

    LOGGER.info("Training %s with params=%s", args.model, params)
    model.fit(X_train, y_train)

    test_metrics = compute_metrics(model, X_test, y_test)
    eval_metrics = compute_metrics(model, X_eval, y_eval)
    LOGGER.info(
        "Test RMSE %.4f R2 %.4f | Eval RMSE %.4f R2 %.4f",
        test_metrics["rmse"],
        test_metrics["r2"],
        eval_metrics["rmse"],
        eval_metrics["r2"],
    )

    model_name = f"{args.model}_window_{args.window}_fold_{args.fold}"
    joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")

    metrics = {
        "window": args.window,
        "fold": args.fold,
        "model": args.model,
        "params": params,
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "eval_rmse": eval_metrics["rmse"],
        "eval_r2": eval_metrics["r2"],
    }
    with (ARTIFACTS_DIR / f"metrics_{model_name}.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()

