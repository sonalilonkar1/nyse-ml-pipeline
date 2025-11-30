#!/usr/bin/env python
"""
Grid search helper for tuning regressors on a single window/fold split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "data" / "normalized"
ARTIFACTS_DIR = ROOT / "artifacts"

MODELS = {
    "ridge": Ridge,
    "gbr": GradientBoostingRegressor,
    "mlp": MLPRegressor,
}


def load_train_split(window: int, fold: int):
    fold_dir = NORMALIZED_DIR / f"window_{window}" / f"fold_{fold}"
    X = pd.read_csv(fold_dir / "X_train.csv")
    y = pd.read_csv(fold_dir / "y_train.csv").values.ravel()
    return X, y


def json_grid(param_grid: str):
    try:
        parsed = json.loads(param_grid)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --param-grid: {param_grid}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--param-grid must decode to a dict")
    return parsed


def serialize(obj):
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search hyperparameters.")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--param-grid", required=True, help='JSON dict, e.g. \'{"alpha":[0.1,1,10]}\'.')
    parser.add_argument("--cv", type=int, default=5, help="Number of CV splits inside GridSearchCV.")
    args = parser.parse_args()

    X, y = load_train_split(args.window, args.fold)
    estimator = MODELS[args.model]()
    grid = json_grid(args.param_grid)

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    search = GridSearchCV(estimator, grid, cv=args.cv, scoring=scorer, n_jobs=-1, refit=True)
    search.fit(X, y)

    result = {
        "window": args.window,
        "fold": args.fold,
        "model": args.model,
        "best_params": search.best_params_,
        "best_score_neg_mse": float(search.best_score_),
        "cv_results": {k: [serialize(v) for v in val] for k, val in search.cv_results_.items()},
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"grid_{args.model}_window_{args.window}_fold_{args.fold}.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"Saved grid search results to {out_path}")


if __name__ == "__main__":
    main()

