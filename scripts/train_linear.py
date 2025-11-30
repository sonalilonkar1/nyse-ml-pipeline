#!/usr/bin/env python
"""
Train a (Ridge) Linear Regression model on normalized data.

Usage examples:

# Train Ridge with CV (default)
python scripts/train_linear.py --window 3 --fold 0

# Train ordinary LinearRegression and save
python scripts/train_linear.py --window 4 --fold 1 --model linear

# Train Ridge with explicit alpha
python scripts/train_linear.py --window 2 --fold 2 --model ridge --alpha 1.0

The script expects normalized data produced by `scripts/build_pipeline.py` at
`data/normalized/window_{window}/fold_{fold}/` and `data/normalized/window_{window}/X_eval.csv`.

Outputs:
- models/{model_name}_window_{window}_fold_{fold}.pkl
- artifacts/metrics_window_{window}_fold_{fold}.json
- reports/figs/predictions_window_{window}_fold_{fold}_[test|eval].png
"""

import argparse
import json
from pathlib import Path
import joblib
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).parent.parent
NORMALIZED_DIR = ROOT / 'data' / 'normalized'
MODELS_DIR = ROOT / 'models'
ARTIFACTS_DIR = ROOT / 'artifacts'
FIGS_DIR = ROOT / 'reports' / 'figs'

for d in (MODELS_DIR, ARTIFACTS_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_fold_data(window: int, fold: int):
    base = NORMALIZED_DIR / f'window_{window}' / f'fold_{fold}'
    X_train = pd.read_csv(base / 'X_train.csv')
    X_test = pd.read_csv(base / 'X_test.csv')
    y_train = pd.read_csv(base / 'y_train.csv')
    y_test = pd.read_csv(base / 'y_test.csv')

    # y may be single-column DataFrame; convert to 1d array
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    return X_train, X_test, y_train, y_test


def load_eval_data(window: int):
    base = NORMALIZED_DIR / f'window_{window}'
    X_eval = pd.read_csv(base / 'X_eval.csv')
    y_eval = pd.read_csv(base / 'y_eval.csv')
    return X_eval, y_eval.values.ravel()


def fit_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    # Use sqrt of MSE for RMSE to remain compatible with older scikit-learn
    rmse_test = float(np.sqrt(mean_squared_error(y_test, preds_test)))
    r2_test = r2_score(y_test, preds_test)
    return model, preds_test, rmse_test, r2_test


def plot_preds(y_true, y_pred, out_path, title):
    """Plot time-series true vs predicted with subsampling and save figure.

    Improvements made for clarity:
    - Subsample when series is long to avoid overplotting
    - Plot rolling mean if series is long
    - Larger figure size and higher DPI for readability
    - Annotate basic stats (RMSE/R2) if available in title
    """
    n = len(y_true)
    # Choose number of points to plot to avoid dense overlapping lines
    max_points = 2000
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
    else:
        idx = np.arange(n)

    plt.figure(figsize=(12, 5), dpi=200)
    plt.plot(idx, np.array(y_true)[idx], label='true', linewidth=1, alpha=0.9)
    plt.plot(idx, np.array(y_pred)[idx], label='pred', linewidth=1, alpha=0.8)

    # Overlay a rolling mean for smoother trend if the series is long
    try:
        if n >= 100:
            window = min(101, max(11, n // 100))
            true_roll = pd.Series(y_true).rolling(window=window, min_periods=1).mean().values
            pred_roll = pd.Series(y_pred).rolling(window=window, min_periods=1).mean().values
            plt.plot(idx, true_roll[idx], label=f'true_roll({window})', linewidth=2, color='tab:blue')
            plt.plot(idx, pred_roll[idx], label=f'pred_roll({window})', linewidth=2, color='tab:orange')
    except Exception:
        pass

    plt.legend()
    plt.title(title)
    plt.xlabel('sample (subsampled)')
    plt.ylabel('target')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter_and_residuals(y_true, y_pred, scatter_path, resid_path, title):
    """Create a scatter plot (true vs pred) and a residual histogram."""
    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(y_true, y_pred, s=3, alpha=0.3)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--', linewidth=1)
    plt.xlabel('true')
    plt.ylabel('pred')
    plt.title(title + ' (true vs pred)')
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # Residuals histogram
    resid = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(6, 4), dpi=200)
    plt.hist(resid, bins=100, alpha=0.8)
    plt.xlabel('residual (true - pred)')
    plt.title(title + ' (residuals)')
    plt.tight_layout()
    plt.savefig(resid_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=3, help='window size to use (2-5)')
    parser.add_argument('--fold', type=int, default=0, help='fold index (0-4)')
    parser.add_argument('--model', choices=['ridge', 'linear'], default='ridge')
    parser.add_argument('--alpha', type=float, default=None, help='alpha for Ridge (if provided uses Ridge(alpha=...))')
    parser.add_argument('--cv_alphas', nargs='+', type=float, default=[0.1, 1.0, 10.0, 100.0], help='alphas for RidgeCV')
    args = parser.parse_args()

    window = args.window
    fold = args.fold

    logger.info(f'Loading normalized data for window={window} fold={fold}...')
    X_train, X_test, y_train, y_test = load_fold_data(window, fold)
    X_eval, y_eval = load_eval_data(window)

    logger.info(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}, Eval shape: {X_eval.shape}')

    # Select model
    if args.model == 'linear':
        model = LinearRegression()
        model_name = 'linear_regression'
    else:
        if args.alpha is not None:
            model = Ridge(alpha=args.alpha)
            model_name = f'ridge_alpha_{args.alpha}'
        else:
            model = RidgeCV(alphas=args.cv_alphas, cv=5)
            model_name = f'ridge_cv'

    logger.info(f'Training model: {model_name}')

    trained_model, preds_test, rmse_test, r2_test = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

    logger.info(f'Test RMSE: {rmse_test:.4f}, Test R2: {r2_test:.4f}')

    # Evaluate on eval set if model exposes predict
    preds_eval = trained_model.predict(X_eval)
    # compute RMSE as sqrt of MSE instead of using `squared` kwarg
    rmse_eval = float(np.sqrt(mean_squared_error(y_eval, preds_eval)))
    r2_eval = r2_score(y_eval, preds_eval)
    logger.info(f'Eval RMSE: {rmse_eval:.4f}, Eval R2: {r2_eval:.4f}')

    # Save model
    model_file = MODELS_DIR / f'{model_name}_window_{window}_fold_{fold}.pkl'
    joblib.dump(trained_model, model_file)
    logger.info(f'Saved model to {model_file}')

    # Save metrics
    metrics = {
        'window': window,
        'fold': fold,
        'model': model_name,
        'test_rmse': float(rmse_test),
        'test_r2': float(r2_test),
        'eval_rmse': float(rmse_eval),
        'eval_r2': float(r2_eval),
    }
    metrics_file = ARTIFACTS_DIR / f'metrics_window_{window}_fold_{fold}.json'
    with open(metrics_file, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f'Saved metrics to {metrics_file}')

    # Save prediction plots
    fig_test = FIGS_DIR / f'predictions_window_{window}_fold_{fold}_test.png'
    fig_eval = FIGS_DIR / f'predictions_window_{window}_fold_{fold}_eval.png'

    plot_preds(y_test, preds_test, fig_test, f'Window {window} Fold {fold} - Test')
    plot_preds(y_eval, preds_eval, fig_eval, f'Window {window} - Eval')
    logger.info(f'Saved prediction plots to {fig_test} and {fig_eval}')

    logger.info('Training complete!')


if __name__ == '__main__':
    main()
