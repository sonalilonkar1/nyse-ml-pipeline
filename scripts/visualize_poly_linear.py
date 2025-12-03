#!/usr/bin/env python
"""
Visualize polynomial regression (degree=2, alpha=10) vs linear regression on eval sets.

This script directly loads eval data, trains models, and generates comparison plots
without modifying the training pipeline.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models import get_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (ROOT already defined above)
NORMALIZED_DIR = ROOT / 'data' / 'normalized'
PLOTS_DIR = ROOT / 'experiments' / 'plots' / 'polynomial_regression'
REPORTS_DIR = ROOT / 'reports'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [2, 3, 4, 5]
FOLD = 0  # Use fold 0 for training


def load_data(window: int):
    """Load eval data and training data (fold 0) for a given window."""
    window_dir = NORMALIZED_DIR / f'window_{window}'
    fold_dir = window_dir / f'fold_{FOLD}'
    
    # Load eval data
    X_eval = pd.read_csv(window_dir / 'X_eval.csv')
    y_eval = pd.read_csv(window_dir / 'y_eval.csv').squeeze()
    
    # Load training data from fold 0
    X_train = pd.read_csv(fold_dir / 'X_train.csv')
    y_train = pd.read_csv(fold_dir / 'y_train.csv').squeeze()
    
    return X_train, y_train, X_eval, y_eval


def compute_metrics(y_true, y_pred):
    """Compute MSE, MAE, and R² metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }


def plot_predictions(y_true, y_pred_poly, y_pred_linear, window, out_path):
    """Plot predictions vs actual for both models."""
    n = len(y_true)
    max_points = 2000
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
    else:
        idx = np.arange(n)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=200)
    
    # Polynomial regression plot
    axes[0].plot(idx, np.array(y_true)[idx], label='Actual', linewidth=1.5, alpha=0.8, color='black')
    axes[0].plot(idx, np.array(y_pred_poly)[idx], label='Polynomial (degree=2, α=10)', 
                 linewidth=1.5, alpha=0.7, color='blue')
    axes[0].set_xlabel('Sample Index', fontsize=11)
    axes[0].set_ylabel('Price', fontsize=11)
    axes[0].set_title(f'Window {window} - Polynomial Regression Predictions vs Actual', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Linear regression plot
    axes[1].plot(idx, np.array(y_true)[idx], label='Actual', linewidth=1.5, alpha=0.8, color='black')
    axes[1].plot(idx, np.array(y_pred_linear)[idx], label='Linear Regression', 
                 linewidth=1.5, alpha=0.7, color='red')
    axes[1].set_xlabel('Sample Index', fontsize=11)
    axes[1].set_ylabel('Price', fontsize=11)
    axes[1].set_title(f'Window {window} - Linear Regression Predictions vs Actual', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prediction plot to {out_path}")


def plot_residuals(y_true, y_pred_poly, y_pred_linear, window, out_path):
    """Plot residual distributions and scatter plots for both models."""
    residuals_poly = y_true - y_pred_poly
    residuals_linear = y_true - y_pred_linear
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    
    # Polynomial residuals histogram
    axes[0, 0].hist(residuals_poly, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Residuals', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Polynomial Regression - Residual Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    
    # Linear residuals histogram
    axes[0, 1].hist(residuals_linear, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Residuals', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Linear Regression - Residual Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    
    # Polynomial residuals scatter
    n = len(y_pred_poly)
    max_points = 2000
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
    else:
        idx = np.arange(n)
    
    axes[1, 0].scatter(np.array(y_pred_poly)[idx], np.array(residuals_poly)[idx], 
                      alpha=0.5, s=10, color='blue')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[1, 0].set_xlabel('Predicted', fontsize=11)
    axes[1, 0].set_ylabel('Residuals', fontsize=11)
    axes[1, 0].set_title('Polynomial Regression - Residuals vs Predicted', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Linear residuals scatter
    axes[1, 1].scatter(np.array(y_pred_linear)[idx], np.array(residuals_linear)[idx], 
                       alpha=0.5, s=10, color='red')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('Predicted', fontsize=11)
    axes[1, 1].set_ylabel('Residuals', fontsize=11)
    axes[1, 1].set_title('Linear Regression - Residuals vs Predicted', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Window {window} - Residual Analysis', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved residual plot to {out_path}")


def plot_performance_comparison(metrics_df, out_path):
    """Plot performance comparison across windows."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    
    windows = metrics_df['window'].values
    x = np.arange(len(windows))
    width = 0.35
    
    # MSE comparison
    axes[0, 0].bar(x - width/2, metrics_df['mse_poly'], width, label='Polynomial', 
                   color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].bar(x + width/2, metrics_df['mse_linear'], width, label='Linear', 
                   color='red', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Window', fontsize=11)
    axes[0, 0].set_ylabel('MSE', fontsize=11)
    axes[0, 0].set_title('MSE Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(windows)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # MAE comparison
    axes[0, 1].bar(x - width/2, metrics_df['mae_poly'], width, label='Polynomial', 
                  color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].bar(x + width/2, metrics_df['mae_linear'], width, label='Linear', 
                  color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Window', fontsize=11)
    axes[0, 1].set_ylabel('MAE', fontsize=11)
    axes[0, 1].set_title('MAE Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(windows)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # R² comparison
    axes[1, 0].bar(x - width/2, metrics_df['r2_poly'], width, label='Polynomial', 
                   color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x + width/2, metrics_df['r2_linear'], width, label='Linear', 
                   color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Window', fontsize=11)
    axes[1, 0].set_ylabel('R²', fontsize=11)
    axes[1, 0].set_title('R² Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(windows)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # RMSE comparison
    axes[1, 1].bar(x - width/2, metrics_df['rmse_poly'], width, label='Polynomial', 
                   color='blue', alpha=0.7, edgecolor='black')
    axes[1, 1].bar(x + width/2, metrics_df['rmse_linear'], width, label='Linear', 
                   color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Window', fontsize=11)
    axes[1, 1].set_ylabel('RMSE', fontsize=11)
    axes[1, 1].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(windows)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Polynomial vs Linear Regression - Performance Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance comparison plot to {out_path}")


def plot_window_comparison(metrics_df, out_path):
    """Plot how metrics change across windows."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    
    windows = metrics_df['window'].values
    
    # MSE across windows
    axes[0, 0].plot(windows, metrics_df['mse_poly'], marker='o', linewidth=2, 
                    markersize=8, label='Polynomial', color='blue')
    axes[0, 0].plot(windows, metrics_df['mse_linear'], marker='s', linewidth=2, 
                    markersize=8, label='Linear', color='red')
    axes[0, 0].set_xlabel('Window Size', fontsize=11)
    axes[0, 0].set_ylabel('MSE', fontsize=11)
    axes[0, 0].set_title('MSE Across Windows', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE across windows
    axes[0, 1].plot(windows, metrics_df['mae_poly'], marker='o', linewidth=2, 
                    markersize=8, label='Polynomial', color='blue')
    axes[0, 1].plot(windows, metrics_df['mae_linear'], marker='s', linewidth=2, 
                    markersize=8, label='Linear', color='red')
    axes[0, 1].set_xlabel('Window Size', fontsize=11)
    axes[0, 1].set_ylabel('MAE', fontsize=11)
    axes[0, 1].set_title('MAE Across Windows', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² across windows
    axes[1, 0].plot(windows, metrics_df['r2_poly'], marker='o', linewidth=2, 
                    markersize=8, label='Polynomial', color='blue')
    axes[1, 0].plot(windows, metrics_df['r2_linear'], marker='s', linewidth=2, 
                    markersize=8, label='Linear', color='red')
    axes[1, 0].set_xlabel('Window Size', fontsize=11)
    axes[1, 0].set_ylabel('R²', fontsize=11)
    axes[1, 0].set_title('R² Across Windows', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMSE across windows
    axes[1, 1].plot(windows, metrics_df['rmse_poly'], marker='o', linewidth=2, 
                    markersize=8, label='Polynomial', color='blue')
    axes[1, 1].plot(windows, metrics_df['rmse_linear'], marker='s', linewidth=2, 
                    markersize=8, label='Linear', color='red')
    axes[1, 1].set_xlabel('Window Size', fontsize=11)
    axes[1, 1].set_ylabel('RMSE', fontsize=11)
    axes[1, 1].set_title('RMSE Across Windows', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Polynomial vs Linear Regression - Window Size Comparison', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved window comparison plot to {out_path}")


def main():
    """Main function to train models, evaluate, and generate visualizations."""
    logger.info("Starting polynomial vs linear regression visualization...")
    
    all_metrics = []
    
    for window in WINDOWS:
        logger.info(f"\nProcessing window {window}...")
        
        # Load data
        X_train, y_train, X_eval, y_eval = load_data(window)
        logger.info(f"  Loaded data: train shape {X_train.shape}, eval shape {X_eval.shape}")
        
        # Train polynomial regression model
        logger.info("  Training polynomial regression (degree=2, alpha=10)...")
        model_poly = get_model("polynomial_regression", {"degree": 2, "alpha": 10.0})
        model_poly.fit(X_train, y_train)
        y_pred_poly = model_poly.predict(X_eval)
        metrics_poly = compute_metrics(y_eval, y_pred_poly)
        logger.info(f"    Polynomial - MSE: {metrics_poly['mse']:.4f}, MAE: {metrics_poly['mae']:.4f}, R²: {metrics_poly['r2']:.4f}")
        
        # Train linear regression model
        logger.info("  Training linear regression...")
        model_linear = get_model("linear_regression", {})
        model_linear.fit(X_train, y_train)
        y_pred_linear = model_linear.predict(X_eval)
        metrics_linear = compute_metrics(y_eval, y_pred_linear)
        logger.info(f"    Linear - MSE: {metrics_linear['mse']:.4f}, MAE: {metrics_linear['mae']:.4f}, R²: {metrics_linear['r2']:.4f}")
        
        # Store metrics
        all_metrics.append({
            'window': window,
            'mse_poly': metrics_poly['mse'],
            'mae_poly': metrics_poly['mae'],
            'r2_poly': metrics_poly['r2'],
            'rmse_poly': metrics_poly['rmse'],
            'mse_linear': metrics_linear['mse'],
            'mae_linear': metrics_linear['mae'],
            'r2_linear': metrics_linear['r2'],
            'rmse_linear': metrics_linear['rmse'],
        })
        
        # Generate plots for this window
        plot_predictions(y_eval, y_pred_poly, y_pred_linear, window,
                        PLOTS_DIR / f'poly_linear_predictions_window_{window}.png')
        plot_residuals(y_eval, y_pred_poly, y_pred_linear, window,
                      PLOTS_DIR / f'poly_linear_residuals_window_{window}.png')
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Generate comparison plots
    logger.info("\nGenerating comparison plots...")
    plot_performance_comparison(metrics_df, PLOTS_DIR / 'poly_linear_performance_comparison.png')
    plot_window_comparison(metrics_df, PLOTS_DIR / 'poly_linear_window_comparison.png')
    
    # Save metrics to CSV
    metrics_csv = REPORTS_DIR / 'poly_linear_eval_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"\nSaved metrics to {metrics_csv}")
    
    logger.info("\nVisualization complete!")
    logger.info(f"All plots saved to {PLOTS_DIR}")
    logger.info(f"Metrics summary saved to {metrics_csv}")


if __name__ == '__main__':
    main()

