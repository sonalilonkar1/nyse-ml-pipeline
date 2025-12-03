#!/usr/bin/env python
"""
Data normalization pipeline builder.

This script creates a scikit-learn Pipeline with StandardScaler to normalize
the processed stock data. It fits the scaler on training data and applies it
consistently across train/test/eval sets for all windows and folds.

The fitted pipeline is saved for reuse, and normalized datasets are saved
to data/normalized/ for model training.
"""

import sys
import logging
from pathlib import Path
import joblib

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
WINDOW_SIZES = [2, 3, 4, 5]
NUM_FOLDS = 5
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "full_processed"
NORMALIZED_DATA_DIR = Path(__file__).parent.parent / "data" / "normalized"


def create_pipeline():
    """Create a pipeline with StandardScaler."""
    pipeline = Pipeline([("scaler", StandardScaler())])
    return pipeline


def normalize_window_fold(window_size, fold, pipeline, fit=False):
    """
    Normalize X and y data for a specific window and fold.

    Args:
        window_size: Size of the sliding window (2-5)
        fold: Fold number (0-4)
        pipeline: Fitted or unfitted pipeline
        fit: If True, fit the pipeline on training data first

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) normalized, or
        (X_train, X_test, y_train, y_test, updated_pipeline) if fit=True
    """
    # Load raw data
    X_train_file = PROCESSED_DATA_DIR / f"X_train_window_{window_size}_fold_{fold}.csv"
    X_test_file = PROCESSED_DATA_DIR / f"X_test_window_{window_size}_fold_{fold}.csv"
    y_train_file = PROCESSED_DATA_DIR / f"y_train_window_{window_size}_fold_{fold}.csv"
    y_test_file = PROCESSED_DATA_DIR / f"y_test_window_{window_size}_fold_{fold}.csv"

    X_train = pd.read_csv(X_train_file)
    X_test = pd.read_csv(X_test_file)
    y_train = pd.read_csv(y_train_file)
    y_test = pd.read_csv(y_test_file)

    # Fit pipeline on training data if requested
    if fit:
        logger.info(f"  Fitting pipeline on fold {fold} training data...")
        pipeline.fit(X_train)

    # Transform data
    X_train_normalized = pipeline.transform(X_train)
    X_test_normalized = pipeline.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)

    if fit:
        return X_train_normalized, X_test_normalized, y_train, y_test, pipeline
    else:
        return X_train_normalized, X_test_normalized, y_train, y_test


def normalize_eval_data(window_size, pipeline):
    """
    Normalize eval data using a fitted pipeline.

    Args:
        window_size: Size of the sliding window (2-5)
        pipeline: Fitted pipeline

    Returns:
        Tuple of (X_eval_normalized, y_eval, dates_eval)
    """
    X_eval = pd.read_csv(PROCESSED_DATA_DIR / f"X_eval_window_{window_size}.csv")
    y_eval = pd.read_csv(PROCESSED_DATA_DIR / f"y_eval_window_{window_size}.csv")
    dates_eval = pd.read_csv(
        PROCESSED_DATA_DIR / f"dates_eval_window_{window_size}.csv"
    )

    X_eval_normalized = pipeline.transform(X_eval)
    X_eval_normalized = pd.DataFrame(X_eval_normalized, columns=X_eval.columns)

    return X_eval_normalized, y_eval, dates_eval


def main():
    logger.info("Starting data normalization pipeline...")
    logger.info(f"Input data directory: {PROCESSED_DATA_DIR}")
    logger.info(f"Output directory: {NORMALIZED_DATA_DIR}")

    # Create output directories
    NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Process each window size
    for window_size in WINDOW_SIZES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing window size: {window_size}")
        logger.info(f"{'=' * 60}")

        window_dir = NORMALIZED_DATA_DIR / f"window_{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)

        # Process each fold
        for fold in range(NUM_FOLDS):
            logger.info(f"\nProcessing fold {fold}...")

            # Create a fresh pipeline for this fold
            pipeline = create_pipeline()

            # Normalize this fold's data
            X_train_norm, X_test_norm, y_train, y_test, pipeline = (
                normalize_window_fold(window_size, fold, pipeline, fit=True)
            )

            # Save normalized train/test data
            fold_dir = window_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            X_train_norm.to_csv(fold_dir / "X_train.csv", index=False)
            X_test_norm.to_csv(fold_dir / "X_test.csv", index=False)
            y_train.to_csv(fold_dir / "y_train.csv", index=False)
            y_test.to_csv(fold_dir / "y_test.csv", index=False)

            # Save the fitted pipeline for this fold
            joblib.dump(pipeline, fold_dir / "scaler_pipeline.pkl")

            logger.info(f"  Saved fold {fold} data and pipeline")

        # Normalize eval data (use pipeline from fold 0 as representative)
        logger.info(f"\nNormalizing eval data for window {window_size}...")
        pipeline_fold0 = joblib.load(window_dir / "fold_0" / "scaler_pipeline.pkl")
        X_eval_norm, y_eval, dates_eval = normalize_eval_data(
            window_size, pipeline_fold0
        )

        X_eval_norm.to_csv(window_dir / "X_eval.csv", index=False)
        y_eval.to_csv(window_dir / "y_eval.csv", index=False)
        dates_eval.to_csv(window_dir / "dates_eval.csv", index=False)
        logger.info(f"  Saved eval data and dates")

    logger.info(f"\n{'=' * 60}")
    logger.info("Normalization pipeline complete!")
    logger.info(f"Normalized data saved to: {NORMALIZED_DATA_DIR}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
