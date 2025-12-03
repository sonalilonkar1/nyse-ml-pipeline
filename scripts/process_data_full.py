#!/usr/bin/env python

import sys
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)
logging.basicConfig(level=sys.argv[1] if len(sys.argv) > 1 else "INFO")

FEATURES = ["open", "close", "low", "high", "volume"]

TRAIN_RATIO = 0.8
EVAL_RATIO = 1 - TRAIN_RATIO

MIN_WINDOW_SIZE = 2
MAX_WINDOW_SIZE = 5

NUM_CROSS_VAL_SPLITS = 5


def split_with_window_size(df, size, keep_date=False):
    df["target"] = df.groupby("symbol")["close"].pct_change(-1)
    logger.debug(df.head())

    logger.debug(f"Processing features for window size {size}...")
    for i in range(1, size + 1):
        for feat in FEATURES:
            df[f"{feat}_t-{i}"] = df.groupby("symbol")[feat].shift(i)
    logger.debug(f"df with new features:\n{df.head(6)}")
    logger.debug(f"df target values\n{df['target'].head(6)}")

    logger.debug(f"Creating final df for window size {size}")
    df_temp = df.dropna()

    exclude_cols = ["symbol"] + FEATURES
    if not keep_date:
        exclude_cols.append("date")

    X = df_temp[[col for col in df_temp.columns if col not in exclude_cols]]
    logger.debug(f"final df:\n{X.head(6)}")

    return X


def main():
    logger.debug("Starting process_data_full.py")
    logger.info("Loading data...")

    logger.debug("Attatching to raw data directory")
    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    logger.debug("Attatching to full processed data directory")
    processed_dir = Path(__file__).parent.parent / "data" / "full_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Attatching to prices.csv")
    prices = raw_dir / "prices.csv"

    logger.debug("Converting prices.csv to pandas df")
    df = pd.read_csv(prices)
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    logger.debug(f"Displaying head:\n{df.head()}")
    logger.debug(f"Displaying info:\n{df.info()}")

    logger.info("Splitting into train and eval...")
    split_date = df["date"].quantile(TRAIN_RATIO)
    train = df[df["date"] < split_date].sort_values(["date"])
    eval = df[df["date"] >= split_date].sort_values(["date"])
    logger.debug(f"Train set tail:\n{train.tail()}")
    logger.debug(f"Eval set tail:\n{eval.tail()}")

    logger.info("Creating multiple windows into data...")
    for window_size in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
        logger.debug(f"Creating window {window_size} eval set...")

        logger.debug(f"Copying eval for window {window_size}...")
        df_temp = eval.copy()
        X_eval = split_with_window_size(df_temp, window_size, keep_date=True)

        logger.debug(
            f"Splitting eval window {window_size} into features and target dfs..."
        )
        y_eval = X_eval["target"]
        dates_eval = X_eval["date"]
        X_eval = X_eval.drop(["target", "date"], axis=1)

        logger.debug(f"Saving final df for window size {window_size}")
        y_eval.to_csv(processed_dir / f"y_eval_window_{window_size}.csv", index=False)
        X_eval.to_csv(processed_dir / f"X_eval_window_{window_size}.csv", index=False)
        dates_eval.to_csv(
            processed_dir / f"dates_eval_window_{window_size}.csv", index=False
        )

        logger.debug(f"Creating window {window_size} train/test set...")

        logger.debug(f"Copying window eval for window {window_size}...")
        df_temp = train.copy()
        X = split_with_window_size(df_temp, window_size)

        logger.debug(
            f"Splitting train window {window_size} into features and target dfs..."
        )
        y = X["target"]
        X = X.drop("target", axis=1)

        logger.debug(f"Splitting train window {window_size} into training folds...")
        tscv = TimeSeriesSplit(n_splits=NUM_CROSS_VAL_SPLITS)
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            logger.debug(f"Saving y_train_window_{window_size}_fold_{i}.csv...")
            y_train.to_csv(
                processed_dir / f"y_train_window_{window_size}_fold_{i}.csv",
                index=False,
            )
            logger.debug(f"Saving y_test_window_{window_size}_fold_{i}.csv...")
            y_test.to_csv(
                processed_dir / f"y_test_window_{window_size}_fold_{i}.csv",
                index=False,
            )

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            logger.debug(f"Saving X_train_window_{window_size}_fold_{i}.csv...")
            X_train.to_csv(
                processed_dir / f"X_train_window_{window_size}_fold_{i}.csv",
                index=False,
            )
            logger.debug(f"Saving X_test_window_{window_size}_fold_{i}.csv...")
            X_test.to_csv(
                processed_dir / f"X_test_window_{window_size}_fold_{i}.csv",
                index=False,
            )
    logger.info("Windows created and data successfully saved!")


if __name__ == "__main__":
    main()
