#!/usr/bin/env python

import sys
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=sys.argv[1] if len(sys.argv) > 1 else "INFO")

FEATURES = ["open", "close", "low", "high", "volume"]

TRAIN_RATIO = 0.8
EVAL_RATIO = 1 - TRAIN_RATIO

MIN_WINDOW_SIZE = 2
MAX_WINDOW_SIZE = 5


def main():
    logger.debug("Starting process_data_full.py")
    logger.info("Loading data...")

    logger.debug("Attatching to raw data directory")
    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    logger.debug("Attatching to full processed data directory")
    processed_dir = Path(__file__).parent.parent / "data"
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

    logger.info("Processing eval set into windows...")
    for window_size in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
        logger.debug(f"Creating window size {window_size} eval set...")

        logger.debug(f"Creating a new df for window size {window_size}...")
        df_temp = df.copy()
        df_temp["target"] = df_temp.groupby("symbol")["close"].shift(-1)
        logger.debug(df_temp.head())

        logger.debug(f"Processing features for window size {window_size}...")
        for i in range(0, window_size):
            for feat in FEATURES:
                df_temp[f"{feat}_t-{i}"] = df_temp.groupby("symbol")[feat].shift(i)
        logger.debug(f"df with new features:\n{df_temp.head(6)}")
        logger.debug(f"df target values\n{df_temp['target'].head(6)}")

        logger.debug(f"Creating final df for window size {window_size}")
        df_temp = df_temp.dropna()
        X = df_temp[
            [col for col in df_temp.columns if col not in ["date", "symbol"] + FEATURES]
        ]
        logger.debug(f"final df:\n{X.head(6)}")

        logger.debug(f"Saving final df for window size {window_size}")
        X.to_csv(processed_dir / f"window_{window_size}_prices.csv", index=False)
    logger.info("Eval processing complete!")


if __name__ == "__main__":
    main()
