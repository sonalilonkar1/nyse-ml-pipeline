#!/usr/bin/env python

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURES = ["open", "close", "low", "high", "volume"]
MIN_WINDOW_SIZE = 2
MAX_WINDOW_SIZE = 5


def main():
    logger.debug("Starting preprocess_data.py")
    logger.info("Loading data...")

    logger.debug("Attatching to raw data directory")
    raw_dir = Path(__file__).parent.parent / "data" / "raw"

    logger.debug("Attatching to processed data directory")
    processed_dir = Path(__file__).parent.parent / "data" / "processed"

    logger.debug("Retrieving prices.csv")
    prices = raw_dir / "prices.csv"

    logger.debug("Converting prices.csv to pandas df")
    df = pd.read_csv(prices)

    logger.debug("Sorting prices df by ticker symbol then date")
    df = df.sort_values(["symbol", "date"])

    logger.debug("Start of main dataprocessing loop")
    for window_size in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
        logger.info(f"Processing data using a window of size {window_size}...")

        logger.debug(f"Creating a new df for window size {window_size}")
        df_temp = df.copy()
        df_temp["target"] = df_temp.groupby("symbol")["close"].shift(-window_size)

        logger.debug(f"Processing features for window size {window_size}")
        for i in range(1, window_size + 1):
            for feat in FEATURES:
                df_temp[f"{feat}_t-{i}"] = df_temp.groupby("symbol")[feat].shift(i)

        logger.debug(f"Creating final df for window size {window_size}")
        df_temp = df_temp.dropna()
        X = df_temp[
            [col for col in df_temp.columns if col not in ["date", "symbol"] + FEATURES]
        ]

        logger.info("Processing complete!")
        logger.info(f"Saving to {processed_dir}/prices_window_{window_size}.csv")
        X.to_csv(processed_dir / f"window_{window_size}.csv", index=False)
        logger.info("Saved successfully!")


if __name__ == "__main__":
    main()
