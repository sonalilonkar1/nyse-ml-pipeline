import argparse
import logging
from pathlib import Path

from src.train import run_cross_validation
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logging.info("Starting main...")

    logging.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        description="Train and evaluate common ML algorithms on a stock market prediction task!"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Optional Path to a results directory.",
    )
    args = parser.parse_args()

    logging.info(f"Config path: {args.config}")
    logging.info(f"Results dir: {args.results_dir}")

    logger.info("Loading config...")
    config = load_config(args.config)

    logging.info(f"Using {config['config_name']} config...")

    logging.info("Running cross validation...")
    for model_conf in config["models"]:
        results_df = run_cross_validation(model_conf, config["windows"])
        print(results_df.head())


if __name__ == "__main__":
    main()
