import argparse
import logging
from pathlib import Path

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


if __name__ == "__main__":
    main()
