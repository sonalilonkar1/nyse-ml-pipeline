"""Generate result visualizations from evaluation data."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

DEFAULT_BUCKET_DAYS = 7
DEFAULT_EMA_SPAN = 20

MARKET_EVENTS = {
    "2016-01-20": "Oil Crash Low",
    "2016-06-23": "Brexit Vote",
    "2016-11-08": "US Election",
}


def get_latest_eval_results(experiment_dir: Path) -> pd.DataFrame:
    """Load most recent eval results for a model."""
    eval_dir = experiment_dir / "eval"
    result_files = sorted(eval_dir.glob("results_*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No eval results found in {eval_dir}")
    return pd.read_csv(result_files[-1], parse_dates=["date"])


def compute_bucketed_r2(
    results_df: pd.DataFrame,
    bucket_days: int = 7,
) -> pd.DataFrame:
    """Compute R^2 for each time bucket and window."""
    results_df = results_df.copy()
    results_df["bucket"] = (
        results_df["date"].dt.to_period(f"{bucket_days}D").dt.start_time
    )

    records = []
    for (window, bucket), group in results_df.groupby(["window", "bucket"]):
        if len(group) < 2:
            continue
        r2 = r2_score(group["target"], group["predicted"])
        records.append({"window": window, "bucket": bucket, "r2": r2})

    return pd.DataFrame(records)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def plot_bucketed_r2(
    bucketed_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    bucket_days: int,
    ema_span: int = 4,
) -> None:
    """Plot R^2 over time with one line per window, smoothed with EMA."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data lines
    for window in sorted(bucketed_df["window"].unique()):
        window_data = bucketed_df[bucketed_df["window"] == window].sort_values("bucket")
        smoothed_r2 = compute_ema(window_data["r2"], span=ema_span)
        ax.plot(
            window_data["bucket"],
            smoothed_r2,
            label=f"Window {window}",
        )

    # Overlay Market Events
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    for date_str, event_name in MARKET_EVENTS.items():
        event_date = pd.Timestamp(date_str)
        # Only plot if event is within the visible x-axis range
        if bucketed_df["bucket"].min() <= event_date <= bucketed_df["bucket"].max():
            ax.axvline(
                x=event_date, color="black", linestyle="--", alpha=0.4, linewidth=1
            )
            # Place text near the bottom of the graph
            ax.text(
                event_date,
                ymin + (y_range * 0.05),
                f"  {event_name}",
                rotation=90,
                verticalalignment="bottom",
                fontsize=9,
                color="black",
                alpha=0.7,
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("R^2 (EMA)")
    ax.set_title(
        f"{model_name}: R^2 by {bucket_days}-Day Buckets (EMA span={ema_span})"
    )
    # Moved legend to upper left to avoid clashing with bottom text
    ax.legend(loc="upper left")  # <--- CHANGED: Move legend up
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def generate_model_plots(
    model_name: str,
    config_name: str,
    bucket_days: int = DEFAULT_BUCKET_DAYS,
    ema_span: int = DEFAULT_EMA_SPAN,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Generate all plots for a single model."""
    experiment_dir = results_dir / model_name / config_name

    results_df = get_latest_eval_results(experiment_dir)
    bucketed_df = compute_bucketed_r2(results_df, bucket_days)

    plot_path = experiment_dir / "eval" / "plots" / "bucketed_r2.png"
    plot_bucketed_r2(bucketed_df, model_name, plot_path, bucket_days, ema_span)


def generate_all_plots(
    results_dir: Path,
    config_name: str,
    bucket_days: int = DEFAULT_BUCKET_DAYS,
    ema_span: int = DEFAULT_EMA_SPAN,
) -> None:
    """Generate plots for all models with eval results."""
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        eval_dir = model_dir / config_name / "eval"
        if not eval_dir.exists():
            continue
        logger.info(f"Generating plots for {model_dir.name}...")
        generate_model_plots(
            model_dir.name, config_name, bucket_days, ema_span, results_dir
        )


def main():
    parser = argparse.ArgumentParser(description="Generate result visualizations.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., linear_regression)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="Config name (default: baseline)",
    )
    parser.add_argument(
        "--bucket-days",
        type=int,
        default=DEFAULT_BUCKET_DAYS,
        help=f"Number of days per bucket (default: {DEFAULT_BUCKET_DAYS})",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=DEFAULT_EMA_SPAN,
        help=f"EMA span for smoothing (default: {DEFAULT_EMA_SPAN})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_model_plots(args.model, args.config, args.bucket_days, args.ema_span)


if __name__ == "__main__":
    main()
