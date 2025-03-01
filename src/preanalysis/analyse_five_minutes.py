import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlmodel import Session, create_engine, select
from pytz import timezone
from tqdm import tqdm

from preanalysis.models import FiveMinutesBar
from preanalysis.constants import DATABASE_URL, TICKERS_HIGH_VOLUME


def get_engine():
    """Create and return SQLModel engine"""
    return create_engine(DATABASE_URL, echo=False)


def setup_intervals():
    """
    Define all possible 5-minute intervals from 7:30-9:25 NY time

    Returns:
        List of time interval strings
    """
    intervals = []
    # Add intervals for 7:30-7:55
    for minute in range(30, 60, 5):
        intervals.append(f"7:{minute:02d}")
    # Add intervals for 8:00-9:25 (excluding 9:30)
    for hour in [8, 9]:
        for minute in range(0, 60, 5):
            if hour == 9 and minute >= 30:  # Skip 9:30
                break
            intervals.append(f"{hour}:{minute:02d}")
    return intervals


def process_ticker_data(
    session, ticker, intervals, trade_count_thresholds, utc_tz, ny_tz
):
    """
    Process data for a single ticker

    Args:
        session: SQLModel session
        ticker: Ticker symbol
        intervals: List of time intervals
        trade_count_thresholds: List of thresholds for low trade count candles
        utc_tz: UTC timezone object
        ny_tz: NY timezone object

    Returns:
        tuple: (volumes by interval, ticker stats, all volumes list, candlestick data)
    """
    # Initialize volume counters for each interval
    volumes = {interval: 0 for interval in intervals}

    # Track ticker statistics
    stats = {
        "total_volume": 0,
        "candle_count": 0,
        "low_trade_count_candles": {
            threshold: 0 for threshold in trade_count_thresholds
        },
        "interval_data": {
            interval: {"volume": 0, "count": 0, "trade_count": 0}
            for interval in intervals
        },
    }

    # Store all volumes for distribution analysis
    all_volumes = []

    # Store candlestick data for outlier analysis
    candlesticks = []

    # Get all data for this ticker
    bars = session.exec(
        select(
            FiveMinutesBar.symbol,
            FiveMinutesBar.datetime_utc,
            FiveMinutesBar.volume,
            FiveMinutesBar.trade_count,
        ).where(FiveMinutesBar.symbol == ticker)
    ).all()

    # Process each bar
    for bar in bars:
        # Convert UTC to NY time
        ny_time = bar.datetime_utc.replace(tzinfo=utc_tz).astimezone(ny_tz)

        # Check if in our target time window (7:30-9:25, excluding 9:30)
        if (
            (ny_time.hour == 7 and ny_time.minute >= 30)
            or (ny_time.hour == 8)
            or (ny_time.hour == 9 and ny_time.minute < 30)  # Exclude 9:30
        ):
            # Calculate which 5-minute interval this falls into
            minutes_since_730am = (ny_time.hour - 7) * 60 + ny_time.minute - 30
            interval_idx = minutes_since_730am // 5

            if 0 <= interval_idx < len(intervals):
                interval = intervals[interval_idx]
                volumes[interval] += bar.volume
                all_volumes.append(bar.volume)

                # Store candlestick data
                candlesticks.append(
                    {
                        "symbol": ticker,
                        "datetime_utc": bar.datetime_utc,
                        "datetime_ny": ny_time,
                        "interval": interval,
                        "volume": bar.volume,
                        "trade_count": bar.trade_count,
                    }
                )

                # Update stats
                stats["total_volume"] += bar.volume
                stats["candle_count"] += 1
                stats["interval_data"][interval]["volume"] += bar.volume
                stats["interval_data"][interval]["count"] += 1
                stats["interval_data"][interval]["trade_count"] += bar.trade_count

                # Check if this is a low trade count candle for each threshold
                for threshold in trade_count_thresholds:
                    if bar.trade_count < threshold:
                        stats["low_trade_count_candles"][threshold] += 1

    return volumes, stats, all_volumes, candlesticks


def save_results(df, pivot, output_dir):
    """
    Save analysis results to CSV files

    Args:
        df: DataFrame with all data
        pivot: Pivot table for heatmap
        output_dir: Directory to save results
    """
    df.to_csv(f"{output_dir}/morning_volume_data.csv", index=False)
    pivot.to_csv(f"{output_dir}/morning_volume_pivot.csv")


def create_line_chart(pivot, output_dir):
    """
    Create line chart visualization

    Args:
        pivot: Pivot table with percentage data
        output_dir: Directory to save visualization
    """
    plt.figure(figsize=(12, 8))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)

    plt.title("Volume Distribution (7:30-9:25 NY Time)", fontsize=16)
    plt.xlabel("Time Interval", fontsize=14)
    plt.ylabel("Percentage of Morning Volume (%)", fontsize=14)
    plt.legend(title="Symbol", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.7, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/line_chart.png", dpi=300)
    plt.close()


def create_heatmap(pivot, output_dir):
    """
    Create heatmap visualization

    Args:
        pivot: Pivot table with percentage data
        output_dir: Directory to save visualization
    """
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "% of Morning Volume"},
    )
    plt.title("Morning Volume Distribution by Ticker (7:30-9:25 NY Time)", fontsize=16)
    plt.ylabel("Time Interval", fontsize=14)
    plt.xlabel("Ticker Symbol", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap.png", dpi=300)
    plt.close()


def create_low_trade_count_charts(ticker_stats, trade_count_thresholds, output_dir):
    """
    Create bar charts for low trade count candles

    Args:
        ticker_stats: Dictionary with ticker statistics
        trade_count_thresholds: List of thresholds for low trade count candles
        output_dir: Directory to save visualizations
    """
    for threshold in trade_count_thresholds:
        low_trade_counts = {
            ticker: stats["low_trade_count_candles"][threshold]
            for ticker, stats in ticker_stats.items()
        }

        plt.figure(figsize=(12, 8))
        plt.bar(low_trade_counts.keys(), low_trade_counts.values(), color="firebrick")

        plt.title(
            f"Low Trade Count Candles by Ticker (< {threshold} trades)", fontsize=16
        )
        plt.xlabel("Ticker Symbol", fontsize=14)
        plt.ylabel("Number of Low Trade Count Candles", fontsize=14)
        plt.grid(True, alpha=0.3, axis="y", linestyle="--")
        plt.xticks(rotation=45)

        # Add value labels on top of each bar
        for ticker, count in low_trade_counts.items():
            plt.text(ticker, count + 0.3, str(count), ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/low_trade_count_candles_{threshold}.png", dpi=300)
        plt.close()


def create_volume_distribution_charts(ticker_volumes, output_dir, num_bins=20):
    """
    Create histograms for volume distributions

    Args:
        ticker_volumes: Dictionary mapping tickers to list of volumes
        output_dir: Directory to save visualizations
        num_bins: Number of bins for histograms
    """
    # Create directory for volume distributions
    volume_dir = f"{output_dir}/volume_distributions"
    os.makedirs(volume_dir, exist_ok=True)

    # Create individual histograms for each ticker
    for ticker, volumes in ticker_volumes.items():
        if not volumes:
            continue

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # 1. Linear scale histogram with quantile-based bins
        if len(volumes) > num_bins:
            # Use quantile-based bins to better distribute the data
            quantiles = np.linspace(0, 100, num_bins + 1)
            bins = np.percentile(volumes, quantiles)
            # Ensure unique bin edges by adding small increments
            for i in range(1, len(bins)):
                if bins[i] <= bins[i - 1]:
                    bins[i] = bins[i - 1] + 0.01
        else:
            # Fallback to regular bins if not enough data points
            bins = num_bins

        n, bins, patches = ax1.hist(
            volumes, bins=bins, alpha=0.75, color="steelblue", edgecolor="black"
        )

        # Add bin count labels where space permits
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        for count, x in zip(n, bin_centers):
            if count > 0:  # Only add label if bin has values
                ax1.text(
                    x,
                    count + 0.1,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax1.set_title(f"{ticker} - Volume Distribution (Linear Scale)", fontsize=14)
        ax1.set_xlabel("Volume", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # 2. Log scale histogram
        # Add a small epsilon to handle zeros
        log_volumes = [max(v, 0.1) for v in volumes]

        # Use log scale on x-axis
        ax2.hist(
            log_volumes, bins=num_bins, alpha=0.75, color="darkgreen", edgecolor="black"
        )
        ax2.set_xscale("log")
        ax2.set_title(f"{ticker} - Volume Distribution (Log Scale)", fontsize=14)
        ax2.set_xlabel("Volume (log scale)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(f"{volume_dir}/{ticker}_volume_distribution.png", dpi=300)
        plt.close()

        # 3. Create a boxplot for each ticker
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            volumes,
            vert=False,
            widths=0.7,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            whiskerprops=dict(color="black"),
            medianprops=dict(color="darkred"),
        )

        # Add a violin plot overlay
        plt.violinplot(volumes, vert=False, showmeans=False, showextrema=False)

        plt.title(f"{ticker} - Volume Distribution (Box & Violin Plot)", fontsize=16)
        plt.xlabel("Volume", fontsize=14)
        plt.yticks([1], [ticker])
        plt.grid(True, alpha=0.3, linestyle="--", axis="x")
        plt.tight_layout()
        plt.savefig(f"{volume_dir}/{ticker}_volume_boxplot.png", dpi=300)
        plt.close()

    # Create combined KDE plots for all tickers
    plt.figure(figsize=(14, 10))

    # Linear scale KDE
    for ticker, volumes in ticker_volumes.items():
        if volumes:
            sns.kdeplot(volumes, label=ticker, fill=True, alpha=0.2)

    plt.title("Volume Distributions by Ticker (Linear Scale)", fontsize=16)
    plt.xlabel("Volume", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{volume_dir}/combined_volume_distributions.png", dpi=300)
    plt.close()

    # Log scale KDE
    plt.figure(figsize=(14, 10))

    for ticker, volumes in ticker_volumes.items():
        if volumes:
            # Add a small epsilon to handle zeros
            log_volumes = [max(v, 0.1) for v in volumes]
            sns.kdeplot(log_volumes, label=ticker, fill=True, alpha=0.2)

    plt.xscale("log")
    plt.title("Volume Distributions by Ticker (Log Scale)", fontsize=16)
    plt.xlabel("Volume (log scale)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{volume_dir}/combined_volume_distributions_log.png", dpi=300)
    plt.close()


def identify_and_save_outliers(ticker_candlesticks, output_dir):
    """
    Identify outlier candlesticks for each ticker and save to CSV.
    Outliers are defined as candlesticks with volume above the 95th percentile.

    Args:
        ticker_candlesticks: Dictionary mapping tickers to list of candlestick data
        output_dir: Directory to save results
    """
    all_outliers = []

    for ticker, candlesticks in ticker_candlesticks.items():
        if not candlesticks or len(candlesticks) < 20:  # Skip if insufficient data
            continue

        # Extract volumes for percentile calculation
        volumes = [candle["volume"] for candle in candlesticks]

        # Calculate 95th percentile
        p95 = np.percentile(volumes, 95)

        # Find outliers (above 95th percentile)
        for candle in candlesticks:
            if candle["volume"] > p95:
                # Mark as outlier and add to list
                candle["is_outlier"] = True
                candle["p95_threshold"] = p95
                candle["percentile"] = 100 * (
                    sum(v < candle["volume"] for v in volumes) / len(volumes)
                )
                # Convert datetime objects to strings for CSV
                candle["datetime_utc"] = candle["datetime_utc"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                candle["datetime_ny"] = candle["datetime_ny"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                all_outliers.append(candle)

    # Save to CSV if outliers found
    if all_outliers:
        outliers_df = pd.DataFrame(all_outliers)
        outliers_df.to_csv(f"{output_dir}/volume_outliers_p95.csv", index=False)
        print(
            f"Found {len(all_outliers)} outlier candlesticks above 95th percentile. Saved to {output_dir}/volume_outliers_p95.csv"
        )
    else:
        print("No volume outliers found above 95th percentile.")


def analyze_morning_volume(
    session: Session,
    tickers: List[str],
    trade_count_thresholds: List[int] = [100],
    output_dir: str = "morning_volume",
):
    """
    Analyze volume distribution from 7:30-9:25 NY time for each ticker in 5-minute intervals

    Args:
        session: SQLModel session
        tickers: List of ticker symbols to analyze
        trade_count_thresholds: List of thresholds for considering low-activity candlesticks
        output_dir: Directory to save results
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    utc_tz = timezone("UTC")
    ny_tz = timezone("America/New_York")

    # Get intervals
    intervals = setup_intervals()

    # Create DataFrame to store all data
    all_data = []

    # Dictionary to store ticker stats
    ticker_stats = {}

    # Dictionary to store volumes for distribution analysis
    ticker_volumes = {}

    # Dictionary to store candlestick data for outlier analysis
    ticker_candlesticks = {}

    # Process each ticker
    for ticker in tqdm(tickers, desc="Processing tickers"):
        volumes, stats, all_volumes, candlesticks = process_ticker_data(
            session, ticker, intervals, trade_count_thresholds, utc_tz, ny_tz
        )

        # Store statistics, volumes, and candlesticks
        ticker_stats[ticker] = stats
        ticker_volumes[ticker] = all_volumes
        ticker_candlesticks[ticker] = candlesticks

        # Calculate total volume for this ticker in the morning period
        total_volume = sum(volumes.values())

        # If there was any volume, add data points for each interval
        if total_volume > 0:
            for interval, volume in volumes.items():
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                all_data.append(
                    {
                        "symbol": ticker,
                        "interval": interval,
                        "volume": volume,
                        "percentage": percentage,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data found for the specified time period")
        return None

    # Create pivot table for the heatmap
    pivot = df.pivot_table(
        index="interval", columns="symbol", values="percentage"
    ).fillna(0)

    # Save results to CSV
    save_results(df, pivot, output_dir)

    # Identify and save outliers
    identify_and_save_outliers(ticker_candlesticks, output_dir)

    # Create visualizations
    create_line_chart(pivot, output_dir)
    create_heatmap(pivot, output_dir)
    create_low_trade_count_charts(ticker_stats, trade_count_thresholds, output_dir)
    create_volume_distribution_charts(ticker_volumes, output_dir)

    print(f"Analysis complete. Results saved to {output_dir}")
    return df, pivot


def main():
    # Get engine and create session
    engine = get_engine()

    with Session(engine) as session:
        # Analyze morning volume distribution
        tickers = TICKERS_HIGH_VOLUME[:5]
        analyze_morning_volume(
            session,
            tickers,
            output_dir=f"morning_volume/{"_".join(tickers)}",
            trade_count_thresholds=[25, 50, 100],
        )


if __name__ == "__main__":
    main()
