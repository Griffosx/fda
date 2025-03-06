import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

TIME_POINTS = [
    "0930",
    "0945",
    "1000",
    "1015",
    "1030",
    "1045",
    "1100",
    "1115",
    "1130",
    "1145",
]


def process_values(filepath, limit=None):
    # Read the CSV file
    df = pd.read_csv(filepath)
    if limit:
        df = df.head(limit)

    # Create a list of close price columns
    close_columns = [f"close_{t}" for t in TIME_POINTS]

    # Create new dataframe with just ticker, day and the normalized values
    result_df = df[["ticker", "day"]].copy()

    # Process each row
    for index, row in df.iterrows():
        # Extract close prices for this row
        close_prices = row[close_columns].values

        # Find min and max values
        min_price = np.min(close_prices)
        max_price = np.max(close_prices)

        # Calculate range
        price_range = max_price - min_price

        # Avoid division by zero if min and max are the same
        if price_range == 0:
            normalized_prices = np.zeros(len(close_prices))
            percentage_change = 0
        else:
            # Normalize to 0-1 range
            normalized_prices = (close_prices - min_price) / price_range

            # Calculate percentage change from min to max
            percentage_change = (max_price - min_price) / min_price * 100

        # Add normalized columns to result dataframe
        for i, col in enumerate(close_columns):
            new_col = f"norm_{col[6:]}"  # Remove 'close_' prefix, add 'norm_' prefix
            result_df.loc[index, new_col] = normalized_prices[i]

        # Add percentage change column
        result_df.loc[index, "min_to_max_percent"] = percentage_change

    return result_df


def draw_normalized_data(df, output_dir="assets/bar_charts", use_alpha: bool = False):
    """
    Create visualizations of the normalized data as colored rectangles.
    Alpha (transparency) is determined by min_to_max_percent.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the dimensions
    rect_width = 20  # pixels
    rect_height = 200  # pixels

    # Find columns that start with 'norm_' and are between 0930 and 1200
    norm_columns = [
        f"norm_{time}" for time in TIME_POINTS if f"norm_{time}" in df.columns
    ]

    # Create a color map (red for low values, green for high values)
    cmap = LinearSegmentedColormap.from_list(
        "value_cmap", ["red", "orange", "yellow", "lightgreen", "green"]
    )

    # Calculate mean and standard deviation of min_to_max_percent
    mean_percent = df["min_to_max_percent"].mean()
    std_percent = df["min_to_max_percent"].std()

    # Calculate thresholds for alpha scaling
    lower_threshold = mean_percent - 2 * std_percent  # Alpha = 0.3
    upper_threshold = mean_percent + 2 * std_percent  # Alpha = 1.0

    # Process each row in the dataframe
    for index, row in df.iterrows():
        # Get ticker and day for filename
        ticker = row["ticker"]
        day = row["day"]

        if use_alpha:
            # Get the min_to_max_percent for this row and calculate alpha
            percent_change = row["min_to_max_percent"]

            # Scale alpha between 0.3 and 1.0 based on percent_change
            # using the 2-standard deviation range
            if percent_change <= lower_threshold:
                alpha = 0.3  # Minimum alpha for values below lower threshold
            elif percent_change >= upper_threshold:
                alpha = 1.0  # Maximum alpha for values above upper threshold
            else:
                # Linear scaling for values between thresholds
                alpha_range = 0.7  # 1.0 - 0.3
                percent_range = upper_threshold - lower_threshold
                alpha = 0.3 + alpha_range * (
                    (percent_change - lower_threshold) / percent_range
                )
        else:
            alpha = 1

        # Create a figure
        fig_width = (
            len(norm_columns) * rect_width / 100
        )  # Convert to inches (assuming 100 dpi)
        fig_height = rect_height / 100  # Convert to inches
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Draw rectangles for each time point
        for i, col in enumerate(norm_columns):
            # Get the normalized value
            value = row[col]

            # Create a rectangle with alpha based on percent change
            rect = patches.Rectangle(
                (i * rect_width, 0),  # (x, y)
                rect_width,  # width
                rect_height,  # height
                facecolor=cmap(value),  # color based on normalized value
                alpha=alpha,  # transparency based on percent change
            )

            # Add the rectangle to the plot
            ax.add_patch(rect)

        # Set the limits of the plot
        ax.set_xlim(0, len(norm_columns) * rect_width)
        ax.set_ylim(0, rect_height)

        # Remove axes
        ax.axis("off")

        # Save the figure
        filename = f"{ticker}_{day}.png"
        plt.savefig(
            os.path.join(output_dir, filename),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        print(f"Generated visualization for {ticker} on {day}")


if __name__ == "__main__":
    result = process_values("assets/top_100_15mins_2023_2024.csv", limit=None)
    print(result.head(20))
    draw_normalized_data(result)
