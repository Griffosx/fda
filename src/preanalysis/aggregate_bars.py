import pandas as pd
from sqlmodel import Session, create_engine, select

from preanalysis.models import FifteenMinutesBar
from preanalysis.constants import DATABASE_URL


def main():
    engine = create_engine(DATABASE_URL, echo=False)

    with Session(engine) as session:

        bars = session.exec(
            select(
                FifteenMinutesBar.symbol,
                FifteenMinutesBar.volume,
                FifteenMinutesBar.close,
                FifteenMinutesBar.datetime_utc,
            )
            # ).where(FifteenMinutesBar.symbol == "AAPL")
        ).all()

        # Convert to pandas DataFrame
        bars = pd.DataFrame(bars)

        # Convert datetime utc to NY timezone
        bars["datetime_utc"] = bars["datetime_utc"].dt.tz_convert("America/New_York")

        # Create a new column with date
        bars["date"] = bars["datetime_utc"].dt.date

        # Extract the hour and minute from the datetime
        bars["time"] = bars["datetime_utc"].dt.strftime("%H%M")

        # Create the pivot table
        pivot_df = bars.pivot_table(
            index=["symbol", "date"],
            columns="time",
            values=["volume", "close"],
            aggfunc="first",  # 'first' to get the value at that specific time
        )

        # Flatten the multi-index columns
        pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]

        # Reset the index to make symbol and date regular columns
        result_df = pivot_df.reset_index()

        # Rename columns for better readability if desired
        result_df = result_df.rename(columns={"symbol": "ticker", "date": "day"})

        # Select only the columns
        final_df = result_df[
            [
                "ticker",
                "day",
                "volume_0930",
                "volume_0945",
                "volume_1000",
                "volume_1015",
                "volume_1030",
                "volume_1045",
                "volume_1100",
                "volume_1115",
                "volume_1130",
                "volume_1145",
                "volume_1200",
                "volume_1215",
                "volume_1230",
                "volume_1245",
                "volume_1300",
                "volume_1315",
                "volume_1330",
                "volume_1345",
                "volume_1400",
                "volume_1415",
                "volume_1430",
                "volume_1445",
                "volume_1500",
                "volume_1515",
                "volume_1530",
                "volume_1545",
                "close_0930",
                "close_0945",
                "close_1000",
                "close_1015",
                "close_1030",
                "close_1045",
                "close_1100",
                "close_1115",
                "close_1130",
                "close_1145",
                "close_1200",
                "close_1215",
                "close_1230",
                "close_1245",
                "close_1300",
                "close_1315",
                "close_1330",
                "close_1345",
                "close_1400",
                "close_1415",
                "close_1430",
                "close_1445",
                "close_1500",
                "close_1515",
                "close_1530",
                "close_1545",
            ]
        ]

        final_df.to_csv("bars.csv", index=False)


if __name__ == "__main__":
    main()
