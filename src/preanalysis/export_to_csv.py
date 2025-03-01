import pandas as pd
from sqlmodel import Session, create_engine, select
from preanalysis.constants import DATABASE_URL
from preanalysis.models import FiveMinutesBar


def get_engine():
    """Create and return SQLModel engine"""
    return create_engine(DATABASE_URL, echo=False)


def export():
    engine = get_engine()

    with Session(engine) as session:
        statement = select(FiveMinutesBar)
        results = session.exec(statement).all()
        print("Data retrieved")
        print(f"Number of records: {len(results)}")

        # Convert to list of dictionaries
        records = [
            {
                "symbol": bar.symbol,
                "volume": bar.volume,
                "vwap": bar.vwap,
                "open": bar.open,
                "close": bar.close,
                "high": bar.high,
                "low": bar.low,
                "trade_count": bar.trade_count,
                "datetime_utc": bar.datetime_utc,
            }
            for bar in results
        ]

        # Save all data to a single CSV
        output_file = f"assets/top_20_five_minutes_2024.csv"
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")


if __name__ == "__main__":
    export()
