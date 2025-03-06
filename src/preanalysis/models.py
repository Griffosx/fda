from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class FiveMinutesBar(SQLModel, table=True):
    """SQLModel representation of the bars table"""

    __tablename__ = "five_minutes_bars"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(max_length=10)
    volume: float
    vwap: float
    open: float
    close: float
    high: float
    low: float
    trade_count: int
    datetime_utc: datetime


class FifteenMinutesBar(SQLModel, table=True):
    """SQLModel representation of the bars table"""

    __tablename__ = "fifteen_minutes_bars"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(max_length=10)
    volume: float
    vwap: float
    open: float
    close: float
    high: float
    low: float
    trade_count: int
    datetime_utc: datetime
