from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from trading.broker.service.broker import Broker
    from trading.candles.service.historical import HistoricalDataService

# Required columns and their expected Polars dtypes (for validation)
_REQUIRED_COLUMNS: set[str] = {"date", "open", "high", "low", "close", "volume"}


def _validate_dataframe(df: pl.DataFrame, source: str) -> None:
    """Raise ``ValueError`` if any required column is missing or has wrong dtype."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataLoader [{source}]: missing required columns: {sorted(missing)}. Got: {df.columns}"
        )
    for col in ("open", "high", "low", "close"):
        if df[col].dtype not in (pl.Float32, pl.Float64):
            # Try casting — the data might be integer-encoded prices
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            except Exception as exc:
                raise ValueError(
                    f"DataLoader [{source}]: column {col!r} cannot be cast to Float64 "
                    f"(current dtype: {df[col].dtype})"
                ) from exc
    if df["volume"].dtype not in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
        try:
            df = df.with_columns(pl.col("volume").cast(pl.Int64))
        except Exception as exc:
            raise ValueError(
                f"DataLoader [{source}]: column 'volume' cannot be cast to Int64 "
                f"(current dtype: {df['volume'].dtype})"
            ) from exc


# ---------------------------------------------------------------------------
# DataLoader ABC
# ---------------------------------------------------------------------------


class DataLoader(ABC):
    """
    Abstract source of historical OHLCV data.

    ``load()`` must return a ``pl.DataFrame`` with exactly these columns
    (extra columns are ignored by callers but must not be absent):

    +---------+--------------------+
    | Column  | Polars dtype       |
    +=========+====================+
    | date    | Datetime (any tz)  |
    +---------+--------------------+
    | open    | Float64, > 0       |
    +---------+--------------------+
    | high    | Float64, > 0       |
    +---------+--------------------+
    | low     | Float64, > 0       |
    +---------+--------------------+
    | close   | Float64, > 0       |
    +---------+--------------------+
    | volume  | Int64, >= 0        |
    +---------+--------------------+

    ``ValueError`` is raised on missing/misnamed columns so the caller
    learns about bad data before the simulation starts.
    """

    @abstractmethod
    def load(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Return OHLCV bars for *symbol* between *start* and *end* inclusive.

        Implementations must raise:
        - ``FileNotFoundError`` if the data source does not exist.
        - ``ValueError`` if required columns are absent or have wrong dtypes.
        """


# ---------------------------------------------------------------------------
# BrokerDataLoader
# ---------------------------------------------------------------------------


class BrokerDataLoader(DataLoader):
    """
    Fetch OHLCV data from a ``Broker`` implementation.

    Works with ``ZerodhaBroker`` (live API) or any mock/stub that
    implements the ``Broker`` ABC.
    """

    def __init__(self, broker: Broker) -> None:
        self._broker = broker

    def load(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        df = self._broker.get_ohlc(symbol, interval, start, end)
        _validate_dataframe(df, f"BrokerDataLoader:{symbol}:{interval}")
        return df


# ---------------------------------------------------------------------------
# FileDataLoader
# ---------------------------------------------------------------------------


class FileDataLoader(DataLoader):
    """
    Load OHLCV data from CSV or Parquet files.

    File path convention::

        {base_dir}/{symbol}/{interval}.csv
        {base_dir}/{symbol}/{interval}.parquet

    Format is auto-detected by extension. Parquet takes precedence if both
    exist.

    CSV example (header required, order does not matter)::

        date,open,high,low,close,volume
        2024-01-02 09:15:00+05:30,1500.0,1510.0,1495.0,1505.0,12000

    The ``date`` column in CSV is parsed as a datetime; timezone-aware
    strings (ISO 8601) are preserved. Parquet ``date`` must be
    ``Datetime`` dtype (any timezone).

    Raises
    ------
    FileNotFoundError
        If neither ``{interval}.csv`` nor ``{interval}.parquet`` exists
        under ``{base_dir}/{symbol}/``.
    ValueError
        If any required column is absent or cannot be cast to the expected
        dtype.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)

    def load(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        symbol_dir = self._base_dir / symbol
        parquet_path = symbol_dir / f"{interval}.parquet"
        csv_path = symbol_dir / f"{interval}.csv"

        if parquet_path.exists():
            df = pl.read_parquet(parquet_path)
            source = str(parquet_path)
        elif csv_path.exists():
            df = pl.read_csv(csv_path, try_parse_dates=True)
            source = str(csv_path)
        else:
            raise FileNotFoundError(
                f"FileDataLoader: no data file for symbol={symbol!r} "
                f"interval={interval!r}. Looked for:\n"
                f"  {parquet_path}\n"
                f"  {csv_path}"
            )

        _validate_dataframe(df, source)

        # Ensure date column is the expected name (some files use 'datetime')
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename({"datetime": "date"})

        # Filter to [start, end] range (inclusive)
        # Cast date to a comparable type if needed
        date_col = df["date"]
        if date_col.dtype == pl.Utf8:
            df = df.with_columns(pl.col("date").str.to_datetime())

        df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        return df.sort("date")


# ---------------------------------------------------------------------------
# ServiceDataLoader
# ---------------------------------------------------------------------------


class ServiceDataLoader(DataLoader):
    """
    Backtest data loader backed by HistoricalDataService.

    Checks the DB first; calls the broker only when bars are not yet
    persisted. Suitable for backtests that should reuse candles already
    stored during live trading rather than re-fetching from the broker.
    """

    def __init__(self, service: HistoricalDataService) -> None:
        self._service = service

    def load(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        result = asyncio.get_event_loop().run_until_complete(
            self._service.fetch(symbol, interval, start, end)
        )
        return result.df
