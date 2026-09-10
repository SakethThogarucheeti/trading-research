"""SyntheticDataBroker — deterministic OHLCV generator for tests."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime, timedelta

import polars as pl
from trading.broker.service.broker import Broker
from trading.core.schemas import OrderType, Side

# NSE session: 09:15 → 15:30, 25 bars at 15min cadence
_SESSION_START_H = 9
_SESSION_START_M = 15
_SESSION_BARS = 25
_SESSION_MINUTES = _SESSION_BARS * 15  # 375 min


class SyntheticDataBroker(Broker):
    """
    Generates deterministic synthetic OHLCV bars from a sine wave + linear drift + noise.

    The same (symbol, interval, start, end) call always returns identical data,
    so tests that use this broker are fully reproducible. Each symbol gets a
    different price series because the seed is mixed with the symbol name.

    Does NOT implement place_order / get_instruments — raises NotImplementedError.
    Wrap with PaperBroker if order execution is needed.
    """

    # No alias — test-only; must not appear in the production Broker registry.

    def __init__(
        self,
        base_price: float = 100.0,
        drift_per_bar: float = 0.02,
        amplitude: float = 5.0,
        period_bars: int = 50,
        volatility: float = 0.3,
        seed: int = 42,
    ) -> None:
        self._base = base_price
        self._drift = drift_per_bar
        self._amp = amplitude
        self._period = period_bars
        self._vol = volatility
        self._seed = seed

    # ── Broker interface ──────────────────────────────────────────────────────

    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        bar_min = _interval_minutes(interval)
        timestamps = _session_timestamps(start, end, bar_min)
        if not timestamps:
            return _empty_df()

        closes = self._generate_closes(symbol, interval, start, len(timestamps))

        rows: list[dict] = []
        prev = closes[0]
        for ts, close in zip(timestamps, closes, strict=True):
            swing = abs(close - prev)
            rows.append(
                {
                    "date": ts,
                    "open": prev,
                    "high": max(prev, close) + swing * 0.3,
                    "low": min(prev, close) - swing * 0.3,
                    "close": close,
                    "volume": 10_000 + int(close * 10) % 5000,
                }
            )
            prev = close

        return pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Datetime("us", "UTC")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        )

    def get_instruments(self) -> pl.DataFrame:
        raise NotImplementedError("SyntheticDataBroker does not support get_instruments")

    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType,
        limit_price: float | None = None,
        instrument_type: str = "EQUITY",
        tick_log_id: int = 0,
    ) -> str:
        raise NotImplementedError("SyntheticDataBroker does not support place_order")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_closes(self, symbol: str, interval: str, start: datetime, n: int) -> list[float]:
        # Deterministic seed: mix global seed + symbol so each symbol differs
        h = int(
            hashlib.md5(
                f"{symbol}:{interval}:{start.isoformat()}:{self._seed}".encode()
            ).hexdigest(),
            16,
        )
        # LCG noise
        state = h
        closes: list[float] = []
        for i in range(n):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            noise = ((state >> 33) / (2**31) - 1.0) * self._vol
            price = (
                self._base
                + i * self._drift
                + self._amp * math.sin(2 * math.pi * i / self._period)
                + noise
            )
            closes.append(max(price, 1.0))
        return closes


# ── Module-level helpers ──────────────────────────────────────────────────────


def _interval_minutes(interval: str) -> int:
    if interval.endswith("min"):
        return int(interval[:-3])
    if interval in ("1h", "60min"):
        return 60
    if interval == "day":
        return _SESSION_MINUTES
    return 15


def _session_timestamps(start: datetime, end: datetime, bar_min: int) -> list[datetime]:
    """Enumerate bar open-timestamps within NSE trading sessions in [start, end]."""
    result: list[datetime] = []

    # Align to first session open on or after start
    cur = start.replace(
        hour=_SESSION_START_H,
        minute=_SESSION_START_M,
        second=0,
        microsecond=0,
        tzinfo=start.tzinfo or UTC,
    )
    if cur < start:
        cur += timedelta(days=1)

    session_end_min = _SESSION_START_H * 60 + _SESSION_START_M + _SESSION_MINUTES

    while cur <= end:
        day_min = cur.hour * 60 + cur.minute
        open_min = _SESSION_START_H * 60 + _SESSION_START_M
        # Skip weekends (Mon=0 … Fri=4)
        if cur.weekday() < 5 and open_min <= day_min < session_end_min:
            result.append(cur)

        cur += timedelta(minutes=bar_min)

        # If we've stepped past session end, jump to next day's open
        day_min_next = cur.hour * 60 + cur.minute
        if day_min_next >= session_end_min or cur.weekday() >= 5:
            next_day = cur + timedelta(days=1)
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            cur = next_day.replace(
                hour=_SESSION_START_H,
                minute=_SESSION_START_M,
                second=0,
                microsecond=0,
            )

    return result


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        }
    )
