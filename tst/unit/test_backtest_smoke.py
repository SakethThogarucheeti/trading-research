"""
Smoke test for BacktestSession.run() against the pinned trading-platform
pipeline, using real Postgres via testcontainers.

Establishes baseline coverage for the run() method that #1/#2/#3 flag as
"no test coverage" — deliberately minimal (tiny synthetic dataset, one
strategy, one symbol) so it runs fast as a correctness check, not a
performance/strategy-quality benchmark.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from research.backtesting.data_loader import DataLoader
from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig, BacktestReport

from trading.config.settings import AlgoSettings

_START = datetime(2024, 1, 2, 9, 15, 0, tzinfo=UTC)


def _synthetic_ohlcv(n_bars: int, start_price: float = 1000.0, step: float = 1.0) -> pl.DataFrame:
    """Small deterministic uptrending OHLCV series — enough bars to clear warmup."""
    rows = []
    price = start_price
    for i in range(n_bars):
        ts = _START + timedelta(minutes=i)
        open_ = price
        close = price + step
        high = close + 0.5
        low = open_ - 0.5
        rows.append((ts, open_, high, low, close, 10_000))
        price = close
    return pl.DataFrame(
        rows,
        schema=["date", "open", "high", "low", "close", "volume"],
        orient="row",
    )


class _InMemoryLoader(DataLoader):
    """DataLoader backed by a pre-built DataFrame — no file I/O."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> pl.DataFrame:
        return self._df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def _algo(name: str = "smoke") -> AlgoSettings:
    return AlgoSettings(
        name=name,
        instruments=["INFY"],
        strategy_id="ema_crossover",
        candle_intervals=["1min"],
    )


async def test_backtest_session_runs_end_to_end(pg_engine, tmp_path):
    """BacktestSession.run() completes and returns a well-formed report."""
    df = _synthetic_ohlcv(n_bars=250)
    config = BacktestConfig(
        algo=_algo(),
        start=_START,
        end=_START + timedelta(minutes=249),
        loader=_InMemoryLoader(df),
        initial_equity=100_000.0,
    )
    session = BacktestSession(config=config, db_engine=pg_engine, results_dir=tmp_path)
    report = await session.run()

    assert isinstance(report, BacktestReport)
    assert report.session_id
    assert report.final_equity > 0
    assert len(report.equity_curve) > 0


async def test_backtest_session_html_report_generated(pg_engine, tmp_path):
    """to_html() must return non-empty Plotly-embedded HTML for a completed run."""
    df = _synthetic_ohlcv(n_bars=250)
    config = BacktestConfig(
        algo=_algo("smoke_html"),
        start=_START,
        end=_START + timedelta(minutes=249),
        loader=_InMemoryLoader(df),
        initial_equity=100_000.0,
    )
    session = BacktestSession(config=config, db_engine=pg_engine, results_dir=tmp_path)
    report = await session.run()

    html = report.to_html()
    assert "<html" in html.lower()
    assert "plotly" in html.lower()
