"""
Smoke test for WalkForwardRunner.run() against the pinned trading-platform
pipeline, using real Postgres via testcontainers.

Establishes baseline coverage for the run() method that #2 flags as "no test
coverage" -- small train/test sizes, not a strategy-quality benchmark.

NOTE: sized for exactly ONE window deliberately. BacktestSession defaults to
db_schema="public" and WalkForwardRunner never overrides it per window, so a
multi-window run makes each window's BacktestSession DROP SCHEMA "public"
CASCADE and recreate it out from under the previous window -- a real,
pre-existing bug (see the filed issue) that produces flaky
DeadlockDetectedError / UniqueViolationError failures unrelated to this
refactor. One window exercises run() end-to-end without hitting it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from trading.config.settings import AlgoSettings

from research.backtesting.data_loader import DataLoader
from research.walk_forward.report import WalkForwardConfig, WalkForwardReport
from research.walk_forward.runner import WalkForwardRunner

_START = datetime(2024, 1, 2, 9, 15, 0, tzinfo=UTC)


def _synthetic_ohlcv(n_bars: int, start_price: float = 1000.0, step: float = 1.0) -> pl.DataFrame:
    """Small deterministic uptrending OHLCV series."""
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
    """DataLoader backed by a pre-built DataFrame -- no file I/O."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> pl.DataFrame:
        return self._df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def _algo(name: str = "wf_smoke") -> AlgoSettings:
    return AlgoSettings(
        name=name,
        instruments=["INFY"],
        strategy_id="ema_crossover",
        candle_intervals=["1min"],
    )


async def test_walk_forward_runner_runs_end_to_end(pg_engine, tmp_path):
    """WalkForwardRunner.run() completes and returns a well-formed report."""
    df = _synthetic_ohlcv(n_bars=45)
    loader = _InMemoryLoader(df)
    config = WalkForwardConfig(
        algo=_algo(),
        loader=loader,
        symbols=["INFY"],
        intervals=["1min"],
        train_bars=35,
        test_bars=10,
        step_bars=10,
        initial_equity=100_000.0,
        session_id="wf-smoke",
    )
    runner = WalkForwardRunner(config=config, db_engine=pg_engine, results_dir=tmp_path)

    report = await runner.run()

    assert isinstance(report, WalkForwardReport)
    assert report.session_id == "wf-smoke"
    assert len(report.windows) == 1  # (45-45)//10 + 1
    assert len(report.combined_equity_curve) > 0


class _MissingLoader(DataLoader):
    """DataLoader that always reports the file as missing."""

    def load(self, symbol: str, interval: str, start: datetime, end: datetime) -> pl.DataFrame:
        raise FileNotFoundError(f"no data for {symbol}/{interval}")


async def test_walk_forward_runner_raises_when_no_data(pg_engine, tmp_path):
    """run() must raise ValueError (not silently produce an empty report) when no data loads."""
    config = WalkForwardConfig(
        algo=_algo("wf_empty"),
        loader=_MissingLoader(),
        symbols=["INFY"],
        intervals=["1min"],
        session_id="wf-empty",
    )
    runner = WalkForwardRunner(config=config, db_engine=pg_engine, results_dir=tmp_path)

    try:
        await runner.run()
        raised = False
    except ValueError:
        raised = True

    assert raised, "run() must raise ValueError when no data loads for any symbol/interval"
