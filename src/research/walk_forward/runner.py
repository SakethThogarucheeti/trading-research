from __future__ import annotations

import logging
import uuid
from datetime import UTC
from pathlib import Path

import polars as pl
from sqlalchemy.ext.asyncio import AsyncEngine

from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig
from research.registry import session_type
from research.session import TestingSession
from research.walk_forward.report import WalkForwardConfig, WalkForwardReport

logger = logging.getLogger(__name__)


@session_type("walk_forward")
class WalkForwardRunner(TestingSession):
    """
    Walk-forward test runner.

    Splits historical data into overlapping train/test windows and runs a
    ``BacktestSession`` on each test window in sequence. Aggregate metrics
    expose whether the strategy degrades outside the training period.

    Window layout::

        |<--- train_bars --->|<-- test_bars -->|
                        |<--- train_bars --->|<-- test_bars -->|
                                        |<--- train_bars --->|<-- test_bars -->|

    Each step advances by ``step_bars`` bars. The combined equity curve
    stitches together only the *test* windows (no train leakage).
    """

    _config_cls = WalkForwardConfig

    def __init__(
        self,
        config: WalkForwardConfig,
        db_engine: AsyncEngine,
        results_dir: Path,
    ) -> None:
        super().__init__(results_dir=results_dir)
        self._config = config
        self._db_engine = db_engine

    def _load_windows(self, config: WalkForwardConfig) -> list[tuple]:
        """Load data once and compute train/test window boundaries."""
        all_data = _load_all_data(config)
        if not all_data:
            raise ValueError("WalkForwardRunner: no data loaded — check symbols/intervals")

        # Use the first (symbol, interval) pair for window indexing
        first_key = next(iter(all_data))
        index_df = all_data[first_key].sort("date")

        windows = _compute_windows(
            index_df, config.train_bars, config.test_bars, config.step_bars
        )
        if not windows:
            raise ValueError(
                f"WalkForwardRunner: not enough bars for even one window. "
                f"Need {config.train_bars + config.test_bars} bars, "
                f"got {len(index_df)}."
            )

        logger.info("WalkForwardRunner: %d windows", len(windows))
        return windows

    async def _run_windows(
        self, config: WalkForwardConfig, windows: list[tuple], session_id: str
    ) -> list:
        """Run a BacktestSession over each window's test range, in sequence."""
        backtest_reports = []
        for w_idx, (_train_start, _train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                "WalkForwardRunner: window %d/%d test=[%s → %s]",
                w_idx + 1,
                len(windows),
                test_start.date(),
                test_end.date(),
            )

            bt_config = BacktestConfig(
                algo=config.algo,
                start=test_start,
                end=test_end,
                loader=config.loader,
                initial_equity=config.initial_equity,
                session_id=f"{session_id}_w{w_idx + 1}",
            )

            bt_session = BacktestSession(
                config=bt_config,
                db_engine=self._db_engine,
                results_dir=self._results_dir,
            )
            bt_report = await bt_session.run()
            backtest_reports.append(bt_report)

        return backtest_reports

    def _build_report(
        self,
        config: WalkForwardConfig,
        backtest_reports: list,
        session_id: str,
        started_at,
    ) -> WalkForwardReport:
        """Aggregate per-window metrics and combine equity curves into a report."""
        all_sharpes = [r.sharpe_ratio for r in backtest_reports]
        all_mdds = [r.max_drawdown for r in backtest_reports]
        all_wrs = [r.win_rate for r in backtest_reports]
        agg_sharpe = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0
        agg_mdd = max(all_mdds) if all_mdds else 0.0
        agg_wr = sum(all_wrs) / len(all_wrs) if all_wrs else 0.0

        combined = _combine_equity_curves(backtest_reports)

        return WalkForwardReport(
            config=config,
            windows=backtest_reports,
            aggregate_sharpe=agg_sharpe,
            aggregate_max_drawdown=agg_mdd,
            aggregate_win_rate=agg_wr,
            combined_equity_curve=combined,
            session_id=session_id,
            session_type="walk_forward",
            started_at=started_at,
            finished_at=self._now(),
        )

    async def run(self) -> WalkForwardReport:
        config = self._config
        session_id = config.session_id or str(uuid.uuid4())
        config.session_id = session_id
        started_at = self._now()

        partial: WalkForwardReport | None = None

        try:
            windows = self._load_windows(config)
            backtest_reports = await self._run_windows(config, windows, session_id)
            report = self._build_report(config, backtest_reports, session_id, started_at)
            partial = report
            return report

        finally:
            if partial is not None:
                await self._persist(partial)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_all_data(
    config: WalkForwardConfig,
) -> dict[tuple[str, str], pl.DataFrame]:
    """Load all (symbol, interval) data without date filtering."""
    from datetime import datetime

    data: dict[tuple[str, str], pl.DataFrame] = {}
    far_past = datetime(2000, 1, 1, tzinfo=UTC)
    far_future = datetime(2100, 1, 1, tzinfo=UTC)

    for symbol in config.symbols:
        for interval in config.intervals:
            try:
                df = config.loader.load(symbol, interval, far_past, far_future)
                data[(symbol, interval)] = df.sort("date")
            except FileNotFoundError:
                logger.warning("WalkForwardRunner: no data for %s/%s", symbol, interval)
    return data


def _compute_windows(
    index_df: pl.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[object, object, object, object]]:
    """
    Compute (train_start, train_end, test_start, test_end) tuples from the index df.
    """
    dates = index_df["date"].to_list()
    n = len(dates)
    window_size = train_bars + test_bars
    windows = []

    pos = 0
    while pos + window_size <= n:
        train_start = dates[pos]
        train_end = dates[pos + train_bars - 1]
        test_start = dates[pos + train_bars]
        test_end = dates[pos + window_size - 1]
        windows.append((train_start, train_end, test_start, test_end))
        pos += step_bars

    return windows


def _combine_equity_curves(reports: list) -> pl.DataFrame:
    """Concatenate per-window equity curves into one continuous curve."""
    frames: list[pl.DataFrame] = []
    for r in reports:
        frames.append(r.equity_curve)

    if not frames:
        return pl.DataFrame({"date": [], "equity": []})

    return pl.concat(frames).sort("date").unique(subset=["date"], keep="last")
