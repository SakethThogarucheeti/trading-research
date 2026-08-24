"""
Baseline coverage for WalkForwardReport.to_html() -- constructs synthetic
report data directly (no need to run a full WalkForwardRunner) since
to_html() only depends on the dataclass's own fields.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from trading.config.settings import AlgoSettings

from research.backtesting.data_loader import DataLoader
from research.backtesting.report import BacktestConfig, BacktestReport
from research.walk_forward.report import WalkForwardConfig, WalkForwardReport

_START = datetime(2024, 1, 2, tzinfo=UTC)


class _NullLoader(DataLoader):
    def load(self, symbol, interval, start, end):  # noqa: ANN001
        raise NotImplementedError


def _algo() -> AlgoSettings:
    return AlgoSettings(name="wf_test", instruments=["INFY"], strategy_id="ema_crossover")


def _backtest_report(sharpe: float, max_dd: float) -> BacktestReport:
    config = BacktestConfig(
        algo=_algo(),
        start=_START,
        end=_START + timedelta(days=1),
        loader=_NullLoader(),
    )
    equity_curve = pl.DataFrame(
        {"date": [_START, _START + timedelta(days=1)], "equity": [100_000.0, 101_000.0]}
    )
    return BacktestReport(
        config=config,
        equity_curve=equity_curve,
        trades=[],
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_drawdown_duration=timedelta(days=0),
        win_rate=0.5,
        profit_factor=1.2,
        cagr=0.1,
        calmar_ratio=0.5,
        total_trades=0,
        final_equity=101_000.0,
    )


def _walk_forward_report() -> WalkForwardReport:
    windows = [_backtest_report(1.2, -0.05), _backtest_report(-0.3, -0.12)]
    config = WalkForwardConfig(algo=_algo(), loader=_NullLoader(), symbols=["INFY"])
    combined = pl.DataFrame(
        {"date": [_START, _START + timedelta(days=1)], "equity": [100_000.0, 102_000.0]}
    )
    return WalkForwardReport(
        config=config,
        windows=windows,
        aggregate_sharpe=0.45,
        aggregate_max_drawdown=-0.12,
        aggregate_win_rate=0.5,
        combined_equity_curve=combined,
        session_id="wf-test-session",
    )


def test_walk_forward_report_html_generated():
    """to_html() must return non-empty Plotly-embedded HTML."""
    report = _walk_forward_report()
    html = report.to_html()
    assert "<html" in html.lower()
    assert "plotly" in html.lower()


def test_walk_forward_report_html_includes_window_count_and_session_id():
    report = _walk_forward_report()
    html = report.to_html()
    assert "wf-test-session" in html
    assert "2" in html  # len(self.windows) rendered in the header


def test_walk_forward_report_html_colors_sharpe_bars_by_sign():
    report = _walk_forward_report()
    html = report.to_html()
    # One positive Sharpe (green) and one negative (red) window in the fixture
    assert "#4CAF50" in html
    assert "#F44336" in html
