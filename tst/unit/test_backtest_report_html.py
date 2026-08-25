"""
Fast baseline coverage for BacktestReport.to_html() -- constructs synthetic
report data directly (no need to run a full BacktestSession) since to_html()
only depends on the dataclass's own fields. Complements the slower end-to-end
coverage in test_backtest_smoke.py (needs Postgres via testcontainers).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from trading.config.settings import AlgoSettings

from research.backtesting.data_loader import DataLoader
from research.backtesting.portfolio import TradeRecord
from research.backtesting.report import BacktestConfig, BacktestReport

_START = datetime(2024, 1, 2, tzinfo=UTC)


class _NullLoader(DataLoader):
    def load(self, symbol, interval, start, end):  # noqa: ANN001
        raise NotImplementedError


def _algo() -> AlgoSettings:
    return AlgoSettings(name="bt_test", instruments=["INFY"], strategy_id="ema_crossover")


def _trade(symbol: str, side: str, pnl: float) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        side=side,
        qty=1,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        pnl=pnl,
        entry_time=_START,
        exit_time=_START + timedelta(hours=1),
    )


def _backtest_report(trades: list[TradeRecord]) -> BacktestReport:
    config = BacktestConfig(
        algo=_algo(),
        start=_START,
        end=_START + timedelta(days=1),
        loader=_NullLoader(),
    )
    equity_curve = pl.DataFrame(
        {
            "date": [_START, _START + timedelta(days=1), _START + timedelta(days=2)],
            "equity": [100_000.0, 98_000.0, 101_000.0],
        }
    )
    return BacktestReport(
        config=config,
        equity_curve=equity_curve,
        trades=trades,
        sharpe_ratio=1.1,
        max_drawdown=-0.02,
        max_drawdown_duration=timedelta(days=1),
        win_rate=0.5,
        profit_factor=1.2,
        cagr=0.1,
        calmar_ratio=0.5,
        total_trades=len(trades),
        final_equity=101_000.0,
    )


def test_backtest_report_html_generated():
    """to_html() must return non-empty Plotly-embedded HTML."""
    report = _backtest_report([_trade("INFY", "BUY", 50.0)])
    html = report.to_html()
    assert "<html" in html.lower()
    assert "plotly" in html.lower()


def test_backtest_report_html_includes_algo_name_and_session_id():
    report = _backtest_report([])
    report.session_id = "bt-test-session"
    html = report.to_html()
    assert "bt_test" in html
    assert "bt-test-session" in html


def test_backtest_report_html_omits_trade_pnl_chart_when_no_trades():
    """With zero trades, the trade-P&L bar chart section must be skipped, not crash."""
    report = _backtest_report([])
    html = report.to_html()
    assert "<html" in html.lower()
    assert "Trade P&L" not in html


def test_backtest_report_html_colors_pnl_bars_by_sign():
    report = _backtest_report([_trade("INFY", "BUY", 50.0), _trade("TCS", "SELL", -20.0)])
    html = report.to_html()
    assert "#4CAF50" in html
    assert "#F44336" in html
