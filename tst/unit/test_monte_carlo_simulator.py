"""
Baseline coverage for MonteCarloSimulator.run() -- no DB/testcontainers needed,
the simulator only takes a list of TradeRecord and writes JSON+HTML locally
via TestingSession._persist (results_dir).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from research.backtesting.portfolio import TradeRecord
from research.monte_carlo.report import MonteCarloConfig, MonteCarloReport
from research.monte_carlo.simulator import MonteCarloSimulator

_START = datetime(2024, 1, 2, tzinfo=UTC)


def _trade(pnl: float) -> TradeRecord:
    return TradeRecord(
        symbol="INFY",
        side="BUY",
        qty=1,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        pnl=pnl,
        entry_time=_START,
        exit_time=_START + timedelta(hours=1),
    )


async def test_monte_carlo_simulator_runs_end_to_end(tmp_path):
    """run() completes and returns a well-formed report; report.json/html get persisted."""
    trades = [_trade(50.0), _trade(-20.0), _trade(30.0), _trade(-10.0), _trade(15.0)]
    config = MonteCarloConfig(n_trials=200, method="bootstrap", seed=42, session_id="mc-test")
    sim = MonteCarloSimulator(config=config, trades=trades, results_dir=tmp_path)

    report = await sim.run()

    assert isinstance(report, MonteCarloReport)
    assert report.session_id == "mc-test"
    assert report.n_trials == 200
    assert len(report.return_distribution) == 200
    assert len(report.drawdown_distribution) == 200
    assert 0.0 <= report.probability_of_ruin <= 1.0

    out_dir = tmp_path / "mc-test"
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.html").exists()


async def test_monte_carlo_simulator_shuffle_method(tmp_path):
    """The 'shuffle' method must also run cleanly (bootstrap is the default)."""
    trades = [_trade(10.0), _trade(-5.0)]
    config = MonteCarloConfig(n_trials=50, method="shuffle", seed=1, session_id="mc-shuffle")
    sim = MonteCarloSimulator(config=config, trades=trades, results_dir=tmp_path)

    report = await sim.run()

    assert report.n_trials == 50


async def test_monte_carlo_simulator_handles_no_trades(tmp_path):
    """Zero trades must not crash -- _sample returns [] and equity never moves."""
    config = MonteCarloConfig(n_trials=10, seed=7, session_id="mc-empty")
    sim = MonteCarloSimulator(config=config, trades=[], results_dir=tmp_path)

    report = await sim.run()

    assert report.n_trials == 10
    assert report.probability_of_ruin == 0.0
