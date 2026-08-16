from __future__ import annotations

import random
import uuid
from pathlib import Path

import polars as pl

from research.backtesting.portfolio import TradeRecord
from research.monte_carlo.report import MonteCarloConfig, MonteCarloReport
from research.registry import session_type
from research.session import TestingSession


@session_type("monte_carlo")
class MonteCarloSimulator(TestingSession):
    """
    Monte Carlo simulation on a set of completed trades.

    Takes the trade list from a backtest and re-orders / resamples them
    across *n_trials* to estimate the distribution of outcomes.

    Methods
    -------
    shuffle:
        Randomly shuffle the order of trades within each trial.
        Simulates sequencing risk — does luck explain good performance?
    bootstrap:
        Sample trades with replacement.
        Simulates variance in the trade sample — is the edge robust?

    Ruin is defined as losing > 50% of initial equity in any trial.
    """

    _config_cls = MonteCarloConfig

    def __init__(
        self,
        config: MonteCarloConfig,
        trades: list[TradeRecord],
        results_dir: Path,
    ) -> None:
        super().__init__(results_dir=results_dir)
        self._config = config
        self._trades = trades

    async def run(self) -> MonteCarloReport:
        config = self._config
        session_id = config.session_id or str(uuid.uuid4())
        config.session_id = session_id
        started_at = self._now()

        partial: MonteCarloReport | None = None

        try:
            rng = random.Random(config.seed)
            pnls = [t.pnl for t in self._trades]
            initial = config.initial_equity

            trial_returns: list[float] = []
            trial_drawdowns: list[float] = []

            for _trial_idx in range(config.n_trials):
                trial_pnls = _sample(pnls, config.method, rng, config.slippage_sigma)
                equity = initial
                peak = initial
                max_dd = 0.0

                for pnl in trial_pnls:
                    equity += pnl
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak if peak > 0 else 0.0
                    if dd > max_dd:
                        max_dd = dd

                trial_returns.append((equity - initial) / initial)
                trial_drawdowns.append(max_dd)

            ret_series = pl.Series("return", trial_returns)
            dd_series = pl.Series("max_drawdown", trial_drawdowns)

            ruin_count = sum(1 for r in trial_returns if r < -0.5)

            sorted_rets = sorted(trial_returns)
            n = len(sorted_rets)
            p5 = sorted_rets[max(0, int(n * 0.05))]
            p95 = sorted_rets[min(n - 1, int(n * 0.95))]
            median_dd = sorted(trial_drawdowns)[n // 2]

            report = MonteCarloReport(
                config=config,
                return_distribution=ret_series,
                drawdown_distribution=dd_series,
                probability_of_ruin=ruin_count / config.n_trials,
                percentile_5_return=p5,
                percentile_95_return=p95,
                median_drawdown=median_dd,
                n_trials=config.n_trials,
                session_id=session_id,
                session_type="monte_carlo",
                started_at=started_at,
                finished_at=self._now(),
            )
            partial = report
            return report

        finally:
            if partial is not None:
                await self._persist(partial)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample(
    pnls: list[float],
    method: str,
    rng: random.Random,
    slippage_sigma: float,
) -> list[float]:
    if not pnls:
        return []

    if method == "shuffle":
        trial = list(pnls)
        rng.shuffle(trial)
    else:  # bootstrap
        trial = [rng.choice(pnls) for _ in pnls]

    if slippage_sigma > 0:
        trial = [p + rng.gauss(0.0, slippage_sigma) for p in trial]

    return trial
