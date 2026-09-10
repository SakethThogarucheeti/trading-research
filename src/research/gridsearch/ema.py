"""
EMA Crossover hyperparameter grid search.

Sweeps fast/slow EMA combinations and ATR multipliers over real data, ranks
by Sharpe ratio, and prints a sorted results table.

Grid defaults (symbols, periods, ATR multipliers, date range, equity) come
from the bundled strategy_config.json (research.gridsearch.config) — override
any of them via flags.

Each completed combo is appended to grid_search_results.csv immediately so
partial progress is preserved if the run is interrupted.

Ported from trading-integ-tests' strategy/strategies/test_hyperparam_search.py.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from research.backtesting.data_loader import FileDataLoader
from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig, BacktestReport
from research.gridsearch.common import append_csv, make_db_engine, parse_date
from research.gridsearch.config import (
    compute_start,
    default_end,
    hyperparam_search_defaults,
    strategy_grid,
)

_CSV_FIELDS = [
    "fast", "slow", "atr_multiplier", "sharpe", "cagr", "max_dd", "calmar",
    "win_rate", "profit_factor", "total_trades", "pnl", "final_equity",
]  # fmt: skip


@dataclass
class GridResult:
    fast: int
    slow: int
    atr_multiplier: float
    sharpe: float
    cagr: float
    max_dd: float
    calmar: float
    win_rate: float
    profit_factor: float
    total_trades: int
    pnl: float
    final_equity: float

    def label(self) -> str:
        return f"EMA({self.fast}/{self.slow}) ATRx{self.atr_multiplier}"

    def to_row(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def add_arguments(p: argparse.ArgumentParser) -> None:
    hp = hyperparam_search_defaults()
    grid = strategy_grid("ema_crossover")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-url", required=True)
    p.add_argument("--db-no-ssl", action="store_true")
    p.add_argument("--symbols", nargs="+", default=hp.get("symbols", []))
    p.add_argument("--interval", default=hp.get("interval", "15min"))
    p.add_argument("--equity", type=float, default=float(hp.get("equity", 10_000.0)))
    p.add_argument("--end", type=parse_date, default=default_end())
    p.add_argument("--months", type=int, default=hp.get("months"))
    p.add_argument("--fast-periods", type=int, nargs="+", default=grid.get("fast_periods", []))
    p.add_argument("--slow-periods", type=int, nargs="+", default=grid.get("slow_periods", []))
    p.add_argument(
        "--atr-multipliers", type=float, nargs="+", default=grid.get("atr_multipliers", [])
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """
    Run all fast/slow/ATR combos — each gets its own Postgres schema so they
    don't interfere. Grid defaults come from the bundled strategy_config.json.
    Results are appended to grid_search_results.csv as each combo finishes.
    """
    start = compute_start(args.end, args.months)
    out_dir = Path(args.results_dir) / "hyperparam_search"
    results_csv = out_dir / "grid_search_results.csv"

    db_engine = make_db_engine(args.db_url, args.db_no_ssl)

    async def _run_one(fast: int, slow: int, atr_multiplier: float) -> BacktestReport:
        from trading.config.settings import AlgoSettings

        schema = f"bt_{fast}_{slow}_{str(atr_multiplier).replace('.', '_')}"
        config = BacktestConfig(
            algo=AlgoSettings(
                name=schema,
                instruments=args.symbols,
                strategy_id="ema_crossover",
                candle_intervals=[args.interval],
                equity=args.equity,
            ),
            start=start,
            end=args.end,
            loader=FileDataLoader(args.data_dir),
            initial_equity=args.equity,
            slippage_pct=0.05,
            strategy_params={"fast": fast, "slow": slow, "atr_multiplier": atr_multiplier},
        )
        session = BacktestSession(
            config=config,
            db_engine=db_engine,
            results_dir=out_dir,
            db_schema=schema,
            keep_schema=True,
        )
        return await session.run()

    def _print_table(results: list[GridResult]) -> None:
        ranked = sorted(results, key=lambda r: r.sharpe, reverse=True)
        W = 95
        header = (
            f"{'Params':<28} {'Sharpe':>8} {'PnL':>8} {'CAGR':>7} {'MaxDD':>7}"
            f" {'Calmar':>7} {'WinR':>6} {'PF':>6} {'Trades':>7}"
        )
        print(f"\n{'=' * W}")
        print("  EMA Crossover Hyperparameter Grid Search")
        print(f"  Symbols : {', '.join(args.symbols)}")
        print(f"  Period  : {start.date()} to {args.end.date()}")
        print(f"  Interval: {args.interval}   Equity: {args.equity:,.0f}")
        print(f"{'=' * W}")
        print(header)
        print("-" * W)
        for r in ranked:
            print(
                f"  {r.label():<26} {r.sharpe:>8.3f} {r.pnl:>+8.0f} {r.cagr:>7.1%}"
                f" {r.max_dd:>7.1%} {r.calmar:>7.2f} {r.win_rate:>6.0%}"
                f" {r.profit_factor:>6.2f} {r.total_trades:>7}"
            )
        print(f"{'=' * W}")
        if ranked:
            best = ranked[0]
            print(
                f"  Best by Sharpe: {best.label()}"
                f"  (Sharpe={best.sharpe:.3f}, PnL={best.pnl:+.0f},"
                f" WinR={best.win_rate:.0%}, PF={best.profit_factor:.2f})"
            )
        print(f"{'=' * W}\n")

    try:
        all_combos = itertools.product(args.fast_periods, args.slow_periods, args.atr_multipliers)
        combos = [(fast, slow, atr_mult) for fast, slow, atr_mult in all_combos if fast < slow]

        sem = asyncio.Semaphore(args.concurrency)
        csv_lock = asyncio.Lock()

        print(f"\n  Running {len(combos)} combos (concurrency={args.concurrency})", flush=True)
        print(
            f"  Grid  : fast={args.fast_periods}  slow={args.slow_periods}  "
            f"atr={args.atr_multipliers}",
            flush=True,
        )
        print(f"  Results: {results_csv}", flush=True)

        async def _run_and_record(i: int, fast: int, slow: int, atr_mult: float) -> GridResult:
            async with sem:
                print(
                    f"  [{i}/{len(combos)}] EMA({fast}/{slow}) ATRx{atr_mult} starting...",
                    flush=True,
                )
                report = await _run_one(fast, slow, atr_mult)
                result = GridResult(
                    fast=fast,
                    slow=slow,
                    atr_multiplier=atr_mult,
                    sharpe=report.sharpe_ratio,
                    cagr=report.cagr,
                    max_dd=report.max_drawdown,
                    calmar=report.calmar_ratio,
                    win_rate=report.win_rate,
                    profit_factor=report.profit_factor,
                    total_trades=report.total_trades,
                    pnl=report.final_equity - args.equity,
                    final_equity=report.final_equity,
                )
                async with csv_lock:
                    append_csv(result.to_row(), results_csv, _CSV_FIELDS)
                print(
                    f"  [{i}/{len(combos)}] EMA({fast}/{slow}) ATRx{atr_mult} done"
                    f" — sharpe={result.sharpe:.3f}  pnl={result.pnl:+.0f}"
                    f"  win%={result.win_rate:.0%}  pf={result.profit_factor:.2f}"
                    f"  trades={result.total_trades}",
                    flush=True,
                )
                return result

        if not combos:
            print("WARNING: No valid (fast, slow) combos to run (need fast < slow)")
            sys.exit(1)

        results: list[GridResult] = await asyncio.gather(
            *[
                _run_and_record(i, fast, slow, atr_mult)
                for i, (fast, slow, atr_mult) in enumerate(combos, 1)
            ]
        )

        _print_table(list(results))

        from sqlalchemy import text

        async with db_engine.begin() as conn:
            for fast, slow, atr_mult in combos:
                schema = f"bt_{fast}_{slow}_{str(atr_mult).replace('.', '_')}"
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    finally:
        await db_engine.dispose()
