"""
RSI Mean-Reversion hyperparameter grid search.

Sweeps oversold/overbought thresholds and ATR multipliers. Grid defaults come
from the bundled strategy_config.json (grids.rsi_mean_reversion).

Each completed combo is appended to rsi_search_results.csv immediately.

Ported from trading-integ-tests' strategy/strategies/test_rsi_search.py.
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
    "oversold", "overbought", "atr_multiplier", "sharpe", "cagr", "max_dd", "calmar",
    "win_rate", "profit_factor", "total_trades", "pnl", "final_equity",
]  # fmt: skip


@dataclass
class GridResult:
    oversold: float
    overbought: float
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
        return f"RSI({self.oversold:.0f}/{self.overbought:.0f}) ATRx{self.atr_multiplier}"

    def to_row(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def add_arguments(p: argparse.ArgumentParser) -> None:
    hp = hyperparam_search_defaults()
    grid = strategy_grid("rsi_mean_reversion")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-url", required=True)
    p.add_argument("--db-no-ssl", action="store_true")
    p.add_argument("--symbols", nargs="+", default=hp.get("symbols", []))
    p.add_argument("--interval", default=hp.get("interval", "15min"))
    p.add_argument("--equity", type=float, default=float(hp.get("equity", 10_000.0)))
    p.add_argument("--end", type=parse_date, default=default_end())
    p.add_argument("--months", type=int, default=hp.get("months"))
    p.add_argument(
        "--oversold-levels", type=float, nargs="+", default=grid.get("oversold_levels", [30])
    )
    p.add_argument(
        "--overbought-levels", type=float, nargs="+", default=grid.get("overbought_levels", [70])
    )
    p.add_argument(
        "--atr-multipliers", type=float, nargs="+", default=grid.get("atr_multipliers", [1.5])
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """RSI mean-reversion grid search — reads grid defaults from strategy_config.json."""
    start = compute_start(args.end, args.months)
    out_dir = Path(args.results_dir) / "rsi_search"
    results_csv = out_dir / "rsi_search_results.csv"

    db_engine = make_db_engine(args.db_url, args.db_no_ssl)

    async def _run_one(oversold: float, overbought: float, atr_mult: float) -> BacktestReport:
        from trading.config.settings import AlgoSettings

        schema = f"bt_rsi_{int(oversold)}_{int(overbought)}_{str(atr_mult).replace('.', '_')}"
        config = BacktestConfig(
            algo=AlgoSettings(
                name=schema,
                instruments=args.symbols,
                strategy_id="rsi_mean_reversion",
                candle_intervals=[args.interval],
                equity=args.equity,
            ),
            start=start,
            end=args.end,
            loader=FileDataLoader(args.data_dir),
            initial_equity=args.equity,
            slippage_pct=0.05,
            strategy_params={
                "oversold": oversold,
                "overbought": overbought,
                "atr_multiplier": atr_mult,
            },
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
            f"{'Params':<30} {'Sharpe':>8} {'PnL':>8} {'CAGR':>7} {'MaxDD':>7}"
            f" {'Calmar':>7} {'WinR':>6} {'PF':>6} {'Trades':>7}"
        )
        print(f"\n{'=' * W}")
        print("  RSI Mean-Reversion Hyperparameter Grid Search")
        print(f"  Symbols : {', '.join(args.symbols)}")
        print(
            f"  Period  : {start.date()} to {args.end.date()}"
            f"  Interval: {args.interval}  Equity: {args.equity:,.0f}"
        )
        print(f"{'=' * W}")
        print(header)
        print("-" * W)
        for r in ranked:
            print(
                f"  {r.label():<28} {r.sharpe:>8.3f} {r.pnl:>+8.0f} {r.cagr:>7.1%}"
                f" {r.max_dd:>7.1%} {r.calmar:>7.2f} {r.win_rate:>6.0%}"
                f" {r.profit_factor:>6.2f} {r.total_trades:>7}"
            )
        print(f"{'=' * W}")
        if ranked:
            best = ranked[0]
            print(
                f"  Best: {best.label()}  Sharpe={best.sharpe:.3f}  PnL={best.pnl:+.0f}"
                f"  WinR={best.win_rate:.0%}  PF={best.profit_factor:.2f}"
            )
        print(f"{'=' * W}\n")

    try:
        all_combos = itertools.product(
            args.oversold_levels, args.overbought_levels, args.atr_multipliers
        )
        combos = [(os_, ob, atr) for os_, ob, atr in all_combos if os_ < ob]

        sem = asyncio.Semaphore(args.concurrency)
        csv_lock = asyncio.Lock()

        print(f"\n  Running {len(combos)} RSI combos (concurrency={args.concurrency})", flush=True)
        print(
            f"  Grid  : oversold={args.oversold_levels}"
            f"  overbought={args.overbought_levels}  atr={args.atr_multipliers}",
            flush=True,
        )

        async def _run_and_record(i: int, os_: float, ob: float, atr: float) -> GridResult:
            async with sem:
                print(
                    f"  [{i}/{len(combos)}] RSI({os_:.0f}/{ob:.0f}) ATRx{atr} starting...",
                    flush=True,
                )
                report = await _run_one(os_, ob, atr)
                result = GridResult(
                    oversold=os_,
                    overbought=ob,
                    atr_multiplier=atr,
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
                    f"  [{i}/{len(combos)}] RSI({os_:.0f}/{ob:.0f}) ATRx{atr} done"
                    f" — sharpe={result.sharpe:.3f}  pnl={result.pnl:+.0f}"
                    f"  win%={result.win_rate:.0%}  trades={result.total_trades}",
                    flush=True,
                )
                return result

        if not combos:
            print("WARNING: No valid (oversold, overbought) combos to run (need oversold < overbought)")
            sys.exit(1)

        results: list[GridResult] = await asyncio.gather(
            *[_run_and_record(i, os_, ob, atr) for i, (os_, ob, atr) in enumerate(combos, 1)]
        )
        _print_table(list(results))

        from sqlalchemy import text

        async with db_engine.begin() as conn:
            for os_, ob, atr in combos:
                schema = f"bt_rsi_{int(os_)}_{int(ob)}_{str(atr).replace('.', '_')}"
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    finally:
        await db_engine.dispose()
