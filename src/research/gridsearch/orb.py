"""
Opening Range Breakout (ORB) hyperparameter grid search.

Sweeps orb_bars (length of opening range in bars) and ATR multipliers. Grid
defaults come from the bundled strategy_config.json (grids.opening_range_breakout).

Each completed combo is appended to orb_search_results.csv immediately.

Ported from trading-integ-tests' strategy/strategies/test_orb_search.py.
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
    "orb_bars", "atr_multiplier", "sharpe", "cagr", "max_dd", "calmar",
    "win_rate", "profit_factor", "total_trades", "pnl", "final_equity",
]  # fmt: skip


@dataclass
class GridResult:
    orb_bars: int
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
        mins = self.orb_bars * 15
        return f"ORB({mins}min) ATRx{self.atr_multiplier}"

    def to_row(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def add_arguments(p: argparse.ArgumentParser) -> None:
    hp = hyperparam_search_defaults()
    grid = strategy_grid("opening_range_breakout")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-url", required=True)
    p.add_argument("--db-no-ssl", action="store_true")
    p.add_argument("--symbols", nargs="+", default=hp.get("symbols", []))
    p.add_argument("--interval", default=hp.get("interval", "15min"))
    p.add_argument("--equity", type=float, default=float(hp.get("equity", 10_000.0)))
    p.add_argument("--end", type=parse_date, default=default_end())
    p.add_argument("--months", type=int, default=hp.get("months"))
    p.add_argument("--orb-bars", type=int, nargs="+", default=grid.get("orb_bars", [4]))
    p.add_argument(
        "--atr-multipliers", type=float, nargs="+", default=grid.get("atr_multipliers", [1.5])
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """ORB grid search — reads orb_bars/ATR grid from strategy_config.json."""
    start = compute_start(args.end, args.months)
    out_dir = Path(args.results_dir) / "orb_search"
    results_csv = out_dir / "orb_search_results.csv"

    db_engine = make_db_engine(args.db_url, args.db_no_ssl)

    async def _run_one(orb_bars: int, atr_mult: float) -> BacktestReport:
        from trading.config.settings import AlgoSettings

        schema = f"bt_orb_{orb_bars}_{str(atr_mult).replace('.', '_')}"
        config = BacktestConfig(
            algo=AlgoSettings(
                name=schema,
                instruments=args.symbols,
                strategy_id="opening_range_breakout",
                candle_intervals=[args.interval],
                equity=args.equity,
            ),
            start=start,
            end=args.end,
            loader=FileDataLoader(args.data_dir),
            initial_equity=args.equity,
            slippage_pct=0.05,
            strategy_params={"orb_bars": orb_bars, "atr_multiplier": atr_mult},
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
        W = 90
        header = (
            f"{'Params':<24} {'Sharpe':>8} {'PnL':>8} {'CAGR':>7} {'MaxDD':>7}"
            f" {'Calmar':>7} {'WinR':>6} {'PF':>6} {'Trades':>7}"
        )
        print(f"\n{'=' * W}")
        print("  Opening Range Breakout Hyperparameter Grid Search")
        print(f"  Symbols: {', '.join(args.symbols)}  Period: {start.date()} to {args.end.date()}")
        print(f"{'=' * W}")
        print(header)
        print("-" * W)
        for r in ranked:
            print(
                f"  {r.label():<22} {r.sharpe:>8.3f} {r.pnl:>+8.0f} {r.cagr:>7.1%}"
                f" {r.max_dd:>7.1%} {r.calmar:>7.2f} {r.win_rate:>6.0%}"
                f" {r.profit_factor:>6.2f} {r.total_trades:>7}"
            )
        print(f"{'=' * W}")
        if ranked:
            best = ranked[0]
            print(
                f"  Best: {best.label()}  Sharpe={best.sharpe:.3f}"
                f"  PnL={best.pnl:+.0f}  WinR={best.win_rate:.0%}"
            )
        print(f"{'=' * W}\n")

    try:
        combos = list(itertools.product(args.orb_bars, args.atr_multipliers))

        sem = asyncio.Semaphore(args.concurrency)
        csv_lock = asyncio.Lock()

        if not combos:
            print("WARNING: No (orb_bars, atr_multiplier) combos to run")
            sys.exit(1)

        print(f"\n  Running {len(combos)} ORB combos (concurrency={args.concurrency})", flush=True)
        print(f"  Grid  : orb_bars={args.orb_bars}  atr={args.atr_multipliers}", flush=True)

        async def _run_and_record(i: int, orb: int, atr: float) -> GridResult:
            async with sem:
                mins = orb * 15
                print(f"  [{i}/{len(combos)}] ORB({mins}min) ATRx{atr} starting...", flush=True)
                report = await _run_one(orb, atr)
                result = GridResult(
                    orb_bars=orb,
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
                    f"  [{i}/{len(combos)}] ORB({mins}min) ATRx{atr} done"
                    f" — sharpe={result.sharpe:.3f}  pnl={result.pnl:+.0f}"
                    f"  win%={result.win_rate:.0%}  trades={result.total_trades}",
                    flush=True,
                )
                return result

        results: list[GridResult] = await asyncio.gather(
            *[_run_and_record(i, orb, atr) for i, (orb, atr) in enumerate(combos, 1)]
        )
        _print_table(list(results))

        from sqlalchemy import text

        async with db_engine.begin() as conn:
            for orb, atr in combos:
                schema = f"bt_orb_{orb}_{str(atr).replace('.', '_')}"
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    finally:
        await db_engine.dispose()
