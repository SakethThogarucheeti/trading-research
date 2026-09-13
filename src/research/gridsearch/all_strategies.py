"""
Real-data backtest — every registered strategy, run once each.

Iterates trading_strategy_sdk.factory.registered_strategies() so any new
strategy added to the SDK is picked up automatically without touching this
file. Runs sanity checks per-strategy (final_equity > 0, drawdown/win-rate in
[0, 1]) and prints a summary.

Ported from trading-integ-tests' strategy/strategies/test_real_data_all_strategies.py.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from trading_strategy_sdk.factory import registered_strategies

from research.backtesting.data_loader import FileDataLoader
from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig
from research.gridsearch.common import make_db_engine, parse_date

_DEFAULT_SYMBOLS = ["INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK"]
_DEFAULT_START = datetime(2026, 4, 25, tzinfo=UTC)
_DEFAULT_END = datetime(2026, 5, 25, tzinfo=UTC)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-url", required=True)
    p.add_argument("--db-no-ssl", action="store_true")
    p.add_argument("--symbols", nargs="+", default=list(_DEFAULT_SYMBOLS))
    p.add_argument("--interval", default="15min")
    p.add_argument("--start", type=parse_date, default=_DEFAULT_START)
    p.add_argument("--end", type=parse_date, default=_DEFAULT_END)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Subset of strategy ids to run (default: all registered strategies)",
    )
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """Run every registered strategy once on real data and print a summary table."""
    from trading.config.settings import AlgoSettings

    strategy_ids = args.strategies or list(registered_strategies().keys())
    if not strategy_ids:
        print("WARNING: No registered strategies found (registered_strategies() is empty)")
        sys.exit(1)

    out_dir = Path(args.results_dir) / "all_strategies"
    db_engine = make_db_engine(args.db_url, args.db_no_ssl)

    def _algo(strategy_id: str) -> AlgoSettings:
        return AlgoSettings(
            name=f"real_{strategy_id}",
            instruments=args.symbols,
            strategy_id=strategy_id,
            candle_intervals=[args.interval],
            equity=args.equity,
        )

    try:
        failures: list[str] = []
        for strategy_id in strategy_ids:
            config = BacktestConfig(
                algo=_algo(strategy_id),
                start=args.start,
                end=args.end,
                loader=FileDataLoader(args.data_dir),
                initial_equity=args.equity,
                slippage_pct=args.slippage,
            )
            session = BacktestSession(
                config=config,
                db_engine=db_engine,
                results_dir=out_dir,
                db_schema=f"bt_real_{strategy_id}",
            )
            report = await session.run()

            if report is None:
                print(f"WARNING: {strategy_id} produced no report")
                failures.append(strategy_id)
                continue
            if not (report.final_equity > 0):
                print(f"WARNING: {strategy_id} final_equity <= 0 ({report.final_equity})")
                failures.append(strategy_id)
            if not (0.0 <= report.max_drawdown <= 1.0):
                print(f"WARNING: {strategy_id} max_drawdown out of range ({report.max_drawdown})")
                failures.append(strategy_id)
            if not (0.0 <= report.win_rate <= 1.0):
                print(f"WARNING: {strategy_id} win_rate out of range ({report.win_rate})")
                failures.append(strategy_id)

            pnl = report.final_equity - args.equity
            print(f"\n{'=' * 65}")
            print(f"  {strategy_id}")
            print(f"  Symbols  : {', '.join(args.symbols)}")
            print(f"  Period   : {args.start.date()} to {args.end.date()}")
            print(f"  Equity   : {args.equity:,.0f} -> {report.final_equity:,.2f}  ({pnl:+,.0f})")
            print(f"  Trades   : {report.total_trades}")
            print(f"  Win rate : {report.win_rate:.1%}")
            print(f"  Max DD   : {report.max_drawdown:.1%}")
            print(f"  Sharpe   : {report.sharpe_ratio:.2f}")
            print(f"  CAGR     : {report.cagr:.1%}")
            print(f"  Calmar   : {report.calmar_ratio:.2f}")
            print(f"  Report   : {out_dir / report.session_id / 'report.html'}")
            print(f"{'=' * 65}")

        if failures:
            print(f"\nWARNING: {len(failures)}/{len(strategy_ids)} strategies failed sanity checks: {failures}")
            sys.exit(1)
    finally:
        await db_engine.dispose()
