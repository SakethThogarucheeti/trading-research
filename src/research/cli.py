"""
trading-research CLI.

Usage
-----
    uv run research backtest --strategy rsi_mean_reversion --symbols INFY TCS \\
        --start 2024-01-01 --end 2025-01-01 --data-dir ../trading-platform/data \\
        --db-url postgresql+asyncpg://user:pass@localhost/trading

    uv run research walk-forward --strategy rsi_mean_reversion --symbols INFY \\
        --start 2024-01-01 --end 2025-01-01 --data-dir ../trading-platform/data \\
        --db-url postgresql+asyncpg://user:pass@localhost/trading \\
        --train-bars 500 --test-bars 100

    uv run research monte-carlo --session-id <backtest-session-id> \\
        --results-dir results --trials 10000 --method bootstrap

Every command writes a SessionReport to results_dir/{session_id}/report.json
(+ .html) — the same contract trading-platform's /api/reports/* endpoints
already serve, so every run is visible on trading-dashboard automatically.

This is a bare M1 skeleton: one strategy, an explicit symbol list, no
universe/feature integration yet (that's M2). --gates/--sizer risk flags
are M4.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from research.backtesting.data_loader import FileDataLoader
from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig
from research.monte_carlo.report import MonteCarloConfig
from research.monte_carlo.simulator import MonteCarloSimulator
from research.walk_forward.report import WalkForwardConfig
from research.walk_forward.runner import WalkForwardRunner


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _make_db_engine(db_url: str, no_ssl: bool):
    connect_args = {"ssl": False} if no_ssl else {}
    return create_async_engine(db_url, connect_args=connect_args)


def _add_common_backtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--strategy", required=True, help="registered strategy id, e.g. rsi_mean_reversion"
    )
    p.add_argument(
        "--symbols", required=True, nargs="+", help="explicit symbol list (universe support is M2)"
    )
    p.add_argument("--start", required=True, type=_parse_date, help="ISO date, e.g. 2024-01-01")
    p.add_argument("--end", required=True, type=_parse_date, help="ISO date, e.g. 2025-01-01")
    p.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Parquet/CSV data dir, e.g. ../trading-platform/data",
    )
    p.add_argument(
        "--db-url", required=True, help="async SQLAlchemy URL, e.g. postgresql+asyncpg://..."
    )
    p.add_argument(
        "--db-no-ssl",
        action="store_true",
        help="disable SSL negotiation on the DB connection — try this if connections reset "
        "intermittently against a local Postgres container on Windows/Docker Desktop "
        "(not a guaranteed fix; that class of flakiness has more than one cause)",
    )
    p.add_argument(
        "--intervals",
        nargs="+",
        default=None,
        help="candle intervals to load, e.g. day 15min (default: 1min 5min 15min — "
        "must match what's actually in --data-dir, missing intervals are silently skipped)",
    )
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--params", default="{}", help="JSON dict of strategy params")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--session-id", default="", help="pin a session id instead of generating one")


def _build_algo_settings(
    strategy: str, symbols: list[str], params: dict, equity: float, intervals: list[str] | None
):
    from trading.config.settings import AlgoSettings

    return AlgoSettings(
        name=f"research-{strategy}",
        instruments=symbols,
        strategy_id=strategy,
        strategy_params=params,
        equity=equity,
        candle_intervals=intervals,
    )


async def _run_backtest(args: argparse.Namespace) -> str:
    algo = _build_algo_settings(
        args.strategy, args.symbols, json.loads(args.params), args.equity, args.intervals
    )
    loader = FileDataLoader(args.data_dir)
    config = BacktestConfig(
        algo=algo,
        start=args.start,
        end=args.end,
        loader=loader,
        initial_equity=args.equity,
        session_id=args.session_id,
    )
    db_engine = _make_db_engine(args.db_url, args.db_no_ssl)
    try:
        session = BacktestSession(config=config, db_engine=db_engine, results_dir=args.results_dir)
        report = await session.run()
        return report.session_id
    finally:
        await db_engine.dispose()


async def _run_walk_forward(args: argparse.Namespace) -> str:
    loader = FileDataLoader(args.data_dir)
    intervals = args.intervals or ["day"]
    algo = _build_algo_settings(
        args.strategy, args.symbols, json.loads(args.params), args.equity, intervals
    )
    config = WalkForwardConfig(
        algo=algo,
        symbols=args.symbols,
        intervals=intervals,
        loader=loader,
        initial_equity=args.equity,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars or args.test_bars,
        session_id=args.session_id,
    )
    db_engine = _make_db_engine(args.db_url, args.db_no_ssl)
    try:
        runner = WalkForwardRunner(config=config, db_engine=db_engine, results_dir=args.results_dir)
        report = await runner.run()
        return report.session_id
    finally:
        await db_engine.dispose()


async def _run_monte_carlo(args: argparse.Namespace) -> str:
    report_path = args.results_dir / args.session_id / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"No backtest report found at {report_path}. "
            "Monte Carlo resamples an existing backtest's trades — run "
            "`research backtest` first and pass its session id."
        )
    data = json.loads(report_path.read_text(encoding="utf-8"))
    from research.backtesting.portfolio import TradeRecord

    trades = [
        TradeRecord(
            symbol=t["symbol"],
            side=t["side"],
            qty=t["qty"],
            entry_price=t["entry_price"],
            exit_price=t["exit_price"],
            pnl=t["pnl"],
            entry_time=datetime.fromisoformat(t["entry_time"]),
            exit_time=datetime.fromisoformat(t["exit_time"]),
        )
        for t in data["trades"]
    ]
    config = MonteCarloConfig(
        n_trials=args.trials,
        method=args.method,
        initial_equity=data.get("initial_equity", 100_000.0),
        seed=args.seed,
        slippage_sigma=args.slippage_sigma,
    )
    simulator = MonteCarloSimulator(config=config, trades=trades, results_dir=args.results_dir)
    report = await simulator.run()
    return report.session_id


def main() -> None:
    if platform.system() == "Windows":
        # asyncpg + SQLAlchemy's connection pool hit spurious
        # ConnectionResetErrors under the default ProactorEventLoop on
        # Windows; the selector loop doesn't have this problem.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(prog="research", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="Run a strategy backtest against real data")
    _add_common_backtest_args(p_bt)

    p_wf = sub.add_parser("walk-forward", help="Rolling train/test window analysis")
    _add_common_backtest_args(p_wf)
    p_wf.add_argument("--train-bars", type=int, required=True)
    p_wf.add_argument("--test-bars", type=int, required=True)
    p_wf.add_argument("--step-bars", type=int, default=0, help="defaults to --test-bars")

    p_mc = sub.add_parser("monte-carlo", help="Resample a completed backtest's trades")
    p_mc.add_argument(
        "--session-id", required=True, help="session id of a completed `research backtest` run"
    )
    p_mc.add_argument("--results-dir", type=Path, default=Path("results"))
    p_mc.add_argument("--trials", type=int, default=10_000)
    p_mc.add_argument("--method", choices=["shuffle", "bootstrap"], default="bootstrap")
    p_mc.add_argument("--seed", type=int, default=42)
    p_mc.add_argument("--slippage-sigma", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "backtest":
        session_id = asyncio.run(_run_backtest(args))
    elif args.command == "walk-forward":
        session_id = asyncio.run(_run_walk_forward(args))
    elif args.command == "monte-carlo":
        session_id = asyncio.run(_run_monte_carlo(args))
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    print(f"session_id: {session_id}")
    print(f"report: {args.results_dir / session_id / 'report.json'}")


if __name__ == "__main__":
    main()
