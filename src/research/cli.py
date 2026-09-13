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
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from research.backtesting.data_loader import FileDataLoader
from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig
from research.gridsearch import all_strategies as _gs_all_strategies
from research.gridsearch import diagnose as _gs_diagnose
from research.gridsearch import ema as _gs_ema
from research.gridsearch import orb as _gs_orb
from research.gridsearch import rsi as _gs_rsi
from research.gridsearch import vwap as _gs_vwap
from research.indicator_research import correlation as _ind_correlation
from research.indicator_research import decay as _ind_decay
from research.indicator_research import ic as _ind_ic
from research.indicator_research import ic_daily as _ind_ic_daily
from research.indicator_research import quintile as _ind_quintile
from research.indicator_research import regime as _ind_regime
from research.indicator_research import wf_ic as _ind_wf_ic
from research.monte_carlo.report import MonteCarloConfig
from research.monte_carlo.simulator import MonteCarloSimulator
from research.walk_forward.report import WalkForwardConfig
from research.walk_forward.runner import WalkForwardRunner

# Subcommand name -> module exposing add_arguments(parser) / async run(args) -> None.
# These commands print their own progress/results and don't produce a
# results_dir/{session_id}/report.json session, unlike backtest/walk-forward/monte-carlo.
_RUN_ONLY_COMMANDS = {
    "indicator-correlation": _ind_correlation,
    "indicator-decay": _ind_decay,
    "indicator-ic": _ind_ic,
    "indicator-ic-daily": _ind_ic_daily,
    "indicator-quintile": _ind_quintile,
    "indicator-regime": _ind_regime,
    "indicator-wf-ic": _ind_wf_ic,
    "hyperparam-search": _gs_ema,
    "rsi-search": _gs_rsi,
    "orb-search": _gs_orb,
    "vwap-search": _gs_vwap,
    "all-strategies": _gs_all_strategies,
    "diagnose-signals": _gs_diagnose,
}


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
    connect_args = {"ssl": False} if args.db_no_ssl else {}
    # A real, isolated scratch schema per invocation — never "public" — so a
    # manual CLI backtest run can't wipe a real database's default schema.
    db_schema = f"bt_{args.session_id or uuid.uuid4().hex[:8]}"
    try:
        session = BacktestSession(
            config=config,
            db_engine=db_engine,
            results_dir=args.results_dir,
            db_schema=db_schema,
            connect_args=connect_args,
        )
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

    p_ind_corr = sub.add_parser(
        "indicator-correlation", help="Cross-indicator correlation matrix over historical signals"
    )
    _ind_correlation.add_arguments(p_ind_corr)

    p_ind_decay = sub.add_parser(
        "indicator-decay", help="Indicator predictive-power decay across forward horizons"
    )
    _ind_decay.add_arguments(p_ind_decay)

    p_ind_ic = sub.add_parser(
        "indicator-ic", help="Information Coefficient (Spearman IC) per indicator"
    )
    _ind_ic.add_arguments(p_ind_ic)

    p_ind_ic_daily = sub.add_parser(
        "indicator-ic-daily", help="Information Coefficient on daily-interval data"
    )
    _ind_ic_daily.add_arguments(p_ind_ic_daily)

    p_ind_quintile = sub.add_parser(
        "indicator-quintile", help="Quintile spread (top vs. bottom bucket forward return)"
    )
    _ind_quintile.add_arguments(p_ind_quintile)

    p_ind_regime = sub.add_parser(
        "indicator-regime", help="Indicator IC broken out by trending/ranging ADX regime"
    )
    _ind_regime.add_arguments(p_ind_regime)

    p_ind_wf_ic = sub.add_parser(
        "indicator-wf-ic", help="Walk-forward IC — indicator IC stability across rolling windows"
    )
    _ind_wf_ic.add_arguments(p_ind_wf_ic)

    p_gs_ema = sub.add_parser(
        "hyperparam-search", help="EMA crossover fast/slow/ATR hyperparameter grid search"
    )
    _gs_ema.add_arguments(p_gs_ema)

    p_gs_rsi = sub.add_parser(
        "rsi-search", help="RSI mean-reversion oversold/overbought/ATR grid search"
    )
    _gs_rsi.add_arguments(p_gs_rsi)

    p_gs_orb = sub.add_parser(
        "orb-search", help="Opening Range Breakout orb-bars/ATR grid search"
    )
    _gs_orb.add_arguments(p_gs_orb)

    p_gs_vwap = sub.add_parser(
        "vwap-search", help="VWAP reversion band/ATR grid search"
    )
    _gs_vwap.add_arguments(p_gs_vwap)

    p_gs_all = sub.add_parser(
        "all-strategies", help="Run every registered strategy once on real data (sanity sweep)"
    )
    _gs_all_strategies.add_arguments(p_gs_all)

    p_gs_diag = sub.add_parser(
        "diagnose-signals", help="Run one backtest and tally signal-rejection reasons from audit_logs"
    )
    _gs_diagnose.add_arguments(p_gs_diag)

    args = parser.parse_args()

    if args.command == "backtest":
        session_id = asyncio.run(_run_backtest(args))
    elif args.command == "walk-forward":
        session_id = asyncio.run(_run_walk_forward(args))
    elif args.command == "monte-carlo":
        session_id = asyncio.run(_run_monte_carlo(args))
    elif args.command in _RUN_ONLY_COMMANDS:
        asyncio.run(_RUN_ONLY_COMMANDS[args.command].run(args))
        return
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    print(f"session_id: {session_id}")
    print(f"report: {args.results_dir / session_id / 'report.json'}")


if __name__ == "__main__":
    main()
