"""
Diagnostic: run one EMA-crossover backtest, keep its schema, then query
audit_logs for signal-rejection reasons.

Runs a single fixed-param backtest (fast/slow/atr_multiplier configurable via
flags, defaults matching the source test's 9/21/1.5 combo), keeps the
schema-isolated DB around long enough to inspect it, tallies "rejected: ..."
messages from audit_logs by reason, prints a summary, then drops the schema.

Ported from trading-integ-tests' strategy/strategies/test_diagnose_signals.py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from research.backtesting.data_loader import FileDataLoader
from research.backtesting.engine import BacktestSession
from research.backtesting.report import BacktestConfig, BacktestReport
from research.gridsearch.common import make_db_engine, parse_date

_DEFAULT_END = datetime(2026, 4, 17, tzinfo=UTC)
_DEFAULT_SYMBOLS = ["INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--db-url", required=True)
    p.add_argument("--db-no-ssl", action="store_true")
    p.add_argument("--symbols", nargs="+", default=list(_DEFAULT_SYMBOLS))
    p.add_argument("--interval", default="15min")
    p.add_argument("--end", type=parse_date, default=_DEFAULT_END)
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--fast", type=int, default=9)
    p.add_argument("--slow", type=int, default=21)
    p.add_argument("--atr-multiplier", type=float, default=1.5)
    p.add_argument("--schema", default=None, help="Defaults to bt_diag_<fast>_<slow>_<atr>")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--keep-schema",
        action="store_true",
        help="Skip the final DROP SCHEMA (leave the isolated DB around for manual inspection)",
    )


async def run(args: argparse.Namespace) -> None:
    """Run one EMA-crossover backtest and diagnose trade counts vs audit_logs rejections."""
    from trading.config.settings import AlgoSettings

    start = args.end - timedelta(days=args.lookback_days)
    schema = args.schema or f"bt_diag_{args.fast}_{args.slow}_{str(args.atr_multiplier).replace('.', '_')}"
    out_dir = Path(args.results_dir) / "diagnose_signals"

    db_engine = make_db_engine(args.db_url, args.db_no_ssl)

    try:
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
            strategy_params={
                "fast": args.fast,
                "slow": args.slow,
                "atr_multiplier": args.atr_multiplier,
            },
        )

        session = BacktestSession(
            config=config,
            db_engine=db_engine,
            results_dir=out_dir,
            db_schema=schema,
            keep_schema=True,
        )
        report: BacktestReport = await session.run()

        print(f"\n{'=' * 60}")
        print(f"  Trades: {report.total_trades}  Final equity: {report.final_equity:,.0f}")
        print(f"{'=' * 60}")

        # ------------------------------------------------------------------
        # Read audit_logs from the schema-isolated DB
        # ------------------------------------------------------------------
        async with db_engine.connect() as conn:
            await conn.execute(text(f'SET search_path TO "{schema}"'))

            rows = (
                await conn.execute(
                    text("SELECT module, level, message FROM audit_logs ORDER BY created_at")
                )
            ).fetchall()

            print(f"\n--- audit_logs ({len(rows)} rows, rejections only) ---")
            reason_counts: dict[str, int] = defaultdict(int)
            for r in rows:
                if "rejected:" not in r.message:
                    continue
                reason = r.message.split("rejected:")[-1].strip()
                reason_counts[reason] += 1
                parts = r.message.split()
                sig_id = parts[1] if len(parts) > 1 else ""
                print(f"  {reason:<24} signal={sig_id[:8]}")

            if reason_counts:
                print("\n--- rejection reason counts ---")
                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    print(f"  {reason:<30} {count}")

        if not args.keep_schema:
            async with db_engine.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))

        print(f"{'=' * 60}\n")
    finally:
        await db_engine.dispose()
