"""Shared helpers for the gridsearch CLI subcommands (ema/rsi/orb/vwap/all_strategies/diagnose)."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def make_db_engine(db_url: str, no_ssl: bool) -> AsyncEngine:
    connect_args = {"ssl": False} if no_ssl else {}
    return create_async_engine(db_url, connect_args=connect_args)


def append_csv(row: dict[str, Any], csv_path: Path, fieldnames: list[str]) -> None:
    """Append one result row to the CSV, writing the header only on first call."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
