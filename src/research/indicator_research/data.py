"""
De-fixture-ed data-loading helpers for indicator-research CLI subcommands.

Ported from trading-integ-tests' ``strategy/indicators/conftest.py`` pytest
fixtures (``data_loader``/``make_store``) as plain callable functions — no
pytest involved, since these are invoked from argparse-driven CLI handlers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from quantindicators.polars_store import PolarsStore
from trading.core.clock import SimulatedClock

from research.backtesting.data_loader import BrokerDataLoader, DataLoader, FileDataLoader
from research.simulators.synthetic_broker import SyntheticDataBroker


def make_data_loader(data_source: str, data_dir: Path) -> DataLoader:
    """
    DataLoader backed by SyntheticDataBroker (``data_source="synthetic"``) or
    real Parquet/CSV files under *data_dir* (``data_source="parquet"``).

    Both return a pl.DataFrame with the same schema — callers see no difference.
    """
    if data_source == "parquet":
        return FileDataLoader(data_dir)
    return BrokerDataLoader(SyntheticDataBroker())


def make_store(
    loader: DataLoader,
    clock: SimulatedClock,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> tuple[PolarsStore, list[dict]]:
    """
    Load bars for the given date window, push every bar into a fresh
    PolarsStore (advancing the simulated clock per bar), and return the
    populated store alongside the raw row list for callers that need it.
    """
    df = loader.load(symbol, interval, start, end)
    store = PolarsStore(maxlen=500)
    rows: list[dict] = []
    for row in df.to_dicts():
        r = {**row, "ts": row["date"]}
        clock.advance(r["ts"])
        store.push(symbol, interval, r)
        rows.append(r)
    return store, rows
