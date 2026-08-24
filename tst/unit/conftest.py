"""
Root conftest.py for trading-research's unit tests.

Provides a session-scoped Postgres fixture for tests that run a full
BacktestSession (or other TestingSession subclasses) against the pinned
trading-platform pipeline. Mirrors the proven pattern in
trading-integ-tests/strategy/conftest.py.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from trading.app.database import init_db


@pytest.fixture(scope="session")
def pg_container():
    """Start a real Postgres container for the test session."""
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest_asyncio.fixture
async def pg_engine(pg_container) -> AsyncEngine:
    """
    Async engine connected to the test Postgres container.

    Tables are created on first use (init_db is idempotent).
    All rows are truncated after each test so tests don't share state.
    """
    url = (
        pg_container.get_connection_url()
        .replace("psycopg2", "asyncpg")
        .replace("postgresql://", "postgresql+asyncpg://")
    )
    eng = create_async_engine(url, echo=False)
    await init_db(eng)
    yield eng
    from sqlalchemy import text

    async with eng.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE TABLE orders, signals, positions, instruments, "
                "heartbeats, audit_logs, decision_logs, tick_logs CASCADE"
            )
        )
    await eng.dispose()
