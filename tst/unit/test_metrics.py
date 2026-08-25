"""
Baseline coverage for research.backtesting.metrics.max_drawdown.

No prior coverage existed for this module. Scoped to max_drawdown since
that's the function this file's fix touches -- the module's other metrics
(sharpe_ratio, win_rate, profit_factor, cagr, calmar_ratio) remain uncovered,
same gap as before this change, not addressed here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from research.backtesting.metrics import max_drawdown


def _equity_curve(values: list[float]) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "equity": values})


def test_max_drawdown_zero_for_monotone_growth():
    """Monotonically growing equity must have zero max drawdown."""
    curve = _equity_curve([100_000 + i * 1000 for i in range(100)])
    mdd = max_drawdown(curve)
    assert mdd == pytest.approx(0.0)


def test_max_drawdown_detects_peak_to_trough():
    """Drawdown from 100k -> 80k is 20%."""
    values = [100_000, 110_000, 100_000, 80_000, 90_000]
    curve = _equity_curve(values)
    mdd = max_drawdown(curve)
    # Peak is 110k, trough is 80k -> (110k - 80k) / 110k ~= 0.272
    assert mdd == pytest.approx((110_000 - 80_000) / 110_000, abs=1e-4)


def test_max_drawdown_handles_integer_equity_values():
    """
    Regression test: an int-typed equity column must not crash map_elements'
    declared Float64 return_dtype (SchemaError: unexpected value while
    building Series of type Float64; found value of type Int64).
    """
    curve = _equity_curve([100_000, 90_000, 95_000])
    mdd = max_drawdown(curve)
    assert mdd == pytest.approx((100_000 - 90_000) / 100_000, abs=1e-4)
