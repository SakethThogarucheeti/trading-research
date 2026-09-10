"""
Walk-forward IC/ICIR evaluation with hyperparam grid.

Each entry in the indicator catalogue is one (indicator, params) combination.
The walk-forward loop:
  1. Splits data into sequential train/test windows (no lookahead).
  2. Feeds train_rows to the indicator for warmup (no scoring).
  3. Computes IC/ICIR only on test_rows (out-of-sample).
  4. Aggregates per-window metrics across symbols.

This combines out-of-sample validation and hyperparam tuning in one pass —
the final ranking by mean_ICIR is the hyperparam selection criterion.

Symbols within each window are evaluated in parallel via a process pool.

Ported from trading-integ-tests' strategy/indicators/test_indicator_wf_ic.py.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from quantindicators.library.aroon import Aroon
from quantindicators.library.bollinger import BollingerBands
from quantindicators.library.candle_body_ratio import CandleBodyRatio
from quantindicators.library.chandelier_exit import ChandelierExit
from quantindicators.library.connors_rsi import ConnorsRSI
from quantindicators.library.coppock_curve import CoppockCurve
from quantindicators.library.distance_from_ma import DistanceFromMA
from quantindicators.library.donchian import DonchianChannels
from quantindicators.library.elder_ray import ElderRay
from quantindicators.library.fisher_transform import FisherTransform
from quantindicators.library.inside_bar import InsideBar
from quantindicators.library.linreg_slope import LinearRegressionSlope
from quantindicators.library.mean_reversion_score import MeanReversionScore
from quantindicators.library.mfi import MFI
from quantindicators.library.normalized_atr import NormalizedATR
from quantindicators.library.opening_range import OpeningRangePosition
from quantindicators.library.price_percentile import PricePercentile
from quantindicators.library.price_vs_52w_high import PriceVs52wHigh
from quantindicators.library.pvt import PVT
from quantindicators.library.rsi import RSI
from quantindicators.library.rsi_divergence import RSIDivergence
from quantindicators.library.rvol import RVOL
from quantindicators.library.session_high_low_pct import SessionHighLowPct
from quantindicators.library.squeeze_momentum import SqueezeMomentum
from quantindicators.library.stochastic import Stochastic
from quantindicators.library.stochastic_rsi import StochasticRSI
from quantindicators.library.tsi import TSI
from quantindicators.library.ultimate_oscillator import UltimateOscillator
from quantindicators.library.upper_shadow_ratio import UpperShadowRatio
from quantindicators.library.volatility_ratio import VolatilityRatio
from quantindicators.library.vroc import VROC
from quantindicators.library.vwap import VWAP
from quantindicators.library.vwap_bands import VWAPBands
from quantindicators.library.williams_r import WilliamsR
from quantindicators.polars_store import PolarsStore
from scipy import stats
from scipy.stats import ConstantInputWarning
from trading.core.clock import SimulatedClock

from research.indicator_research.data import make_data_loader
from research.walk_forward.runner import _compute_windows

_DEFAULT_SYMBOLS = [
    "INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "SBIN", "BAJFINANCE", "BAJAJFINSV", "WIPRO", "HCLTECH", "TECHM", "LT",
    "MARUTI", "SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "TITAN", "ASIANPAINT",
    "NESTLEIND", "HINDUNILVR", "BRITANNIA", "POWERGRID", "NTPC", "ONGC",
    "COALINDIA", "ITC", "TATASTEEL",
]  # fmt: skip

_DEFAULT_HORIZONS = [5, 15, 30, 50]

_NEG = "neg"
_RSI = "rsi"
_ID = "id"

_SESSION_CLASSES = (VWAP, VWAPBands, SessionHighLowPct)


# ---------------------------------------------------------------------------
# Indicator catalogue — the hyperparam grid.
# ---------------------------------------------------------------------------


def _catalogue() -> list[tuple[str, Any, Any, Any]]:
    """Return (label, cls, params, extractor) tuples.

    Special labels handled separately in _evaluate_window/_extract:
      "VWAP_dev"   — compute(); deviation (vwap - close) / close
      "BB_*"       — compute_full(); -pct_b
      "Donchian_*" — compute_full(); -(middle-lower)/width
    """
    return [
        # RSI — period sweep
        ("RSI_7", RSI, RSI.Parameters(period=7), _RSI),
        ("RSI_10", RSI, RSI.Parameters(period=10), _RSI),
        ("RSI_14", RSI, RSI.Parameters(period=14), _RSI),
        ("RSI_21", RSI, RSI.Parameters(period=21), _RSI),
        # Stochastic — k_period sweep
        ("Stoch_9", Stochastic, Stochastic.Parameters(k_period=9, d_period=3), _RSI),
        ("Stoch_14", Stochastic, Stochastic.Parameters(k_period=14, d_period=3), _RSI),
        ("Stoch_21", Stochastic, Stochastic.Parameters(k_period=21, d_period=3), _RSI),
        # VWAP deviation
        ("VWAP_dev", VWAP, VWAP.Parameters(), None),
        # Bollinger %B — std multiplier sweep
        ("BB_20_1.5", BollingerBands, BollingerBands.Parameters(period=20, k=1.5), None),
        ("BB_20_2.0", BollingerBands, BollingerBands.Parameters(period=20, k=2.0), None),
        ("BB_20_2.5", BollingerBands, BollingerBands.Parameters(period=20, k=2.5), None),
        # MFI — period sweep
        ("MFI_10", MFI, MFI.Parameters(period=10), _RSI),
        ("MFI_14", MFI, MFI.Parameters(period=14), _RSI),
        ("MFI_21", MFI, MFI.Parameters(period=21), _RSI),
        # Williams %R — period sweep
        ("WR_10", WilliamsR, WilliamsR.Parameters(period=10), _NEG),
        ("WR_14", WilliamsR, WilliamsR.Parameters(period=14), _NEG),
        # Donchian — period sweep
        ("Donchian_10", DonchianChannels, DonchianChannels.Parameters(period=10), None),
        ("Donchian_20", DonchianChannels, DonchianChannels.Parameters(period=20), None),
        # ConnorsRSI — period sweeps
        (
            "CRSI_3_2_50",
            ConnorsRSI,
            ConnorsRSI.Parameters(rsi_period=3, streak_period=2, rank_period=50),
            _RSI,
        ),
        (
            "CRSI_3_2_100",
            ConnorsRSI,
            ConnorsRSI.Parameters(rsi_period=3, streak_period=2, rank_period=100),
            _RSI,
        ),
        # Fisher Transform — period sweep
        ("Fisher_5", FisherTransform, FisherTransform.Parameters(period=5), _NEG),
        ("Fisher_10", FisherTransform, FisherTransform.Parameters(period=10), _NEG),
        ("Fisher_20", FisherTransform, FisherTransform.Parameters(period=20), _NEG),
        # Ultimate Oscillator
        (
            "UltOsc_7_14_28",
            UltimateOscillator,
            UltimateOscillator.Parameters(period1=7, period2=14, period3=28),
            _RSI,
        ),
        (
            "UltOsc_4_8_14",
            UltimateOscillator,
            UltimateOscillator.Parameters(period1=4, period2=8, period3=14),
            _RSI,
        ),
        # TSI — fast/slow sweep
        ("TSI_5_13", TSI, TSI.Parameters(fast=5, slow=13), _NEG),
        ("TSI_13_25", TSI, TSI.Parameters(fast=13, slow=25), _NEG),
        # VWAP Bands — std sweep
        ("VWAPBands_1.5", VWAPBands, VWAPBands.Parameters(num_std=1.5), _NEG),
        ("VWAPBands_2.0", VWAPBands, VWAPBands.Parameters(num_std=2.0), _NEG),
        ("VWAPBands_2.5", VWAPBands, VWAPBands.Parameters(num_std=2.5), _NEG),
        # Session HL %
        ("SessionHLPct", SessionHighLowPct, SessionHighLowPct.Parameters(), _NEG),
        # Opening Range Position — range_bars sweep
        ("OR_2bar", OpeningRangePosition, OpeningRangePosition.Parameters(range_bars=2), _NEG),
        ("OR_4bar", OpeningRangePosition, OpeningRangePosition.Parameters(range_bars=4), _NEG),
        # Volume
        ("RVOL_10", RVOL, RVOL.Parameters(period=10), _ID),
        ("RVOL_20", RVOL, RVOL.Parameters(period=20), _ID),
        ("VROC_10", VROC, VROC.Parameters(period=10), _ID),
        ("VROC_14", VROC, VROC.Parameters(period=14), _ID),
        ("PVT_14", PVT, PVT.Parameters(period=14), _ID),
        ("PVT_20", PVT, PVT.Parameters(period=20), _ID),
        # Volatility context
        (
            "VolRatio_14_50",
            VolatilityRatio,
            VolatilityRatio.Parameters(atr_period=14, smooth_period=50),
            _ID,
        ),
        (
            "VolRatio_7_20",
            VolatilityRatio,
            VolatilityRatio.Parameters(atr_period=7, smooth_period=20),
            _ID,
        ),
        ("SqueezeMom_20", SqueezeMomentum, SqueezeMomentum.Parameters(period=20), _NEG),
        ("NormATR_14", NormalizedATR, NormalizedATR.Parameters(period=14), _NEG),
        # Swing indicators
        (
            "StochRSI_14",
            StochasticRSI,
            StochasticRSI.Parameters(rsi_period=14, stoch_period=14),
            _NEG,
        ),
        (
            "RSIDivergence",
            RSIDivergence,
            RSIDivergence.Parameters(rsi_period=14, divergence_window=10),
            _NEG,
        ),
        ("CoppockCurve", CoppockCurve, CoppockCurve.Parameters(), _ID),
        ("ElderRay_13", ElderRay, ElderRay.Parameters(period=13), _NEG),
        ("Aroon_25", Aroon, Aroon.Parameters(period=25), _NEG),
        ("PricePercentile", PricePercentile, PricePercentile.Parameters(period=50), _NEG),
        ("DistFromMA_20", DistanceFromMA, DistanceFromMA.Parameters(period=20), _NEG),
        (
            "LinRegSlope_20",
            LinearRegressionSlope,
            LinearRegressionSlope.Parameters(period=20),
            _NEG,
        ),
        ("MeanRevScore", MeanReversionScore, MeanReversionScore.Parameters(), _NEG),
        ("Chandelier_22", ChandelierExit, ChandelierExit.Parameters(period=22), _NEG),
        ("CandleBody_5", CandleBodyRatio, CandleBodyRatio.Parameters(period=5), _NEG),
        ("UpperShadow_5", UpperShadowRatio, UpperShadowRatio.Parameters(period=5), _NEG),
        ("InsideBar_10", InsideBar, InsideBar.Parameters(period=10), _ID),
        ("PriceVs52w", PriceVs52wHigh, PriceVs52wHigh.Parameters(period=252), _NEG),
    ]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _ic(signals: np.ndarray, fwd: np.ndarray) -> float:
    if len(signals) < 4:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r, _ = stats.spearmanr(signals, fwd)
    return float(r) if np.isfinite(r) else float("nan")


def _icir_from_series(signals: np.ndarray, fwd: np.ndarray, window: int) -> float:
    rolling: list[float] = []
    step = max(1, window // 2)
    for start in range(0, len(signals) - window + 1, step):
        sl = slice(start, start + window)
        ic = _ic(signals[sl], fwd[sl])
        if np.isfinite(ic):
            rolling.append(ic)
    if len(rolling) < 3:
        return float("nan")
    arr = np.array(rolling)
    std = arr.std(ddof=1)
    return float(arr.mean() / std) if std > 0 else float("nan")


def _qspread(signals: np.ndarray, fwd: np.ndarray) -> float:
    if len(signals) < 10:
        return float("nan")
    try:
        q_lo, q_hi = np.percentile(signals, [20, 80])
        top = fwd[signals >= q_hi]
        bot = fwd[signals <= q_lo]
        if len(top) == 0 or len(bot) == 0:
            return float("nan")
        return float(top.mean() - bot.mean())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Evaluate one indicator on one (train, test) window for one symbol
# ---------------------------------------------------------------------------


async def _evaluate_window(
    label: str,
    cls: Any,
    params: Any,
    extractor: Any,
    train_rows: list[dict],
    test_rows: list[dict],
    symbol: str,
    clock: SimulatedClock,
    horizon: int,
    interval: str,
    ic_window: int,
) -> dict[str, float]:
    store = PolarsStore(maxlen=max(500, len(train_rows) + len(test_rows)))

    if cls in _SESSION_CLASSES:
        ind = cls(store, symbol, interval, clock)
    else:
        ind = cls(store, symbol, interval)

    for row in train_rows:
        clock.advance(row["ts"])
        store.push(symbol, interval, row)

    test_closes = np.array([r["close"] for r in test_rows], dtype=float)
    signals: list[float] = []
    bar_indices: list[int] = []

    for i, row in enumerate(test_rows):
        clock.advance(row["ts"])
        store.push(symbol, interval, row)
        try:
            sig = await _extract(ind, label, params, extractor, row)
        except Exception:
            sig = None
        if sig is None or not np.isfinite(sig):
            continue
        signals.append(sig)
        bar_indices.append(i)

    nan = float("nan")
    if len(signals) < 10:
        return {"IC": nan, "ICIR": nan, "Qspread": nan}

    fwd = np.array(
        [
            (test_closes[i + horizon] - test_closes[i]) / test_closes[i]
            if i + horizon < len(test_closes)
            else np.nan
            for i in bar_indices
        ]
    )
    mask = ~np.isnan(fwd)
    sig_arr, fwd_arr = np.array(signals)[mask], fwd[mask]

    if len(sig_arr) < 10:
        return {"IC": nan, "ICIR": nan, "Qspread": nan}

    return {
        "IC": _ic(sig_arr, fwd_arr),
        "ICIR": _icir_from_series(sig_arr, fwd_arr, ic_window),
        "Qspread": _qspread(sig_arr, fwd_arr),
    }


async def _extract(ind: Any, label: str, params: Any, extractor: Any, row: dict) -> float | None:
    """Compute and transform one indicator value for the given row."""
    if label == "VWAP_dev":
        vwap = await ind.compute(params)
        close = float(row["close"])
        if vwap is None or close == 0:
            return None
        return (vwap - close) / close

    if label.startswith("BB_"):
        r = await ind.compute_full(params)
        if r is None:
            return None
        _upper, _mid, _lower, _bw, pct_b = r
        return -pct_b

    if label.startswith("Donchian_"):
        r = await ind.compute_full(params)
        if r is None:
            return None
        upper, middle, lower = r
        width = upper - lower
        return -(middle - lower) / width if width != 0 else None

    raw = await ind.compute(params)
    if raw is None:
        return None
    if extractor == _NEG:
        return -raw
    if extractor == _RSI:
        return 100.0 - raw
    return raw  # _ID


# ---------------------------------------------------------------------------
# Multiprocessing worker — module-level so it's picklable.
# Evaluates all catalogue indicators for one (symbol, window) pair.
# ---------------------------------------------------------------------------


def _worker_window(
    args: tuple[str, list[dict], list[dict], str, list[int], int],
) -> list[tuple[str, int, float]]:
    """Returns list of (label, horizon, IC) for this symbol x window."""
    symbol, train_rows, test_rows, interval, horizons, ic_window = args

    async def _run() -> list[tuple[str, int, float]]:
        catalogue = _catalogue()
        out: list[tuple[str, int, float]] = []
        for label, cls, params, extractor in catalogue:
            for h in horizons:
                clock = SimulatedClock()
                m = await _evaluate_window(
                    label, cls, params, extractor, train_rows, test_rows, symbol, clock, h,
                    interval, ic_window,
                )
                if np.isfinite(m["IC"]):
                    out.append((label, h, m["IC"]))
        return out

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-source", choices=["synthetic", "parquet"], default="parquet")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    p.add_argument("--interval", default="15min")
    p.add_argument("--horizons", type=int, nargs="+", default=_DEFAULT_HORIZONS)
    p.add_argument("--train-bars", type=int, default=200)
    p.add_argument("--test-bars", type=int, default=50)
    p.add_argument("--step-bars", type=int, default=50)
    p.add_argument("--ic-window", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """
    Walk-forward IC/ICIR sweep across all symbols, windows, and hyperparam variants.
    Each (symbol, window) pair is evaluated in a separate process.
    """
    far_past = datetime(2000, 1, 1, tzinfo=UTC)
    far_future = datetime(2100, 1, 1, tzinfo=UTC)
    horizons = args.horizons

    loader = make_data_loader(args.data_source, args.data_dir)

    symbol_rows: dict[str, list[dict]] = {}
    for symbol in args.symbols:
        try:
            df = loader.load(symbol, args.interval, far_past, far_future)
        except FileNotFoundError:
            print(f"  WARNING: no data for {symbol}, skipping")
            continue
        if df.is_empty():
            continue
        rows = [{**row, "ts": row["date"]} for row in df.to_dicts()]
        symbol_rows[symbol] = rows

    if not symbol_rows:
        print("WARNING: No data available")
        sys.exit(1)

    first_rows = next(iter(symbol_rows.values()))
    index_df = pl.DataFrame({"date": [r["ts"] for r in first_rows]})
    windows = _compute_windows(index_df, args.train_bars, args.test_bars, args.step_bars)

    if not windows:
        print(
            f"WARNING: Not enough bars for walk-forward. "
            f"Need {args.train_bars + args.test_bars}, got {len(first_rows)}."
        )
        sys.exit(1)

    print(
        f"\n  {len(windows)} windows  |  train={args.train_bars}  test={args.test_bars}  "
        f"step={args.step_bars}"
    )
    print(
        f"  {len(symbol_rows)} symbols  |  {sum(len(v) for v in symbol_rows.values())} total bars\n"
    )

    all_tasks: list[tuple[str, list[dict], list[dict], int]] = []  # (sym, train, test, w_idx)
    for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        for symbol, all_rows in symbol_rows.items():
            train_rows = [r for r in all_rows if train_start <= r["ts"] <= train_end]
            test_rows = [r for r in all_rows if test_start <= r["ts"] <= test_end]
            if len(train_rows) < args.train_bars // 2 or len(test_rows) < 10:
                continue
            all_tasks.append((symbol, train_rows, test_rows, w_idx))

    print(f"  {len(all_tasks)} total (symbol x window) tasks — dispatching to process pool ...\n")

    results: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _worker_window,
                (sym, train, test, args.interval, horizons, args.ic_window),
            )
            for sym, train, test, _ in all_tasks
        ]
        w_indices = [w_idx for _, _, _, w_idx in all_tasks]
        completed = await asyncio.gather(*futures)

    for w_idx, triples in zip(w_indices, completed, strict=True):
        print(f"  Window {w_idx + 1} done ({len(triples)} IC values)")
        for label, h, ic in triples:
            results[label][h].append(ic)

    summary: list[dict] = []
    for label, h_ics in results.items():
        row: dict = {"indicator": label}
        mean_icirs = []
        for h in horizons:
            ics = np.array(h_ics.get(h, []))
            if len(ics) == 0:
                row[f"IC_{h}"] = float("nan")
                row[f"ICIR_{h}"] = float("nan")
            else:
                row[f"IC_{h}"] = float(ics.mean())
                std = ics.std(ddof=1) if len(ics) > 1 else 0.0
                icir = float(ics.mean() / std) if std > 0 else float("nan")
                row[f"ICIR_{h}"] = icir
                if np.isfinite(icir):
                    mean_icirs.append(icir)
        row["mean_ICIR"] = float(np.mean(mean_icirs)) if mean_icirs else float("nan")
        stability_h = horizons[1] if len(horizons) > 1 else horizons[0]
        ic_stab = np.array(h_ics.get(stability_h, []))
        if len(ic_stab) > 1:
            sign = np.sign(ic_stab.mean())
            row["stability"] = float((np.sign(ic_stab) == sign).mean())
        else:
            row["stability"] = float("nan")
        summary.append(row)

    summary.sort(
        key=lambda r: r["mean_ICIR"] if np.isfinite(r["mean_ICIR"]) else -999,
        reverse=True,
    )

    sep = "=" * 105
    print(f"\n{sep}")
    print(
        f"  Walk-Forward IC  |  {len(windows)} windows  "
        f"train={args.train_bars}  test={args.test_bars}  step={args.step_bars}  |  {args.interval}"
    )
    print(sep)
    h_ic = "  ".join(f"IC_{h:>2}" for h in horizons)
    h_icir = "  ".join(f"ICIR_{h}" for h in horizons)
    print(f"  {'Indicator':<16}  {h_ic}    {h_icir}    mean_ICIR  stability")
    print(f"  {'-' * 100}")
    for r in summary:
        ics = "  ".join(
            f"{r[f'IC_{h}']:+.3f}" if np.isfinite(r[f"IC_{h}"]) else "   nan" for h in horizons
        )
        icirs = "  ".join(
            f"{r[f'ICIR_{h}']:+.3f}" if np.isfinite(r[f"ICIR_{h}"]) else "    nan" for h in horizons
        )
        micir = f"{r['mean_ICIR']:+.3f}" if np.isfinite(r["mean_ICIR"]) else "    nan"
        stab = f"{r['stability']:.2f}" if np.isfinite(r["stability"]) else "  nan"
        print(f"  {r['indicator']:<16}  {ics}    {icirs}    {micir}      {stab}")
    print(sep)
    print("\n  IC: |IC| > 0.05 useful, > 0.10 strong")
    print("  ICIR: |ICIR| > 0.5 consistent, > 1.0 excellent")
    print(
        "  stability: fraction of windows where IC sign matches overall direction "
        "(1.0 = always same sign)"
    )

    out_dir = Path(args.results_dir) / "indicator_wf_ic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "indicator_wf_ic_results.csv"
    if summary:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print(f"\n  Results written to {csv_path}")

    best = max(
        (r["mean_ICIR"] for r in summary if np.isfinite(r["mean_ICIR"])),
        default=float("nan"),
    )
    if not (np.isfinite(best) and best > 0):
        print("WARNING: Expected at least one indicator with positive walk-forward ICIR")
        sys.exit(1)
