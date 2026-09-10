"""
Indicator Information Coefficient (IC) evaluation — 15min bars, intraday swing horizons.

For each indicator computes:
  IC       — mean Spearman rank correlation between indicator value at bar t
             and forward return at bar t+N
  ICIR     — IC / std(IC) across rolling 20-bar windows (consistency)
  Q-spread — mean return of top quintile minus bottom quintile (monotonicity)

Default forward horizons: 5, 15, 30, 50 bars.

Ported from trading-integ-tests' strategy/indicators/test_indicator_ic.py. Unlike
correlation.py/decay.py this module carries its own local indicator catalogue
(broader than the shared JSON catalogue) rather than importing
research.indicator_research.catalogue.load_catalogue — preserved as-is from source.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from quantindicators.library.adx import ADX
from quantindicators.library.aroon import Aroon
from quantindicators.library.atr import ATR
from quantindicators.library.bollinger import BollingerBands
from quantindicators.library.candle_body_ratio import CandleBodyRatio
from quantindicators.library.cci import CCI
from quantindicators.library.chaikin_volatility import ChaikinVolatility
from quantindicators.library.chandelier_exit import ChandelierExit
from quantindicators.library.cmf import CMF
from quantindicators.library.connors_rsi import ConnorsRSI
from quantindicators.library.coppock_curve import CoppockCurve
from quantindicators.library.distance_from_ma import DistanceFromMA
from quantindicators.library.donchian import DonchianChannels
from quantindicators.library.dpo import DPO
from quantindicators.library.elder_ray import ElderRay
from quantindicators.library.ema import EMA
from quantindicators.library.fisher_transform import FisherTransform
from quantindicators.library.gap import GapSize
from quantindicators.library.historical_volatility import HistoricalVolatility
from quantindicators.library.inside_bar import InsideBar
from quantindicators.library.keltner import KeltnerChannels
from quantindicators.library.linreg_slope import LinearRegressionSlope
from quantindicators.library.macd import MACD
from quantindicators.library.mean_reversion_score import MeanReversionScore
from quantindicators.library.mfi import MFI
from quantindicators.library.momentum import Momentum
from quantindicators.library.normalized_atr import NormalizedATR
from quantindicators.library.obv import OBV
from quantindicators.library.opening_range import OpeningRangePosition
from quantindicators.library.parabolic_sar import ParabolicSAR
from quantindicators.library.price_percentile import PricePercentile
from quantindicators.library.price_vs_52w_high import PriceVs52wHigh
from quantindicators.library.pvt import PVT
from quantindicators.library.roc import ROC
from quantindicators.library.rsi import RSI
from quantindicators.library.rsi_divergence import RSIDivergence
from quantindicators.library.rvol import RVOL
from quantindicators.library.session_high_low_pct import SessionHighLowPct
from quantindicators.library.squeeze_momentum import SqueezeMomentum
from quantindicators.library.stochastic import Stochastic
from quantindicators.library.stochastic_rsi import StochasticRSI
from quantindicators.library.supertrend import Supertrend
from quantindicators.library.tsi import TSI
from quantindicators.library.ultimate_oscillator import UltimateOscillator
from quantindicators.library.upper_shadow_ratio import UpperShadowRatio
from quantindicators.library.volatility_ratio import VolatilityRatio
from quantindicators.library.vroc import VROC
from quantindicators.library.vwap import VWAP
from quantindicators.library.vwap_bands import VWAPBands
from quantindicators.library.vwma import VWMA
from quantindicators.library.weekly_rsi import WeeklyRSI
from quantindicators.library.williams_r import WilliamsR
from quantindicators.polars_store import PolarsStore
from scipy import stats
from trading.core.clock import SimulatedClock

from research.indicator_research.data import make_data_loader, make_store

_DEFAULT_SYMBOLS = [
    "INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
    "SBIN", "BAJFINANCE", "BAJAJFINSV", "WIPRO", "HCLTECH", "TECHM", "LT",
    "MARUTI", "SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "TITAN", "ASIANPAINT",
    "NESTLEIND", "HINDUNILVR", "BRITANNIA", "POWERGRID", "NTPC", "ONGC",
    "COALINDIA", "ITC", "TATASTEEL",
]  # fmt: skip

_DEFAULT_HORIZONS = [5, 15, 30, 50]  # 1.25h, 3.75h, 7.5h, 12.5h — intraday swing
_DEFAULT_END = datetime(2026, 4, 30, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Extractor sentinels — picklable strings, no lambdas crossing process boundary
# ---------------------------------------------------------------------------
_NEG = "neg"  # negate the value: -v
_RSI = "rsi"  # 100 - v  (mean-reversion flip for RSI-scale oscillators)
_ID = "id"  # identity: use value as-is

# Session-aware indicator classes that require a clock argument
_SESSION_CLASSES = (VWAP, VWAPBands, SessionHighLowPct)


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Indicator catalogue
# Returns (label, cls, params, extractor) tuples.
# extractor: _NEG | _RSI | _ID | None  (None = special-case in _evaluate_indicator)
# ---------------------------------------------------------------------------


def _catalogue() -> list[tuple[str, Any, Any, Any]]:
    """Return (label, cls, params, extractor) tuples.

    Special labels handled separately in _evaluate_indicator:
      "EMA_cross_9_21" — cls and params are None; two EMAs constructed inline
      "MACD_hist"      — compute_full() used; histogram (index 2) returned
      "VWAP_dev"       — compute() called; deviation from close computed
      "PSAR"           — compute_full() called; 1.0 if bullish else -1.0
    """
    return [
        ("EMA_cross_9_21", None, None, None),
        ("MACD_hist", MACD, MACD.Parameters(fast=12, slow=26, signal=9), None),
        ("Supertrend", Supertrend, Supertrend.Parameters(period=10, multiplier=3.0), _ID),
        ("ADX_14", ADX, ADX.Parameters(period=14), _ID),
        ("Momentum_10", Momentum, Momentum.Parameters(period=10), _NEG),
        ("ROC_10", ROC, ROC.Parameters(period=10), _NEG),
        ("RSI_14", RSI, RSI.Parameters(period=14), _RSI),
        ("RSI_7", RSI, RSI.Parameters(period=7), _RSI),
        ("Stochastic_14", Stochastic, Stochastic.Parameters(k_period=14, d_period=3), _RSI),
        ("Williams_R_14", WilliamsR, WilliamsR.Parameters(period=14), _NEG),
        ("CCI_20", CCI, CCI.Parameters(period=20), _NEG),
        ("MFI_14", MFI, MFI.Parameters(period=14), _RSI),
        ("CMF_20", CMF, CMF.Parameters(period=20), _ID),
        ("Bollinger_%B", BollingerBands, BollingerBands.Parameters(period=20, k=2.0), _NEG),
        (
            "Keltner_%",
            KeltnerChannels,
            KeltnerChannels.Parameters(ema_period=20, atr_period=10, k=2.0),
            _ID,
        ),
        ("Donchian_%", DonchianChannels, DonchianChannels.Parameters(period=20), _ID),
        ("OBV_20", OBV, OBV.Parameters(period=20), _ID),
        ("VWMA_20", VWMA, VWMA.Parameters(period=20), _ID),
        ("VWAP_dev", VWAP, VWAP.Parameters(), None),
        ("ATR_14", ATR, ATR.Parameters(period=14), _NEG),
        (
            "ChaikinVol_10",
            ChaikinVolatility,
            ChaikinVolatility.Parameters(ema_period=10, roc_period=10),
            _ID,
        ),
        ("HistVol_20", HistoricalVolatility, HistoricalVolatility.Parameters(period=20), _NEG),
        ("PSAR", ParabolicSAR, ParabolicSAR.Parameters(), None),
        ("ConnorsRSI", ConnorsRSI, ConnorsRSI.Parameters(), _RSI),
        ("Fisher_10", FisherTransform, FisherTransform.Parameters(period=10), _NEG),
        (
            "UltimateOsc",
            UltimateOscillator,
            UltimateOscillator.Parameters(period1=7, period2=14, period3=28),
            _RSI,
        ),
        ("DPO_20", DPO, DPO.Parameters(period=20), _NEG),
        ("TSI", TSI, TSI.Parameters(), _NEG),
        ("GapSize", GapSize, GapSize.Parameters(), _NEG),
        ("OpeningRange", OpeningRangePosition, OpeningRangePosition.Parameters(range_bars=4), _NEG),
        ("VWAPBands", VWAPBands, VWAPBands.Parameters(), _NEG),
        ("SessionHLPct", SessionHighLowPct, SessionHighLowPct.Parameters(), _NEG),
        ("RVOL_20", RVOL, RVOL.Parameters(period=20), _ID),
        ("VROC_14", VROC, VROC.Parameters(period=14), _ID),
        ("PVT_20", PVT, PVT.Parameters(period=20), _ID),
        ("VolRatio", VolatilityRatio, VolatilityRatio.Parameters(), _ID),
        ("SqueezeMom", SqueezeMomentum, SqueezeMomentum.Parameters(), _NEG),
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
        ("WeeklyRSI_14", WeeklyRSI, WeeklyRSI.Parameters(rsi_period=14), _RSI),
        ("PriceVs52w", PriceVs52wHigh, PriceVs52wHigh.Parameters(period=252), _NEG),
    ]


# ---------------------------------------------------------------------------
# IC maths
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.spearmanr(x, y)
    return float(r) if np.isfinite(r) else float("nan")


def _icir(ics: np.ndarray) -> float:
    valid = ics[~np.isnan(ics)]
    if len(valid) < 4:
        return float("nan")
    mean, std = valid.mean(), valid.std(ddof=1)
    return float(mean / std) if std > 0 else float("nan")


def _quintile_spread(signals: np.ndarray, fwd: np.ndarray) -> float:
    q_lo, q_hi = np.percentile(signals, [20, 80])
    top = fwd[signals >= q_hi]
    bot = fwd[signals <= q_lo]
    if len(top) == 0 or len(bot) == 0:
        return float("nan")
    return float(top.mean() - bot.mean())


# ---------------------------------------------------------------------------
# Per-indicator evaluation
# ---------------------------------------------------------------------------


async def _evaluate_indicator(
    label: str,
    cls: Any,
    params: Any,
    extractor: Any,
    rows: list[dict],
    symbol: str,
    clock: SimulatedClock,
    interval: str,
    warmup: int,
    horizons: list[int],
    ic_window: int,
) -> dict[str, Any]:
    store = PolarsStore(maxlen=500)
    closes = np.array([r["close"] for r in rows], dtype=float)

    fwd_returns: dict[int, np.ndarray] = {}
    for h in horizons:
        ret = np.full(len(closes), np.nan)
        for i in range(len(closes) - h):
            ret[i] = (closes[i + h] - closes[i]) / closes[i]
        fwd_returns[h] = ret

    is_ema_cross = label == "EMA_cross_9_21"
    if is_ema_cross:
        fast_ind = EMA(store, symbol, interval)
        slow_ind = EMA(store, symbol, interval)
        fast_params = EMA.Parameters(period=9)
        slow_params = EMA.Parameters(period=21)
    elif cls in _SESSION_CLASSES:
        ind = cls(store, symbol, interval, clock)
    else:
        ind = cls(store, symbol, interval)

    prev_fast: float | None = None
    prev_slow: float | None = None
    signals: list[float] = []
    bar_indices: list[int] = []

    for i, row in enumerate(rows):
        clock.advance(row["ts"])
        store.push(symbol, interval, row)

        if i < warmup:
            if is_ema_cross:
                prev_fast = await fast_ind.compute(fast_params)
                prev_slow = await slow_ind.compute(slow_params)
            continue

        try:
            if is_ema_cross:
                fast = await fast_ind.compute(fast_params)
                slow = await slow_ind.compute(slow_params)
                sig = (
                    (fast - slow)
                    if (
                        fast is not None
                        and slow is not None
                        and prev_fast is not None
                        and prev_slow is not None
                    )
                    else None
                )
                prev_fast, prev_slow = fast, slow

            elif label == "MACD_hist":
                r = await ind.compute_full(params)
                sig = r[2] if r is not None else None

            elif label == "PSAR":
                r = await ind.compute_full(params)
                sig = (1.0 if r[1] else -1.0) if r is not None else None

            elif label == "VWAP_dev":
                vwap = await ind.compute(params)
                close = float(row["close"])
                sig = (vwap - close) / close if (vwap is not None and close != 0) else None

            elif label == "Bollinger_%B":
                r = await ind.compute_full(params)
                if r is None:
                    sig = None
                else:
                    _upper, _mid, _lower, _bw, pct_b = r
                    sig = -pct_b

            elif label == "Keltner_%":
                r = await ind.compute_full(params)
                if r is None:
                    sig = None
                else:
                    upper, middle, lower = r
                    width = upper - lower
                    sig = -(middle - lower) / width if width != 0 else None

            elif label == "Donchian_%":
                r = await ind.compute_full(params)
                if r is None:
                    sig = None
                else:
                    upper, middle, lower = r
                    width = upper - lower
                    sig = -(middle - lower) / width if width != 0 else None

            else:
                raw = await ind.compute(params)
                if raw is None:
                    sig = None
                elif extractor == _NEG:
                    sig = -raw
                elif extractor == _RSI:
                    sig = 100.0 - raw
                else:
                    sig = raw  # _ID or None extractor

        except Exception:
            sig = None

        if sig is None or not np.isfinite(sig):
            continue
        signals.append(sig)
        bar_indices.append(i)

    _nan = float("nan")
    empty: dict[str, Any] = {
        "label": label,
        "n": len(signals),
        **{f"IC_{h}": _nan for h in horizons},
        **{f"ICIR_{h}": _nan for h in horizons},
        **{f"Qspread_{h}": _nan for h in horizons},
    }
    if len(signals) < 20:
        return empty

    sig_arr = np.array(signals)
    result: dict[str, Any] = {"label": label, "n": len(signals)}

    for h in horizons:
        fwd = fwd_returns[h]
        paired_fwd = np.array([fwd[i] for i in bar_indices])
        mask = ~np.isnan(paired_fwd)
        s, f = sig_arr[mask], paired_fwd[mask]

        if len(s) < 20:
            result[f"IC_{h}"] = _nan
            result[f"ICIR_{h}"] = _nan
            result[f"Qspread_{h}"] = _nan
            continue

        result[f"IC_{h}"] = _spearman(s, f)

        rolling: list[float] = []
        for start in range(0, len(s) - ic_window + 1, ic_window // 2):
            ic = _spearman(s[start : start + ic_window], f[start : start + ic_window])
            if np.isfinite(ic):
                rolling.append(ic)
        result[f"ICIR_{h}"] = _icir(np.array(rolling))
        result[f"Qspread_{h}"] = _quintile_spread(s, f)

    return result


# ---------------------------------------------------------------------------
# Multiprocessing worker — module-level so it's picklable
# ---------------------------------------------------------------------------


def _worker(
    args: tuple[str, list[dict], str, int, list[int], int],
) -> list[dict[str, Any]]:
    """
    Evaluate all catalogue indicators for one symbol.
    Runs in a subprocess with its own event loop — no shared state.
    """
    symbol, rows, interval, warmup, horizons, ic_window = args
    print(f"\n  Evaluating {symbol} ({len(rows)} bars) ...", flush=True)

    async def _run() -> list[dict[str, Any]]:
        catalogue = _catalogue()
        results = []
        for label, cls, params, extractor in catalogue:
            clock = SimulatedClock()
            res = await _evaluate_indicator(
                label, cls, params, extractor, rows, symbol, clock, interval, warmup,
                horizons, ic_window,
            )
            res["symbol"] = symbol
            results.append(res)
        return results

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-source", choices=["synthetic", "parquet"], default="parquet")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    p.add_argument("--interval", default="15min")
    p.add_argument("--end", type=_parse_date, default=_DEFAULT_END)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lookback-days", type=int, default=400)
    p.add_argument("--horizons", type=int, nargs="+", default=_DEFAULT_HORIZONS)
    p.add_argument("--ic-window", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """
    Evaluate every indicator's IC / ICIR / Q-spread across multiple symbols.
    Each symbol is evaluated in a separate process via ProcessPoolExecutor.
    Prints a ranked table sorted by mean ICIR and writes indicator_ic_results.csv.
    """
    month_end = args.end
    month_start = month_end - timedelta(days=args.lookback_days)
    horizons = args.horizons

    loader = make_data_loader(args.data_source, args.data_dir)
    clock = SimulatedClock()

    symbol_rows: list[tuple[str, list[dict]]] = []
    for symbol in args.symbols:
        _store, rows = make_store(loader, clock, symbol, args.interval, month_start, month_end)
        if not rows:
            print(f"  WARNING: no data for {symbol}")
            continue
        symbol_rows.append((symbol, rows))

    print(f"\n  Dispatching {len(symbol_rows)} symbols to process pool ...")

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            loop.run_in_executor(
                executor, _worker, (sym, rows, args.interval, args.warmup, horizons, args.ic_window)
            )
            for sym, rows in symbol_rows
        ]
        per_symbol = await asyncio.gather(*futures)

    all_results = [r for sym_results in per_symbol for r in sym_results]

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        lbl = r["label"]
        for h in horizons:
            for metric in (f"IC_{h}", f"ICIR_{h}", f"Qspread_{h}"):
                v = r.get(metric, float("nan"))
                if np.isfinite(v):
                    agg[lbl][metric].append(v)

    summary: list[dict[str, Any]] = []
    for lbl, metrics in agg.items():
        row: dict[str, Any] = {"indicator": lbl}
        for h in horizons:
            for metric in (f"IC_{h}", f"ICIR_{h}", f"Qspread_{h}"):
                vals = metrics.get(metric, [])
                row[metric] = float(np.mean(vals)) if vals else float("nan")
        icir_vals = [row[f"ICIR_{h}"] for h in horizons if np.isfinite(row[f"ICIR_{h}"])]
        row["mean_ICIR"] = float(np.mean(icir_vals)) if icir_vals else float("nan")
        summary.append(row)

    summary.sort(
        key=lambda r: r["mean_ICIR"] if np.isfinite(r["mean_ICIR"]) else -999,
        reverse=True,
    )

    print(f"\n{'=' * 95}")
    print(
        f"  Indicator IC Evaluation  |  {month_start.date()} to {month_end.date()}  |  "
        f"{args.interval}  |  {len(args.symbols)} symbols"
    )
    print(f"{'=' * 95}")
    h_ics = "  ".join(f"IC_{h:>2}" for h in horizons)
    h_icirs = "  ".join(f"ICIR_{h}" for h in horizons)
    print(f"  {'Indicator':<18}  {h_ics}    {h_icirs}    mean_ICIR")
    print(f"  {'-' * 90}")
    for r in summary:
        ics = "  ".join(
            f"{r[f'IC_{h}']:+.3f}" if np.isfinite(r[f"IC_{h}"]) else "   nan" for h in horizons
        )
        icirs = "  ".join(
            f"{r[f'ICIR_{h}']:+.3f}" if np.isfinite(r[f"ICIR_{h}"]) else "    nan" for h in horizons
        )
        micir = f"{r['mean_ICIR']:+.3f}" if np.isfinite(r["mean_ICIR"]) else "    nan"
        print(f"  {r['indicator']:<18}  {ics}    {icirs}    {micir}")
    print(f"{'=' * 95}")
    print("\n  IC: |IC| > 0.05 useful, > 0.10 strong")
    print("  ICIR: |ICIR| > 0.5 consistent, > 1.0 excellent")

    out_dir = Path(args.results_dir) / "indicator_ic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "indicator_ic_results.csv"
    if summary:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        print(f"\n  Results written to {csv_path}")

    html_path = out_dir / "indicator_ic_15min.html"

    def _cell_color(v: float) -> str:
        if not np.isfinite(v):
            return "#fff"
        if v > 1.0:
            return "#4caf50"
        if v > 0.5:
            return "#a5d6a7"
        if v < 0:
            return "#ef9a9a"
        return "#fff"

    h_cols = (
        [f"IC_{h}" for h in horizons] + [f"ICIR_{h}" for h in horizons] + [f"Qspread_{h}" for h in horizons]
    )
    html_rows = []
    for r in summary:
        cells = [f"<td>{r['indicator']}</td>"]
        for col in h_cols:
            v = r.get(col, float("nan"))
            color = _cell_color(v) if "ICIR" in col else "#fff"
            fmt = f"{v:+.4f}" if np.isfinite(v) else "nan"
            cells.append(f'<td style="background:{color}">{fmt}</td>')
        micir = r["mean_ICIR"]
        color = _cell_color(micir)
        fmt = f"{micir:+.4f}" if np.isfinite(micir) else "nan"
        cells.append(f'<td style="background:{color}"><b>{fmt}</b></td>')
        html_rows.append(f"<tr>{''.join(cells)}</tr>")

    th_cols = ["Indicator"] + h_cols + ["mean_ICIR"]
    header = "".join(f"<th>{c}</th>" for c in th_cols)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Indicator IC — {args.interval}</title>
<style>
  body {{ font-family: monospace; font-size: 13px; margin: 20px; }}
  h2 {{ margin-bottom: 8px; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }}
  th {{ background: #e0e0e0; position: sticky; top: 0; }}
  tr:hover {{ background: #f5f5f5 !important; }}
</style>
</head>
<body>
<h2>Indicator IC Evaluation — {args.interval} | {month_start.date()} to {month_end.date()} | {len(args.symbols)} symbols</h2>
<p>Sorted by mean_ICIR. Green &gt; 1.0, light-green &gt; 0.5, red &lt; 0.</p>
<table>
<thead><tr>{header}</tr></thead>
<tbody>
{"".join(html_rows)}
</tbody>
</table>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  HTML report written to {html_path}")

    best_ic5_key = f"IC_{horizons[0]}"
    best_ic5 = max(
        (r[best_ic5_key] for r in summary if np.isfinite(r[best_ic5_key])),
        default=float("nan"),
    )
    if not (np.isfinite(best_ic5) and best_ic5 > 0.0):
        print(
            f"WARNING: Expected at least one indicator with positive IC at "
            f"{horizons[0]}-bar horizon"
        )
        sys.exit(1)
