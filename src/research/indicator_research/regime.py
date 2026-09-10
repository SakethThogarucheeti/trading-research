"""
Indicator IC split by market regime.

Checks whether indicator predictive power (IC) changes between:
  - Trending vs ranging bars (split on ADX(14) >= 25 / <= 20)
  - High-vol vs low-vol bars (split on HistoricalVolatility(20) above/below median)

Uses a 5-bar forward horizon only, for speed.

Ported from trading-integ-tests' strategy/indicators/test_indicator_regime.py.
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
from quantindicators.library.historical_volatility import HistoricalVolatility
from quantindicators.library.session_high_low_pct import SessionHighLowPct
from quantindicators.library.vwap import VWAP
from quantindicators.library.vwap_bands import VWAPBands
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

_DEFAULT_END = datetime(2026, 4, 30, tzinfo=UTC)

_SESSION_CLASSES = (VWAP, VWAPBands, SessionHighLowPct)


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.spearmanr(x, y)
    return float(r) if np.isfinite(r) else float("nan")


def _icir(ics: list[float]) -> float:
    arr = np.array([v for v in ics if np.isfinite(v)])
    if len(arr) < 4:
        return float("nan")
    mean, std = arr.mean(), arr.std(ddof=1)
    return float(mean / std) if std > 0 else float("nan")


# ---------------------------------------------------------------------------
# Worker — module-level for pickling
# ---------------------------------------------------------------------------


def _worker_regime(
    args: tuple[str, list[dict], str, int, int, float, float],
) -> dict[str, Any]:
    """
    For each indicator, collect (signal, fwd_h) per bar, then return
    regime-split IC lists.

    Returns dict keyed by label -> {
        "trending": [(sig, fwd), ...],
        "ranging":  [(sig, fwd), ...],
        "high_vol": [(sig, fwd), ...],
        "low_vol":  [(sig, fwd), ...],
    }
    """
    symbol, rows, interval, warmup, horizon, adx_trending, adx_ranging = args
    print(f"  {symbol} ...", flush=True)

    async def _run() -> dict[str, Any]:
        from research.indicator_research.catalogue import load_catalogue

        catalogue = load_catalogue(interval)
        closes = np.array([r["close"] for r in rows], dtype=float)

        fwd_h = np.full(len(closes), np.nan)
        for i in range(len(closes) - horizon):
            fwd_h[i] = (closes[i + horizon] - closes[i]) / closes[i]

        adx_store = PolarsStore(maxlen=500)
        hv_store = PolarsStore(maxlen=500)
        adx_ind = ADX(adx_store, symbol, interval)
        hv_ind = HistoricalVolatility(hv_store, symbol, interval)
        adx_params = ADX.Parameters(period=14)
        hv_params = HistoricalVolatility.Parameters(period=20)
        regime_clock = SimulatedClock()

        adx_vals: list[float | None] = []
        hv_vals: list[float | None] = []

        for i, row in enumerate(rows):
            regime_clock.advance(row["ts"])
            adx_store.push(symbol, interval, row)
            hv_store.push(symbol, interval, row)
            if i < warmup:
                adx_vals.append(None)
                hv_vals.append(None)
            else:
                try:
                    av = await adx_ind.compute(adx_params)
                    adx_vals.append(float(av) if av is not None else None)
                except Exception:
                    adx_vals.append(None)
                try:
                    hv = await hv_ind.compute(hv_params)
                    hv_vals.append(float(hv) if hv is not None else None)
                except Exception:
                    hv_vals.append(None)

        valid_hvs = [v for v in hv_vals if v is not None]
        hv_median = float(np.median(valid_hvs)) if valid_hvs else 0.0

        result: dict[str, Any] = {}

        for label, cls, params, extractor in catalogue:
            if cls is None:
                continue  # skip EMA_cross_9_21

            clock = SimulatedClock()
            store = PolarsStore(maxlen=500)

            if cls in _SESSION_CLASSES:
                ind = cls(store, symbol, interval, clock)
            else:
                ind = cls(store, symbol, interval)

            trending_pairs: list[tuple[float, float]] = []
            ranging_pairs: list[tuple[float, float]] = []
            high_vol_pairs: list[tuple[float, float]] = []
            low_vol_pairs: list[tuple[float, float]] = []

            for i, row in enumerate(rows):
                clock.advance(row["ts"])
                store.push(symbol, interval, row)

                if i < warmup:
                    continue

                fwd = fwd_h[i]
                if not np.isfinite(fwd):
                    continue

                try:
                    if extractor == "macd_hist":
                        r = await ind.compute_full(params)
                        sig = r[2] if r is not None else None
                    elif extractor == "psar":
                        r = await ind.compute_full(params)
                        sig = (1.0 if r[1] else -1.0) if r is not None else None
                    elif extractor == "vwap_dev":
                        vwap = await ind.compute(params)
                        close = float(row["close"])
                        sig = (vwap - close) / close if (vwap is not None and close != 0) else None
                    elif extractor == "bollinger_pctb":
                        r = await ind.compute_full(params)
                        sig = -r[4] if r is not None else None
                    elif extractor == "keltner_pct":
                        r = await ind.compute_full(params)
                        if r is None:
                            sig = None
                        else:
                            upper, middle, lower = r
                            width = upper - lower
                            sig = -(middle - lower) / width if width != 0 else None
                    elif extractor == "donchian_pct":
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
                        elif extractor == "neg":
                            sig = -raw
                        elif extractor == "rsi":
                            sig = 100.0 - raw
                        else:
                            sig = raw
                except Exception:
                    sig = None

                if sig is None or not np.isfinite(sig):
                    continue

                pair = (float(sig), float(fwd))

                av = adx_vals[i]
                if av is not None:
                    if av >= adx_trending:
                        trending_pairs.append(pair)
                    elif av <= adx_ranging:
                        ranging_pairs.append(pair)

                hv = hv_vals[i]
                if hv is not None:
                    if hv > hv_median:
                        high_vol_pairs.append(pair)
                    else:
                        low_vol_pairs.append(pair)

            result[label] = {
                "trending": trending_pairs,
                "ranging": ranging_pairs,
                "high_vol": high_vol_pairs,
                "low_vol": low_vol_pairs,
            }

        return result

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
    p.add_argument("--horizon", type=int, default=5, help="5-bar forward return only")
    p.add_argument("--adx-trending", type=float, default=25.0)
    p.add_argument("--adx-ranging", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """
    Evaluate IC by market regime (trending/ranging, high-vol/low-vol) for all indicators.
    Writes indicator_regime_results.csv and indicator_regime.html.
    """
    month_end = args.end
    month_start = month_end - timedelta(days=args.lookback_days)
    _nan = float("nan")

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
                executor,
                _worker_regime,
                (sym, rows, args.interval, args.warmup, args.horizon, args.adx_trending, args.adx_ranging),
            )
            for sym, rows in symbol_rows
        ]
        per_symbol: list[dict[str, Any]] = await asyncio.gather(*futures)

    agg: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {"trending": [], "ranging": [], "high_vol": [], "low_vol": []}
    )
    for sym_result in per_symbol:
        for label, regimes in sym_result.items():
            for regime, pairs in regimes.items():
                agg[label][regime].extend(pairs)

    _IC_WINDOW = 20
    summary: list[dict[str, Any]] = []

    for label in sorted(agg.keys()):
        row: dict[str, Any] = {"indicator": label}
        for regime in ("trending", "ranging", "high_vol", "low_vol"):
            pairs = agg[label][regime]
            if len(pairs) < 20:
                row[f"IC_{regime}"] = _nan
                row[f"ICIR_{regime}"] = _nan
                row[f"n_{regime}"] = len(pairs)
                continue
            sigs = np.array([p[0] for p in pairs])
            fwds = np.array([p[1] for p in pairs])
            row[f"IC_{regime}"] = _spearman(sigs, fwds)
            row[f"n_{regime}"] = len(pairs)

            ic_list: list[float] = []
            for start in range(0, len(sigs) - _IC_WINDOW + 1, _IC_WINDOW // 2):
                ic = _spearman(sigs[start : start + _IC_WINDOW], fwds[start : start + _IC_WINDOW])
                if np.isfinite(ic):
                    ic_list.append(ic)
            row[f"ICIR_{regime}"] = _icir(ic_list)

        summary.append(row)

    def _best_ic(r: dict) -> float:
        vals = [
            abs(r[f"IC_{reg}"]) for reg in ("trending", "ranging") if np.isfinite(r[f"IC_{reg}"])
        ]
        return max(vals) if vals else 0.0

    summary.sort(key=_best_ic, reverse=True)

    print(f"\n{'=' * 110}")
    print(
        f"  Indicator Regime IC  |  {month_start.date()} to {month_end.date()}  |  "
        f"{args.interval}  |  {len(args.symbols)} symbols  |  Horizon={args.horizon}"
    )
    print(f"{'=' * 110}")
    print(
        f"  {'Indicator':<20}  {'IC_trend':>9}  {'IC_range':>9}  {'IC_hvol':>9}  {'IC_lvol':>9}  "
        f"{'ICIR_trend':>11}  {'ICIR_range':>11}"
    )
    print(f"  {'-' * 100}")
    for r in summary:

        def _fmt(v: float) -> str:
            return f"{v:+.3f}" if np.isfinite(v) else "    nan"

        print(
            f"  {r['indicator']:<20}  "
            f"{_fmt(r['IC_trending']):>9}  {_fmt(r['IC_ranging']):>9}  "
            f"{_fmt(r['IC_high_vol']):>9}  {_fmt(r['IC_low_vol']):>9}  "
            f"{_fmt(r['ICIR_trending']):>11}  {_fmt(r['ICIR_ranging']):>11}"
        )
    print(f"{'=' * 110}")

    out_dir = Path(args.results_dir) / "indicator_regime"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "indicator_regime_results.csv"

    if summary:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(f"\n  CSV written to {csv_path}")

    def _icir_color(v: float) -> str:
        if not np.isfinite(v):
            return "#ffffff"
        if v > 1.0:
            return "#4caf50"
        if v > 0.5:
            return "#a5d6a7"
        if v < -0.5:
            return "#ef9a9a"
        return "#ffffff"

    def _ic_color(v: float) -> str:
        if not np.isfinite(v):
            return "#ffffff"
        if abs(v) > 0.10:
            return "#c8e6c9"
        if abs(v) > 0.05:
            return "#f0f4c3"
        return "#ffffff"

    regime_cols = [
        ("IC_trending", "IC_trending"),
        ("IC_ranging", "IC_ranging"),
        ("IC_high_vol", "IC_high_vol"),
        ("IC_low_vol", "IC_low_vol"),
        ("ICIR_trending", "ICIR_trending"),
        ("ICIR_ranging", "ICIR_ranging"),
        ("n_trending", "n_trending"),
        ("n_ranging", "n_ranging"),
    ]
    th = "<th>indicator</th>" + "".join(f"<th>{col[1]}</th>" for col in regime_cols)
    html_rows = []
    for r in summary:
        cells = [f"<td><b>{r['indicator']}</b></td>"]
        for key, _ in regime_cols:
            v = r.get(key, _nan)
            if key.startswith("ICIR"):
                color = _icir_color(v)
            elif key.startswith("IC"):
                color = _ic_color(v)
            else:
                color = "#ffffff"
            if key.startswith("n_"):
                fmt = str(int(v)) if np.isfinite(v) else "nan"
            else:
                fmt = f"{v:+.4f}" if np.isfinite(v) else "nan"
            cells.append(f'<td style="background:{color}">{fmt}</td>')
        html_rows.append(f"<tr>{''.join(cells)}</tr>")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Indicator Regime IC — {args.interval}</title>
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
<h2>Indicator Regime IC — {args.interval} | {month_start.date()} to {month_end.date()} | {len(args.symbols)} symbols | Horizon={args.horizon}</h2>
<p>ADX&ge;{args.adx_trending:.0f} → trending, ADX&le;{args.adx_ranging:.0f} → ranging. HV above/below median → high_vol/low_vol.</p>
<p>Sorted by max|IC| across trending/ranging. Green ICIR &gt; 1.0, light-green &gt; 0.5.</p>
<table>
<thead><tr>{th}</tr></thead>
<tbody>
{"".join(html_rows)}
</tbody>
</table>
</body>
</html>
"""
    html_path = out_dir / "indicator_regime.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  HTML report written to {html_path}")

    if not (len(summary) > 0):
        print("WARNING: Expected regime analysis to produce at least one indicator row")
        sys.exit(1)
    n_finite = sum(
        1
        for r in summary
        if np.isfinite(r.get("IC_trending", _nan)) or np.isfinite(r.get("IC_ranging", _nan))
    )
    if not (n_finite > 0):
        print("WARNING: Expected at least one indicator with finite regime IC")
        sys.exit(1)
