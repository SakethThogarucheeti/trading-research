"""
Indicator IC Decay at Fibonacci Horizons.

Computes IC at Fibonacci horizons (default [1, 2, 3, 5, 8, 13, 21, 34, 50] bars) to find:
  - Peak horizon (argmax |IC|)
  - IC half-life (where |IC| drops below |IC_peak| / 2)

This helps identify the optimal holding period for each indicator signal.

Ported from trading-integ-tests' strategy/indicators/test_indicator_decay.py.
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
from quantindicators.library.ema import EMA
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

_DEFAULT_HORIZONS = [1, 2, 3, 5, 8, 13, 21, 34, 50]  # Fibonacci
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


def _halflife(horizons: list[int], ics: list[float]) -> float:
    """
    Find the horizon at which |IC| drops below |IC_peak| / 2.
    Uses linear interpolation between adjacent horizons.
    Returns nan if IC doesn't decay within the measured range.
    """
    finite_abs = [abs(ic) for ic in ics if np.isfinite(ic)]
    if not finite_abs:
        return float("nan")
    peak_ic = max(finite_abs)
    if peak_ic == 0:
        return float("nan")
    threshold = peak_ic / 2

    for i in range(len(horizons) - 1):
        ai = abs(ics[i]) if np.isfinite(ics[i]) else 0.0
        ai1 = abs(ics[i + 1]) if np.isfinite(ics[i + 1]) else 0.0
        if ai >= threshold and ai1 < threshold:
            denom = ai - ai1
            if denom == 0:
                return float(horizons[i])
            frac = (ai - threshold) / denom
            return float(horizons[i]) + frac * float(horizons[i + 1] - horizons[i])

    return float("nan")  # IC doesn't decay within measured range


# ---------------------------------------------------------------------------
# Per-indicator evaluation
# ---------------------------------------------------------------------------


async def _evaluate_indicator_decay(
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
) -> dict[str, Any]:
    """
    Collect signals and compute IC at each Fibonacci horizon.
    Returns dict with IC_<h> per horizon, peak_horizon, IC_peak, half_life.
    """
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
        prev_fast: float | None = None
        prev_slow: float | None = None
    elif cls in _SESSION_CLASSES:
        ind = cls(store, symbol, interval, clock)
    else:
        ind = cls(store, symbol, interval)

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

            elif extractor == "macd_hist":
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
                if r is None:
                    sig = None
                else:
                    _upper, _mid, _lower, _bw, pct_b = r
                    sig = -pct_b

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
        signals.append(sig)
        bar_indices.append(i)

    _nan = float("nan")
    empty: dict[str, Any] = {
        "label": label,
        "n": len(signals),
        **{f"IC_{h}": _nan for h in horizons},
        "peak_horizon": _nan,
        "IC_peak": _nan,
        "half_life": _nan,
    }
    if len(signals) < 20:
        return empty

    sig_arr = np.array(signals)
    result: dict[str, Any] = {"label": label, "n": len(signals)}

    ic_values: list[float] = []
    for h in horizons:
        fwd = fwd_returns[h]
        paired_fwd = np.array([fwd[i] for i in bar_indices])
        mask = ~np.isnan(paired_fwd)
        s, f = sig_arr[mask], paired_fwd[mask]

        if len(s) < 20:
            result[f"IC_{h}"] = _nan
        else:
            result[f"IC_{h}"] = _spearman(s, f)

        ic_values.append(result[f"IC_{h}"])

    finite_pairs = [(h, ic) for h, ic in zip(horizons, ic_values, strict=True) if np.isfinite(ic)]
    if finite_pairs:
        peak_h, peak_ic = max(finite_pairs, key=lambda x: abs(x[1]))
        result["peak_horizon"] = float(peak_h)
        result["IC_peak"] = peak_ic
    else:
        result["peak_horizon"] = _nan
        result["IC_peak"] = _nan

    result["half_life"] = _halflife(horizons, ic_values)
    return result


# ---------------------------------------------------------------------------
# Worker — module-level for pickling
# ---------------------------------------------------------------------------


def _worker_decay(
    args: tuple[str, list[dict], str, int, list[int]],
) -> list[dict[str, Any]]:
    """Evaluate IC decay for all catalogue indicators for one symbol."""
    symbol, rows, interval, warmup, horizons = args
    print(f"  {symbol} ...", flush=True)

    async def _run() -> list[dict[str, Any]]:
        from research.indicator_research.catalogue import load_catalogue

        catalogue = load_catalogue(interval)
        results = []
        for label, cls, params, extractor in catalogue:
            clock = SimulatedClock()
            res = await _evaluate_indicator_decay(
                label, cls, params, extractor, rows, symbol, clock, interval, warmup, horizons
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results-dir", type=Path, default=Path("results"))


async def run(args: argparse.Namespace) -> None:
    """
    Compute IC at Fibonacci horizons for all indicators. Identifies peak
    horizon and IC half-life. Writes indicator_decay_results.csv and
    indicator_decay.html.
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
                executor, _worker_decay, (sym, rows, args.interval, args.warmup, horizons)
            )
            for sym, rows in symbol_rows
        ]
        per_symbol: list[list[dict[str, Any]]] = await asyncio.gather(*futures)

    all_results = [r for sym_results in per_symbol for r in sym_results]

    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        lbl = r["label"]
        for h in horizons:
            v = r.get(f"IC_{h}", float("nan"))
            if np.isfinite(v):
                agg[lbl][f"IC_{h}"].append(v)
        if np.isfinite(r.get("peak_horizon", float("nan"))):
            agg[lbl]["peak_horizon"].append(r["peak_horizon"])
        if np.isfinite(r.get("half_life", float("nan"))):
            agg[lbl]["half_life"].append(r["half_life"])

    summary: list[dict[str, Any]] = []
    _nan = float("nan")

    for lbl, metrics in agg.items():
        row: dict[str, Any] = {"indicator": lbl}
        ic_values: list[float] = []
        for h in horizons:
            vals = metrics.get(f"IC_{h}", [])
            ic = float(np.mean(vals)) if vals else _nan
            row[f"IC_{h}"] = ic
            ic_values.append(ic)

        finite_pairs = [
            (h, ic) for h, ic in zip(horizons, ic_values, strict=True) if np.isfinite(ic)
        ]
        if finite_pairs:
            peak_h, peak_ic = max(finite_pairs, key=lambda x: abs(x[1]))
            row["peak_horizon"] = float(peak_h)
            row["IC_peak"] = peak_ic
        else:
            row["peak_horizon"] = _nan
            row["IC_peak"] = _nan

        row["half_life"] = _halflife(horizons, ic_values)
        summary.append(row)

    summary.sort(
        key=lambda r: abs(r["IC_peak"]) if np.isfinite(r["IC_peak"]) else 0.0,
        reverse=True,
    )

    print(f"\n{'=' * 115}")
    print(
        f"  Indicator IC Decay  |  {month_start.date()} to {month_end.date()}  |  "
        f"{args.interval}  |  {len(args.symbols)} symbols"
    )
    print(f"  Horizons (bars): {horizons}")
    print(f"{'=' * 115}")
    h_header = "  ".join(f"IC_{h:>2}" for h in horizons)
    print(f"  {'Indicator':<22}  {h_header}  {'Peak_H':>7}  {'IC_peak':>8}  {'Half-life':>9}")
    print(f"  {'-' * 110}")
    for r in summary:
        ic_str = "  ".join(
            f"{r[f'IC_{h}']:+.3f}" if np.isfinite(r[f"IC_{h}"]) else "   nan" for h in horizons
        )
        ph = f"{int(r['peak_horizon'])}" if np.isfinite(r["peak_horizon"]) else "nan"
        icp = f"{r['IC_peak']:+.4f}" if np.isfinite(r["IC_peak"]) else "     nan"
        hl = f"{r['half_life']:.1f}" if np.isfinite(r["half_life"]) else "      nan"
        print(f"  {r['indicator']:<22}  {ic_str}  {ph:>7}  {icp}  {hl:>9}")
    print(f"{'=' * 115}")
    print("\n  IC: |IC| > 0.05 useful, > 0.10 strong")
    print(
        "  Half-life: bars until |IC| drops to half its peak value "
        f"(nan = doesn't decay within {horizons[-1]} bars)"
    )

    out_dir = Path(args.results_dir) / "indicator_decay"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "indicator_decay_results.csv"

    if summary:
        fieldnames = (
            ["indicator"] + [f"IC_{h}" for h in horizons] + ["peak_horizon", "IC_peak", "half_life"]
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
    print(f"\n  CSV written to {csv_path}")

    def _ic_color(v: float) -> str:
        if not np.isfinite(v):
            return "#ffffff"
        av = abs(v)
        if av > 0.10:
            return "#4caf50"
        if av > 0.05:
            return "#c8e6c9"
        return "#f5f5f5"

    def _peak_color(v: float) -> str:
        if not np.isfinite(v):
            return "#ffffff"
        av = abs(v)
        if av > 0.10:
            return "#4caf50"
        if av > 0.05:
            return "#a5d6a7"
        return "#ffffff"

    h_cols = [f"IC_{h}" for h in horizons]
    th_list = ["indicator"] + h_cols + ["peak_horizon", "IC_peak", "half_life"]
    th = "".join(f"<th>{c}</th>" for c in th_list)

    html_rows = []
    for r in summary:
        cells = [f"<td><b>{r['indicator']}</b></td>"]
        for h in horizons:
            v = r[f"IC_{h}"]
            color = _ic_color(v)
            fmt = f"{v:+.4f}" if np.isfinite(v) else "nan"
            cells.append(f'<td style="background:{color}">{fmt}</td>')

        ph = r["peak_horizon"]
        cells.append(f"<td>{int(ph) if np.isfinite(ph) else 'nan'}</td>")

        icp = r["IC_peak"]
        color = _peak_color(icp)
        fmt = f"{icp:+.4f}" if np.isfinite(icp) else "nan"
        cells.append(f'<td style="background:{color}"><b>{fmt}</b></td>')

        hl = r["half_life"]
        fmt = f"{hl:.1f}" if np.isfinite(hl) else "nan"
        cells.append(f"<td>{fmt}</td>")

        html_rows.append(f"<tr>{''.join(cells)}</tr>")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Indicator IC Decay — {args.interval}</title>
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
<h2>Indicator IC Decay — {args.interval} | {month_start.date()} to {month_end.date()} | {len(args.symbols)} symbols</h2>
<p>Horizons: {horizons} bars. Sorted by |IC_peak|.</p>
<p>Dark green |IC| &gt; 0.10 (strong), light green |IC| &gt; 0.05 (useful), grey = weak.</p>
<p>Half-life: bars until |IC| falls to half its peak. nan = signal persists beyond {horizons[-1]} bars.</p>
<table>
<thead><tr>{th}</tr></thead>
<tbody>
{"".join(html_rows)}
</tbody>
</table>
</body>
</html>
"""
    html_path = out_dir / "indicator_decay.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  HTML report written to {html_path}")

    if not (len(summary) > 0):
        print("WARNING: Expected IC decay analysis to produce at least one row")
        sys.exit(1)
    finite_peak_ics = [r["IC_peak"] for r in summary if np.isfinite(r["IC_peak"])]
    if not (len(finite_peak_ics) > 0):
        print("WARNING: Expected at least one indicator with a finite peak IC")
        sys.exit(1)
