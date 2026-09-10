"""
Shared catalogue loader for indicator research CLI subcommands.

Reads indicator_catalogue.json and returns structured tuples for use by all
indicator-research modules. Each entry becomes (label, cls, params_instance, extractor_str).

The extractor string is a picklable sentinel used in ProcessPoolExecutor workers:
  "neg"           → signal = -raw
  "rsi"           → signal = 100.0 - raw
  "id"            → signal = raw as-is
  "macd_hist"     → calls compute_full(); returns index 2 (histogram)
  "psar"          → calls compute_full(); returns 1.0 if bullish else -1.0
  "vwap_dev"      → calls compute(); returns (vwap - close) / close, negated
  "bollinger_pctb"→ calls compute_full(); returns -pct_b (index 4)
  "keltner_pct"   → calls compute_full(); returns -(mid - lower) / width
  "donchian_pct"  → calls compute_full(); returns -(mid - lower) / width

Session-aware indicators (VWAP, VWAPBands, SessionHighLowPct) require a clock
to be passed at construction time — the loader cannot construct those instances
itself. It returns the class and params; callers supply the clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantindicators.base import Indicator, IndicatorParameters

_CATALOGUE_PATH = Path(__file__).parent / "indicator_catalogue.json"

# Aliases for session-aware indicators — callers must pass clock at construction
_SESSION_ALIASES = frozenset({"vwap", "vwap_bands", "session_hl_pct"})


def load_catalogue(
    interval: str,
) -> list[tuple[str, type[Indicator] | None, IndicatorParameters | None, str]]:
    """
    Load indicator entries for the given interval section.

    Returns a list of (label, cls, params, extractor) tuples.
    For entries with alias=null (e.g. EMA_cross_9_21), cls and params are None.
    For session-aware indicators, cls is returned but callers must construct
    instances with an additional clock argument.

    Args:
        interval: Key in indicator_catalogue.json ("15min", "day", "wf_sweep").

    Raises:
        KeyError: if interval is not in the catalogue.
        ValueError: if an alias does not resolve to a registered indicator.
    """
    import importlib
    import pkgutil

    import quantindicators.library as _lib_pkg

    for _info in pkgutil.iter_modules(_lib_pkg.__path__):
        importlib.import_module(f"quantindicators.library.{_info.name}")

    raw: dict[str, list[dict[str, Any]]] = json.loads(_CATALOGUE_PATH.read_text(encoding="utf-8"))
    entries = raw[interval]

    result: list[tuple[str, type[Indicator] | None, IndicatorParameters | None, str]] = []
    for entry in entries:
        label: str = entry["label"]
        alias: str | None = entry["alias"]
        params_dict: dict[str, Any] = entry["params"]
        extractor: str = entry["extractor"]

        if alias is None or is_session_aware(alias):
            cls = Indicator.lookup(alias) if alias is not None else None
            result.append((label, cls, None, extractor))
            continue

        cls = Indicator.lookup(alias)
        params = cls.Parameters(**params_dict)
        result.append((label, cls, params, extractor))

    return result


def is_session_aware(alias: str | None) -> bool:
    """Return True if the alias belongs to a session-aware indicator."""
    return alias in _SESSION_ALIASES
