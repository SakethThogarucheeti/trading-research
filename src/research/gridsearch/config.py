"""
Grid-search config loader.

Reads a bundled copy of trading-platform's strategy_config.json (the
hyperparam-search grids and their symbol/interval/equity/date defaults) that
ships inside trading-research itself, alongside this module.

Deviation from the original plan: the plan's sketch called for basing this on
``trading.config.strategy_config``'s own path resolution directly. In practice
that module resolves its JSON file relative to *its own* __file__ four levels
up (``config/ -> trading/ -> src/ -> project root``) — which only lands on a
real ``strategy_config.json`` when trading-platform is installed *editable*
from its own checkout, as trading-integ-tests does. trading-research pins
trading-platform as a git-tagged (non-editable) dependency, so
``trading.config.strategy_config.__file__`` resolves to
``.venv/lib/python3.X/site-packages/trading/config/strategy_config.py`` and
``parents[3]`` lands inside the venv tree, nowhere near a strategy_config.json
(verified: ``_DEFAULT_PATH.exists()`` is False under trading-research's own
install). The source test files' own workaround —
``Path(__file__).parents[3] / "trading-platform" / "strategy_config.json"`` in
e.g. test_rsi_search.py — is exactly the fragile sibling-repo-depth hack the
plan called out as something to avoid.

The fix used here follows the same pattern already applied to
indicator_catalogue.json: bundle a copy of the JSON file inside
trading-research (co-located with this module) and load it directly, so
resolution is independent of trading-research's own file-tree depth or
install mode.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent / "strategy_config.json"


def load_full_config() -> dict[str, Any]:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def strategy_grid(strategy_id: str) -> dict[str, Any]:
    return load_full_config()["hyperparam_search"]["grids"][strategy_id]


def hyperparam_search_defaults() -> dict[str, Any]:
    """symbols/interval/equity/months/end_date shared by all grid-search subcommands."""
    return load_full_config()["hyperparam_search"]


def default_end() -> datetime:
    hp = hyperparam_search_defaults()
    end_date = hp.get("end_date")
    if end_date:
        return datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    return datetime(2026, 4, 17, tzinfo=UTC)


def compute_start(end: datetime, months: int | None) -> datetime:
    """Mirrors the source grid-search tests' ``_start()`` helper."""
    if months is None:
        return datetime(2025, 6, 1, tzinfo=UTC)
    return (end - timedelta(days=months * 30)).replace(hour=0, minute=0, second=0, microsecond=0)
