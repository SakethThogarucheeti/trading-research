from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research.session import SessionConfig, TestingSession

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, _SessionTypeEntry] = {}


@dataclass
class _SessionTypeEntry:
    session_cls: type[TestingSession]
    config_cls: type[SessionConfig]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def session_type(name: str):
    """
    Class decorator that registers a ``TestingSession`` subclass.

    Usage::

        @session_type("backtest")
        class BacktestSession(TestingSession):
            _config_cls = BacktestConfig
            ...

    The ``name`` must match the ``type`` literal on the paired
    ``SessionConfig`` subclass.

    Adding a new test type requires only this decorator — no registry
    changes needed elsewhere.
    """

    def decorator(cls: type[TestingSession]) -> type[TestingSession]:
        _REGISTRY[name] = _SessionTypeEntry(
            session_cls=cls,
            config_cls=cls._config_cls,
        )
        return cls

    return decorator


def registered_names() -> list[str]:
    """Return all registered session type names."""
    return list(_REGISTRY.keys())
