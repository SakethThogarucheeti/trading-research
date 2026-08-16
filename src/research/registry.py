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
    ``SessionConfig`` subclass so YAML / JSON deserialisation can resolve
    the right config class via ``config_types()``.

    Adding a new session type requires only this decorator — the CLI and
    the dashboard's report registry both dispatch by ``session_type``
    string, with no hardcoded type list to update.
    """

    def decorator(cls: type[TestingSession]) -> type[TestingSession]:
        _REGISTRY[name] = _SessionTypeEntry(
            session_cls=cls,
            config_cls=cls._config_cls,
        )
        return cls

    return decorator


def build_session(config: SessionConfig, **kwargs: object) -> TestingSession:
    """
    Instantiate the right ``TestingSession`` for *config*.

    Dispatches via the global registry — no ``isinstance`` checks required.
    Extra keyword arguments (``db_engine``, ``redis``, ``bus``, etc.) are
    forwarded to the session constructor.

    Raises ``KeyError`` if ``config.type`` has not been registered.
    """
    entry = _REGISTRY[config.type]
    return entry.session_cls(config=config, **kwargs)  # type: ignore[call-arg]


def config_types() -> list[type[SessionConfig]]:
    """
    Return all registered config classes.

    Useful for building a discriminated union at parse time (YAML/JSON
    config loading) — new session types appear automatically here with
    no changes needed elsewhere.
    """
    return [e.config_cls for e in _REGISTRY.values()]


def registered_names() -> list[str]:
    """Return all registered session type names."""
    return list(_REGISTRY.keys())
