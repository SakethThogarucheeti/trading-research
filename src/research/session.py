from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from research.report_base import SessionReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionConfig ABC
# ---------------------------------------------------------------------------


class SessionConfig(ABC):
    """
    Base for all session configuration objects.

    Subclasses must set ``type`` as a ``Literal[<name>]`` class attribute
    that matches the name passed to ``@session_type("<name>")``.

    ``session_id`` is auto-generated when empty; override to pin a run ID.
    """

    type: str
    session_id: str = ""


# ---------------------------------------------------------------------------
# Progress event
# ---------------------------------------------------------------------------


class SessionProgressEvent(BaseModel):
    session_id: str
    session_type: str
    pct_complete: float  # 0.0 – 1.0
    bars_processed: int
    signals_generated: int
    timestamp: datetime


# ---------------------------------------------------------------------------
# TestingSession ABC
# ---------------------------------------------------------------------------


class TestingSession(ABC):
    """
    Base class for all testing session types (backtest, Monte Carlo, etc.).

    Lifecycle
    ---------
    1. Construct with ``config`` and ``results_dir``.
    2. Call ``await run()`` — returns a ``SessionReport``.
    3. Partial results are persisted in ``finally`` so a crash/kill does not
       lose all progress.

    Extensibility
    -------------
    Decorated with ``@session_type("name")`` to register in the global
    registry so callers (CLI commands, the dashboard's report registry)
    can dispatch without ``isinstance`` checks. Each subclass must also
    set::

        _config_cls: ClassVar[type[SessionConfig]] = MyConfig
    """

    _config_cls: ClassVar[type[SessionConfig]]

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    @abstractmethod
    async def run(self) -> SessionReport:
        """
        Execute the session and return a completed report.

        Implementations must wrap the body in ``try/finally`` and call
        ``await self._persist(partial_report)`` in the ``finally`` block so
        partial results survive a keyboard interrupt or process kill.
        """

    async def _persist(self, report: SessionReport) -> None:
        """
        Write ``report`` as JSON + HTML to ``results_dir/{session_id}/``.

        Creates the directory if it does not exist. Silently logs on failure
        so a persist error never masks the original exception.
        """
        try:
            out_dir = self._results_dir / report.session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            json_path = out_dir / "report.json"
            html_path = out_dir / "report.html"

            json_path.write_text(
                json.dumps(report.to_dict(), default=str, indent=2),
                encoding="utf-8",
            )
            html_path.write_text(report.to_html(), encoding="utf-8")
            logger.info("TestingSession: persisted report to %s", out_dir)
        except Exception:
            logger.exception("TestingSession: failed to persist report")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)
