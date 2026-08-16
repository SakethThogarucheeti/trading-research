from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import polars as pl

from trading.core.schemas import CandleEvent
from trading.core.lifecycle.component import Component
from trading.core.lifecycle.runtime import Runtime
from trading.candles.service.bar_accumulator import SymbolConfig

logger = logging.getLogger(__name__)


class CandlePlayer(Component):
    """
    Replay historical OHLCV data as ``CandleEvent`` objects via a callback.

    Loads all (symbol × interval) DataFrames in ``_setup()``, merges them
    into a single globally time-sorted event queue, then replays each row
    by calling ``on_candle(event)`` in ``_run()``.

    After the last bar, calls ``runtime.stop()`` so the attached
    ``Runtime`` (and all its components) shuts down cleanly.

    No-lookahead guarantee
    ----------------------
    All events are sorted by ``date`` globally before replay. Each bar is
    delivered only after previous bars (across all symbols) in the same
    time bucket have been delivered. ``tick_log_id`` is set to 0 on every
    backtest candle — it is never written to the ``tick_logs`` table.

    Progress callback
    -----------------
    ``on_progress(bars_done, bar_ts)`` is called after every bar, allowing
    ``BacktestSession`` to emit progress updates without coupling this
    component to the session layer.
    """

    def __init__(
        self,
        symbols: list[SymbolConfig],
        intervals: list[str],
        start: datetime,
        end: datetime,
        runtime: Runtime,
        on_candle: Callable[[CandleEvent], Awaitable[None]],
        on_progress: Callable[[int, datetime], Awaitable[None]],
        data: dict[tuple[str, str], pl.DataFrame],  # (symbol, interval) → OHLCV df
        replay_delay_secs: float = 0.0,
        on_bar_price: Callable[[str, float], None] | None = None,
    ) -> None:
        super().__init__(name="candle_player")
        self._symbols = symbols
        self._intervals = intervals
        self._start = start
        self._end = end
        self._runtime = runtime
        self._on_candle = on_candle
        self._on_progress = on_progress
        self._data = data
        self._replay_delay_secs = replay_delay_secs
        self._on_bar_price = on_bar_price
        self._event_queue: list[tuple[datetime, str, str, CandleEvent]] = []

    async def _setup(self) -> None:
        """Build the globally sorted event queue from pre-loaded DataFrames."""
        symbol_map = {sc.symbol: sc for sc in self._symbols}

        for (symbol, interval), df in self._data.items():
            sym_config = symbol_map.get(symbol)
            if sym_config is None:
                logger.warning("CandlePlayer: symbol %r not in symbols list — skipping", symbol)
                continue

            instr_type = sym_config.instrument_type

            for row in df.iter_rows(named=True):
                ts: datetime = row["date"]
                if not isinstance(ts, datetime):
                    ts = datetime.fromisoformat(str(ts)).replace(tzinfo=UTC)

                event = CandleEvent(
                    symbol=symbol,
                    instrument_type=instr_type,
                    interval=interval,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    timestamp=ts,
                    tick_log_id=0,  # backtest candles never write to tick_logs
                )
                self._event_queue.append((ts, symbol, interval, event))

        # Global sort by timestamp → no lookahead bias across symbols/intervals
        self._event_queue.sort(key=lambda x: x[0])
        logger.info(
            "CandlePlayer: loaded %d events across %d symbol-interval pairs",
            len(self._event_queue),
            len(self._data),
        )

    async def _run(self) -> None:
        bars_done = 0
        for bar_ts, symbol, _interval, event in self._event_queue:
            # Advance the clock BEFORE dispatching so risk gates and indicators
            # (VWAP session_open_utc, intraday cutoff) see this bar's timestamp.
            bars_done += 1
            try:
                await self._on_progress(bars_done, bar_ts)
            except Exception:
                logger.debug("CandlePlayer: on_progress callback raised", exc_info=True)

            if self._on_bar_price is not None:
                self._on_bar_price(symbol, event.close)

            await self._on_candle(event)

            if self._replay_delay_secs > 0:
                await asyncio.sleep(self._replay_delay_secs)

        logger.info("CandlePlayer: replay complete — %d bars published", bars_done)
        await asyncio.sleep(0.05)
        self._runtime.stop()

    async def _teardown(self) -> None:
        logger.debug("CandlePlayer: teardown")
