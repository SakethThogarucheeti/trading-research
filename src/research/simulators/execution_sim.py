from __future__ import annotations

import asyncio
import logging
import math
import random
from datetime import datetime
from uuid import uuid4

import polars as pl

from trading.broker.service.broker import Broker
from trading.broker.service.paper_broker import AbstractPriceStore
from trading.core.schemas import OrderType, Side

logger = logging.getLogger(__name__)


class SlippageFillSimulator(Broker):
    """
    Simulated broker for backtesting that models real-world execution imperfections.

    Plugs into ``DirectExecutionEngine`` in place of ``ZerodhaBroker``. All
    market-data methods (``get_instruments``, ``get_ohlc``) return empty
    DataFrames — only ``place_order`` is meaningful during replay.

    Slippage
    --------
    ``fill_price = price × (1 + slippage_pct/100)`` for BUY.
    ``fill_price = price × (1 − slippage_pct/100)`` for SELL.

    Partial fills
    -------------
    With probability ``partial_fill_prob`` a fill is split: the first fill
    covers ``floor(qty × 0.5)`` units immediately; the remainder arrives
    after ``latency_secs``. The second fill uses a second ``place_order``
    call from the ``DirectExecutionEngine``, so idempotency is handled
    upstream.

    Latency
    -------
    ``await asyncio.sleep(latency_secs)`` before each fill notification.

    Reproducibility
    ---------------
    ``seed`` controls the internal ``random.Random`` instance so partial-fill
    decisions are deterministic across runs.
    """

    def __init__(
        self,
        price_store: AbstractPriceStore,
        slippage_pct: float = 0.05,
        partial_fill_prob: float = 0.0,
        latency_secs: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._price_store = price_store
        self._slippage_pct = slippage_pct
        self._partial_fill_prob = partial_fill_prob
        self._latency_secs = latency_secs
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Broker ABC — market data (unused in backtesting)
    # ------------------------------------------------------------------

    def get_instruments(self) -> pl.DataFrame:
        return pl.DataFrame()

    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        return pl.DataFrame()

    # ------------------------------------------------------------------
    # Broker ABC — order placement
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType,
        limit_price: float | None = None,
        instrument_type: str = "EQUITY",
        tick_log_id: int = 0,
    ) -> str:
        order_id = f"SIM_{uuid4().hex[:12].upper()}"
        price = self._get_fill_price(symbol, side, limit_price)

        if price is None:
            logger.warning(
                "SlippageFillSimulator: no price for %s — order %s rejected",
                symbol,
                order_id,
            )
            return order_id

        if self._latency_secs > 0:
            await asyncio.sleep(self._latency_secs)

        is_partial = self._partial_fill_prob > 0 and self._rng.random() < self._partial_fill_prob

        if is_partial:
            first_qty = math.floor(qty * 0.5)
            remaining_qty = qty - first_qty
            await self._notify_fill(order_id, price, first_qty, symbol, side)
            # Schedule the remainder asynchronously
            asyncio.get_running_loop().create_task(
                self._delayed_fill(order_id, price, remaining_qty, symbol, side)
            )
        else:
            await self._notify_fill(order_id, price, qty, symbol, side)

        return order_id

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_fill_price(
        self,
        symbol: str,
        side: Side,
        limit_price: float | None,
    ) -> float | None:
        base_price = limit_price if limit_price is not None else self._price_store.get(symbol)
        if base_price is None:
            return None

        slippage = self._slippage_pct / 100.0
        if side == Side.BUY:
            return base_price * (1 + slippage)
        return base_price * (1 - slippage)

    async def _notify_fill(
        self,
        order_id: str,
        price: float,
        qty: int,
        symbol: str,
        side: Side,
    ) -> None:
        """
        Update PriceStore with fill price so downstream equity tracking sees it.
        The actual FillEvent is published by DirectExecutionEngine.handle_fill().
        """
        self._price_store.update(symbol, price)
        logger.debug(
            "SlippageFillSimulator: fill order=%s side=%s qty=%d price=%.4f",
            order_id,
            side.value,
            qty,
            price,
        )

    async def _delayed_fill(
        self,
        order_id: str,
        price: float,
        qty: int,
        symbol: str,
        side: Side,
    ) -> None:
        await asyncio.sleep(self._latency_secs)
        await self._notify_fill(order_id, price, qty, symbol, side)
