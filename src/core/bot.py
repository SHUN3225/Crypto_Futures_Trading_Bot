"""Main trading bot orchestrator."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger

from src.core.config import Settings, TradingMode
from src.core.exchange import ExchangeClient
from src.core.models import Position, Side, Signal, SignalAction, TradeResult
from src.core.risk_manager import RiskManager
from src.strategies.base import BaseStrategy
from src.strategies.multi_signal import MultiSignalStrategy
from src.strategies.scalping import ScalpingStrategy


def _build_strategy(settings: Settings) -> BaseStrategy:
    name = settings.strategy_name
    strategies: dict[str, BaseStrategy] = {
        "multi_signal": MultiSignalStrategy(regime_filter=settings.regime_filter),
        "scalping": ScalpingStrategy(),
    }
    if name not in strategies:
        logger.warning("Unknown strategy '{}', defaulting to multi_signal", name)
        return strategies["multi_signal"]
    return strategies[name]


class TradingBot:
    """Orchestrates the full trading loop: fetch → analyze → execute → manage."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exchange = ExchangeClient(settings)
        self._strategy = _build_strategy(settings)
        self._risk: RiskManager | None = None
        self._positions: list[Position] = []
        self._running = False
        self._last_daily_reset: str = ""

    def _init_risk_manager(self) -> None:
        if self._settings.mode == TradingMode.PAPER:
            balance = 10_000.0
            logger.info("[PAPER] Using simulated balance: ${:,.2f}", balance)
        else:
            balance = self._exchange.get_free_balance()
            logger.info("Account balance: ${:,.2f}", balance)

        self._risk = RiskManager(self._settings, balance)

    def start(self) -> None:
        logger.info(
            "Starting {} bot | Strategy: {} | Symbol: {} | Leverage: {}x | Mode: {}",
            self._settings.exchange_id.value,
            self._strategy.name,
            self._settings.symbol,
            self._settings.leverage,
            self._settings.mode.value,
        )
        self._init_risk_manager()
        self._exchange.set_leverage()
        self._running = True

        logger.info(
            "Risk params: SL={:.1%} TP={:.1%} TrailSL={:.1%} MaxDD={:.1%} R:R={:.1f}",
            self._settings.stop_loss_pct,
            self._settings.take_profit_pct,
            self._settings.trailing_stop_pct,
            self._settings.max_drawdown_pct,
            self._risk.get_risk_reward_ratio() if self._risk else 0,
        )

        try:
            while self._running:
                self._tick()
                time.sleep(self._settings.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            self._shutdown()

    async def start_async(self) -> None:
        logger.info("Starting bot (async mode)")
        self._init_risk_manager()
        self._exchange.set_leverage()
        self._running = True

        try:
            while self._running:
                self._tick()
                await asyncio.sleep(self._settings.poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Bot cancelled")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    def _tick(self) -> None:
        try:
            self._maybe_daily_reset()
            assert self._risk is not None
            if self._risk.state.is_halted:
                logger.warning("Trading halted: {}", self._risk.state.halt_reason)
                return

            df = self._exchange.fetch_ohlcv()
            current_price = float(df["close"].iloc[-1])

            self._manage_open_positions(current_price)

            signal = self._strategy.analyze(df)
            logger.debug(
                "Signal: {} score={:.3f} reason={}",
                signal.action.value,
                signal.score,
                signal.reason,
            )

            if signal.action == SignalAction.HOLD:
                return
            if signal.score < self._settings.min_signal_score:
                return

            can_open, deny = self._risk.can_open_position(self._positions)
            if not can_open:
                logger.info("Cannot open position: {}", deny)
                return

            self._open_position(signal, current_price)

        except Exception:
            logger.exception("Error in tick")

    def _open_position(self, signal: Signal, price: float) -> None:
        assert self._risk is not None
        side = Side.LONG if signal.action == SignalAction.BUY else Side.SHORT

        atr_sl = self._settings.atr_sl_multiplier
        atr_tp = self._settings.atr_tp_multiplier
        if atr_sl and atr_tp and signal.atr_value and signal.atr_value > 0:
            sl_dist = signal.atr_value * atr_sl
            tp_dist = signal.atr_value * atr_tp
            if side == Side.LONG:
                sl = round(price - sl_dist, 2)
                tp = round(price + tp_dist, 2)
            else:
                sl = round(price + sl_dist, 2)
                tp = round(price - tp_dist, 2)
            risk_amount = self._risk.state.current_balance * self._settings.max_position_pct
            qty = round(risk_amount / sl_dist, 6) if sl_dist > 0 else 0.0
        else:
            qty = self._risk.calculate_position_size(price, side)
            sl = self._risk.compute_stop_loss(price, side)
            tp = self._risk.compute_take_profit(price, side)
        if qty <= 0:
            return

        order = self._exchange.create_market_order(side, qty)
        self._exchange.create_stop_order(side, qty, sl)
        self._exchange.create_take_profit_order(side, qty, tp)

        pos = Position(
            symbol=self._settings.symbol,
            side=side,
            entry_price=price,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            order_id=order.get("id", ""),
            highest_price=price,
            lowest_price=price,
        )
        self._positions.append(pos)
        logger.info(
            "Opened {} {:.6f} {} @ {:.2f} | SL={:.2f} TP={:.2f} | Score={:.2f}",
            side.value,
            qty,
            self._settings.symbol,
            price,
            sl,
            tp,
            signal.score,
        )

    def _manage_open_positions(self, current_price: float) -> None:
        assert self._risk is not None
        closed: list[int] = []

        for i, pos in enumerate(self._positions):
            pos.update_trailing(current_price, self._settings.trailing_stop_pct)
            should_close, reason = self._risk.should_close_position(pos, current_price)

            if should_close:
                self._close_position(pos, current_price, reason)
                closed.append(i)

        for idx in reversed(closed):
            self._positions.pop(idx)

    def _close_position(self, pos: Position, exit_price: float, reason: str) -> None:
        assert self._risk is not None
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        self._exchange.create_market_order(close_side, pos.quantity, reduce_only=True)
        self._exchange.cancel_all_orders()

        if pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        result = TradeResult(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            opened_at=pos.opened_at,
            reason=reason,
        )
        self._risk.record_trade(result)

    def _maybe_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_daily_reset:
            if self._risk:
                self._risk.reset_daily()
            self._last_daily_reset = today

    def _shutdown(self) -> None:
        logger.info("Shutting down bot…")
        if self._positions:
            logger.warning("{} positions still open at shutdown", len(self._positions))
        self._running = False
        if self._risk:
            state = self._risk.state
            logger.info(
                "Session stats: PnL=${:.2f} | Balance=${:.2f} | Drawdown={:.2%}",
                state.total_pnl,
                state.current_balance,
                state.current_drawdown,
            )
