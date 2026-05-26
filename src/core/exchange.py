"""Exchange adapter wrapping ccxt for futures trading."""

from __future__ import annotations

from typing import Any

import ccxt
import pandas as pd
from loguru import logger

from src.core.config import ExchangeId, Settings, TradingMode
from src.core.models import Side


class ExchangeClient:
    """Thin wrapper around ccxt for unified futures API access."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._exchange = self._create_exchange()

    def _create_exchange(self) -> ccxt.Exchange:
        exchange_cls = getattr(ccxt, self._settings.exchange_id.value)
        params: dict[str, Any] = {
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        }

        if self._settings.mode == TradingMode.PAPER:
            logger.info("[PAPER] Using public API — no credentials required")
        else:
            params["apiKey"] = self._settings.api_key
            params["secret"] = self._settings.api_secret
            if self._settings.api_passphrase:
                params["password"] = self._settings.api_passphrase
            if self._settings.testnet:
                params["sandbox"] = True

        return exchange_cls(params)

    @property
    def exchange(self) -> ccxt.Exchange:
        return self._exchange

    def fetch_ohlcv(
        self, symbol: str | None = None, timeframe: str | None = None, limit: int = 300
    ) -> pd.DataFrame:
        sym = symbol or self._settings.symbol
        tf = timeframe or self._settings.timeframe
        raw = self._exchange.fetch_ohlcv(sym, tf, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        sym = symbol or self._settings.symbol
        return self._exchange.fetch_ticker(sym)  # type: ignore[no-any-return]

    def fetch_balance(self) -> dict[str, Any]:
        return self._exchange.fetch_balance()  # type: ignore[no-any-return]

    def get_free_balance(self, currency: str = "USDT") -> float:
        balance = self.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0.0))

    def set_leverage(self, leverage: int | None = None, symbol: str | None = None) -> None:
        lev = leverage or self._settings.leverage
        sym = symbol or self._settings.symbol
        try:
            self._exchange.set_leverage(lev, sym)
            logger.info("Leverage set to {}x for {}", lev, sym)
        except Exception as e:
            logger.warning("Could not set leverage: {}", e)

    def create_market_order(
        self,
        side: Side,
        quantity: float,
        symbol: str | None = None,
        reduce_only: bool = False,
        sl_price: float | None = None,
        tp_price: float | None = None,
    ) -> dict[str, Any]:
        sym = symbol or self._settings.symbol
        order_side = "buy" if side == Side.LONG else "sell"
        params: dict[str, Any] = {}
        if reduce_only:
            params["reduceOnly"] = True

        if self._settings.mode == TradingMode.PAPER:
            return self._simulate_order(sym, order_side, quantity)

        # Bybit V5: attach SL/TP directly to the opening order (no separate orders needed)
        if not reduce_only and self._settings.exchange_id == ExchangeId.BYBIT:
            if sl_price:
                params["stopLoss"] = str(sl_price)
            if tp_price:
                params["takeProfit"] = str(tp_price)

        order = self._exchange.create_market_order(sym, order_side, quantity, params=params)
        logger.info(
            "Market {} order placed: {} {} @ market | SL={} TP={}",
            order_side, quantity, sym, sl_price or "-", tp_price or "-",
        )
        return order  # type: ignore[no-any-return]

    def create_stop_order(
        self,
        side: Side,
        quantity: float,
        stop_price: float,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Fallback for non-Bybit exchanges. Bybit uses SL in create_market_order."""
        sym = symbol or self._settings.symbol
        order_side = "sell" if side == Side.LONG else "buy"

        if self._settings.mode == TradingMode.PAPER:
            return self._simulate_order(sym, order_side, quantity, stop_price)

        if self._settings.exchange_id == ExchangeId.BYBIT:
            # Bybit: update SL on existing position via set_trading_stop
            try:
                self._exchange.set_trading_stop(sym, params={"stopLoss": str(stop_price)})
                logger.info("Bybit SL updated: {} @ {}", sym, stop_price)
            except Exception as e:
                logger.warning("Could not update Bybit SL: {}", e)
            return {}

        stop_params: dict[str, Any] = {"reduceOnly": True, "stopPrice": stop_price}
        order = self._exchange.create_order(
            sym, "stop_market", order_side, quantity, params=stop_params,
        )
        logger.info("Stop order placed: {} {} @ {}", order_side, quantity, stop_price)
        return order  # type: ignore[no-any-return]

    def create_take_profit_order(
        self,
        side: Side,
        quantity: float,
        tp_price: float,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Fallback for non-Bybit exchanges. Bybit uses TP in create_market_order."""
        sym = symbol or self._settings.symbol
        order_side = "sell" if side == Side.LONG else "buy"

        if self._settings.mode == TradingMode.PAPER:
            return self._simulate_order(sym, order_side, quantity, tp_price)

        if self._settings.exchange_id == ExchangeId.BYBIT:
            try:
                self._exchange.set_trading_stop(sym, params={"takeProfit": str(tp_price)})
                logger.info("Bybit TP updated: {} @ {}", sym, tp_price)
            except Exception as e:
                logger.warning("Could not update Bybit TP: {}", e)
            return {}

        tp_params: dict[str, Any] = {"reduceOnly": True, "stopPrice": tp_price}
        order = self._exchange.create_order(
            sym, "take_profit_market", order_side, quantity, params=tp_params,
        )
        logger.info("Take-profit order placed: {} {} @ {}", order_side, quantity, tp_price)
        return order  # type: ignore[no-any-return]

    def cancel_all_orders(self, symbol: str | None = None) -> None:
        sym = symbol or self._settings.symbol
        try:
            self._exchange.cancel_all_orders(sym)
            logger.info("All orders cancelled for {}", sym)
        except Exception as e:
            logger.warning("Could not cancel orders: {}", e)

    def fetch_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        sym = symbol or self._settings.symbol
        positions = self._exchange.fetch_positions([sym])
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    def fetch_realized_pnl(self, symbol: str | None = None, since_ms: int | None = None) -> float | None:
        """Fetch realized PnL for trades after `since_ms` (position open time).

        Returns summed realized PnL in USDT, or None if unavailable.
        """
        sym = symbol or self._settings.symbol
        try:
            trades = self._exchange.fetch_my_trades(sym, since=since_ms, limit=20)
            if not trades:
                return None
            total_pnl = 0.0
            found = False
            for trade in trades:
                info = trade.get("info", {})
                # Bybit returns closedPnl in trade info
                pnl_str = info.get("closedPnl") or info.get("realizedPnl") or info.get("closed_pnl")
                if pnl_str is not None:
                    total_pnl += float(pnl_str)
                    found = True
            return total_pnl if found else None
        except Exception as e:
            logger.warning("Could not fetch realized PnL: {}", e)
            return None

    @staticmethod
    def _simulate_order(
        symbol: str, side: str, quantity: float, price: float | None = None
    ) -> dict[str, Any]:
        logger.info("[PAPER] {} {} qty={} price={}", side, symbol, quantity, price or "market")
        return {
            "id": "paper-sim",
            "symbol": symbol,
            "side": side,
            "amount": quantity,
            "price": price,
            "status": "closed",
            "info": {"paper": True},
        }

    def get_exchange_id(self) -> ExchangeId:
        return self._settings.exchange_id
