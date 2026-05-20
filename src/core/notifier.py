"""Discord webhook notifier for trade alerts."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger


class DiscordNotifier:
    """Sends trade alerts to a Discord channel via webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url
        self._enabled = bool(webhook_url)
        if self._enabled:
            logger.info("Discord notifications enabled")

    def _send(self, payload: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            resp = requests.post(self._url, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                logger.warning("Discord webhook returned {}", resp.status_code)
        except Exception as e:
            logger.warning("Discord send failed: {}", e)

    def _send_async(self, payload: dict[str, Any]) -> None:
        threading.Thread(target=self._send, args=(payload,), daemon=True).start()

    def signal_alert(
        self,
        symbol: str,
        action: str,
        score: float,
        price: float,
        reason: str,
        atr: float | None = None,
    ) -> None:
        color = 0x00FF00 if action == "buy" else 0xFF0000
        direction = "LONG" if action == "buy" else "SHORT"
        fields = [
            {"name": "심볼", "value": symbol, "inline": True},
            {"name": "방향", "value": direction, "inline": True},
            {"name": "현재가", "value": f"${price:,.2f}", "inline": True},
            {"name": "시그널 점수", "value": f"{score:.3f}", "inline": True},
        ]
        if atr:
            fields.append({"name": "ATR", "value": f"${atr:,.2f}", "inline": True})
        fields.append({"name": "근거", "value": reason[:200], "inline": False})

        self._send_async({
            "embeds": [{
                "title": f"[Signal] {direction} — {symbol}",
                "color": color,
                "fields": fields,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        })

    def position_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        sl: float,
        tp: float,
        balance: float,
    ) -> None:
        side_str = "LONG" if side == "long" else "SHORT"
        color = 0x2ECC71 if side == "long" else 0xE74C3C
        sl_pct = abs(sl - price) / price * 100
        tp_pct = abs(tp - price) / price * 100

        self._send_async({
            "embeds": [{
                "title": f"[Entry] {side_str} — {symbol}",
                "color": color,
                "fields": [
                    {"name": "심볼", "value": symbol, "inline": True},
                    {"name": "진입가", "value": f"${price:,.2f}", "inline": True},
                    {"name": "수량", "value": f"{qty:.6f}", "inline": True},
                    {"name": "손절 (SL)", "value": f"${sl:,.2f} ({sl_pct:.2f}%)", "inline": True},
                    {"name": "익절 (TP)", "value": f"${tp:,.2f} ({tp_pct:.2f}%)", "inline": True},
                    {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        })

    def position_closed(
        self,
        symbol: str,
        side: str,
        entry: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
        balance: float,
    ) -> None:
        win = pnl > 0
        result = "WIN" if win else "LOSS"
        color = 0x27AE60 if win else 0xC0392B
        side_str = "LONG" if side == "long" else "SHORT"

        self._send_async({
            "embeds": [{
                "title": f"[Exit] {side_str} {result} — {symbol}",
                "color": color,
                "fields": [
                    {"name": "심볼", "value": symbol, "inline": True},
                    {"name": "진입가", "value": f"${entry:,.2f}", "inline": True},
                    {"name": "청산가", "value": f"${exit_price:,.2f}", "inline": True},
                    {"name": "손익", "value": f"${pnl:+,.2f} ({pnl_pct:+.2f}%)", "inline": True},
                    {"name": "청산 사유", "value": reason, "inline": True},
                    {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        })

    def halted(self, reason: str, balance: float) -> None:
        self._send_async({
            "embeds": [{
                "title": "[HALT] Trading Stopped",
                "color": 0xFF0000,
                "description": f"**사유:** {reason}\n**잔고:** ${balance:,.2f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        })

    def bot_started(self, symbol: str, mode: str, balance: float) -> None:
        self._send_async({
            "embeds": [{
                "title": "[Bot] Started",
                "color": 0x3498DB,
                "fields": [
                    {"name": "심볼", "value": symbol, "inline": True},
                    {"name": "모드", "value": mode.upper(), "inline": True},
                    {"name": "잔고", "value": f"${balance:,.2f}", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        })

    def bot_stopped(self, balance: float, total_pnl: float) -> None:
        self._send_async({
            "embeds": [{
                "title": "[Bot] Stopped",
                "color": 0x95A5A6,
                "fields": [
                    {"name": "최종 잔고", "value": f"${balance:,.2f}", "inline": True},
                    {"name": "총 손익", "value": f"${total_pnl:+,.2f}", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        })
