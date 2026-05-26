"""
분기별 코인 스크리닝 — 시총 상위 알트 자동 백테스트 + 교체 추천

사용법:
    pip install -e .
    python run_screening.py

출력:
    1) 시총 상위 알트코인 자동 탐색
    2) 전체 코인 2년 백테스트 (롱/숏 분리)
    3) 기존 포트폴리오 대비 교체 추천
"""
import time
import sys
import logging
from datetime import datetime, timezone

logging.disable(logging.CRITICAL)
from loguru import logger  # noqa: E402
logger.remove()

import ccxt  # noqa: E402
import pandas as pd  # noqa: E402

from src.core.config import Settings  # noqa: E402
from src.core.backtester import Backtester  # noqa: E402
from src.core.models import Side  # noqa: E402
from src.strategies.multi_signal import MultiSignalStrategy  # noqa: E402


# ──────────────────────── 현재 포트폴리오 ────────────────────────
CURRENT_PORTFOLIO = [
    "DOGE/USDT:USDT", "OP/USDT:USDT", "ARB/USDT:USDT", "ETH/USDT:USDT",
    "ADA/USDT:USDT", "DOT/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
]

EXCLUDED_BASES = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDD",
    "WBTC", "WETH", "STETH", "CBETH",
}

TOP_N_CANDIDATES = 20  # 시총 상위 몇 개까지 스캔
PORTFOLIO_SIZE = 8     # 최종 포트폴리오 크기
MIN_TRADES = 30        # 최소 거래 수 (통계 신뢰)
MIN_24H_VOL = 10_000_000  # 최소 24h 거래량 ($10M)


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    all_data: list[list] = []
    cursor = since_ms
    last_ts = -1
    for _ in range(500):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        batch.sort(key=lambda x: x[0])
        if batch[-1][0] == last_ts:
            break
        last_ts = batch[-1][0]
        all_data.extend(batch)
        cursor = batch[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000 + 0.1)
        if len(batch) < 1000:
            break
    if not all_data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def discover_candidates(exchange: ccxt.Exchange) -> list[str]:
    """Find top altcoins by 24h volume from futures markets."""
    exchange.load_markets(reload=True)
    scored: list[tuple[str, float]] = []

    candidates = []
    for symbol, market in exchange.markets.items():
        if not market.get("swap"):
            continue
        if market.get("quote") != "USDT":
            continue
        base = market.get("base", "")
        if base in EXCLUDED_BASES:
            continue
        if not symbol.endswith(":USDT"):
            continue
        candidates.append(symbol)

    try:
        tickers = exchange.fetch_tickers(candidates)
    except Exception:
        tickers = {}
        for sym in candidates[:50]:
            try:
                tickers[sym] = exchange.fetch_ticker(sym)
            except Exception:
                continue

    for sym in candidates:
        ticker = tickers.get(sym, {})
        vol = float(ticker.get("quoteVolume", 0) or 0)
        if vol >= MIN_24H_VOL:
            scored.append((sym, vol))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scored[:TOP_N_CANDIDATES]]


def run_backtest_coin(
    exchange: ccxt.Exchange, symbol: str, since_ms: int,
    settings: Settings, strategy: MultiSignalStrategy,
) -> dict | None:
    coin = symbol.split("/")[0]
    try:
        df = fetch_ohlcv(exchange, symbol, "4h", since_ms)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None
    if len(df) < 300:
        print(f"    데이터 부족 ({len(df)} candles)")
        return None

    days = (df.index[-1] - df.index[0]).days
    mkt_start = float(df["close"].iloc[0])
    mkt_end = float(df["close"].iloc[-1])
    mkt_chg = (mkt_end / mkt_start - 1) * 100

    bt = Backtester(strategy, settings, atr_sl_mult=1.5, atr_tp_mult=2.0)
    m = bt.run(df, min_signal_score=0.3)

    longs = [t for t in m.trades if t.side == Side.LONG]
    shorts = [t for t in m.trades if t.side == Side.SHORT]
    long_pnl = sum(t.pnl for t in longs)
    short_pnl = sum(t.pnl for t in shorts)
    long_wins = sum(1 for t in longs if t.pnl > 0)
    short_wins = sum(1 for t in shorts if t.pnl > 0)

    return {
        "symbol": symbol,
        "coin": coin,
        "days": days,
        "candles": len(df),
        "mkt_chg": mkt_chg,
        "trades": m.total_trades,
        "wins": m.wins,
        "losses": m.losses,
        "wr": m.win_rate * 100,
        "roi": m.roi * 100,
        "pf": m.profit_factor,
        "max_dd": m.max_drawdown * 100,
        "sharpe": m.sharpe_ratio,
        "long_n": len(longs),
        "short_n": len(shorts),
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "long_wr": (long_wins / len(longs) * 100) if longs else 0,
        "short_wr": (short_wins / len(shorts) * 100) if shorts else 0,
        "both_profitable": long_pnl > 0 and short_pnl > 0,
    }


def main() -> None:
    print("=" * 90)
    print("  분기별 코인 스크리닝")
    print(f"  현재 포트폴리오: {', '.join(s.split('/')[0] for s in CURRENT_PORTFOLIO)}")
    print(f"  실행 시각: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    print("=" * 90)

    # 1) Exchange 초기화
    try:
        exchange = ccxt.okx({"options": {"defaultType": "swap"}, "enableRateLimit": True})
        exchange.fetch_ohlcv("BTC/USDT:USDT", "4h", limit=5)
    except Exception:
        try:
            exchange = ccxt.bybit({"options": {"defaultType": "linear"}, "enableRateLimit": True})
            exchange.fetch_ohlcv("BTC/USDT:USDT", "4h", limit=5)
            print("  (OKX 접근 불가 — Bybit 사용)")
        except Exception as e:
            print(f"\n❌ 거래소 접근 불가: {e}")
            sys.exit(1)

    now = exchange.milliseconds()
    ms_per_4h = 14_400_000
    since = now - (int(2 * 365 * 6) * ms_per_4h)

    # 2) 후보 탐색
    print("\n  시총/거래량 상위 후보 탐색 중...", flush=True)
    top_symbols = discover_candidates(exchange)
    current_set = set(CURRENT_PORTFOLIO)
    new_candidates = [s for s in top_symbols if s not in current_set]
    all_symbols = list(dict.fromkeys(CURRENT_PORTFOLIO + new_candidates))

    print(f"  현재 포트폴리오: {len(CURRENT_PORTFOLIO)}코인")
    print(f"  신규 후보: {len(new_candidates)}코인 ({', '.join(s.split('/')[0] for s in new_candidates[:10])}...)")
    print(f"  총 스크리닝: {len(all_symbols)}코인\n")

    # 3) 백테스트
    settings = Settings(min_signal_score=0.3, regime_filter=True, max_drawdown_pct=0.10)
    strategy = MultiSignalStrategy(regime_filter=True)

    results: list[dict] = []
    for i, sym in enumerate(all_symbols, 1):
        coin = sym.split("/")[0]
        tag = "현재" if sym in current_set else "후보"
        print(f"  [{i}/{len(all_symbols)}] [{tag}] {coin}...", end=" ", flush=True)
        r = run_backtest_coin(exchange, sym, since, settings, strategy)
        if r:
            r["tag"] = tag
            results.append(r)
            sign = "+" if r["both_profitable"] else "-"
            print(
                f"{r['trades']}거래 WR={r['wr']:.0f}% ROI={r['roi']:+.1f}% PF={r['pf']:.2f} "
                f"롱${r['long_pnl']:+,.0f} 숏${r['short_pnl']:+,.0f} [{sign}양방향]"
            )
        else:
            print("SKIP")

    if not results:
        print("\n  결과 없음!")
        return

    # 4) 결과표
    results.sort(key=lambda x: x["roi"], reverse=True)

    print(f"\n{'='*100}")
    print(f"  전체 순위 (ROI 기준)")
    print(f"{'='*100}")
    print(
        f"{'#':>2} {'Tag':>4} {'Coin':<6} {'Mkt':>7} {'Trades':>6} {'WR':>5} "
        f"{'ROI':>9} {'PF':>5} {'MaxDD':>6} {'롱PnL':>10} {'숏PnL':>10} {'양방향':>6}"
    )
    print(f"{'-'*100}")
    for i, r in enumerate(results, 1):
        both = "✓" if r["both_profitable"] else "✗"
        star = "★" if r["roi"] > 0 and r["pf"] >= 1.2 and r["trades"] >= MIN_TRADES else " "
        print(
            f"{i:>2} {r['tag']:>4} {r['coin']:<6} {r['mkt_chg']:>+6.0f}% {r['trades']:>6} "
            f"{r['wr']:>4.0f}% {r['roi']:>+8.1f}% {r['pf']:>5.2f} {r['max_dd']:>5.1f}% "
            f"${r['long_pnl']:>+9,.0f} ${r['short_pnl']:>+9,.0f}   {both}  {star}"
        )
    print(f"{'='*100}")

    # 5) 추천 포트폴리오
    qualified = [
        r for r in results
        if r["roi"] > 0 and r["pf"] >= 1.1 and r["trades"] >= MIN_TRADES and r["both_profitable"]
    ]
    recommended = qualified[:PORTFOLIO_SIZE]

    print(f"\n{'='*100}")
    print(f"  추천 포트폴리오 (상위 {PORTFOLIO_SIZE}코인, ROI>0 & PF≥1.1 & 거래≥{MIN_TRADES} & 양방향수익)")
    print(f"{'='*100}")
    if recommended:
        for i, r in enumerate(recommended, 1):
            change = "유지" if r["tag"] == "현재" else "🆕 신규"
            print(
                f"  {i}. {r['coin']:<6} ROI={r['roi']:+.1f}% | PF={r['pf']:.2f} | "
                f"WR={r['wr']:.0f}% | {r['trades']}거래 | "
                f"롱${r['long_pnl']:+,.0f} 숏${r['short_pnl']:+,.0f} | {change}"
            )

        rec_coins = [r["coin"] for r in recommended]
        cur_coins = [s.split("/")[0] for s in CURRENT_PORTFOLIO]
        added = [c for c in rec_coins if c not in cur_coins]
        removed = [c for c in cur_coins if c not in rec_coins]

        print(f"\n  추천 심볼:")
        rec_symbols = ",".join(r["symbol"] for r in recommended)
        print(f"  BOT_SYMBOLS={rec_symbols}")

        avg_roi = sum(r["roi"] for r in recommended) / len(recommended)
        print(f"\n  평균 ROI: {avg_roi:+.1f}%")
        if added:
            print(f"  추가: {', '.join(added)}")
        if removed:
            print(f"  제거: {', '.join(removed)}")
        if not added and not removed:
            print(f"  변경 없음 — 현재 포트폴리오 유지")
    else:
        print("  조건 충족 코인 부족 — 현재 포트폴리오 유지 권장")

    print(f"\n{'='*100}")
    print(f"  .env에 BOT_SYMBOLS 복사 후 봇 재시작하면 적용됩니다.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
