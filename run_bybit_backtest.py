"""
Bybit 데이터 기반 6코인 백테스트 + OKX 교차검증

사용법:
    pip install -e .
    python run_bybit_backtest.py

출력: Bybit vs OKX 비교표 + 최종 결론
"""
import time
import sys

import ccxt
import pandas as pd

from src.core.config import Settings
from src.core.backtester import Backtester
from src.strategies.multi_signal import MultiSignalStrategy


COINS = [
    "DOGE/USDT:USDT", "ETH/USDT:USDT", "ADA/USDT:USDT",
    "DOT/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
]


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    all_data: list[list] = []
    cursor = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=200)
        if not batch:
            break
        all_data.extend(batch)
        cursor = batch[-1][0] + 1
        time.sleep(0.35)
        if len(batch) < 200:
            break
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def run_backtest(
    exchange_name: str,
    exchange: ccxt.Exchange,
    coins: list[str],
) -> list[dict]:
    now = exchange.milliseconds()
    ms_per_4h = 14_400_000
    candles_needed = int(2 * 365 * 6)  # 2 years of 4h candles
    since = now - (candles_needed * ms_per_4h)

    settings = Settings(min_signal_score=0.3, regime_filter=True, max_drawdown_pct=0.10)
    strategy = MultiSignalStrategy(regime_filter=True)

    print(f"\n{'='*70}")
    print(f"  {exchange_name} — 6코인 2년 백테스트")
    print(f"  MaxDD=10% | ATR SL=1.5x, TP=2.0x, Trail=ATR*0.5")
    print(f"  Fee=0.1% + Slip=0.05% | Entry=next bar open | score>=0.3")
    print(f"{'='*70}\n")

    results = []
    for sym in coins:
        print(f"  [{exchange_name}] {sym}...", end=" ", flush=True)
        try:
            df = fetch_ohlcv(exchange, sym, "4h", since)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        if len(df) < 300:
            print(f"데이터 부족 ({len(df)} candles)")
            continue

        days = (df.index[-1] - df.index[0]).days
        mkt_start = float(df["close"].iloc[0])
        mkt_end = float(df["close"].iloc[-1])
        mkt_chg = (mkt_end / mkt_start - 1) * 100

        bt = Backtester(strategy, settings, atr_sl_mult=1.5, atr_tp_mult=2.0)
        m = bt.run(df, min_signal_score=0.3)

        r = {
            "coin": sym.split("/")[0],
            "days": days,
            "mkt_chg": mkt_chg,
            "trades": m.total_trades,
            "wins": m.wins,
            "losses": m.losses,
            "wr": m.win_rate * 100,
            "roi": m.roi * 100,
            "pf": m.profit_factor,
            "max_dd": m.max_drawdown * 100,
            "sharpe": m.sharpe_ratio,
        }
        results.append(r)
        print(f"{m.total_trades}거래, WR={m.win_rate:.0%}, ROI={m.roi:+.2%}, PF={m.profit_factor:.2f}")

    if not results:
        print("  결과 없음!")
        return []

    print(f"\n{'='*70}")
    print(f"{'Coin':<6} {'Mkt':>7} {'Trades':>6} {'W/L':>7} {'WR':>5} {'ROI':>9} {'PF':>5} {'MaxDD':>6} {'Sharpe':>6}")
    print(f"{'-'*70}")
    for r in results:
        print(
            f"{r['coin']:<6} {r['mkt_chg']:>+6.0f}% {r['trades']:>6} "
            f"{r['wins']:>3}/{r['losses']:<3} {r['wr']:>4.0f}% {r['roi']:>+8.1f}% "
            f"{r['pf']:>5.1f} {r['max_dd']:>5.1f}% {r['sharpe']:>6.2f}"
        )
    print(f"{'='*70}")

    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    avg_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_roi = sum(r["roi"] for r in results) / len(results)
    coins_profitable = sum(1 for r in results if r["roi"] > 0)

    print(f"\n  총 거래: {total_trades} ({total_wins}W)")
    print(f"  평균 승률: {avg_wr:.1f}%")
    print(f"  평균 ROI: {avg_roi:+.1f}%")
    print(f"  수익 코인: {coins_profitable}/{len(results)}")
    return results


def print_comparison(r_bybit: list[dict], r_okx: list[dict]) -> None:
    print(f"\n{'='*78}")
    print(f"  BYBIT vs OKX 교차검증 비교")
    print(f"{'='*78}")
    print(f"{'Coin':<6} {'Bybit ROI':>10} {'OKX ROI':>9} {'B거래':>6} {'O거래':>6} {'B PF':>6} {'O PF':>6} {'B WR':>5} {'O WR':>5}")
    print(f"{'-'*78}")

    bybit_map = {r["coin"]: r for r in r_bybit}
    okx_map = {r["coin"]: r for r in r_okx}
    all_coins = list(dict.fromkeys(r["coin"] for r in r_bybit))

    for coin in all_coins:
        b = bybit_map.get(coin)
        o = okx_map.get(coin)
        if b and o:
            print(
                f"{coin:<6} {b['roi']:>+9.1f}% {o['roi']:>+8.1f}% "
                f"{b['trades']:>6} {o['trades']:>6} "
                f"{b['pf']:>5.2f} {o['pf']:>6.2f} "
                f"{b['wr']:>4.0f}% {o['wr']:>4.0f}%"
            )

    print(f"{'='*78}")
    avg_b = sum(r["roi"] for r in r_bybit) / len(r_bybit)
    avg_o = sum(r["roi"] for r in r_okx) / len(r_okx)
    print(f"  평균 ROI: Bybit={avg_b:+.1f}%  vs  OKX={avg_o:+.1f}%")
    diff = abs(avg_b - avg_o)
    print(f"  차이: {diff:.1f}%", end="")
    if diff < 3:
        print(" — 거의 동일 (거래소 간 차이 무시 가능)")
    elif diff < 5:
        print(" — 미미한 차이 (실전 영향 없음)")
    else:
        print(" — 유의미한 차이 (원인 확인 필요)")

    print(f"\n{'='*78}")
    print(f"  결론")
    print(f"{'='*78}")
    if diff < 5:
        print("  ✓ Bybit과 OKX 결과가 거의 동일합니다.")
        print("  ✓ OKX 데이터로 검증된 전략을 Bybit에서 그대로 사용 가능합니다.")
        print(f"  ✓ 6코인 평균 ROI: Bybit {avg_b:+.1f}% / OKX {avg_o:+.1f}%")
        print("  → Bybit 실전 투입 OK")
    else:
        print("  ⚠ 거래소 간 차이가 존재합니다.")
        print("  → Bybit 결과를 기준으로 운용하세요.")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    from loguru import logger
    logger.remove()

    print("=" * 78)
    print("  Bybit vs OKX 교차검증 백테스트")
    print("  6코인: DOGE, ETH, ADA, DOT, AVAX, LINK")
    print("=" * 78)

    # 1) Bybit 백테스트
    try:
        bybit = ccxt.bybit({"options": {"defaultType": "linear"}, "enableRateLimit": True})
        bybit.fetch_ohlcv("BTC/USDT:USDT", "4h", limit=5)  # connectivity test
    except Exception as e:
        print(f"\n❌ Bybit API 접근 불가: {e}")
        print("VPN이 필요하거나 지역 차단일 수 있습니다.")
        sys.exit(1)

    r_bybit = run_backtest("Bybit", bybit, COINS)

    # 2) OKX 백테스트 (교차검증)
    try:
        okx = ccxt.okx({"options": {"defaultType": "swap"}, "enableRateLimit": True})
        r_okx = run_backtest("OKX", okx, COINS)
    except Exception as e:
        print(f"\n⚠ OKX 접근 불가: {e}")
        print("Bybit 결과만 사용합니다.")
        r_okx = []

    # 3) 비교
    if r_bybit and r_okx:
        print_comparison(r_bybit, r_okx)
    elif r_bybit:
        print("\n  OKX 비교 불가 — Bybit 결과만 참고하세요.")
