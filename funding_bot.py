import time
import sys
from datetime import datetime, timezone

import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

API_URL = "http://localhost:3001/api/funding-arb"
POLL_INTERVAL = 30  # seconds

console = Console()


def fetch_data():
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "백엔드 서버 연결 실패 (localhost:3001)"
    except requests.exceptions.Timeout:
        return None, "요청 시간 초과"
    except Exception as e:
        return None, str(e)


def fmt_countdown(next_funding_iso: str) -> str:
    try:
        target = datetime.fromisoformat(next_funding_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = int((target - now).total_seconds())
        if diff <= 0:
            return "[yellow]곧[/yellow]"
        h, rem = divmod(diff, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "-"


def fmt_premium(val: float) -> Text:
    text = Text(f"{val:+.4f}%")
    text.stylize("bold green" if val <= 0 else "bold red")
    return text


def fmt_funding(val: float) -> Text:
    text = Text(f"{val:.4f}%")
    if val >= 0.05:
        text.stylize("bold bright_green")
    elif val >= 0.02:
        text.stylize("green")
    else:
        text.stylize("dim green")
    return text


def build_table(data: dict) -> Table:
    krw_rate = data.get("krwRate", 0)
    ts = data.get("ts", 0)
    opportunities = data.get("opportunities", [])
    updated = datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S") if ts else "-"

    table = Table(
        title=f"[bold cyan]펀딩비 차익 봇[/bold cyan]  "
              f"[dim]KRW/USDT: {krw_rate:,.2f}  |  업데이트: {updated}  |  {len(opportunities)}개 코인[/dim]",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold white",
        show_lines=False,
        expand=True,
    )

    table.add_column("코인",         justify="center", style="bold white",  min_width=6)
    table.add_column("업비트 (KRW)", justify="right",  style="white",       min_width=14)
    table.add_column("바이낸스 (USDT)", justify="right", style="white",     min_width=14)
    table.add_column("김프",         justify="right",  min_width=10)
    table.add_column("펀딩비 (8h)",  justify="right",  min_width=10)
    table.add_column("펀딩비 (연)",  justify="right",  style="dim",         min_width=8)
    table.add_column("다음 펀딩",    justify="center", min_width=10)
    table.add_column("전략",         justify="center", style="dim cyan",     min_width=22)

    if not opportunities:
        table.add_row(
            "[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]",
            "[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]",
            "[dim]-[/dim]",
            "[dim]진입 조건 없음[/dim]",
        )
        return table

    for opp in opportunities:
        table.add_row(
            f"[bold]{opp['base']}[/bold]",
            f"{opp['upbitAskKrw']:,.0f}",
            f"{opp['binanceMidUsdt']:,.4f}",
            fmt_premium(opp["kimchiPremium"]),
            fmt_funding(opp["fundingRate"]),
            f"{opp['fundingRateAnnualized']:.2f}%",
            fmt_countdown(opp["nextFundingTime"]),
            "업비트 매수 + 바이낸스 숏",
        )

    return table


def run():
    console.clear()
    last_data = None
    last_error = None
    next_poll = 0  # poll immediately on start

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            now = time.time()

            # 폴링 주기마다 데이터 갱신
            if now >= next_poll:
                data, err = fetch_data()
                if data:
                    last_data = data
                    last_error = None
                else:
                    last_error = err
                next_poll = now + POLL_INTERVAL

            # 렌더링
            remaining = max(0, int(next_poll - time.time()))

            if last_error:
                panel = Panel(
                    f"[bold red]오류:[/bold red] {last_error}\n\n[dim]{remaining}초 후 재시도...[/dim]",
                    title="[bold cyan]펀딩비 차익 봇[/bold cyan]",
                    border_style="red",
                )
                live.update(panel)
            elif last_data:
                table = build_table(last_data)
                footer = Text(f" q 종료  |  다음 갱신: {remaining}초 후 ", style="dim", justify="right")
                from rich.console import Group
                live.update(Group(table, footer))
            else:
                live.update(Panel("[dim]데이터 로딩 중...[/dim]", border_style="cyan"))

            time.sleep(1)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[dim]종료되었습니다.[/dim]")
        sys.exit(0)
