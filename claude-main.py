"""
claude-main.py - a small, single-file starting point.

Step 1 toward the "AI stock watch" idea from CLAUDE.md / AGENTS.md: pull a
quick price snapshot for a handful of tickers with yfinance, print it as a
table, then ask Claude for a short, plain-English read on what the numbers
show. This is research commentary, not financial advice.

This intentionally skips the fuller architecture AGENTS.md describes for the
eventual rebuild (config.py/llm.py/data_tools.py/agents.py/report.py/main.py)
- it's one file to read end-to-end and build on incrementally.
"""

import os
import sys

import pandas as pd  # yfinance가 내부적으로 pandas를 쓰기 때문에 이미 설치돼 있음
import yfinance as yf
from anthropic import Anthropic
from dotenv import load_dotenv

# Same default roster AGENTS.md settled on for the eventual CLI.
# Yahoo Finance tickers: KRX-listed stocks use a ".KS" suffix.
DEFAULT_WATCHLIST = ["005930.KS", "066570.KS", "GOOGL", "META", "MSFT", "NVDA"]

CLAUDE_MODEL = "claude-opus-5"


def get_price_history(ticker: str, period: str = "3mo") -> pd.DataFrame | None:
    """최근 N개월치 일별 시세(시가/고가/저가/종가/거래량)를 가져온다.

    RSI 같은 지표는 하루짜리 스냅샷(fast_info)만으로는 계산할 수 없고
    최근 며칠~몇 주치 종가의 흐름이 있어야 계산된다.
    """
    try:
        history = yf.Ticker(ticker).history(period=period)
        if history.empty:
            return None
        return history
    except Exception:
        return None


def calculate_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """RSI(상대강도지수, Relative Strength Index)를 계산한다.

    - 하루 전 대비 가격 변화(delta)를 오른 날(gain)과 내린 날(loss)로 나눈다.
    - 최근 `period`일 동안의 평균 상승폭(avg_gain), 평균 하락폭(avg_loss)을 구한다.
    - RS(상대강도) = avg_gain / avg_loss
    - RSI = 100 - (100 / (1 + RS))  →  0~100 사이 값

    관습적으로 70 이상이면 "과매수", 30 이하면 "과매도"라고 부르지만,
    이건 절대적인 법칙이 아니라 경험적으로 통용되는 기준선일 뿐이다.
    """
    delta = closes.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_snapshot(ticker: str) -> dict:
    """Pull a small, JSON-friendly price snapshot for one ticker.

    yfinance calls are flaky (network hiccups, rate limits, delisted
    tickers), so this is wrapped in try/except - callers can trust it
    never raises, and just check for an "error" key instead.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        price = info["last_price"]
        prev_close = info["previous_close"]
        change_pct = (price - prev_close) / prev_close * 100

        snapshot = {
            "ticker": ticker,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(info["day_high"], 2),
            "day_low": round(info["day_low"], 2),
            "year_high": round(info["year_high"], 2),
            "year_low": round(info["year_low"], 2),
            "volume": int(info["last_volume"]),
        }

        # RSI(14일 기준)는 최소 15일치 데이터가 있어야 의미있는 값이 나온다.
        history = get_price_history(ticker)
        if history is not None and len(history) >= 15:
            latest_rsi = calculate_rsi(history["Close"]).iloc[-1]
            if pd.notna(latest_rsi):
                snapshot["rsi"] = round(latest_rsi, 1)

        return snapshot
    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def print_table(snapshots: list[dict]) -> None:
    """Print a simple fixed-width table - a preview of the daily-email format."""
    header = f"{'Ticker':<10}{'Price':>10}{'Chg %':>9}{'RSI':>7}{'Day Range':>22}{'52w Range':>22}"
    print(header)
    print("-" * len(header))
    for s in snapshots:
        if "error" in s:
            print(f"{s['ticker']:<10}error: {s['error']}")
            continue
        day_range = f"{s['day_low']}-{s['day_high']}"
        year_range = f"{s['year_low']}-{s['year_high']}"
        rsi_display = s.get("rsi", "-")  # 데이터가 부족하면 rsi 키 자체가 없을 수 있음
        print(
            f"{s['ticker']:<10}{s['price']:>10}{s['change_pct']:>+8.2f}%"
            f"{rsi_display:>7}{day_range:>22}{year_range:>22}"
        )


def ask_claude_for_commentary(snapshots: list[dict]) -> str:
    """Send the day's numbers to Claude and get back a short, grounded read."""
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    usable = [s for s in snapshots if "error" not in s]
    if not usable:
        return "No usable price data to comment on."

    system_prompt = (
        "You are a research assistant reading a table of stock price "
        "snapshots. For each ticker, give 1-2 plain-English sentences on "
        "what today's numbers show (e.g. proximity to the 52-week high/low, "
        "size of the daily move). Do not invent numbers that aren't in the "
        "data. This is research commentary, not financial advice - do not "
        "issue a buy/hold/sell call from a single day's price snapshot."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": str(usable)}],
    )

    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    load_dotenv()
    tickers = sys.argv[1:] or DEFAULT_WATCHLIST

    snapshots = [get_snapshot(t) for t in tickers]
    print_table(snapshots)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nSet ANTHROPIC_API_KEY in a .env file to also get Claude's commentary.")
        return

    print("\nClaude's read on today's numbers:\n")
    print(ask_claude_for_commentary(snapshots))


if __name__ == "__main__":
    main()
