"""
brief.py - a daily morning brief for your own watchlist.

Reads watchlist.csv, pulls a year of daily prices from Yahoo Finance
(yfinance) and builds a short brief that:

- highlights only what is unusual today, judged against each stock's own
  typical daily move rather than one fixed percentage;
- shows exchange rates, and each stock's 1-year return converted into NZD;
- teaches one concept per day using that day's real numbers;
- repeats the "why" and "wrong if" you wrote for any flagged stock.

    python brief.py           # print the brief and save an HTML copy
    python brief.py --send    # also email it through Gmail (see .env.example)

A learning aid, not financial advice. It reads prices; it never trades.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import smtplib
import ssl
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from email.message import EmailMessage
from html import escape
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
WATCHLIST_PATH = ROOT / "watchlist.csv"
BRIEF_DIR = ROOT / "reports" / "briefs"

HOME_CURRENCY = "NZD"
BENCHMARK = "VOO"  # S&P 500 ETF: the "could I have just bought the index?" yardstick
FX_PAIRS = ["NZDUSD=X", "NZDKRW=X", "USDKRW=X"]

UNUSUAL_MOVE_MULTIPLE = 2.0  # a day bigger than 2x the stock's typical daily move gets flagged
EARNINGS_WINDOW_DAYS = 7
TRADING_DAYS_PER_MONTH = 21

CURRENCY_SYMBOLS = {"USD": "$", "KRW": "₩", "NZD": "NZ$"}
MARKET_NAMES = {"USD": "US", "KRW": "Korea", "NZD": "NZ"}
NOT_WRITTEN = "(not written yet - add it in watchlist.csv)"


@dataclass
class StockLine:
    ticker: str
    name: str
    why: str = ""
    wrong_if: str = ""
    currency: str | None = None
    price: float | None = None
    asof: date | None = None
    day_pct: float | None = None
    month_pct: float | None = None
    year_pct: float | None = None
    year_pct_home: float | None = None
    from_high_pct: float | None = None
    from_low_pct: float | None = None
    typical_move_pct: float | None = None
    next_earnings: date | None = None
    flags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class FxLine:
    pair: str
    rate: float | None = None
    day_pct: float | None = None
    year_pct: float | None = None
    error: str | None = None

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:6]


@dataclass
class Brief:
    today: date
    lines: list[StockLine]
    fx: list[FxLine]
    concept: tuple[str, str] | None

    @property
    def flagged(self) -> list[StockLine]:
        return [line for line in self.lines if line.flags or line.error]


# --- Watchlist -------------------------------------------------------------


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[StockLine]:
    """Read watchlist.csv, skipping '#' comment lines and rows with no ticker."""
    with path.open(encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel adds a BOM
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        return [
            StockLine(
                ticker=row["ticker"].strip().upper(),
                name=(row.get("name") or row["ticker"]).strip(),
                why=(row.get("why") or "").strip(),
                wrong_if=(row.get("wrong_if") or "").strip(),
            )
            for row in rows
            if (row.get("ticker") or "").strip()
        ]


# --- Calculations (pure, no network) ----------------------------------------


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or pd.isna(new) or pd.isna(old) or old == 0:
        return None
    return (float(new) / float(old) - 1) * 100


def describe_closes(closes: pd.Series) -> dict:
    """Turn about a year of daily closes into the few numbers the brief uses."""
    closes = closes.dropna()
    if len(closes) < 2:
        return {}
    last = float(closes.iloc[-1])
    daily_moves = closes.pct_change().dropna() * 100
    month_ago = closes.iloc[-(TRADING_DAYS_PER_MONTH + 1)] if len(closes) > TRADING_DAYS_PER_MONTH else closes.iloc[0]
    return {
        "price": last,
        "asof": closes.index[-1].date(),
        "day_pct": pct_change(last, closes.iloc[-2]),
        "month_pct": pct_change(last, month_ago),
        "year_pct": pct_change(last, closes.iloc[0]),
        "from_high_pct": pct_change(last, closes.max()),
        "from_low_pct": pct_change(last, closes.min()),
        # Standard deviation of daily % moves: how big an ordinary day is for this stock.
        "typical_move_pct": float(daily_moves.std()) if len(daily_moves) >= 20 else None,
    }


def to_home_return(local_pct: float | None, fx_start: float | None, fx_end: float | None) -> float | None:
    """Convert a return in the stock's currency into an NZD return.

    fx is units of the stock's currency per 1 NZD (NZDUSD = 0.56 means 1 NZD buys
    0.56 USD). A weaker NZD (fx falls) makes the same foreign stock worth more NZD.
    """
    if local_pct is None or not fx_start or not fx_end:
        return None
    return ((1 + local_pct / 100) * (fx_start / fx_end) - 1) * 100


def find_flags(line: StockLine, today: date) -> list[str]:
    """Only events worth a look today; a quiet stock gets no flags."""
    flags = []
    if line.day_pct is not None and line.typical_move_pct:
        if abs(line.day_pct) >= UNUSUAL_MOVE_MULTIPLE * line.typical_move_pct:
            flags.append(f"big day: {line.day_pct:+.1f}% against a typical ±{line.typical_move_pct:.1f}%")
    if line.from_high_pct is not None and line.from_high_pct >= 0:
        flags.append("closed at a new 1-year high")
    if line.from_low_pct is not None and line.from_low_pct <= 0:
        flags.append("closed at a new 1-year low")
    if line.next_earnings is not None:
        days = (line.next_earnings - today).days
        if 0 <= days <= EARNINGS_WINDOW_DAYS:
            when = "today" if days == 0 else f"in {days} day{'s' if days > 1 else ''}"
            flags.append(f"earnings {when} ({line.next_earnings:%a %d %b} in the market's time zone; may be an estimate)")
    return flags


# --- Concept of the day -----------------------------------------------------
# Each returns (title, text) built from today's numbers, or None if today's
# data can't illustrate it. The brief rotates through them by date.


def _priced(lines: list[StockLine]) -> list[StockLine]:
    return [line for line in lines if line.error is None and line.price is not None]


def concept_typical_move(lines: list[StockLine]) -> tuple[str, str] | None:
    ok = [line for line in _priced(lines) if line.typical_move_pct]
    if len(ok) < 2:
        return None
    wild = max(ok, key=lambda line: line.typical_move_pct)
    calm = min(ok, key=lambda line: line.typical_move_pct)
    return (
        "Typical daily move (volatility)",
        f"{wild.name} usually moves about ±{wild.typical_move_pct:.1f}% a day; "
        f"{calm.name} about ±{calm.typical_move_pct:.1f}%. A 3% day is ordinary for "
        f"{wild.name} and big news for {calm.name}, so this brief judges each stock "
        "against its own normal.",
    )


def concept_drawdown(lines: list[StockLine]) -> tuple[str, str] | None:
    ok = [line for line in _priced(lines) if line.from_high_pct is not None and line.year_pct is not None]
    if not ok:
        return None
    deepest = min(ok, key=lambda line: line.from_high_pct)
    if deepest.from_high_pct > -10:
        return None
    return (
        "Distance from the high (drawdown)",
        f"{deepest.name} is {abs(deepest.from_high_pct):.0f}% below its 1-year high, while its "
        f"1-year change is {deepest.year_pct:+.0f}%. A fall from the high describes the price "
        "path. It doesn't tell you whether the business is now worth more or less.",
    )


def concept_currency(lines: list[StockLine]) -> tuple[str, str] | None:
    ok = [
        line for line in _priced(lines)
        if line.year_pct_home is not None and line.year_pct is not None and line.currency != HOME_CURRENCY
    ]
    if not ok:
        return None
    biggest = max(ok, key=lambda line: abs(line.year_pct_home - line.year_pct))
    if abs(biggest.year_pct_home - biggest.year_pct) < 1:
        return None
    return (
        "Currency is a second bet",
        f"Over the past year {biggest.name} returned {biggest.year_pct:+.0f}% in {biggest.currency} "
        f"but about {biggest.year_pct_home:+.0f}% in NZD. Buying a foreign stock also means "
        "holding its currency, so exchange rates add to or subtract from what you earn.",
    )


def concept_benchmark(lines: list[StockLine]) -> tuple[str, str] | None:
    bench = next((line for line in _priced(lines) if line.ticker == BENCHMARK), None)
    others = [line for line in _priced(lines) if line.ticker != BENCHMARK and line.year_pct is not None]
    if bench is None or bench.year_pct is None or not others:
        return None
    beat = [line.name for line in others if line.year_pct > bench.year_pct]
    names = ", ".join(beat) if beat else "none of them"
    return (
        "Keeping score against the index",
        f"{len(beat)} of {len(others)} stocks on your list beat the S&P 500 ({bench.year_pct:+.0f}%) "
        f"over the past year: {names}. Professionals judge a stock pick against what an index "
        "fund would have earned.",
    )


def concept_trend_split(lines: list[StockLine]) -> tuple[str, str] | None:
    ok = [
        line for line in _priced(lines)
        if line.month_pct is not None and line.year_pct is not None and line.month_pct * line.year_pct < 0
    ]
    if not ok:
        return None
    split = max(ok, key=lambda line: abs(line.month_pct - line.year_pct))
    return (
        "Short trend against long trend",
        f"{split.name} is {split.month_pct:+.0f}% over the past month but {split.year_pct:+.0f}% "
        "over the year. Short and long trends often disagree. Your reason for owning a stock "
        "should say which one matters to you.",
    )


CONCEPTS: list[Callable[[list[StockLine]], tuple[str, str] | None]] = [
    concept_typical_move,
    concept_drawdown,
    concept_currency,
    concept_benchmark,
    concept_trend_split,
]


def pick_concept(lines: list[StockLine], today: date) -> tuple[str, str] | None:
    """Rotate by date, skipping any concept today's numbers can't illustrate."""
    start = today.toordinal() % len(CONCEPTS)
    for offset in range(len(CONCEPTS)):
        concept = CONCEPTS[(start + offset) % len(CONCEPTS)](lines)
        if concept:
            return concept
    return None


# --- Market data (network) -------------------------------------------------


def fetch_closes(symbol: str, cache: dict[str, pd.Series]) -> pd.Series:
    """One year of dividend- and split-adjusted daily closes, fetched once per run."""
    if symbol not in cache:
        cache[symbol] = yf.Ticker(symbol).history(period="1y", auto_adjust=True)["Close"].dropna()
    return cache[symbol]


def _currency(ticker: yf.Ticker) -> str | None:
    try:
        value = ticker.fast_info["currency"]
        return str(value).upper() if value else None
    except Exception:
        return None


def _next_earnings(ticker: yf.Ticker) -> date | None:
    try:
        calendar = ticker.calendar
        dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
        return min(dates) if dates else None
    except Exception:  # ETFs have no earnings, and the field is scraped
        return None


def fill_from_market(line: StockLine, cache: dict[str, pd.Series], today: date) -> None:
    """Fill one watchlist line in place. Never raises; failures land in line.error."""
    try:
        ticker = yf.Ticker(line.ticker)
        stats = describe_closes(fetch_closes(line.ticker, cache))
        if not stats:
            line.error = "no recent prices returned - check the ticker"
            return
        for key, value in stats.items():
            setattr(line, key, value)
        line.currency = _currency(ticker)
        line.next_earnings = _next_earnings(ticker)
    except Exception as exc:  # yfinance is a flaky network boundary
        line.error = f"{type(exc).__name__}: {exc}"
        return

    if line.currency == HOME_CURRENCY:
        line.year_pct_home = line.year_pct
    elif line.currency:
        try:
            fx = fetch_closes(f"{HOME_CURRENCY}{line.currency}=X", cache)
            if len(fx) >= 2:
                line.year_pct_home = to_home_return(line.year_pct, float(fx.iloc[0]), float(fx.iloc[-1]))
        except Exception:
            pass  # leave the NZD column empty rather than guess
    line.flags = find_flags(line, today)


def fetch_fx(pair: str, cache: dict[str, pd.Series]) -> FxLine:
    try:
        stats = describe_closes(fetch_closes(pair, cache))
    except Exception as exc:
        return FxLine(pair, error=f"{type(exc).__name__}: {exc}")
    if not stats:
        return FxLine(pair, error="no rates returned")
    return FxLine(pair, rate=stats["price"], day_pct=stats["day_pct"], year_pct=stats["year_pct"])


def build_brief(lines: list[StockLine], today: date) -> Brief:
    cache: dict[str, pd.Series] = {}
    fx = [fetch_fx(pair, cache) for pair in FX_PAIRS]
    for line in lines:
        fill_from_market(line, cache, today)
    return Brief(today=today, lines=lines, fx=fx, concept=pick_concept(lines, today))


# --- Rendering ---------------------------------------------------------------


def fmt_price(value: float | None, currency: str | None) -> str:
    if value is None:
        return "-"
    digits = 0 if currency == "KRW" else 2
    number = f"{value:,.{digits}f}"
    symbol = CURRENCY_SYMBOLS.get(currency or "")
    return f"{symbol}{number}" if symbol else f"{number} {currency or ''}".strip()


def fmt_pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:+.{digits}f}%"


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.4f}" if value < 10 else f"{value:,.2f}"


def asof_summary(lines: list[StockLine]) -> str:
    """'US close 28 Oct · Korea close 29 Oct': markets close on different days."""
    latest: dict[str, date] = {}
    for line in lines:
        if line.asof and line.currency:
            latest[line.currency] = max(latest.get(line.currency, line.asof), line.asof)
    parts = [f"{MARKET_NAMES.get(cur, cur)} close {day:%d %b}" for cur, day in sorted(latest.items(), key=lambda kv: kv[0] != "USD")]
    return " · ".join(parts) or "no prices returned"


def fx_hint(fx: FxLine) -> str:
    """Plain-English meaning of the 1-year move for an NZD-based investor."""
    if fx.base != HOME_CURRENCY or fx.year_pct is None or abs(fx.year_pct) < 0.5:
        return ""
    if fx.year_pct < 0:
        return f"NZD weaker this year: your {fx.quote} stocks are worth more in NZD"
    return f"NZD stronger this year: your {fx.quote} stocks are worth less in NZD"


def subject_line(brief: Brief) -> str:
    count = len(brief.flagged)
    status = "nothing unusual" if count == 0 else f"{count} to look at"
    return f"Stock brief {brief.today:%a %d %b}: {status}"


def render_text(brief: Brief) -> str:
    out = [subject_line(brief), f"Prices as of: {asof_summary(brief.lines)}", ""]

    out.append("LOOK AT TODAY")
    if not brief.flagged:
        out.append("  Nothing unusual. Quiet days are normal; no action needed.")
    for line in brief.flagged:
        out.append(f"  {line.name} ({line.ticker})  {fmt_price(line.price, line.currency)}  {fmt_pct(line.day_pct)}")
        if line.error:
            out.append(f"    - data problem: {line.error}")
        out.extend(f"    - {flag}" for flag in line.flags)
        out.append(f"    Why you watch it:  {line.why or NOT_WRITTEN}")
        out.append(f"    You'd be wrong if: {line.wrong_if or NOT_WRITTEN}")
    out.append("")

    out.append("ALL STOCKS")
    out.append(f"  {'Name':<24}{'Price':>13}{'Day':>8}{'1M':>8}{'1Y':>8}{'1Y NZD':>9}{'From high':>11}")
    for line in brief.lines:
        if line.error:
            out.append(f"  {line.name[:23]:<24}  data problem: {line.error}")
            continue
        out.append(
            f"  {line.name[:23]:<24}{fmt_price(line.price, line.currency):>13}{fmt_pct(line.day_pct):>8}"
            f"{fmt_pct(line.month_pct, 0):>8}{fmt_pct(line.year_pct, 0):>8}{fmt_pct(line.year_pct_home, 0):>9}"
            f"{fmt_pct(line.from_high_pct, 0):>11}"
        )
    out.append("")

    out.append("EXCHANGE RATES")
    for fx in brief.fx:
        if fx.error:
            out.append(f"  {fx.base}/{fx.quote}: data problem: {fx.error}")
            continue
        hint = fx_hint(fx)
        out.append(
            f"  1 {fx.base} = {fmt_rate(fx.rate)} {fx.quote}   day {fmt_pct(fx.day_pct, 2)}   "
            f"1 year {fmt_pct(fx.year_pct, 1)}" + (f"   ({hint})" if hint else "")
        )
    out.append("")

    if brief.concept:
        title, text = brief.concept
        out += [f"CONCEPT OF THE DAY: {title}", f"  {text}", ""]

    missing = sum(1 for line in brief.lines if not line.why)
    if missing:
        out += [f'YOUR RULES: {missing} of {len(brief.lines)} stocks have no "why" yet. Add them in watchlist.csv.', ""]

    out.append(
        "Learning aid, not financial advice. Data: Yahoo Finance via yfinance. Returns use "
        "dividend-adjusted prices; NZD returns are approximate; earnings dates may be estimates."
    )
    return "\n".join(out)


def _td(text: str, align: str = "right", extra: str = "") -> str:
    return f'<td style="padding:6px 10px;border-bottom:1px solid #e3e7ee;text-align:{align};{extra}">{text}</td>'


def _pct_cell(value: float | None, digits: int = 1) -> str:
    color = "#5b6472" if value is None else ("#11744a" if value >= 0 else "#b3302b")
    return _td(escape(fmt_pct(value, digits)), extra=f"color:{color};font-variant-numeric:tabular-nums;")


def render_html(brief: Brief) -> str:
    """Email-safe HTML: inline styles only, because mail apps drop <style> blocks."""
    h2 = 'style="font-size:15px;margin:22px 0 8px;color:#121720;letter-spacing:.02em;text-transform:uppercase"'
    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#121720;'
        'max-width:720px;margin:0 auto;padding:8px 4px;line-height:1.5">',
        f'<h1 style="font-size:21px;margin:0 0 4px">{escape(subject_line(brief))}</h1>',
        f'<p style="margin:0;color:#5b6472;font-size:13px">Prices as of: {escape(asof_summary(brief.lines))}</p>',
        f"<h2 {h2}>Look at today</h2>",
    ]
    if not brief.flagged:
        parts.append("<p style='margin:0'>Nothing unusual. Quiet days are normal; no action needed.</p>")
    for line in brief.flagged:
        bullets = ([f"data problem: {line.error}"] if line.error else []) + line.flags
        parts.append(
            '<div style="border-left:3px solid #1d5c8f;padding:4px 0 4px 12px;margin:0 0 12px">'
            f"<b>{escape(line.name)}</b> <span style='color:#5b6472'>{escape(line.ticker)}</span> "
            f"&nbsp;{escape(fmt_price(line.price, line.currency))} &nbsp;{escape(fmt_pct(line.day_pct))}"
            + "".join(f"<div>• {escape(b)}</div>" for b in bullets)
            + f"<div style='color:#5b6472;font-size:13px;margin-top:4px'>Why you watch it: {escape(line.why or NOT_WRITTEN)}<br>"
            f"You'd be wrong if: {escape(line.wrong_if or NOT_WRITTEN)}</div></div>"
        )

    head = "".join(
        f'<th style="padding:6px 10px;text-align:{a};font-size:11px;color:#5b6472;text-transform:uppercase;'
        f'border-bottom:1px solid #c9d0db">{t}</th>'
        for t, a in [("Stock", "left"), ("Price", "right"), ("Day", "right"), ("1M", "right"),
                     ("1Y", "right"), ("1Y in NZD", "right"), ("From high", "right")]
    )
    rows = []
    for line in brief.lines:
        name = f"{escape(line.name)}<br><span style='color:#5b6472;font-size:12px'>{escape(line.ticker)}</span>"
        if line.error:
            rows.append(f"<tr>{_td(name, 'left')}<td colspan='6' style='padding:6px 10px;color:#b3302b'>"
                        f"data problem: {escape(line.error)}</td></tr>")
            continue
        rows.append(
            f"<tr>{_td(name, 'left')}{_td(escape(fmt_price(line.price, line.currency)))}"
            f"{_pct_cell(line.day_pct)}{_pct_cell(line.month_pct, 0)}{_pct_cell(line.year_pct, 0)}"
            f"{_pct_cell(line.year_pct_home, 0)}{_pct_cell(line.from_high_pct, 0)}</tr>"
        )
    parts += [
        f"<h2 {h2}>All stocks</h2>",
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-size:14px">',
        f"<tr>{head}</tr>", *rows, "</table></div>",
        f"<h2 {h2}>Exchange rates</h2>",
    ]
    for fx in brief.fx:
        if fx.error:
            parts.append(f"<div>{fx.base}/{fx.quote}: data problem: {escape(fx.error)}</div>")
            continue
        hint = fx_hint(fx)
        parts.append(
            f"<div>1 {fx.base} = <b>{fmt_rate(fx.rate)} {fx.quote}</b> &nbsp;"
            f"<span style='color:#5b6472'>day {fmt_pct(fx.day_pct, 2)} · 1 year {fmt_pct(fx.year_pct, 1)}</span>"
            + (f"<div style='color:#5b6472;font-size:13px'>{escape(hint)}</div>" if hint else "")
            + "</div>"
        )
    if brief.concept:
        title, text = brief.concept
        parts += [
            f"<h2 {h2}>Concept of the day</h2>",
            f'<div style="background:#e4eef7;padding:12px 14px;border-radius:8px"><b>{escape(title)}</b>'
            f"<div>{escape(text)}</div></div>",
        ]
    missing = sum(1 for line in brief.lines if not line.why)
    if missing:
        parts.append(
            f"<p style='margin:18px 0 0'><b>Your rules:</b> {missing} of {len(brief.lines)} stocks have "
            "no &ldquo;why&rdquo; yet. Add them in watchlist.csv.</p>"
        )
    parts.append(
        "<p style='margin:22px 0 0;color:#5b6472;font-size:12px'>Learning aid, not financial advice. "
        "Data: Yahoo Finance via yfinance. Returns use dividend-adjusted prices; NZD returns are "
        "approximate; earnings dates may be estimates.</p></div>"
    )
    return "\n".join(parts)


# --- Email & CLI ------------------------------------------------------------


def send_email(subject: str, text: str, html: str, *, sender: str, password: str, to: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ₩ and ± on a Windows console
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # e.g. "no fundamentals" 404s for ETFs

    parser = argparse.ArgumentParser(description="Build the daily watchlist brief.")
    parser.add_argument("--send", action="store_true", help="also email the brief through Gmail")
    parser.add_argument("--watchlist", type=Path, default=WATCHLIST_PATH, help="path to watchlist.csv")
    args = parser.parse_args(argv)

    # Check email settings before spending a minute downloading prices.
    if args.send:
        missing = [key for key in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD") if not os.getenv(key)]
        if missing:
            print(f"--send needs {', '.join(missing)} in .env (see .env.example).", file=sys.stderr)
            return 2

    today = date.today()
    brief = build_brief(load_watchlist(args.watchlist), today)
    text, html = render_text(brief), render_html(brief)
    print(text)

    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    saved = BRIEF_DIR / f"{today.isoformat()}.html"
    saved.write_text(html, encoding="utf-8")
    print(f"\nSaved {saved}")

    if args.send:
        sender = os.environ["GMAIL_ADDRESS"]
        to = os.getenv("BRIEF_TO") or sender
        try:
            send_email(subject_line(brief), text, html, sender=sender, password=os.environ["GMAIL_APP_PASSWORD"], to=to)
        except (smtplib.SMTPException, OSError) as exc:
            print(f"Email failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"Emailed to {to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
