"""Market boundaries plus a per-company evidence ledger shared across analysts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from config import Settings
from contracts import Evidence, Instrument, Tool

TOOLS = {
    name: Tool(name, description, {}) for name, description in {
        "get_prices": "Read the latest completed daily close, date, currency and change.",
        "get_fundamentals": "Read annual statements and valuation inputs with periods.",
        "get_news": "Read dated company headlines and summaries; not full articles.",
        "get_ecosystem": "Read configured relationship hypotheses and related news.",
        "get_dossier": "Read all retained evidence and errors collected for this company.",
    }.items()
}
TOOLS["value_scenario"] = Tool(
    "value_scenario", "Calculate a conditional earnings-multiple valuation from annual EPS. "
    "All growth, multiple and required-return inputs are assumptions, not forecasts.",
    {"annual_growth": {"type": "number", "minimum": -0.5, "maximum": 0.5},
     "exit_pe": {"type": "number", "minimum": 3, "maximum": 60},
     "required_return": {"type": "number", "minimum": 0.01, "maximum": 0.5},
     "assumption_reason": {"type": "string"}})


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def completed_prices(rows: list[dict[str, Any]], instrument: Instrument,
                     now: datetime) -> dict[str, Any]:
    """Drop an in-progress daily bar; use regular close + 20 minute buffer.

    Conservative on early-close days: include that day only after normal close.
    Exchange holidays are represented by absent rows, not invented observations.
    """
    market_now = now.astimezone(ZoneInfo(instrument.timezone))
    cutoff = datetime.combine(market_now.date(), time(instrument.close_hour,
                              instrument.close_minute), tzinfo=market_now.tzinfo)
    cutoff += timedelta(minutes=20)
    eligible = sorted((r for r in rows if finite(r.get("close")) is not None
                       and r["date"] <= market_now.date().isoformat()
                       and (r["date"] < market_now.date().isoformat()
                            or market_now >= cutoff)), key=lambda r: r["date"])
    if not eligible or eligible[-1]["close"] <= 0:
        raise ValueError("No usable completed-session close")
    latest = eligible[-1]
    previous = eligible[-2] if len(eligible) > 1 else None
    change = None
    if previous and finite(latest.get("adjusted")) is not None:
        prior = finite(previous.get("adjusted"))
        if prior and prior > 0:
            change = (latest["adjusted"] / prior - 1) * 100
    return {"close": latest["close"], "session_date": latest["date"],
            "change_pct": change,
            "change_basis": "adjusted close vs previous observed session",
            "price_basis": "unadjusted completed regular-session daily close",
            "currency": instrument.currency,
            "age_days": (market_now.date() - datetime.fromisoformat(latest["date"]).date()).days}


def normalize_news(items: list[dict[str, Any]], now: datetime,
                   days: int, limit: int) -> list[dict[str, Any]]:
    result, seen = [], set()
    for item in items:
        raw = item.get("content") or item
        title = raw.get("title")
        url = (raw.get("canonicalUrl") or {}).get("url") or raw.get("link")
        published = raw.get("pubDate") or raw.get("providerPublishTime")
        try:
            stamp = (datetime.fromtimestamp(published, timezone.utc)
                     if isinstance(published, (float, int))
                     else datetime.fromisoformat(str(published).replace("Z", "+00:00")))
            if stamp.tzinfo is None:
                continue
        except (ValueError, TypeError, OverflowError, OSError):
            continue
        if not title or not url or not str(url).startswith("https://"):
            continue
        if not now - timedelta(days=days) <= stamp <= now:
            continue
        identity = str(url).split("?")[0]
        if identity in seen:
            continue
        seen.add(identity)
        provider = raw.get("provider") or {}
        result.append({"title": str(title)[:350], "url": url,
                       "published_at": stamp.isoformat(),
                       "publisher": provider.get("displayName") or raw.get("publisher"),
                       "summary": str(raw.get("summary") or "")[:900],
                       "coverage": "headline/summary only; article not retrieved"})
    return sorted(result, key=lambda n: n["published_at"], reverse=True)[:limit]


class YahooSource:
    """Lazy yfinance access, fixed ticker/query allowlist supplied by config."""

    label = "yfinance / Yahoo Finance"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: dict[str, Any] = {}

    def company(self, ticker: str):
        import yfinance as yf
        if ticker not in self.cache:
            self.cache[ticker] = yf.Ticker(ticker)
        return self.cache[ticker]

    def fetch(self, kind: str, item: Instrument, now: datetime) -> dict[str, Any]:
        import yfinance as yf
        if kind == "news":
            raw = (self.company(item.ticker).news if item.ticker else
                   yf.Search(item.topics[0], max_results=0,
                             news_count=self.settings.max_news, timeout=20).news)
            return {"articles": normalize_news(raw or [], now, self.settings.news_days,
                                               self.settings.max_news)}
        if kind == "ecosystem":
            raw, errors = [], []
            # Bound expansion: two configured topics + at most two related tickers.
            for query in list(item.topics[:2]) + list(item.related[:2]):
                try:
                    raw.extend(yf.Search(query, max_results=0, news_count=3,
                                         timeout=20).news or [])
                except Exception as exc:
                    errors.append(f"Query unavailable: {query} ({type(exc).__name__})")
            return {"research_leads": list(item.related),
                    "relationship_status": "hypotheses; not verified supplier contracts",
                    "articles": normalize_news(raw, now, self.settings.news_days,
                                               self.settings.max_news),
                    "query_errors": errors}
        if not item.ticker:
            raise ValueError("No direct listed instrument configured")
        company = self.company(item.ticker)
        info = company.get_info()
        name = str(info.get("longName") or info.get("shortName") or "").lower()
        if (info.get("quoteType") != "EQUITY" or info.get("symbol") != item.ticker
                or not any(term in name for term in item.identity_terms)):
            raise ValueError("Ticker identity did not match expected company equity")
        if info.get("currency") != item.currency:
            raise ValueError("Unexpected or missing trading currency")
        if kind == "prices":
            history = company.history(period="1mo", interval="1d", auto_adjust=False,
                                      timeout=20, raise_errors=True)
            rows = [{"date": index.date().isoformat(), "close": finite(row["Close"]),
                     "adjusted": finite(row.get("Adj Close"))}
                    for index, row in history.iterrows()]
            data = completed_prices(rows, item, now)
            data["resolved_name"] = info.get("longName") or info.get("shortName")
            data["identity_verified"] = True
            data["source_url"] = f"https://finance.yahoo.com/quote/{item.ticker}/history/"
            return data
        income, cash = company.income_stmt, company.cashflow
        if income is None or income.empty:
            raise ValueError("No annual income statement available")
        period = sorted(income.columns, reverse=True)[0]

        def row_value(frame: Any, names: list[str]) -> float | None:
            if frame is not None and not frame.empty and period in frame.columns:
                for label in names:
                    if label in frame.index:
                        return finite(frame.loc[label, period])
            return None

        financial_currency = info.get("financialCurrency")
        return {"period_end": period.date().isoformat(), "period_type": "annual",
                "currency": financial_currency,
                "revenue": row_value(income, ["Total Revenue"]),
                "operating_income": row_value(income, ["Operating Income"]),
                "net_income": row_value(income, ["Net Income"]),
                "diluted_eps": row_value(income, ["Diluted EPS"]),
                "operating_cash_flow": row_value(cash, ["Operating Cash Flow"]),
                "capital_expenditure": row_value(cash, ["Capital Expenditure"]),
                "source_url": f"https://finance.yahoo.com/quote/{item.ticker}/financials/",
                "source_grade": "aggregator; primary filing not retrieved"}


class EvidenceBook:
    def __init__(self, item: Instrument, source: Any, settings: Settings, now: datetime):
        self.item, self.source, self.settings, self.now = item, source, settings, now
        self.records: dict[str, Evidence] = {}
        self.tool_cache: dict[str, Evidence] = {}

    def collect(self, kind: str) -> Evidence:
        if kind in self.tool_cache:
            return self.tool_cache[kind]
        stamp = self.now.isoformat()
        try:
            if kind in {"prices", "fundamentals"} and not self.item.ticker:
                raise ValueError("No direct listed instrument configured")
            data = self.source.fetch(kind, self.item, self.now)
            json.dumps(data, allow_nan=False)
            as_of = data.get("session_date") or data.get("period_end")
            warnings = []
            if kind in {"news", "ecosystem"}:
                warnings.append("Headlines/summaries only; relevance and causality require review.")
            if kind == "fundamentals":
                warnings.append("Annual aggregator data; verify critical facts against original filings.")
            if kind == "prices" and data.get("change_pct") is None:
                warnings.append("Comparable adjusted daily change unavailable.")
            evidence = Evidence(f"{self.item.key}:{kind}", kind, self.item.key,
                                self.source.label, stamp, as_of, data, warnings)
        except Exception as exc:
            # Raw exception strings may contain request headers/keys; do not log them.
            evidence = Evidence(f"{self.item.key}:{kind}", kind, self.item.key,
                                self.source.label, stamp, None, {},
                                error=f"Data unavailable ({type(exc).__name__})")
        self.tool_cache[kind] = evidence
        self.records[evidence.id] = evidence
        return evidence

    def execute(self, name: str, arguments: Any, allowed: set[str]) -> dict[str, Any]:
        if name not in allowed or name not in TOOLS:
            return {"error": "Unknown or disallowed tool"}
        properties = TOOLS[name].properties
        if not isinstance(arguments, dict) or set(arguments) != set(properties):
            return {"error": "Arguments must exactly match the tool schema"}
        if name == "get_dossier":
            return {"evidence": [e.payload() for e in self.records.values()]}
        if name == "value_scenario":
            return self.value_scenario(arguments)
        kind = {"get_prices": "prices", "get_fundamentals": "fundamentals",
                "get_news": "news", "get_ecosystem": "ecosystem"}[name]
        return {"evidence": [self.collect(kind).payload()]}

    def value_scenario(self, arguments: dict[str, Any]) -> dict[str, Any]:
        for key, low, high in (("annual_growth", -0.5, 0.5), ("exit_pe", 3, 60),
                               ("required_return", 0.01, 0.5)):
            value = arguments[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return {"error": f"{key} must be numeric"}
            if not math.isfinite(value) or not low <= value <= high:
                return {"error": f"{key} is outside this teaching tool's bounds"}
        reason = arguments["assumption_reason"]
        if not isinstance(reason, str) or not 20 <= len(reason) <= 1500:
            return {"error": "Give a 20-1500 character explanation of assumptions"}
        fundamentals = self.collect("fundamentals")
        data = fundamentals.data
        eps = finite(data.get("diluted_eps"))
        if (fundamentals.error or eps is None or eps <= 0
                or data.get("currency") != self.item.currency):
            return {"error": "Positive annual EPS in trading currency is required; no FX assumed"}
        years = self.settings.horizon_years
        future = eps * (1 + arguments["annual_growth"]) ** years * arguments["exit_pe"]
        present = future / (1 + arguments["required_return"]) ** years
        suffix = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()[:10]
        evidence = Evidence(f"{self.item.key}:valuation:{suffix}", "valuation",
                            self.item.key, "Python conditional calculation",
                            self.now.isoformat(), fundamentals.as_of,
                            {"inputs": arguments, "input_evidence_id": fundamentals.id,
                             "annual_eps": eps, "horizon_years": years,
                             "future_price": round(future, 4),
                             "present_value": round(present, 4),
                             "currency": self.item.currency,
                             "method": "EPS*(1+growth)^years*exit_PE/(1+required_return)^years",
                             "assumption_status": "model assumptions, not verified forecasts"},
                            ["Simplified earnings multiple; no dividends, FX, or explicit dilution model.",
                             "Not a universal method: cyclical/negative-earnings firms may require abstention."])
        self.records[evidence.id] = evidence
        return {"evidence": [fundamentals.payload(), evidence.payload()]}
