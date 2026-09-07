"""FICTIONAL data + scripted models, routed through the real loop and gate.

This demonstrates software behavior. No prices, news or stances are real research.
"""

import json
from datetime import timedelta

from contracts import Call, Turn
from data_tools import completed_prices


class DemoSource:
    label = "FICTIONAL DEMO DATA"

    def fetch(self, kind, item, now):
        price = 60000.0 if item.currency == "KRW" else 100.0
        if kind == "prices":
            rows = [{"date": (now.date() - timedelta(days=days)).isoformat(),
                     "close": price, "adjusted": price}
                    for days in range(10)
                    if (now.date() - timedelta(days=days)).weekday() < 5]
            data = completed_prices(rows, item, now)
            return {**data, "identity_verified": True,
                    "resolved_name": "FICTIONAL " + item.name,
                    "price_basis": "fictional completed daily close",
                    "change_basis": "fictional adjusted daily change"}
        if kind == "fundamentals":
            return {"period_end": f"{now.year - 1}-12-31", "period_type": "annual",
                    "currency": item.currency, "revenue": 1000000.0,
                    "operating_income": 200000.0, "net_income": 150000.0,
                    "diluted_eps": price / 20, "operating_cash_flow": 180000.0,
                    "capital_expenditure": -40000.0,
                    "source_grade": "invented teaching fixture"}
        article = {"title": "FICTIONAL: Example demand and capacity update",
                   "summary": "Invented news to demonstrate the report format.",
                   "url": "https://example.com/fictional-news",
                   "published_at": now.isoformat(), "publisher": "Demo only",
                   "coverage": "fictional summary"}
        return {"articles": [article], "research_leads": list(item.related),
                "relationship_status": "hypotheses, not verified contracts",
                "query_errors": []}


class DemoSession:
    def __init__(self, task):
        parsed = json.loads(task)
        self.role = parsed["role"]
        self.company = parsed["company"]
        self.turn = 0
        self.evidence = {}

    def send(self, results):
        for _, payload in results:
            for record in payload.get("evidence", []):
                if not record.get("error"):
                    self.evidence[record["id"]] = record
        self.turn += 1
        if self.turn == 1:
            names = {"fundamental": ["get_prices", "get_fundamentals"],
                     "news": ["get_news"], "ecosystem": ["get_ecosystem"],
                     "decision": ["get_dossier"], "skeptic": ["get_dossier"]}[self.role]
            return Turn(calls=[Call(f"demo-{self.role}-{n}", n, {}) for n in names])
        if self.role == "decision" and self.turn == 2:
            # Deliberately illustrate several branches of the decision gate.
            sell = self.company == "Tesla"
            return Turn(calls=[Call("demo-valuation", "value_scenario", {
                "annual_growth": 0.0 if sell else 0.14,
                "exit_pe": 12 if sell else 24, "required_return": 0.10,
                "assumption_reason": "FICTIONAL teaching assumptions; not company forecasts."})])
        refs = list(self.evidence)
        valuations = [key for key, value in self.evidence.items() if value["kind"] == "valuation"]
        stance = "HOLD"
        if "Acushnet" in self.company:
            stance = "BUY"
        elif self.company == "Tesla":
            stance = "SELL"
        summaries = {
            "fundamental": "FICTIONAL example: compare earnings with cash after equipment spending before assuming growth benefits shareholders.",
            "news": "FICTIONAL example: the supplied demand headline sounds positive, but it does not establish higher margins or a share-price rise.",
            "ecosystem": "FICTIONAL example: investigate whether capacity expansion and competing supply offset stronger demand. Related tickers remain research leads.",
            "decision": "FICTIONAL example: the stance below follows an assumed valuation, and changes if those assumptions fail.",
            "skeptic": "FICTIONAL scripted approval: the software demonstrates a review gate; no real company thesis has been verified.",
        }
        rationale = "The next evidence to seek is realized pricing, cash conversion, and the timing of competing supply."
        if self.role == "decision" and valuations:
            valuation = self.evidence[valuations[-1]]["data"]
            rationale = (f"DEMO {stance}: conditional present value is {valuation['present_value']:.2f} "
                         f"{valuation['currency']} under the stated growth, exit multiple and required return. "
                         "This calculation is sensitive to its assumptions and excludes dividends/dilution.")
        return Turn(text=json.dumps({
            "summary": summaries[self.role],
            "claims": [{"text": "This fixture provides invented figures and news for demonstrating the flow.",
                        "evidence_ids": refs}],
            "risks": ["No real research was performed; never act on this example."],
            "stance": stance if self.role == "decision" else "ABSTAIN",
            "rationale": rationale,
            "invalidation": "In this fictional case, reassess if realized cash generation or pricing fails to support the assumed earnings growth. Verify real evidence before any action.",
            "valuation_id": valuations[-1] if self.role == "decision" and valuations else None,
            "approved": self.role == "skeptic",
        }))


class DemoFactory:
    def create(self, route, system, task, tools):
        return DemoSession(task)
