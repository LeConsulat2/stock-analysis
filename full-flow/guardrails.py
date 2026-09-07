"""Deterministic publication gate. This is not proof of investment profitability."""

from datetime import datetime

from config import Settings
from contracts import AgentResult
from data_tools import EvidenceBook, finite


def publication_gate(book: EvidenceBook, analysts: list[AgentResult],
                     decision: AgentResult, skeptic: AgentResult,
                     settings: Settings) -> tuple[str, list[str]]:
    blocks = []
    if not book.item.ticker:
        return "CONTEXT ONLY", ["No direct listed instrument configured"]
    prices, fundamentals = book.collect("prices"), book.collect("fundamentals")
    if prices.error or prices.data.get("identity_verified") is not True:
        blocks.append("A matching company price could not be verified")
    price = finite(prices.data.get("close"))
    age = finite(prices.data.get("age_days"))
    if price is None or price <= 0 or age is None or not 0 <= age <= settings.max_price_age_days:
        blocks.append("Price is missing, future-dated, or older than the configured tolerance")
    if fundamentals.error:
        blocks.append("Fundamental data unavailable")
    try:
        statement_age = (book.now.date() - datetime.fromisoformat(fundamentals.as_of).date()).days
        if not 0 <= statement_age <= settings.max_statement_age_days:
            blocks.append("Annual statements are too old or future-dated")
    except (TypeError, ValueError):
        blocks.append("Financial reporting period unavailable")
    if prices.data.get("currency") != book.item.currency:
        blocks.append("Price currency does not match the configured instrument")
    if any(result.status != "complete" for result in analysts + [decision, skeptic]):
        blocks.append("At least one required analyst or reviewer failed")
    if skeptic.status == "complete" and skeptic.content.get("approved") is not True:
        blocks.append("Skeptical reviewer rejected the draft: " + skeptic.content.get("rationale", ""))
    action = decision.content.get("stance", "ABSTAIN")
    if action in {"BUY", "SELL"}:
        valuation = book.records.get(decision.content.get("valuation_id"))
        if valuation is None or valuation.kind != "valuation" or valuation.error:
            blocks.append("BUY/SELL needs a calculated, cited valuation scenario")
        else:
            value = finite(valuation.data.get("present_value"))
            if value is None or price is None:
                blocks.append("Cannot compare valuation with price")
            elif action == "BUY" and value < price * (1 + settings.buy_cushion):
                blocks.append(f"BUY does not meet the example's {settings.buy_cushion:.0%} valuation cushion")
            elif action == "SELL" and value >= price:
                blocks.append("SELL is inconsistent with the selected valuation scenario")
    if blocks:
        return "ABSTAIN", blocks
    return action, []
