"""Read this file first: it shows the entire sequential research workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agents import run_agent
from config import ROOT, ROLES, Route, Settings, check_keys, load_routes
from contracts import AgentResult, CompanyReport, Instrument
from data_tools import EvidenceBook, YahooSource
from demo import DemoFactory, DemoSource
from guardrails import publication_gate
from providers import Budget, ProviderFactory
from report import previous_run, write_reports
from watchlist import select_watchlist


def skipped(role: str) -> AgentResult:
    return AgentResult(role, "none", "none", "skipped")


def analyze_company(item: Instrument, source: Any, factory: Any,
                    routes: dict[str, Route], settings: Settings, now: datetime,
                    trace: Callable[[str], None] = print) -> CompanyReport:
    book = EvidenceBook(item, source, settings, now)
    # The dashboard needs every price even if a model later fails. Analysts still
    # explicitly request tools and see the same cached evidence, not hidden values.
    snapshot = book.collect("prices")
    if not item.ticker or snapshot.error:
        book.collect("news")
        action = "CONTEXT ONLY" if not item.ticker else "ABSTAIN"
        return CompanyReport(item, snapshot, [], skipped("skeptic"), skipped("decision"),
                             action, [item.note or "Matching completed-session price unavailable"],
                             list(book.records.values()))

    analysts = [run_agent(role, routes[role], factory, book, settings, {}, trace)
                for role in ("fundamental", "news", "ecosystem")]
    context = {"analysts": [asdict(result) for result in analysts]}
    decision = run_agent("decision", routes["decision"], factory, book, settings, context, trace)
    # Skeptic sees the proposed decision too: it can reject a new unsupported claim
    # introduced during synthesis, not just objections in earlier analyst notes.
    context["draft_decision"] = asdict(decision)
    skeptic = run_agent("skeptic", routes["skeptic"], factory, book, settings, context, trace)
    action, blocks = publication_gate(book, analysts, decision, skeptic, settings)
    return CompanyReport(item, snapshot, analysts, skeptic, decision, action, blocks,
                         list(book.records.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Educational daily investment research flow")
    parser.add_argument("--demo", action="store_true", help="Fictional data and scripted AI; no network or keys")
    parser.add_argument("--provider", choices=["openai", "claude", "google", "mixed"], default="openai")
    parser.add_argument("--only", nargs="+", help="Watchlist keys/tickers, e.g. acushnet samsung NVDA")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--timezone", default="Pacific/Auckland")
    parser.add_argument("--max-requests", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = Settings(display_timezone=args.timezone, max_requests=args.max_requests)
        from zoneinfo import ZoneInfo
        ZoneInfo(settings.display_timezone)  # Fail before any paid calls if invalid.
        items = select_watchlist(args.only)
        budget = Budget(settings.max_requests)
        if args.demo:
            source, factory = DemoSource(), DemoFactory()
            routes = {role: Route("demo", "scripted-fiction") for role in ROLES}
        else:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")  # Predictable even in Windows Task Scheduler.
            routes = load_routes(args.provider)
            check_keys(routes)
            source, factory = YahooSource(settings), ProviderFactory(settings, budget)
        now = datetime.now(timezone.utc)
        prior = previous_run(args.output, args.demo)
        trace = (lambda text: None) if args.quiet else print
        reports = [analyze_company(item, source, factory, routes, settings, now, trace)
                   for item in items]
        paths = write_reports(reports, args.output, now, settings.display_timezone,
                              args.demo, budget.used, prior, settings.horizon_years)
        for path in paths:
            print(path)
        failed = any(r.instrument.ticker and (r.snapshot.error or r.blocks) for r in reports)
        return 1 if failed else 0  # Reports still exist for partial runs.
    except (ValueError, ImportError, KeyError, OSError) as exc:
        # Our setup ValueErrors contain only argument/key names, never secret values.
        detail = str(exc) if type(exc) is ValueError else type(exc).__name__
        print(f"Setup/output failed: {detail}. Check dependencies (including tzdata on Windows), settings and output.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
