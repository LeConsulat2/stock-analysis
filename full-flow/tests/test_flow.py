"""Offline behavioral tests. No network calls, API keys, or paid inference."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import run_agent, validate_note
from config import ROLES, Route, Settings, check_keys, load_routes
from contracts import Call, Turn
from data_tools import EvidenceBook, TOOLS, YahooSource, completed_prices, normalize_news
from demo import DemoFactory, DemoSource, DemoSession
from guardrails import publication_gate
from main import analyze_company
from providers import Budget, ClaudeSession, GoogleSession, OpenAISession
from report import previous_run, write_reports
from watchlist import select_watchlist

NOW = datetime(2026, 9, 8, 20, 30, tzinfo=timezone.utc)
SETTINGS = Settings()
ITEM = select_watchlist(["acushnet"])[0]
ROUTES = {role: Route("demo", "fiction") for role in ROLES}


class QueueFactory:
    def __init__(self, turns):
        self.turns = iter(turns)

    def create(self, *args):
        return self

    def send(self, results):
        return next(self.turns)


class FlowTests(unittest.TestCase):
    def book(self, item=ITEM, source=None):
        return EvidenceBook(item, source or DemoSource(), SETTINGS, NOW)

    def test_demo_buy_sell_and_context_only(self):
        reports = [analyze_company(item, DemoSource(), DemoFactory(), ROUTES,
                                   SETTINGS, NOW, lambda _: None)
                   for item in select_watchlist(["acushnet", "tesla", "pga", "samsung"])]
        self.assertEqual([r.published_action for r in reports],
                         ["BUY", "SELL", "CONTEXT ONLY", "HOLD"])
        self.assertNotIn("close", reports[2].snapshot.data)
        self.assertEqual(reports[3].snapshot.data["currency"], "KRW")

    def test_report_keeps_evidence_and_escapes_html(self):
        report = analyze_company(ITEM, DemoSource(), DemoFactory(), ROUTES, SETTINGS, NOW, lambda _: None)
        report.decision.content["summary"] = "<script>alert('bad')</script>"
        temporary_root = Path(__file__).resolve().parents[1] / "reports"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            paths = write_reports([report], Path(directory), NOW, "Pacific/Auckland", True, 0, None)
            rendered = paths[1].read_text(encoding="utf-8")
            self.assertNotIn("<script>", rendered)
            self.assertIn("&lt;script&gt;", rendered)
            self.assertIn("<table>", rendered)
            saved = previous_run(Path(directory), True)
            self.assertEqual(saved["companies"][0]["published_action"], "BUY")
            self.assertIsNone(previous_run(Path(directory), False))
            self.assertTrue(saved["companies"][0]["evidence"])

    def test_no_model_call_after_bad_price(self):
        class Broken(DemoSource):
            def fetch(self, kind, item, now):
                if kind == "prices":
                    raise RuntimeError("secret token must not leak")
                return super().fetch(kind, item, now)
        report = analyze_company(ITEM, Broken(), QueueFactory([]), ROUTES, SETTINGS, NOW, lambda _: None)
        self.assertEqual(report.published_action, "ABSTAIN")
        self.assertNotIn("secret", report.snapshot.error)

    def test_answer_without_tools_is_rejected(self):
        result = run_agent("news", ROUTES["news"], QueueFactory([Turn("{}")]),
                           self.book(), SETTINGS, {}, lambda _: None)
        self.assertEqual(result.status, "failed")

    def test_unknown_tool_and_bad_args_do_not_execute(self):
        book = self.book()
        self.assertIn("error", book.execute("shell", {"command": "x"}, {"get_prices"}))
        self.assertIn("error", book.execute("get_prices", {"ticker": "MSFT"}, {"get_prices"}))
        self.assertFalse(book.records)

    def test_repeated_calls_and_turn_exhaustion_fail(self):
        for turns, settings in [
            ([Turn(calls=[Call("1", "get_news", {})]), Turn(calls=[Call("2", "get_news", {})])], SETTINGS),
            ([Turn(calls=[Call("1", "get_news", {})])], replace(SETTINGS, max_turns=1)),
            ([Turn(terminal_error="truncated")], SETTINGS),
            ([Turn(calls=[Call("", "get_news", {})])], SETTINGS),
        ]:
            result = run_agent("news", ROUTES["news"], QueueFactory(turns),
                               self.book(), settings, {}, lambda _: None)
            self.assertEqual(result.status, "failed")

    def test_unseen_citations_rejected(self):
        demo = DemoSession(json.dumps({"role": "news", "company": "Test"}))
        call = demo.send([]).calls[0]
        note = demo.send([(call, {"evidence": [{"id": "fake", "kind": "news"}]})]).text
        with self.assertRaises(ValueError):
            validate_note(note, {"different"})

    def test_gate_rejects_skeptic_failure_staleness_and_wrong_valuation(self):
        report = analyze_company(ITEM, DemoSource(), DemoFactory(), ROUTES, SETTINGS, NOW, lambda _: None)
        book = self.book()
        book.records = {e.id: e for e in report.evidence}
        book.tool_cache = {e.kind: e for e in report.evidence if e.kind != "valuation"}
        report.skeptic.content["approved"] = False
        self.assertEqual(publication_gate(book, report.analysts, report.decision,
                                          report.skeptic, SETTINGS)[0], "ABSTAIN")
        report.skeptic.content["approved"] = True
        book.tool_cache["prices"].data["age_days"] = 20
        self.assertEqual(publication_gate(book, report.analysts, report.decision,
                                          report.skeptic, SETTINGS)[0], "ABSTAIN")
        book.tool_cache["prices"].data["age_days"] = 1
        report.decision.content["valuation_id"] = "acushnet:news"
        self.assertEqual(publication_gate(book, report.analysts, report.decision,
                                          report.skeptic, SETTINGS)[0], "ABSTAIN")

    def test_valuation_math_and_nan_rejection(self):
        book = self.book()
        args = {"annual_growth": 0.1, "exit_pe": 20, "required_return": 0.1,
                "assumption_reason": "Illustrative stable multiple and matching growth/discount rates."}
        value = book.value_scenario(args)["evidence"][-1]["data"]["present_value"]
        self.assertAlmostEqual(value, 100)
        args["annual_growth"] = float("nan")
        self.assertIn("error", book.value_scenario(args))

    def test_market_cutoff_and_split_adjusted_change(self):
        rows = [{"date": "2026-09-04", "close": 200, "adjusted": 100},
                {"date": "2026-09-08", "close": 102, "adjusted": 102}]
        before = completed_prices(rows, ITEM, datetime(2026, 9, 8, 18, tzinfo=timezone.utc))
        self.assertEqual(before["session_date"], "2026-09-04")
        after = completed_prices(rows, ITEM, NOW)
        self.assertEqual(after["session_date"], "2026-09-08")
        self.assertAlmostEqual(after["change_pct"], 2)

    def test_news_dedup_dates_and_untrusted_urls(self):
        base = {"title": "Example", "link": "https://example.com/a", "providerPublishTime": NOW.timestamp()}
        items = [base, dict(base), {**base, "link": "javascript:alert(1)"},
                 {**base, "link": "https://example.com/b", "providerPublishTime": 0},
                 {**base, "link": "https://example.com/c", "providerPublishTime": NOW.timestamp() + 50}]
        self.assertEqual(len(normalize_news(items, NOW, 7, 6)), 1)

    def test_only_selected_provider_needs_key(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "fake"}, clear=True):
            check_keys(load_routes("google"))
            with self.assertRaises(ValueError):
                check_keys(load_routes("mixed"))

    def test_budget_is_bounded(self):
        budget = Budget(1)
        budget.take()
        with self.assertRaises(RuntimeError):
            budget.take()


class AdapterTests(unittest.TestCase):
    def test_openai_retains_reasoning_and_correlates_multiple_results(self):
        reasoning = NS(type="reasoning", encrypted_content="opaque")
        calls = [NS(type="function_call", call_id=f"c{i}", name="get_prices", arguments="{}") for i in range(2)]
        requests = []
        responses = iter([NS(output=[reasoning] + calls, output_text="", status="completed"),
                          NS(output=[], output_text="done", status="completed")])
        def create(**kwargs):
            requests.append(list(kwargs["input"]))
            return next(responses)
        session = OpenAISession(NS(responses=NS(create=create)), Route("openai", "test"),
                                "system", "task", [TOOLS["get_prices"]], SETTINGS, Budget(4))
        turn = session.send([])
        self.assertEqual(len(turn.calls), 2)
        session.send([(c, {"error": "unavailable"}) for c in turn.calls])
        self.assertIs(requests[-1][1], reasoning)
        self.assertEqual([x["call_id"] for x in requests[-1][-2:]], ["c0", "c1"])

    def test_claude_preserves_thinking_and_error_result(self):
        class Block(NS):
            def model_dump(self, **kwargs):
                return vars(self)
        response = NS(content=[Block(type="thinking", thinking="opaque", signature="sig"),
                               Block(type="tool_use", id="c1", name="get_news", input={})], stop_reason="tool_use")
        requests = []
        def create(**kwargs):
            requests.append(list(kwargs["messages"]))
            return response
        session = ClaudeSession(NS(messages=NS(create=create)), Route("claude", "test"),
                                "system", "task", [TOOLS["get_news"]], SETTINGS, Budget(4))
        turn = session.send([])
        session.send([(turn.calls[0], {"error": "unavailable"})])
        self.assertEqual(requests[-1][1]["content"][0]["signature"], "sig")
        self.assertTrue(requests[-1][2]["content"][0]["is_error"])
        self.assertEqual(requests[-1][2]["content"][0]["tool_use_id"], "c1")

    def test_google_native_types_preserve_signatures_and_ids(self):
        from google.genai import types
        content = types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(id="g1", name="get_news", args={}),
            thought_signature=b"opaque")])
        requests = []
        def create(**kwargs):
            requests.append(list(kwargs["contents"]))
            return NS(candidates=[NS(content=content, finish_reason="STOP")])
        session = GoogleSession(NS(models=NS(generate_content=create)), types,
                                Route("google", "test"), "system", "task",
                                [TOOLS["get_news"]], SETTINGS, Budget(4))
        turn = session.send([])
        session.send([(turn.calls[0], {"error": "unavailable"})])
        self.assertIs(requests[-1][1], content)
        self.assertEqual(requests[-1][1].parts[0].thought_signature, b"opaque")
        self.assertEqual(requests[-1][2].parts[0].function_response.id, "g1")
        types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1))

    def test_yahoo_identity_and_annual_statement_normalization(self):
        import pandas as pd
        date = pd.Timestamp("2025-12-31")
        income = pd.DataFrame({date: [1000, 2]}, index=["Total Revenue", "Diluted EPS"])
        fake = NS(get_info=lambda: {"longName": "Acushnet Holdings", "quoteType": "EQUITY",
                                   "symbol": "GOLF", "currency": "USD", "financialCurrency": "USD"},
                  income_stmt=income, cashflow=pd.DataFrame())
        source = YahooSource(SETTINGS)
        source.company = lambda ticker: fake
        result = source.fetch("fundamentals", ITEM, NOW)
        self.assertEqual(result["diluted_eps"], 2)
        self.assertIsNone(result["operating_cash_flow"])
        fake.get_info = lambda: {"longName": "Unrelated ETF", "quoteType": "ETF", "symbol": "GOLF", "currency": "USD"}
        with self.assertRaises(ValueError):
            source.fetch("fundamentals", ITEM, NOW)


if __name__ == "__main__":
    unittest.main()
