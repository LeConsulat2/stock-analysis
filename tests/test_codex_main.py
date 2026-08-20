"""Offline contract tests for the OpenAI-first walking skeleton."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[1] / "codex-main.py"
SPEC = importlib.util.spec_from_file_location("codex_main_sample", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
sample = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sample
SPEC.loader.exec_module(sample)


def function_call(call_id: str, ticker: str = "NVDA") -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name="get_price_snapshot",
        call_id=call_id,
        arguments=json.dumps({"ticker": ticker}),
    )


def response(*items: Any, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(output=list(items), output_text=text)


class FakeResponses:
    def __init__(self, queued: list[SimpleNamespace]) -> None:
        self.queued = list(queued)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        recorded = dict(kwargs)
        recorded["input"] = list(kwargs["input"])
        self.requests.append(recorded)
        return self.queued.pop(0)


class FakeClient:
    def __init__(self, *queued: SimpleNamespace) -> None:
        self.responses = FakeResponses(list(queued))


class FakeSeries:
    def __init__(self, values: list[float], dates: list[datetime]) -> None:
        self._values = values
        self.index = dates
        self.iloc = self

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

    def dropna(self) -> "FakeSeries":
        return self


class FakeHistory:
    empty = False

    def __init__(self) -> None:
        self.closes = FakeSeries(
            [100.0, 102.0],
            [
                datetime(2026, 8, 19, tzinfo=timezone.utc),
                datetime(2026, 8, 20, tzinfo=timezone.utc),
            ],
        )

    def __getitem__(self, key: str) -> FakeSeries:
        if key != "Close":
            raise KeyError(key)
        return self.closes


class FakeTicker:
    fast_info = {"currency": "USD"}

    def history(self, **_: Any) -> FakeHistory:
        return FakeHistory()


class OpenAIToolLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = {
            "ok": True,
            "ticker": "NVDA",
            "source": "fake-market-data",
            "data_as_of": "2026-08-20",
            "latest_close": 123.45,
        }

    def test_preserves_output_and_correlates_tool_result(self) -> None:
        reasoning = SimpleNamespace(type="reasoning", id="reasoning-1")
        call = function_call("call-1")
        client = FakeClient(
            response(reasoning, call),
            response(SimpleNamespace(type="message"), text="## Evidence\nGrounded."),
        )

        answer = sample.run_openai_agent(
            client,
            "NVDA",
            tool_handlers={"get_price_snapshot": lambda ticker: self.evidence},
            trace=None,
        )

        self.assertEqual(answer, "## Evidence\nGrounded.")
        self.assertEqual(len(client.responses.requests), 2)
        second_input = client.responses.requests[1]["input"]
        self.assertIn(reasoning, second_input)
        self.assertIn(call, second_input)
        tool_output = next(
            item
            for item in second_input
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        )
        self.assertEqual(tool_output["call_id"], "call-1")
        self.assertEqual(json.loads(tool_output["output"]), self.evidence)
        self.assertTrue(sample.OPENAI_TOOLS[0]["strict"])
        self.assertEqual(client.responses.requests[0]["tool_choice"], "required")
        self.assertEqual(client.responses.requests[1]["tool_choice"], "auto")
        self.assertEqual(client.responses.requests[0]["reasoning"], {"effort": "low"})
        self.assertFalse(client.responses.requests[0]["store"])

    def test_handles_multiple_calls_in_one_response(self) -> None:
        client = FakeClient(
            response(function_call("call-a"), function_call("call-b", "MSFT")),
            response(SimpleNamespace(type="message"), text="Grounded comparison."),
        )

        sample.run_openai_agent(
            client,
            "NVDA",
            tool_handlers={"get_price_snapshot": lambda ticker: self.evidence},
            trace=None,
        )

        outputs = [
            item
            for item in client.responses.requests[1]["input"]
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual([item["call_id"] for item in outputs], ["call-a", "call-b"])

    def test_rejects_final_answer_before_any_tool_call(self) -> None:
        client = FakeClient(response(SimpleNamespace(type="message"), text="Ungrounded."))

        with self.assertRaises(sample.GroundingError):
            sample.run_openai_agent(client, "NVDA", trace=None)

    def test_fails_closed_when_tool_returns_no_evidence(self) -> None:
        client = FakeClient(
            response(function_call("call-1")),
            response(SimpleNamespace(type="message"), text="Invented price: 999."),
        )
        failure = {
            "ok": False,
            "error": {"code": "offline", "message": "No network."},
        }

        answer = sample.run_openai_agent(
            client,
            "NVDA",
            tool_handlers={"get_price_snapshot": lambda ticker: failure},
            trace=None,
        )

        self.assertIn("No usable market-data evidence", answer)
        self.assertNotIn("999", answer)

    def test_turn_limit_is_an_explicit_failure(self) -> None:
        client = FakeClient(response(function_call("call-1")))

        with self.assertRaisesRegex(sample.AgentRunError, "turn limit"):
            sample.run_openai_agent(
                client,
                "NVDA",
                config=sample.SampleConfig(max_turns=1),
                tool_handlers={"get_price_snapshot": lambda ticker: self.evidence},
                trace=None,
            )

    def test_rejects_non_json_tool_result(self) -> None:
        client = FakeClient(
            response(function_call("call-1")),
            response(SimpleNamespace(type="message"), text="Invented."),
        )

        answer = sample.run_openai_agent(
            client,
            "NVDA",
            tool_handlers={
                "get_price_snapshot": lambda ticker: {
                    "ok": True,
                    "when": datetime(2026, 8, 20, tzinfo=timezone.utc),
                }
            },
            trace=None,
        )

        self.assertIn("No usable market-data evidence", answer)
        self.assertNotIn("Invented", answer)


class PriceToolTests(unittest.TestCase):
    def test_returns_compact_sourced_snapshot(self) -> None:
        result = sample.get_price_snapshot(
            "nvda",
            ticker_factory=lambda ticker: FakeTicker(),
            now=datetime(2026, 8, 20, 5, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["ticker"], "NVDA")
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["latest_close"], 102.0)
        self.assertEqual(result["change_pct"], 2.0)
        self.assertEqual(result["data_as_of"], "2026-08-20T00:00:00+00:00")

    def test_returns_error_payload_for_bad_ticker(self) -> None:
        result = sample.get_price_snapshot("not a ticker!")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "invalid_ticker")

    def test_wraps_data_source_failure(self) -> None:
        def fail(_: str) -> Any:
            raise ConnectionError("offline")

        result = sample.get_price_snapshot("NVDA", ticker_factory=fail)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "data_source_error")


if __name__ == "__main__":
    unittest.main()
