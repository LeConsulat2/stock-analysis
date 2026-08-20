"""A small, OpenAI-first lesson in grounded tool calling.

This is intentionally a walking skeleton, not the full stock-analysis rebuild.
It demonstrates one complete and trustworthy path:

    question -> OpenAI tool call -> Python/yfinance -> tool result -> answer

Run it after placing ``OPENAI_API_KEY`` in the untracked ``.env`` file:

    python codex-main.py 005930.KS

The default model is deliberately configurable:

    python codex-main.py NVDA --model gpt-5.6-terra
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_TICKER = "005930.KS"
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.^=\-]{0,19}$")


@dataclass(frozen=True)
class SampleConfig:
    """The few controls worth changing in this first lesson."""

    model: str = DEFAULT_MODEL
    max_turns: int = 4
    max_output_tokens: int = 1_200
    reasoning_effort: str = "low"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if self.reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("reasoning_effort is not supported")


SYSTEM_INSTRUCTIONS = """You are a careful market-data research assistant.
This is educational research, not financial advice.

Rules:
1. Call get_price_snapshot before stating any price, date, or price change.
2. Treat the tool result as the only source of market facts. Never invent a
   missing value, currency, timestamp, or explanation.
3. If the tool reports an error or missing field, state that limitation plainly.
4. Keep the answer short and use exactly these headings: Evidence,
   Interpretation, Limits.
5. Do not issue a BUY, HOLD, or SELL recommendation in this teaching sample.
"""


# OpenAI function tools are ordinary JSON Schemas. Strict mode makes the model
# adhere to the contract; Python still validates and catches errors at runtime.
OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_price_snapshot",
        "description": (
            "Return a compact, sourced snapshot of recent unadjusted daily "
            "closing prices for one ticker."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Yahoo Finance ticker, for example NVDA or 005930.KS.",
                }
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


class AgentRunError(RuntimeError):
    """The bounded agent loop could not produce a valid terminal answer."""


class GroundingError(AgentRunError):
    """The model tried to answer before requesting evidence."""


TickerFactory = Callable[[str], Any]
ToolHandler = Callable[..., dict[str, Any]]
TraceWriter = Callable[[str], None]


def _default_ticker_factory(ticker: str) -> Any:
    """Import yfinance only when the real tool is actually used."""

    import yfinance as yf

    return yf.Ticker(ticker)


def _error_result(
    ticker: object,
    code: str,
    message: str,
    retrieved_at: str,
) -> dict[str, Any]:
    """Use the same small, JSON-safe error shape for every tool failure."""

    return {
        "ok": False,
        "ticker": str(ticker)[:40],
        "source": "yfinance",
        "retrieved_at": retrieved_at,
        "error": {"code": code, "message": message[:300]},
    }


def _read_currency(market: Any) -> str | None:
    """Read optional yfinance metadata without guessing a fallback currency."""

    try:
        fast_info = getattr(market, "fast_info", None)
        if fast_info is None:
            return None
        value = (
            fast_info.get("currency")
            if hasattr(fast_info, "get")
            else getattr(fast_info, "currency", None)
        )
    except Exception:  # yfinance metadata is a network-backed boundary.
        return None
    return str(value) if value else None


def get_price_snapshot(
    ticker: str,
    *,
    ticker_factory: TickerFactory | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch a compact evidence envelope instead of returning a raw dataframe."""

    retrieved_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    normalized = ticker.strip().upper() if isinstance(ticker, str) else ""
    if not normalized or not TICKER_PATTERN.fullmatch(normalized):
        return _error_result(
            ticker,
            "invalid_ticker",
            "Ticker must contain 1-20 supported Yahoo Finance symbol characters.",
            retrieved_at,
        )

    make_ticker = ticker_factory or _default_ticker_factory
    try:
        market = make_ticker(normalized)
        history = market.history(period="5d", interval="1d", auto_adjust=False)
        if history is None or getattr(history, "empty", True):
            return _error_result(
                normalized,
                "no_price_data",
                "No recent daily price rows were returned.",
                retrieved_at,
            )

        closes = history["Close"].dropna()
        if len(closes) == 0:
            return _error_result(
                normalized,
                "no_close_data",
                "Recent rows contained no usable closing prices.",
                retrieved_at,
            )

        latest_close = float(closes.iloc[-1])
        previous_close = float(closes.iloc[-2]) if len(closes) > 1 else None
        change_pct = None
        if previous_close not in (None, 0.0):
            change_pct = ((latest_close / previous_close) - 1.0) * 100.0

        data_index = closes.index[-1]
        data_as_of = (
            data_index.isoformat()
            if hasattr(data_index, "isoformat")
            else str(data_index)
        )
        currency = _read_currency(market)
        warnings = [] if currency else ["Currency metadata was unavailable; none was assumed."]

        return {
            "ok": True,
            "ticker": normalized,
            "source": "yfinance",
            "retrieved_at": retrieved_at,
            "data_as_of": data_as_of,
            "window": "5d / 1d",
            "price_type": "unadjusted daily close",
            "currency": currency,
            "latest_close": round(latest_close, 4),
            "previous_close": (
                round(previous_close, 4) if previous_close is not None else None
            ),
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "observations": len(closes),
            "warnings": warnings,
        }
    except Exception as exc:  # External data failures become evidence, not crashes.
        return _error_result(
            normalized,
            "data_source_error",
            f"{type(exc).__name__}: {exc}",
            retrieved_at,
        )


def _tool_error(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "source": "local_tool_dispatch",
        "error": {"code": code, "message": message[:300]},
    }


def _execute_tool_call(
    call: Any,
    handlers: Mapping[str, ToolHandler],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Validate one model request and dispatch only explicitly allowed tools."""

    name = str(getattr(call, "name", ""))
    call_id = str(getattr(call, "call_id", ""))
    if not call_id:
        raise AgentRunError("OpenAI returned a function call without a call_id.")

    raw_arguments = getattr(call, "arguments", "{}")
    try:
        arguments = (
            raw_arguments
            if isinstance(raw_arguments, dict)
            else json.loads(raw_arguments)
        )
    except (TypeError, json.JSONDecodeError) as exc:
        return (
            name,
            call_id,
            {},
            _tool_error("invalid_arguments", f"Arguments were not valid JSON: {exc}"),
        )

    if not isinstance(arguments, dict):
        return (
            name,
            call_id,
            {},
            _tool_error("invalid_arguments", "Tool arguments must be a JSON object."),
        )

    handler = handlers.get(name)
    if handler is None:
        return (
            name,
            call_id,
            arguments,
            _tool_error("unknown_tool", f"Tool is not allowed: {name or '<empty>'}"),
        )

    try:
        result = handler(**arguments)
    except Exception as exc:  # A custom handler must not tear down the loop.
        result = _tool_error("tool_execution_error", f"{type(exc).__name__}: {exc}")

    if not isinstance(result, dict):
        result = _tool_error("invalid_tool_result", "Tool result must be a dictionary.")
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        result = _tool_error(
            "invalid_tool_result",
            f"Tool result was not JSON-serializable: {exc}",
        )
    return name, call_id, arguments, result


def _render_no_evidence(ticker: str, failures: Sequence[str]) -> str:
    details = "; ".join(failures[:3]) or "The data tool returned no usable result."
    return (
        "## Evidence\n"
        f"- No usable market-data evidence was returned for {ticker}.\n"
        f"- Tool detail: {details}\n\n"
        "## Interpretation\n"
        "- A price interpretation would be unsupported, so none is provided.\n\n"
        "## Limits\n"
        "- Retry after checking the ticker, network access, and data source."
    )


def run_openai_agent(
    client: Any,
    ticker: str,
    *,
    config: SampleConfig | None = None,
    tool_handlers: Mapping[str, ToolHandler] | None = None,
    trace: TraceWriter | None = print,
) -> str:
    """Run a bounded Responses API loop while preserving every output item."""

    active_config = config or SampleConfig()
    normalized = ticker.strip().upper()
    handlers = dict(tool_handlers or {"get_price_snapshot": get_price_snapshot})
    input_items: list[Any] = [
        {
            "role": "user",
            "content": (
                f"Give me a grounded recent price snapshot for {normalized}. "
                "Use the available tool before answering."
            ),
        }
    ]
    tool_was_requested = False
    successful_evidence = False
    failures: list[str] = []

    for turn in range(1, active_config.max_turns + 1):
        if trace:
            trace(f"[openai] turn {turn}/{active_config.max_turns}")

        response = client.responses.create(
            model=active_config.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=input_items,
            tools=OPENAI_TOOLS,
            # Require evidence on the first turn, then let the model either
            # request more evidence or finish. The loop still verifies this.
            tool_choice="required" if not tool_was_requested else "auto",
            reasoning={"effort": active_config.reasoning_effort},
            max_output_tokens=active_config.max_output_tokens,
            store=False,
        )

        # This line is subtle and important: preserve all OpenAI output items,
        # including reasoning items, before adding correlated tool results.
        output_items = list(getattr(response, "output", []) or [])
        input_items.extend(output_items)
        function_calls = [
            item for item in output_items if getattr(item, "type", None) == "function_call"
        ]

        if function_calls:
            tool_was_requested = True
            for call in function_calls:
                name, call_id, arguments, result = _execute_tool_call(call, handlers)
                if trace:
                    trace(
                        f"[tool] {name or '<empty>'} "
                        f"{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
                    )

                if result.get("ok") is True:
                    successful_evidence = True
                    if trace:
                        trace(
                            "[tool result] ok "
                            f"source={result.get('source')} "
                            f"data_as_of={result.get('data_as_of')}"
                        )
                else:
                    error = result.get("error", {})
                    failure = f"{error.get('code', 'tool_error')}: {error.get('message', '')}"
                    failures.append(failure)
                    if trace:
                        trace(f"[tool result] error {failure}")

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            result,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
            continue

        final_text = str(getattr(response, "output_text", "") or "").strip()
        if not final_text:
            raise AgentRunError("OpenAI returned neither function calls nor final text.")
        if not tool_was_requested:
            raise GroundingError("The model answered before requesting market evidence.")
        if not successful_evidence:
            return _render_no_evidence(normalized, failures)
        return final_text

    raise AgentRunError(
        f"The agent reached its {active_config.max_turns}-turn limit without a final answer."
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI boundary: configuration in, one grounded answer out."""

    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(
        description="Run the OpenAI-first grounded tool-calling lesson."
    )
    parser.add_argument("ticker", nargs="?", default=DEFAULT_TICKER)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model ID (default: OPENAI_MODEL or %(default)s)",
    )
    parser.add_argument("--max-turns", type=int, default=4)
    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is missing. Add it to the untracked .env file.",
            file=sys.stderr,
        )
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "The openai package is missing. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    try:
        answer = run_openai_agent(
            OpenAI(),
            args.ticker,
            config=SampleConfig(model=args.model, max_turns=args.max_turns),
        )
    except (AgentRunError, ValueError) as exc:
        print(f"Sample stopped safely: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Keep API/auth/network failures readable at the CLI.
        print(f"OpenAI request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\n" + answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
