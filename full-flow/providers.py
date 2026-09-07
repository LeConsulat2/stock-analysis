"""Native SDK sessions preserve native assistant turns, including thinking blocks.

Only this file knows provider wire formats. SDKs import lazily: demo/tests need none.
"""

from __future__ import annotations

import json
import os
from typing import Any

from config import KEYS, Route, Settings
from contracts import Call, Tool, Turn


class Budget:
    """A request ceiling, not an exact dollar budget. Configure vendor spend limits."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def take(self) -> None:
        if self.used >= self.limit:
            raise RuntimeError("Run request budget exhausted")
        self.used += 1


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def parse_arguments(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return None  # The dispatcher returns a correlated argument error.


class OpenAISession:
    def __init__(self, client: Any, route: Route, system: str, task: str,
                 tools: list[Tool], settings: Settings, budget: Budget):
        self.client, self.route, self.system = client, route, system
        self.settings, self.budget = settings, budget
        self.messages: list[Any] = [{"role": "user", "content": task}]
        self.tools = [{"type": "function", "name": t.name,
                       "description": t.description, "parameters": t.schema(),
                       "strict": True} for t in tools]

    def send(self, results: list[tuple[Call, dict[str, Any]]]) -> Turn:
        self.messages.extend({"type": "function_call_output", "call_id": c.id,
                              "output": encode(value)} for c, value in results)
        self.budget.take()
        response = self.client.responses.create(
            model=self.route.model, instructions=self.system, input=self.messages,
            tools=self.tools, max_output_tokens=self.settings.max_output_tokens,
            store=False, include=["reasoning.encrypted_content"])
        # Retain reasoning/output items, not just output_text.
        self.messages.extend(response.output)
        if response.status != "completed":
            return Turn(terminal_error="OpenAI response incomplete or failed")
        calls = [Call(x.call_id, x.name, parse_arguments(x.arguments))
                 for x in response.output if x.type == "function_call"]
        return Turn(response.output_text or "", calls)


class ClaudeSession:
    def __init__(self, client: Any, route: Route, system: str, task: str,
                 tools: list[Tool], settings: Settings, budget: Budget):
        self.client, self.route, self.system = client, route, system
        self.settings, self.budget = settings, budget
        self.messages: list[Any] = [{"role": "user", "content": task}]
        self.tools = [{"name": t.name, "description": t.description,
                       "input_schema": t.schema()} for t in tools]

    def send(self, results: list[tuple[Call, dict[str, Any]]]) -> Turn:
        if results:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c.id,
                 "content": encode(value), "is_error": bool(value.get("error"))}
                for c, value in results]})
        self.budget.take()
        response = self.client.messages.create(
            model=self.route.model, system=self.system, messages=self.messages,
            tools=self.tools, max_tokens=self.settings.max_output_tokens)
        content = [b.model_dump(exclude_none=True) for b in response.content]
        self.messages.append({"role": "assistant", "content": content})
        if response.stop_reason not in {"end_turn", "tool_use"}:
            return Turn(terminal_error="Claude refused, paused, or truncated")
        calls = [Call(b.id, b.name, b.input) for b in response.content
                 if b.type == "tool_use"]
        text = "\n".join(b.text for b in response.content if b.type == "text")
        return Turn(text, calls)


class GoogleSession:
    def __init__(self, client: Any, types: Any, route: Route, system: str,
                 task: str, tools: list[Tool], settings: Settings, budget: Budget):
        self.client, self.types, self.route = client, types, route
        self.settings, self.budget = settings, budget
        self.messages = [types.Content(role="user", parts=[types.Part(text=task)])]
        self.native_ids: dict[str, str | None] = {}
        self.turn = 0
        self.config = types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=settings.max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description,
                                          parameters_json_schema=t.schema())
                for t in tools])])

    def send(self, results: list[tuple[Call, dict[str, Any]]]) -> Turn:
        types = self.types
        if results:
            parts = []
            for call, value in results:
                fields = {"name": call.name, "response": value}
                if self.native_ids.get(call.id):
                    fields["id"] = self.native_ids[call.id]
                parts.append(types.Part(function_response=types.FunctionResponse(**fields)))
            self.messages.append(types.Content(role="user", parts=parts))
        self.budget.take()
        response = self.client.models.generate_content(
            model=self.route.model, contents=self.messages, config=self.config)
        if not response.candidates:
            return Turn(terminal_error="Google returned no candidate")
        candidate = response.candidates[0]
        finish = getattr(candidate.finish_reason, "value", candidate.finish_reason)
        if finish != "STOP" or candidate.content is None:
            return Turn(terminal_error="Google refused or truncated")
        # Preserve all native parts and thought signatures, even non-text parts.
        self.messages.append(candidate.content)
        self.turn += 1
        calls, texts = [], []
        for index, part in enumerate(candidate.content.parts or []):
            if part.function_call:
                fc = part.function_call
                native_id = getattr(fc, "id", None)
                call_id = native_id or f"google-{self.turn}-{index}"
                self.native_ids[call_id] = native_id
                calls.append(Call(call_id, fc.name, dict(fc.args or {})))
            elif part.text and not getattr(part, "thought", False):
                texts.append(part.text)
        return Turn("\n".join(texts), calls)


class ProviderFactory:
    def __init__(self, settings: Settings, budget: Budget):
        self.settings, self.budget = settings, budget
        self.clients: dict[str, Any] = {}

    def create(self, route: Route, system: str, task: str, tools: list[Tool]):
        if route.provider not in self.clients:
            key = os.environ[KEYS[route.provider]]
            if route.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=key, timeout=self.settings.timeout_seconds,
                                max_retries=0)
            elif route.provider == "claude":
                from anthropic import Anthropic
                client = Anthropic(api_key=key, timeout=self.settings.timeout_seconds,
                                   max_retries=0)
            else:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key, http_options=types.HttpOptions(
                    timeout=self.settings.timeout_seconds * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1)))
            self.clients[route.provider] = client
        client = self.clients[route.provider]
        args = (route, system, task, tools, self.settings, self.budget)
        if route.provider == "openai":
            return OpenAISession(client, *args)
        if route.provider == "claude":
            return ClaudeSession(client, *args)
        from google.genai import types
        return GoogleSession(client, types, *args)
