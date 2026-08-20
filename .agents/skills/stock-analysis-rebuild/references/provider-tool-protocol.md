# Provider and Tool Protocol

Read this for `llm.py`, `agents.py`, native adapters, tool dispatch, or provider-specific failures.

## One Semantic Contract

The four providers do not need identical SDK code; they must expose identical behavior to the agent loop. The internal contract needs enough information to represent:

- assistant text, including an intentionally empty text response;
- zero, one, or multiple tool calls in stable order;
- a unique call ID, tool name, and parsed argument object for every call;
- provider stop/finish state, including refusal, truncation, and ordinary completion;
- usage or diagnostic metadata when available, without making it required for control flow.

Keep SDK response objects inside adapters. The loop should consume internal dataclasses or protocols and should not branch on provider names.

## Bounded State Machine

Conceptually, one agent run is:

```text
send conversation + allowed tool schemas
  -> final text: validate grounding requirement, return
  -> tool calls: validate each call, execute registered handler, append correlated results, repeat
  -> refusal/truncation/empty invalid response: terminate explicitly
  -> max turns or repeated no-progress state: terminate explicitly
```

Append the complete assistant tool-call turn before appending results. Return one result for every requested call, preserving its call ID even when the handler fails. Multiple calls in one turn are part of the protocol, not an edge case to discard.

## Dispatch Safety

- Execute only exact names in the role's allowed registry.
- Validate arguments against the declared schema before calling Python.
- Unknown tool, invalid JSON, schema mismatch, handler exception, empty data, and timeout become compact tool-error payloads. They are not uncaught crashes and not fake data.
- Do not silently coerce a ticker, period, numeric bound, or enum into a different request.
- Tool handlers in v1 are read-only. If a side-effecting tool is ever added, it needs a separate authorization gate; a generic tool loop is not permission.
- Cap turns and any retry policy. Detect repeated identical calls/results when another round cannot add information, and return a stalled outcome rather than spending until the cap.

## Adapter Responsibilities

Verify current native SDK documentation when implementing exact syntax; SDK shapes can change. Preserve these semantics:

- **Anthropic:** assistant `tool_use` blocks and matching user `tool_result` blocks remain paired and ordered.
- **OpenAI:** the assistant message carrying `tool_calls` is retained, and each tool-role result references the correct call ID.
- **Google:** function calls and function responses retain the model turn and correlation expected by the current `google-genai` API.
- **Local/Ollama:** reuse the OpenAI-compatible transport only after a readiness check confirms `ollama serve`, the exact installed tag, and tool-call support. “Local inference” does not prove the whole research flow is offline.

Do not reduce all provider responses to text too early. That destroys tool-call structure and makes later turns impossible to serialize correctly.

## Terminal Outcomes

Distinguish at least:

- completed with final text;
- refused;
- truncated;
- invalid provider response;
- tool failure with no usable evidence;
- stalled/no progress;
- turn limit exhausted.

A caller must not render stalled or exhausted output as an approved analysis. If partial analyst text is retained for diagnostics, label its outcome separately.

## Observability Without Leakage

Print or record a concise trace: role, turn, provider, tool name, safe arguments, result status, and elapsed time. Redact secrets and avoid raw financial-statement dumps or full model responses. Correlation IDs should make it possible to answer “which tool result supported this analyst output?”

## Conformance Tests

Run the same mocked scenarios against every adapter:

1. final text without a tool call;
2. one tool call and correlated result;
3. two tool calls in one assistant turn;
4. tool error returned to the model;
5. malformed arguments and unknown tool;
6. refusal, truncation, and empty content;
7. turn exhaustion and repeated-call stall;
8. usage metadata absent without breaking control flow.

Provider parity means these scenarios have the same internal outcomes. It does not mean the providers produce identical prose.
