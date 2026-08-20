# Rebuild Execution Plan

**Last reconciled:** 2026-08-20  
**Status:** Active plan; the six-module application rebuild remains unimplemented. An OpenAI-only educational spike now exists and does not advance a production gate.  
**Living record:** Update statuses and dated rulings here as work lands. Do not rewrite settled decisions without new evidence or a user ruling.

## Bottom Line

The first job is not to build five analyst personalities. It is to prove one bounded, provider-neutral tool round trip and its failure states with a fake client. Once that spine is trustworthy, add normalized market tools, then provider adapters, then roles, synthesis, reporting, and the CLI. The largest foreseeable trust failure is a polished recommendation whose ticker, date, currency, calculation, or tool provenance cannot be reconstructed.

## Scope and Authority

- Root `AGENTS.md` and `.claude/NewStart.md` define v1.
- Current files and tests prove implementation status.
- `CLAUDE.md` explains the legacy experiments and why `archive/` is reference-only.
- `thoughts/codex-thoughts-at-start.md` is adopted as direction, not v1 scope. Primary-source research, SQLite memory, portfolio management, and a UI are parked until the lean rebuild is proven.
- This plan orders work; it does not replace those documents.

## Confirmed Rulings

| ID | Ruling | Status |
|---|---|---|
| R1 | Active production code is six flat root modules plus tests. | CONFIRMED 2026-08-20 |
| R2 | Use native Anthropic, OpenAI, Google, and Ollama-compatible adapters behind one interface. No LiteLLM. | CONFIRMED 2026-08-20 |
| R3 | Analysts request real tools dynamically; do not prefetch prose inputs or let agents write from a ticker alone. | CONFIRMED 2026-08-20 |
| R4 | V1 orchestration is synchronous and sequential. | CONFIRMED 2026-08-20 |
| R5 | `archive/` is immutable design evidence, not the active entry point. | CONFIRMED 2026-08-20 |
| R6 | Tool payloads are compact, JSON-serializable, and return structured errors for unstable yfinance access. | CONFIRMED 2026-08-20 |
| R7 | Default roster is `005930.KS`, `066570.KS`, `GOOGL`, `META`, `MSFT`, and `NVDA`; `GOOGL` is deliberate. | CONFIRMED 2026-08-20 |
| R8 | Reports are timestamped Markdown under `reports/` and labeled research, not financial advice. | CONFIRMED 2026-08-20 |
| R9 | Numeric target, stop, or position size is never invented to satisfy a report field; use an explicit unavailable/conditional value when inputs or method are absent. | GUARDRAIL INTERPRETATION 2026-08-20 |
| R10 | Research, memory, portfolio, automation, and UI ideas in `thoughts/` remain post-v1 until separately approved. | SCOPE RECONCILIATION 2026-08-20 |

## Educational Spike — OpenAI Tool Round Trip

`codex-main.py` and `tests/test_codex_main.py` are a deliberately self-contained learning spike, not active production architecture. They demonstrate the current OpenAI Responses API function-call round trip with one compact yfinance tool, strict arguments, full output-item preservation, call-ID correlation, a bounded loop, visible traces, and fail-closed no-evidence behavior.

- Offline contract tests in repository `env` (Python 3.13.6): **PASS**, 9 tests on 2026-08-20.
- CLI parsing/help: **PASS** on 2026-08-20.
- Installed OpenAI SDK surface used by the sample: **PASS** (`openai` 3.3.1).
- Live yfinance/OpenAI round trip: **NOT RUN**; `.env` and `OPENAI_API_KEY` were absent.
- Production gates P0-P5: **UNCHANGED**. The spike intentionally does not create `llm.py`, `agents.py`, or the provider-neutral internal protocol.

Its default `gpt-5.6-terra` model is an overrideable learning choice balancing capability and cost. It does not settle D1 for the four production adapters.

## Phase 0 — Contract Spine

**Outcome:** A fake provider can request a registered fake tool, receive its correlated result, and return final text through the same public loop future providers will use.

Likely files:

- `llm.py`: internal tool/request/response types and `LLMClient` interface;
- `agents.py`: bounded `run_agent` state machine and dispatch boundary;
- `tests/test_agent_loop.py`: deterministic fake client/tool scenarios.

Proof:

- final text without a tool;
- one and multiple tool calls;
- full assistant call turn retained before tool results;
- unknown tool, malformed arguments, handler exception;
- refusal/truncation/empty response;
- repeated no-progress call and max-turn exhaustion are not returned as success.

**Gate P0:** Do not begin native adapters until the fake-client contract and terminal outcomes pass.

## Phase 1 — Configuration and Grounded Tools

**Outcome:** Configuration is typed and selected-provider-aware; each market tool returns a compact evidence-bearing result or explicit error.

Likely files:

- `config.py`;
- `data_tools.py`;
- `tests/test_config.py` and `tests/test_data_tools.py`.

Proof:

- environment precedence and only-selected-provider key validation;
- representative mocked KRX and US yfinance shapes;
- missing rows, empty frames, zero denominators, insufficient history, and upstream exceptions;
- finite JSON output and explicit date/currency/unit metadata;
- indicator warm-up, risk assumptions, and no-look-ahead backtest fixtures;
- one bounded live smoke for a KRX ticker and one for a US ticker, recorded separately from unit tests.

**Gate P1:** Do not let analyst prompts consume a tool until its success, partial, and error shapes are testable and distinguishable.

## Phase 2 — Native Provider Adapters

**Outcome:** `claude`, `openai`, `google`, and `local` translate the same internal protocol without leaking wire-format differences into `agents.py`.

Likely files:

- `llm.py`;
- `tests/test_llm_contract.py` and provider-focused test files if useful.

Proof:

- all adapters pass the same mocked scenarios from the provider protocol;
- multiple calls and result correlation survive every translation;
- refusal, truncation, malformed output, absent usage, and SDK errors remain explicit;
- factory selects exactly one configured client;
- local readiness reports missing server/model/tool support clearly.

**Gate P2:** Mocked parity permits integration work; it does not qualify a live provider. Record live qualification later and independently.

## Phase 3 — Analyst Roles and Grounded Synthesis

**Outcome:** Technical, fundamental, risk, forecast, and sentiment roles use only their allowed tools, disclose gaps, and feed a synthesizer that adds no new facts.

Likely files:

- `agents.py`;
- role and synthesis tests using scripted fake responses and tool evidence.

Proof:

- each analyst must call relevant evidence before factual completion;
- a failed tool narrows or blocks the analyst conclusion;
- headline-only inputs are not described as article-body analysis;
- bull/base/bear cases include conditions, horizon, and invalidation;
- conflicting analysts remain visible;
- target/stop/size can be explicitly unestablished without breaking the result contract.

**Gate P3:** Do not treat fluent role output as grounded unless the test can trace it to the supplied tool result.

## Phase 4 — Report and CLI

**Outcome:** A ticker override and the default roster orchestrate sequential analyses and render safe, traceable reports.

Likely files:

- `report.py`, `main.py`;
- `tests/test_report.py`, `tests/test_main.py`.

Proof:

- safe filename and exact destination;
- KRX/US currency regression fixtures;
- generated time versus data-as-of time;
- error/missing/`nan`/`inf` cannot masquerade as facts;
- research disclaimer, disagreements, and evidence limitations appear;
- CLI override and default roster parse deterministically without live calls.

**Gate P4:** A structurally valid report is not yet a live-qualified research result.

## Phase 5 — Qualification

Run in this order:

1. compile and deterministic tests;
2. live data tools for one KRX and one US ticker;
3. one ticker end-to-end with one selected provider and visible trace;
4. repeat provider-by-provider where credentials/services exist;
5. run the six-ticker roster only after single-ticker success;
6. audit at least one produced report with `$stock-analysis-review`.

Record each gate as `PASS`, `FAIL`, `NOT RUN`, or `BLOCKED`. Never compress four provider statuses into “multi-provider works.”

## Parked Until V1 Earns Them

- `research.py`, primary-source fetching, and shared evidence search;
- SQLite research memory and thesis-change tracking;
- portfolio ledger, target weights, and portfolio-aware actions;
- spreadsheet/dashboard UI;
- schedules, alerts, async/concurrent orchestration, autonomous trading;
- extra frameworks or a generalized plugin architecture.

These may be valuable. Their absence is not a current defect.

## Open Decisions

| ID | Decision | Resolve when |
|---|---|---|
| D1 | Exact default model IDs for each provider | At adapter implementation, against current native docs and account availability |
| D2 | Exact internal message/type representation | Phase 0, before adapters |
| D3 | Use standard-library `unittest` for the initial contract spine; add no test dependency without demonstrated need. | RESOLVED 2026-08-20 after the educational spike proved the approach |
| D4 | Precise normalized tool metadata fields | Phase 1, with one KRX and one US payload in hand |
| D5 | Backtest execution convention, benchmark, costs, and shorting policy | Before claiming `run_backtest` works |
| D6 | Recommendation schema when evidence is insufficient | Phase 3; preserve explicit unavailable state |
| D7 | Repository `env/` exists and runs Python 3.13.6 with `openai`, `yfinance`, and `dotenv` installed. | RESOLVED 2026-08-20 by direct environment check |

## Next Action

Use `.agents/commands/plan-next-slice.md` to re-check the tree and plan Phase 0 only. If Phase 0 has already landed, update this status and select the first incomplete gate rather than replaying the plan from memory.
