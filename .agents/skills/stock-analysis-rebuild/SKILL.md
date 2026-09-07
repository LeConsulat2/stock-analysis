---
name: stock-analysis-rebuild
description: Plan, implement, modify, or debug the active flat-Python stock-analysis rebuild in this repository. Use for config.py, llm.py, data_tools.py, agents.py, report.py, main.py, provider adapters, dynamic tool calling, yfinance-backed evidence, reports, and rebuild tests. Do not use for archived CrewAI/LangChain experiments or for general investment advice.
---

# Stock Analysis Rebuild

## Mission

Build the smallest defensible slice of the active rebuild. Preserve a visible chain from market evidence to analyst claim to synthesis to report, while keeping provider-specific mechanics behind one bounded interface.

This skill is model-neutral. A smaller model should be able to succeed by following the contracts and gates rather than guessing the architecture.

## Establish What Is True Now

Before planning or editing:

1. Read the root `AGENTS.md` for current repository rules.
2. Read `.claude/NewStart.md` for the approved v1 design.
3. Inspect the current files and tests. A planning document describes intent, not proof that code exists or still behaves as described.
4. Use `CLAUDE.md` for repository history and archive interpretation.
5. Treat `thoughts/codex-thoughts-at-start.md` as a product north star, not permission to add `research.py`, storage, portfolio features, web search, or other post-v1 scope.
6. Treat `archive/` as read-only design evidence. Never make it the active entry point or restore its dependency stack incidentally.

When sources disagree materially, identify the conflict and preserve the narrower approved v1 unless the user rules otherwise. Do not silently turn an idea into a requirement.

## Choose the Work Path

- **Plan:** Reconcile the requested outcome with the current tree, select one dependency-complete slice, and define proof before implementation.
- **Implement or modify:** Make the smallest coherent change, including tests at the network and model boundaries.
- **Diagnose:** Reproduce or trace the failure and report the cause; do not fix unless requested.
- **Review:** Use `$stock-analysis-review`. A review request alone does not authorize edits.

Do not delegate unless the user explicitly asks for subagents or parallel agent work. If delegation is requested, reserve final architectural and evidence judgments for the parent agent.

## Preserve the V1 Invariants

- Active code remains the six flat root modules: `config.py`, `llm.py`, `data_tools.py`, `agents.py`, `report.py`, and `main.py`, plus `tests/`.
- Providers are `claude`, `openai`, `google`, and `local`, implemented through native SDK adapters behind one `LLMClient` contract.
- Do not add CrewAI, LangChain, LiteLLM, vector databases, memory/RAG, async orchestration, or speculative framework layers.
- The agent loop is dynamic and bounded: the model requests tools, the harness executes only registered tools, returns correlated results, and stops on final text or an explicit terminal failure.
- An analyst must obtain relevant real tool evidence before making a factual market claim. A failed or empty tool result is a limitation, never permission to improvise.
- Tool responses are compact and JSON-serializable. Unstable `yfinance` access returns a structured `{"error": ...}` payload instead of crashing the workflow.
- The synthesizer may compare and reason over analyst outputs; it may not introduce new external facts or hide disagreement.
- A successful tool call establishes data access, not support for every later sentence. Material claims must match the actual returned evidence; numerical calculations belong in deterministic tools. Prompt instructions alone do not prove these properties are enforced.
- Agent roles and provider agreement are not independent evidence. Preserve originating sources and unresolved contradictions instead of turning agreement into a confidence score.
- V1 orchestration is sequential. Local inference uses Ollama's OpenAI-compatible endpoint and an exact installed model tag confirmed at runtime.
- Reports go to `reports/<TICKER>_analysis_<YYYYMMDD_HHMMSS>.md` and state that they support research rather than provide financial advice.

## Work in Evidence-Bearing Slices

Before editing, state the observable outcome, owning modules, contract affected, realistic failure scenario, and proof command. Prefer a walking skeleton that closes one real loop over many disconnected stubs.

Keep these boundaries sharp:

- provider adapters translate wire formats;
- `agents.py` owns loop state and role policy;
- `data_tools.py` owns market retrieval and calculations;
- `report.py` renders supplied results and performs no research;
- `main.py` orchestrates and contains no hidden analysis logic.

Never report a skipped or unavailable test as passed. Distinguish deterministic verification, live-data smoke tests, hosted-provider checks, and local-Ollama checks.

## Load Only the Relevant Reference

- For module ownership, dependency order, scope conflicts, or slice planning, read [architecture-contract.md](references/architecture-contract.md).
- For `llm.py`, `agents.py`, provider adapters, or tool-call failures, read [provider-tool-protocol.md](references/provider-tool-protocol.md).
- For `data_tools.py`, analyst prompts, synthesis, recommendations, or report claims, read [market-data-grounding.md](references/market-data-grounding.md).
- For tests, smoke checks, release qualification, or claims that something works, read [verification-matrix.md](references/verification-matrix.md).

Load more than one only when the change crosses those boundaries.

## Handoff

End implementation work with:

- the outcome now supported;
- files changed and why;
- verification actually run and its result;
- live providers or markets not verified;
- any decision still requiring the user;
- the next dependency-complete slice, if one is obvious.
