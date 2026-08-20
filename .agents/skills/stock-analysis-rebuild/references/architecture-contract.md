# Architecture Contract

Read this for planning, module ownership, dependency order, or scope reconciliation.

## Authority and Drift

The approved v1 is the six-module flat design in root `AGENTS.md` and `.claude/NewStart.md`. The current tree proves implementation state. `CLAUDE.md` explains why archived approaches were rejected. `thoughts/codex-thoughts-at-start.md` records a broader product thesis; its proposed research, storage, portfolio, and UI modules remain later ideas until the user promotes them into the active plan.

Do not keep saying “planning only” after files land. Re-check the tree. Conversely, the existence of code does not make an accidental architectural change approved.

## Ownership Boundaries

| Module | Owns | Must not own |
|---|---|---|
| `config.py` | Typed analysis/provider settings, environment parsing, validation | SDK calls, market retrieval, orchestration |
| `llm.py` | Provider-neutral request/response types, four native adapters, client factory | Analyst policy, tool execution, financial calculations |
| `data_tools.py` | Tool specs/registry, yfinance access, normalization, calculations | Provider wire formats, prose recommendations |
| `agents.py` | Bounded tool loop, tool dispatch, role prompts, synthesis inputs | Direct yfinance/SDK calls, report file I/O |
| `report.py` | Deterministic Markdown rendering and safe filename creation | New research, hidden calculations, model calls |
| `main.py` | CLI parsing and sequential orchestration | Business rules duplicated from other modules |

Keep dependencies pointing toward contracts and away from orchestration. A useful shape is `config` -> `llm`/`data_tools` -> `agents` -> `report` -> `main`, with shared types placed deliberately to avoid cycles. Do not create a seventh production module merely to avoid making a clear ownership decision.

## Dependency-Complete Delivery Order

Use the living plan at `.agents/plans/rebuild-execution.md`. Its core logic is:

1. Prove the provider-neutral tool loop with a fake client and fake tool before involving any SDK or network.
2. Prove market-data normalization and calculations with mocked yfinance shapes, then one KRX and one US smoke case.
3. Make each provider adapter pass the same mocked conformance contract before comparing live models.
4. Add analyst roles only after the tools and loop can demonstrate evidence-bearing round trips.
5. Add synthesis, report, and CLI after their inputs have stable contracts.
6. Qualify one ticker and one provider before the default roster; qualify providers independently rather than hiding missing credentials.

This order makes failures attributable. Building all tools, adapters, prompts, and reporting at once produces an end-to-end failure with no trustworthy boundary.

## Walking-Skeleton Rule

A slice is complete only if one observable behavior crosses every boundary it claims to establish. For the first slice, a fake model should request a registered fake tool, receive a correlated JSON result, and return final text; tests must also prove unknown tools, malformed arguments, and turn exhaustion do not become false success.

Avoid placeholder methods that return plausible values. If a later boundary is not implemented, make it explicit and unreachable from a claimed working path.

## Scope Tensions Already Present

- NewStart requires `BUY/HOLD/SELL` plus target, stop, and position size. The system may keep these report fields, but must say `not established from available evidence` when no valuation, technical level, or portfolio-risk basis exists. A filled field is not more complete than an honest gap.
- The thoughts note prefers portfolio actions such as `WATCH`, `INITIATE`, and `TRIM`. Those require portfolio state and remain post-v1.
- The thoughts note proposes primary-source web research and an evidence ledger. Preserve provenance-friendly tool contracts now, but do not add a research subsystem to v1 without a user ruling.
- The archive contains reusable formulas, not trusted modules. Port the idea, add tests, and correct known defects; do not copy architecture or dependency assumptions wholesale.

## Definition of Done for Any Slice

- The public contract is typed and has one owner.
- Success, partial data, and terminal failure are distinguishable.
- The realistic failure scenario has a deterministic regression test.
- No forbidden dependency or archive mutation was introduced.
- Logs expose the state transition without exposing credentials or dumping large payloads.
- Documentation or the living plan is updated only when a real ruling or implemented state changed.
