---
name: stock-analysis-review
description: Conduct a read-only, evidence-first review of the active stock-analysis rebuild, its plan, tests, tool traces, or generated reports. Use for code review, architecture audit, grounding or hallucination checks, provider-parity review, release readiness, report trustworthiness, and “what did we miss?” requests. Do not use to implement fixes or to review archived experiments as active code.
---

# Stock Analysis Review

## Mission

Decide what is actually trustworthy now. Re-verify claims against current files and observable behavior, trace report claims back to evidence, and give the user an ordered gate—not a reassuring summary.

This is a review skill. Do not edit application code, plans, or reports unless the user separately asks for changes. Suggested fixes are not authorization to implement them.

## Establish the Review Baseline

1. Name the review target: working-tree diff, named files, plan, tool transcript, report, or end-to-end run.
2. Read root `AGENTS.md` and `.claude/NewStart.md`; inspect current code and tests before repeating any claim from those documents.
3. Use `CLAUDE.md` to distinguish active work from archived reference code.
4. Treat `thoughts/codex-thoughts-at-start.md` as longer-term product judgment. Do not report missing post-v1 ideas as v1 defects.
5. State what was not available: live data, API keys, Ollama, a provider, a generated report, or a reproducible trace.

When revisiting an earlier review, confirm what still holds and focus on the delta. A previous finding is not current evidence.

## Apply Independent Lenses

Read only the role cards relevant to the target:

- [architecture-reviewer.md](../../agents/architecture-reviewer.md) for module boundaries, scope, state flow, and forbidden complexity.
- [grounding-reviewer.md](../../agents/grounding-reviewer.md) for market evidence, calculations, recommendations, and report claims.
- [provider-parity-reviewer.md](../../agents/provider-parity-reviewer.md) for native adapter and tool-round-trip semantics.
- [verification-reviewer.md](../../agents/verification-reviewer.md) for test meaning, failure injection, and qualification claims.
- [release-judge.md](../../agents/release-judge.md) only when consolidating a full review or explicit multi-agent review.

Run the lenses yourself by default. If the user explicitly requests subagents and the platform supports them, give each independent reviewer one complete role card and the same raw target. The parent must inspect the evidence, remove duplicates, resolve contradictions openly, and own the verdict.

## Evidence Rules

Match the verdict to what was actually examined: tool-loop correctness, report factual support, and predictive performance are different claims. Passing one does not establish the others. Do not require new research infrastructure for v1, but do require a report to limit conclusions to the evidence it actually has.

Check both provenance and whether the cited passage or tool value supports the claim. Several AI roles agreeing on one source do not provide independent corroboration. Separate a validated citation's existence from support for the inference it accompanies.

Every finding must contain:

- a stable ID;
- a concrete, falsifiable claim;
- a current `file:line`, test output, tool trace, or report excerpt;
- the exact state or input that triggers failure;
- the wrong outcome and why it matters;
- evidence status: observed, reproduced, inferred, or unverified;
- the smallest correction direction and a regression test.

Do not create a finding from style preference, hypothetical perfection, or a planning idea that v1 deliberately excludes. Separate verified defects, unverified risks, opportunities, and user decisions.

## Gate Findings by Consequence

- **G0 — trust blocker:** Fix before a generated recommendation or report can be treated as grounded. Examples: fabricated facts, wrong ticker/currency, look-ahead backtest, lost tool-call correlation, secrets in output, or an error payload treated as data.
- **G1 — expansion blocker:** Fix before claiming support for all providers, both KRX and US markets, or the default roster. Examples: one adapter drops parallel tool calls or one market's missing fields crash a tool.
- **G2 — hardening:** A concrete maintainability, diagnostic, cost, or edge-case defect that does not invalidate the current proven gate.

The condition after the label is mandatory. Severity without a named gate is unfinished judgment.

## Produce an Actionable Verdict

Lead with the bottom line and the highest live gate. List findings in gate order, then unverified risks and opportunities separately. End with the smallest safe correction order and one of:

- `READY FOR THE NEXT GATE`
- `CONDITIONALLY READY — <conditions>`
- `NOT READY — <blocking IDs>`
- `INSUFFICIENT EVIDENCE — <missing proof>`

If there are no findings, say so plainly and identify residual untested surfaces. Never turn “not tested” into “works.”

For a generated report, also read [report-integrity.md](references/report-integrity.md). For the exact finding and verdict shape, read [review-output.md](references/review-output.md).
