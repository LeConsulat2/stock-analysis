# Stock Analysis Agent Pack

**Established:** 2026-08-20  
**Research guidance extended:** 2026-09-08

**Scope:** Repository-local operating guidance only; no application code is implemented here.

This folder turns the repository's design intent into a repeatable way to plan, build, verify, and review the rebuild. It borrows the strongest habits from the Fable material—re-verify instead of inheriting, bind severity to a real gate, keep specialist lenses independent, preserve human rulings, and end with a bottom line—without copying product-specific vocabulary that does not belong in a stock-analysis CLI.

## What Is Here

```text
.agents/
|-- README.md
|-- plans/
|   |-- rebuild-execution.md
|   `-- personal-research-roadmap.md
|-- commands/
|   |-- plan-next-slice.md
|   |-- implement-next-slice.md
|   |-- review-current-change.md
|   |-- qualify-workflow.md
|   |-- research-company.md
|   |-- challenge-thesis.md
|   `-- plan-personal-workflow.md
|-- agents/
|   |-- architecture-reviewer.md
|   |-- grounding-reviewer.md
|   |-- provider-parity-reviewer.md
|   |-- verification-reviewer.md
|   |-- release-judge.md
|   |-- company-investigator.md
|   `-- thesis-challenger.md
`-- skills/
    |-- stock-analysis-rebuild/
    |-- stock-analysis-review/
    |-- stock-analysis-roadmap/
    `-- stock-research/
```

The research additions support work in an assistant session. They do not implement new application agents, search tools, or portfolio features.

## Use It

- Invoke `$stock-analysis-rebuild` for planning, implementation, modification, or diagnosis of the active rebuild.
- Invoke `$stock-analysis-review` for a read-only review, report audit, provider-parity check, or readiness verdict.
- Invoke `$stock-analysis-roadmap` for product direction, practical starting milestones, architecture tradeoffs, and growth planning.
- Invoke `$stock-research` for a company thesis, a newsletter/video claim, or a dated research update with primary evidence and a skeptical challenge.
- Use the command playbooks as saved starting prompts. Give the playbook a target when needed.
- Treat `plans/rebuild-execution.md` as the dated execution and ruling ledger. Root `AGENTS.md` and `.claude/NewStart.md` remain the v1 product/architecture authorities.

The files are repository-local skill definitions. This documentation does not verify whether the current client has refreshed its skill picker; a saved playbook can also be supplied directly by path.

Future models should load only the reference routed by the selected skill. Do not read every file in this pack for every task. The pack is designed so Luna or Terra can follow explicit contracts, failure cases, and proof gates without being asked to imitate a persona.

## Important Format Boundary

Verified against official Codex documentation on 2026-08-20:

- Repository skills are discovered from `.agents/skills/<name>/SKILL.md`, which is why the new skills live there.
- Native project custom-agent configurations are discovered from `.codex/agents/*.toml`, not `.agents/agents/`.
- Codex slash commands are product commands; this repository location does not register custom slash commands.

Re-check time-sensitive formats in the official [skills](https://learn.chatgpt.com/docs/build-skills), [subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents), and [developer commands](https://learn.chatgpt.com/docs/developer-commands) documentation before converting this pack.

This pack retains its existing `.agents` layout: `agents/*.md` are reusable **role cards**, and `commands/*.md` are **prompt playbooks**. The skills route to the relevant cards. These files do not register runtime agents or slash commands. A future conversion to native agent configuration should verify the then-current format and match the user's requested scope.

## Operating Principles

1. Current files outrank remembered claims about current files.
2. Approved v1 scope outranks attractive later ideas.
3. An error or absence is evidence of a limit, not a neutral value.
4. Every material report claim needs identity, time, currency/unit, source, and a traceable result.
5. Provider parity means equivalent state transitions, not identical prose.
6. Review and implementation stay separate unless the user asks for both.
7. Deterministic tests precede live data; one ticker precedes the roster.
8. “Not run” and “insufficient evidence” are valid outcomes. False completion is not.

## Sources of Truth

Use this order for different questions rather than pretending one document owns everything:

- root `AGENTS.md`: current contributor constraints;
- `.claude/NewStart.md`: approved v1 technical design;
- current code/tests/tool traces: present implementation behavior;
- `CLAUDE.md`: repository and archive history;
- `thoughts/codex-thoughts-at-start.md`: longer-term product thesis and unresolved ideas;
- `.agents/plans/rebuild-execution.md`: implementation sequence, gates, and dated rulings.
- `.agents/plans/personal-research-roadmap.md`: dated product recommendations and proposed later milestones; not an implementation authorization.

When two sources genuinely conflict, surface the choice. Do not resolve it by quietly expanding scope.
