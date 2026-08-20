# Codex Thoughts at the Start

**Last updated:** 2026-08-20  
**Status:** Working product and architecture thesis—not a fixed specification.

## Purpose of This Note

This file records the reasoning behind the rebuild so future work does not lose the original intent. It should answer three questions: what are we building, why are we building it this way, and what should happen next?

Use the repository documents with this precedence:

1. [`CLAUDE.md`](../CLAUDE.md) describes the repository and its history.
2. [`.claude/NewStart.md`](../.claude/NewStart.md) is the active technical build plan.
3. This note holds longer-lived product judgment, research principles, and unresolved ideas.

When an idea here becomes a confirmed implementation decision, copy the decision—not the whole discussion—into `NewStart.md` before coding.

## Product Direction

The most useful version of this project is not a collection of AI agents giving `BUY/HOLD/SELL` opinions. It is a personal investment research and portfolio decision system, initially focused on US equities, that:

- gathers current, traceable evidence;
- converts evidence into an explicit investment thesis;
- tests that thesis from multiple viewpoints;
- connects conclusions to the user's actual portfolio;
- remembers what was believed, decided, and later learned.

The key output is not a prediction. It is a better decision record: the evidence available at the time, the assumptions made, what would invalidate them, and what action—if any—fits the portfolio.

## Lessons From the Reference Material

The four [김현준 주식 - 꿀직장TV videos](#references) and [`thepublic-example.txt`](../thepublic-example.txt) suggest a consistent research style.

First, surface similarities are dangerous. A historical price chart, familiar cycle, or company analogy is only useful after comparing business model, capital intensity, shareholder returns, competitive structure, and market conditions.

Second, distinguish what the market already knows from what is genuinely new. A strong future quarter can be correctly forecast and still be fully reflected in the price. Profit level, profit-growth rate, and share-price peaks are different events.

Third, capital expenditure needs interpretation. High capex can be a valuable reinvestment opportunity, a defensive necessity, or late-cycle overbuilding. The important questions are expected long-term return, funding capacity, and whether the spending widens a moat.

Fourth, normalize cyclical businesses. Current peak earnings should not automatically receive a normal multiple. Use through-cycle earnings and let durability, cyclicality, balance-sheet risk, and competitive advantage influence the valuation range.

Finally, research should move from macro flows to industries, beneficiaries, companies, and portfolios. The recurring question is: **where is value being created or transferred, and what structural change is hidden beneath the headline?**

## Why Generic Web Search Is Not Enough

Adding a `web_search()` tool by itself will mostly produce better-sounding summaries. The system needs a research protocol around search:

```text
Investment question
  -> search plan
  -> original sources
  -> evidence and contradiction ledger
  -> specialist analysis
  -> skeptical review
  -> portfolio-aware action
  -> cited report and decision record
```

Research should prefer primary material: SEC filings, company investor-relations pages, earnings releases and calls, official industry data, and original analyst research where accessible. News and social sources are useful for discovery and sentiment, but should not silently become proof.

Every meaningful claim should carry structured provenance:

```text
claim | source URL | publication date | fact/opinion
supports/contradicts | confidence | retrieved_at | valid_as_of
```

Each analyst should actively search for disconfirming evidence. The final synthesis must show important disagreements and uncertainty instead of averaging them into false confidence. Sources also need freshness rules, because an old fact can be correctly cited but no longer decision-useful.

`LLM_PROVIDER` and `SEARCH_PROVIDER` should be separate settings. Anthropic, OpenAI, Gemini, and Ollama do not have identical search behavior or terms. A shared evidence set is necessary when comparing model quality; otherwise we are comparing both models and their different information access. Ollama inference may be local, but its hosted web-search capability still requires an external service, so “local model” must not be presented as “fully offline research.”

## Recommended Architecture

Keep the first implementation understandable and mostly synchronous. The current flat Python design remains appropriate, with three additions:

```text
config.py       environment, provider, and policy settings
llm.py          common model interface for four providers
data_tools.py   market and fundamental data helpers
research.py     search, fetching, source ranking, evidence ledger
agents.py       bounded analyst roles and skeptical review
report.py       cited research and decision reports
storage.py      SQLite schema, migrations, and repositories
portfolio.py    positions, exposure, targets, and action context
main.py         CLI orchestration
```

Avoid restoring CrewAI, LangChain, a vector database, async orchestration, or a large UI until a simpler implementation proves they solve a real bottleneck. Agent roles should be explicit functions with typed inputs and outputs, not autonomous personalities exchanging unstructured prose.

The report should separate:

- facts and source-backed observations;
- interpretation and assumptions;
- bull, base, and bear cases;
- valuation inputs and sensitivity;
- catalysts, risks, and thesis invalidation conditions;
- missing or conflicting evidence;
- portfolio implication and next review trigger.

## Portfolio and Management Layer

The referenced Google Sheet appears to represent the desired management experience, but it was not anonymously accessible during review. The exact tabs and workflows should be inspected after the sheet is shared as link-viewable or exported as `.xlsx`/CSV.

Conceptually, many spreadsheet tabs should be views over one reliable data model—not separate copies of truth. SQLite is a suitable first source of truth, with spreadsheet import/export and a dashboard added later.

Minimum entities and views:

- accounts and immutable transactions;
- derived positions, cash, cost basis, realized/unrealized returns, and FX;
- watchlist and company thesis records;
- catalysts, risks, invalidation conditions, confidence, horizon, and review date;
- sector, geography, currency, factor, and concentration exposure;
- target weights and a rebalance queue;
- research runs, claims, sources, and model/provider metadata;
- a decision journal linking research to an explicit action.

Portfolio actions should be `WATCH`, `INITIATE`, `ADD`, `HOLD`, `TRIM`, or `EXIT`. Each proposed action should include a target weight or range, rationale, invalidation condition, and review trigger. The same company can be attractive in isolation but inappropriate in an already concentrated portfolio, so research and portfolio context must remain distinct but connected.

Positions must only change through an explicit user action or an imported broker transaction. Research agents should never place trades or quietly mutate the ledger.

## Suggested Delivery Sequence

### Phase 1 — Grounded single-company research

Build the provider abstraction, common research backend, evidence schema, contradiction search, and one cited single-ticker report. Support point-in-time timestamps from the beginning to reduce look-ahead errors later.

**Success condition:** another person can trace each important factual claim to a source and understand why the conclusion could be wrong.

### Phase 2 — Memory and thesis tracking

Persist companies, research runs, evidence, theses, catalysts, risks, and review dates in SQLite. Add source freshness and thesis-change comparisons.

**Success condition:** a new report can explain what changed since the previous report rather than starting from zero.

### Phase 3 — Portfolio ledger and decision journal

Add accounts, transactions, positions, exposures, target weights, and portfolio-aware action language.

**Success condition:** the system can show how a research conclusion affects concentration, cash, and existing theses without altering holdings automatically.

### Phase 4 — Usable management interface

Add spreadsheet imports/exports and then a focused dashboard with research, thesis, portfolio, exposure, and review-queue views. Reproduce useful concepts from the example Sheet after its layout is available; do not mechanically copy every tab.

### Phase 5 — Automation only where earned

Add scheduled refreshes, alerts, broker integrations, or richer orchestration only after manual workflows are stable and auditable.

## Guardrails and Non-Goals

- Never invent a missing fact or citation; report the gap.
- Preserve publication, retrieval, and “valid as of” dates.
- Store the evidence used for a decision so later results do not rewrite history.
- Treat paywalls, failed fetches, and unavailable originals as explicit limitations.
- Put bounds on search depth, tool calls, tokens, latency, and cost.
- Keep API keys in environment variables and portfolio data local by default.
- Present outputs as research support, not personalized financial advice.
- Do not optimize for a confident recommendation; optimize for a traceable decision.
- Do not build the complete portfolio UI before the research and source model is stable.

## Near-Term Decisions to Resolve

1. Choose the first shared search/fetch backend and define its source-ranking rules.
2. Define the v1 evidence, thesis, research-run, and report schemas.
3. Choose one US stock as the end-to-end reference case.
4. Decide the minimum valuation methods needed for that case rather than building a universal valuation engine.
5. Obtain a viewable/exported copy of the portfolio Sheet and identify the few workflows worth reproducing first.
6. Decide which provider is the default while keeping the report contract provider-neutral.

## References

- [Apple three-stage and Samsung analogy](https://www.youtube.com/watch?v=sB53fwLcQcc)
- [Semiconductor cycle and peak interpretation](https://www.youtube.com/watch?v=BJwLNXSniP4&t=1458s)
- [Buffett, Alphabet, free cash flow, and capex](https://www.youtube.com/watch?v=dexBntXfNbY)
- [Long-term agreements, AI capex, and normalized PER](https://www.youtube.com/watch?v=t5h5ybKJ5WY)
- [Portfolio-management Google Sheet](https://docs.google.com/spreadsheets/d/1lgU20V2zDO7OBfY44y6E_xDtPb34beeet6GnWi1uSc0/edit?gid=421427572#gid=421427572) — access was unavailable during review.
- [OpenAI web search](https://openai.com/index/new-tools-for-building-agents/)
- [Anthropic web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
- [Gemini grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search)
- [Ollama web search](https://docs.ollama.com/capabilities/web-search)
