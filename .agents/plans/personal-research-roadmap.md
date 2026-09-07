# Personal Stock Research Roadmap

Date: 2026-09-08.
Status: recommendations for the owner; not a replacement for approved application scope.

## The Product Worth Building

Build a personal research workbench that helps answer: what does this company do, why might it become more valuable, what price makes that attractive, what could invalidate the idea, and what changed since the last review?

Trust should mean that facts and calculations are checkable, assumptions are visible, and missing evidence changes the output. It cannot mean that every investment works. The owner remains responsible for the decision; the software should make the reasoning easier to inspect.

The practical starting point is a small watchlist, one repeatable company note, and a weekly review habit. Initially use one familiar US company, such as Alphabet, as an engineering reference case, then Samsung Electronics to expose Korean-market differences. This is a choice of research examples, not a recommendation to buy either company. The approved six-company default roster remains unchanged; AAPL can later be a ticker override.

## What Exists Now

Inspected on 2026-09-08: the six planned production modules are absent. `codex-main.py` and `claude-main.py` are educational experiments; `tests/test_codex_main.py` exists. Their live behavior was not qualified in this planning pass. `archive/` contains earlier experiments and remains reference-only.

The approved implementation sequence is [rebuild-execution.md](rebuild-execution.md). This roadmap adds product judgment and research procedures available to the assistant; it does not add primary-source fetching, storage, or portfolio functionality to the application.

## What to Learn First

The owner does not need a complete finance course. Learn enough to challenge each report:

1. Business economics: who pays, why customers choose the business, and what controls revenue and margins.
2. Cash versus profit: reported earnings and cash available after reinvestment can differ substantially.
3. Price versus value: a strong business may still require unrealistic future performance at the observed price.
4. Supply and competition: a growing industry need not produce growing profits for every supplier.
5. Portfolio exposure: a company can be attractive individually while increasing an existing concentration.

Professional golf experience can help generate research questions about equipment, distribution, facilities, or customer behavior. Those observations are hypotheses to verify against company economics; they are not an automatic investing edge.

Time horizon, liquidity needs, and ability to withstand losses shape allocation. A collection of familiar technology names is not automatically diversified. Broad funds are worth considering separately from the stock-research project; narrowly focused funds may also concentrate exposure. See [Investor.gov on allocation and diversification](https://www.investor.gov/introduction-investing/getting-started/asset-allocation). No personal allocation or position size is proposed without the owner's circumstances.

## Use the Reference Investor as a Source of Questions

The supplied July and September newsletters and translated semiconductor video emphasize interpreting business changes beneath headlines. Preserve the questions, then independently test their answers. The local texts were inspected; their company figures were not independently verified in this planning task. The YouTube channel page could not be fetched, so this is not a claim to have watched its catalogue.

For example, the semiconductor transcript suggests asking how capacity lead times, competing supply, pricing, and contract coverage affect a demand boom. The September newsletter suggests following AI revenue through capital spending and financing to cash flow. Its retail discussion suggests distinguishing initial store stocking from repeat purchases. The reusable protocol lives in [research-method.md](../skills/stock-research/references/research-method.md).

## First Useful Workflow

Recommended starting cadence: a 45-60 minute weekly research session, adjusted to the owner's actual time. Revisit a thesis after earnings or a material development; daily price commentary need not be the default.

1. Select one question, such as whether increased investment is strengthening a company's earning power.
2. Read a dated evidence note: business model, recent financial context, price, and material gaps.
3. Inspect the handful of facts that carry the conclusion using their original sources.
4. Examine the strongest objection and the assumptions behind any valuation range.
5. Record the current view and a specific observation that would change it.
6. At the next review, compare facts and assumptions with the dated earlier note.

The first reference note can be produced manually with the assistant's research skill before building automatic document retrieval. This reveals what the application actually needs to produce. It does not bypass the engineering gates for application code.

## Architecture and Growth

Recommendation: retain Python for the research core. It matches the repository and its existing numerical/data work. TypeScript becomes useful if a richer browser interface is desired; that interface can call Python without rewriting financial calculations.

The initial application structure remains:

```text
stock-analysis/
  config.py       settings and validation
  llm.py          common model contract and provider adapters
  data_tools.py   retrieval, normalization, financial calculations
  agents.py       bounded tool use and analytical reasoning
  report.py       render supplied evidence and conclusions
  main.py         command-line entry point
  tests/          fake model/data boundaries and meaningful failure cases
  reports/        generated notes, with ignore policy checked before saving
  archive/        historical experiments
  .agents/        development skills, roles, plans, and saved prompts
```

These six modules are planned, not newly created by this roadmap. Keep model reasoning, financial arithmetic, and rendering separate. Store compact provenance in tool results from the first implementation; a citation added after the conclusion is not equivalent.

Later, once the responsibilities exist and file size or change coupling makes the flat layout awkward, move them into a package. An illustrative destination is:

```text
src/stock_analysis/
  providers/      native model adapters
  data/           market data and filing readers
  research/       claims, thesis, valuation, review
  portfolio/      transactions, positions, exposure, when needed
  reporting/      output formats
  cli.py
tests/
web/              optional TypeScript interface
```

Do not create these folders as empty promises. Add SQLite when comparing and querying dated research records becomes a real requirement. Add CSV/spreadsheet views before a custom dashboard if they solve the interaction. Consider a server database only when shared access and concurrent writes justify it; consider concurrency only after measuring delays. Ordinary personal watchlist use does not justify distributed infrastructure by itself.

Keep all four provider adapters as the approved goal, while proving one provider end to end first. Compare providers using the same evidence and judge citation support, arithmetic, useful objections, latency, and cost. Several models agreeing is not independent financial evidence. Select model IDs at implementation time rather than hard-coding a fashionable choice into this roadmap.

## Milestones and Proof

| Milestone | Useful result | Acceptance evidence | Status |
|---|---|---|---|
| Reference research note | Owner can understand and challenge one company thesis | Material facts trace to passages; assumptions and strongest objection visible | Recommended manual next step |
| Lean application | One ticker runs through tools, analysis, and a dated report | Existing engineering gates, explicit data failures, currency/period checks, inspected live trace | Approved, unimplemented |
| Primary-source integration | Thesis-critical company facts supported by original disclosures | Correct issuer/period, passage retained, conflicts shown, missing originals disclosed | Proposed after v1; manual research can be used now |
| Thesis history | A new note explains what changed | Prior evidence and assumptions remain available; corrections do not rewrite history | Proposed |
| Portfolio context | Research shows existing exposure and conditional implications | Reconciled holdings, cash, currency and no inferred transactions | Proposed |
| Convenient interface | Owner completes the weekly review with less effort | Observed usability improvement; numbers agree with underlying records | Proposed |

There are two different acceptance questions. Engineering asks whether data and tool calls flow correctly. Research asks whether the evidence supports the investment reasoning. A yfinance-only report can pass the first while lacking material evidence for the second. Full primary-source automation can wait; missing primary evidence must still limit the report's claims.

Use [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [OpenDART](https://engopendart.fss.or.kr/guide/main.do?apiGrpCd=DE003), and company investor relations for disclosures. [yfinance's own documentation](https://ranaroussi.github.io/yfinance/) describes its research/personal-use role and lack of affiliation with Yahoo. It is a practical prototype source, not a substitute for verifying consequential facts.

Review a small collection of real reports plus deliberate failures: missing Korean data, stale statements, incorrect currency, tool failure, unsupported target, misleading citation, and conflicting sources. Fix consequential factual or arithmetic errors before relying on a report. A prompt telling a model to be careful is not a tested enforcement mechanism. Track research quality separately from later investment returns; short-term gains do not establish an edge.

## Decisions to Refine With the Owner

The provisional framing is personal, longer-term company research with manual decisions. Clarify the intended holding horizon, US/Korean priority, weekly time, and preferred reading interface before tailoring the product further. Resolve model/API spending budget before an extended paid run. Portfolio sizing needs additional financial context only when that feature is in scope.

Recommended next deliverable: one understandable reference note for Alphabet or Samsung, using the new research protocol, then use that note to assess whether the approved report contract needs revision. Application implementation remains governed by the existing execution plan.
