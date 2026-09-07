# Market-Data Grounding Contract

Read this for data tools, financial calculations, analyst prompts, synthesis, or report generation.

## Evidence Before Interpretation

Every claim-bearing tool payload should make the following recoverable, using compact fields appropriate to the tool:

- requested and resolved ticker;
- source/provider;
- retrieval time and data “as of” time;
- request window or reporting period;
- currency and units where numeric values need them;
- normalized data;
- missing fields and warnings;
- a structured error when the request failed.

Unknown, unavailable, not applicable, and zero are different states. Preserve that distinction. An old but correctly cited value is still stale; a current value without units can still be misleading.

Source provenance and claim support are separate checks. A source can be correctly cited while not supporting the sentence, and an official management forecast remains a forecast. Reposts of the same announcement are not independent corroboration. A tool-call requirement is necessary but insufficient: audit what the analyst inferred from its result.

Do not infer subsidiary or segment economics from a parent ticker's consolidated data. User-supplied newsletters and transcripts provide research leads and attributed opinions; uncertain translations and hypothetical prices cannot become verified company facts. New source-fetching infrastructure remains subject to approved scope; if required evidence is unavailable, narrow the claim.

## Identity, Time, Currency, and Units

- Confirm the returned instrument matches the requested Yahoo symbol. KRX suffixes are part of identity.
- Use exchange or source timestamps and explicit timezone handling. A New Zealand run date is not automatically the market-data date.
- Obtain currency from source metadata when possible. Never hardcode KRW because early examples used Korean tickers, and never render `$` for `005930.KS` or `066570.KS`.
- Distinguish price currency from financial-statement presentation currency and from any converted portfolio currency.
- State whether price history uses raw close or adjusted close. Splits and dividends can materially change returns and backtests.
- State annual versus quarterly statement period and the source column date. Do not compare a trailing price with an unlabeled historical accounting value.

The existing 2025 report that prints Korean prices with dollar signs is a concrete regression case. A report test should fail if the symbol and source currency disagree.

## Tool-Specific Failure Modes

### Price and technicals

- Require enough observations for every indicator; warm-up rows are not signals.
- Return the exact window and last valid observation date.
- Keep raw indicator values separate from derived direction labels.
- Avoid presenting support, resistance, target, or stop levels unless the calculation is defined and the value exists.

### Financial statements

- yfinance row labels and availability vary across issuers and markets. Map known alternatives deliberately and report missing rows.
- Guard division by zero, negative-denominator interpretation, mixed periods, and mismatched units.
- Do not turn an absent KRX analyst field into evidence that analyst sentiment is neutral.

### Risk metrics

- State return frequency, annualization assumption, sample size, VaR confidence and sign convention, and risk-free-rate assumption.
- Handle zero variance and insufficient history without `inf`, `nan`, or a confident prose rating.
- VaR is a historical estimate under assumptions, not a maximum possible loss.

### News and outlook

- Normalize nested yfinance news shapes, missing links, publisher, and publication time; deduplicate repeated stories.
- A headline is evidence that something was reported, not proof that the underlying assertion is true.
- Analyst coverage is uneven, especially across markets. Include observation count and date range before summarizing upgrades or targets.
- Keep consensus price target distinct from the system's own valuation and label upcoming earnings dates as provisional when the source does.

### Backtest

- Compute signals only from information available at that timestamp and execute no earlier than the next permitted price; otherwise the test looks ahead.
- Define position sizing, trade accounting, open-position treatment, fees/slippage, benchmark, and whether shorting is allowed.
- Count closed trades when calculating win rate. An entry event is not a losing trade merely because it has no P&L field.
- Separate in-sample description from out-of-sample evidence. A backtest over the same window used to design thresholds is not predictive validation.
- Reject `nan` and `inf` before JSON serialization and report insufficient samples honestly.

The archived backtest is design evidence, not a trusted implementation: it trades at the same bar's close, counts all trade events in the win-rate denominator, omits quantity and costs, and can leave positions open. Porting it “verbatim” must not bypass correctness tests.

## Analyst and Synthesis Rules

- Each specialist names the tools actually used and the material gaps returned.
- Facts, calculations, interpretations, and scenarios must remain distinguishable.
- Bull/base/bear cases are conditional scenarios, not three unsupported predictions. Each needs a catalyst, invalidation condition, horizon, and evidence link.
- The sentiment analyst may assess the supplied headlines; it may not claim to have read article bodies that were not fetched.
- The synthesizer may reconcile conflicting evidence and lower confidence. It may not invent a catalyst, target, stop, portfolio percentage, or current fact absent from analyst evidence.
- `BUY/HOLD/SELL` must include horizon, evidence date, uncertainty, and what would change the call. When evidence is insufficient, the honest result is “not established,” even if the report schema has a field.
- Position size requires portfolio value, risk budget, and a defined sizing rule. Without them, give a conditional method or mark it unavailable—never a generic “5%”.

## Report Contract

Every report should expose:

- generated time and market-data as-of time;
- ticker/company identity and currency;
- analyst sections with evidence status and limitations;
- disagreements and missing evidence;
- recommendation horizon and invalidation conditions;
- whether target/stop/size are observed, calculated, consensus, conditional, or unavailable;
- a research-not-financial-advice label.

Rendering must not silently convert an error, missing value, `nan`, or `inf` into polished prose. The more consequential the recommendation, the more visible the evidence chain and uncertainty must be.
