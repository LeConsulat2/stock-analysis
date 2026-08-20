# Report Integrity Review

Use this reference when reviewing a generated Markdown report, synthesis output, or recommendation contract.

## Trace Claims Backward

Start with the most consequential statements: recommendation, current price, target, stop, position size, catalysts, risk labels, and forecast. For each, trace:

```text
report sentence -> synthesizer input -> analyst statement -> tool result -> source field and as-of time
```

A broken or missing link is a grounding finding. A citation-like label without retained evidence is not a link.

## Integrity Checks

- Ticker, company, exchange, currency symbol/code, and numeric units agree.
- Generated time is separate from each source's data-as-of time and uses an explicit timezone.
- Current price is actually current enough for the stated use and says raw versus adjusted where relevant.
- Financial values identify period and currency; ratios do not mix incompatible periods.
- News analysis does not imply article-body review when only headlines were supplied.
- Analyst targets show count, date range, and that they are external consensus—not the system's valuation.
- Risk metrics state their assumptions and do not become categorical labels through arbitrary undocumented thresholds.
- Backtest results disclose no-look-ahead treatment, sample window, costs, benchmark, closed-trade count, and limitations.
- Bull/base/bear cases have conditions and invalidation evidence rather than decorative percentage ranges.
- Disagreement and missing evidence remain visible after synthesis.
- `nan`, `inf`, `None`, exception text, and tool-error payloads are not rendered as normal facts.
- Recommendation horizon and invalidation conditions are present.
- Numeric target, stop, and position size have a named method and inputs. Otherwise they are explicitly unavailable.
- The report says research support, not financial advice.

## Known Regression Anchors

The committed legacy report `reports/066570.KS_analysis_20250127_225937.md` demonstrates failures worth preserving as tests: dollar symbols on a KRW security, generic catalysts with no source, precise entry/exit/stop values without a visible method, and a blanket 5% portfolio allocation without portfolio context.

Do not review a new report by asking whether it sounds more polished than that file. Review whether every material claim is traceable and decision-useful.

## Verdict

A report is not trusted when any G0 claim lacks evidence, uses the wrong instrument/currency/time basis, or presents a failed calculation as a recommendation. A report may be structurally valid but still `INSUFFICIENT EVIDENCE` when live data or tool traces were not retained.
