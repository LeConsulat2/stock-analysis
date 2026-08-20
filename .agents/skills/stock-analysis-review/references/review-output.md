# Review Output Contract

Use this shape for substantial reviews. Keep it compact when only one or two findings exist.

## Bottom Line

```markdown
Verdict: NOT READY — GND-1, PRV-2
Highest live gate: G0 — reports are not trustworthy until currency and tool-result correlation are fixed.
Scope reviewed: [diff/files/report/run]
Evidence unavailable: [providers/live data/etc.]
```

## Finding

```markdown
### GND-1 — KRX prices are rendered as USD
Gate: G0 — fix before any report can be treated as grounded.
Evidence: `report.py:123` formats every current price with `$`; reproduced report excerpt: `...`.
Failure scenario: `066570.KS` returns `currency=KRW` -> report labels 85,100 KRW as `$85,100` -> the user reads the wrong currency.
Evidence status: reproduced.
Correction direction: render the normalized source currency; reject or label unknown currency.
Regression proof: a KRX fixture renders `KRW`/`₩`, while a US fixture renders `USD`/`$`.
```

## Sections After Findings

1. **Unverified risks** — plausible but not proven; name the missing check.
2. **Opportunities** — useful ideas outside the current defect set or v1 scope.
3. **Decisions needed** — material choices only the user can make.
4. **Correction order** — dependency ordered, with a proof gate after each step.
5. **Final verdict** — repeat the verdict in one standalone sentence.

Use stable prefixes by lens: `ARC`, `GND`, `PRV`, and `VER`. Preserve IDs when a later review rechecks the same defect. If the user explicitly asks to record a ruling, stamp the living plan with the date and outcome; otherwise propose the entry without silently editing the record.
