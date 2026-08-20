# Implement the Next Rebuild Slice

This is a saved prompt playbook, not a registered slash command.

```text
Use $stock-analysis-rebuild. Re-verify the current tree and .agents/plans/rebuild-execution.md, then implement only the first incomplete dependency gate (or the named slice: <TARGET>).

Before editing, state the observable outcome, owning contracts, and failure proof. Keep archive/ untouched and preserve the six-module v1 boundary. Add deterministic tests at SDK/yfinance boundaries, run the applicable verification, and do not disguise NOT RUN live checks as success.

After implementation, update the living plan only for status or rulings that actually changed. Report files changed, commands/results, unverified providers/markets, and the next gate. Do not begin the next gate in the same task unless the user asked for it.
```
