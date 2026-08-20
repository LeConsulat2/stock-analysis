# Plan the Next Rebuild Slice

This is a saved prompt playbook, not a registered slash command.

```text
Use $stock-analysis-rebuild. Do not edit application code.

Re-read root AGENTS.md, .claude/NewStart.md, the current source/tests, and .agents/plans/rebuild-execution.md. Re-verify implementation status instead of inheriting the plan's status line. Select exactly one smallest dependency-complete slice from the first incomplete gate.

Return:
1. what is actually present now;
2. the one observable outcome this slice will add;
3. files and public contracts involved;
4. realistic failure scenarios and deterministic tests;
5. commands that would prove completion;
6. explicit deferrals and any user decision needed.

Do not expand into post-v1 research, storage, portfolio, UI, async, or framework work. End with one bottom-line recommendation.
```
