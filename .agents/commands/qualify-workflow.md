# Qualify the Workflow

This is a saved prompt playbook, not a registered slash command.

```text
Use $stock-analysis-rebuild and read its verification matrix. Verify the current rebuild without changing application behavior unless the user separately requests fixes.

Run gates in order: static compile -> deterministic tests -> live data for one KRX and one US ticker -> one selected provider on one ticker -> other configured providers -> default roster. Stop at a failed dependency gate instead of producing noisy downstream failures.

For every gate, record PASS, FAIL, NOT RUN, or BLOCKED with the exact command and relevant output. Never print credentials. For local, confirm ollama serve and the exact installed model tag before the run. Finish by naming the highest proven gate and the first missing proof.
```
