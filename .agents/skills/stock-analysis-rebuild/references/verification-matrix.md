# Verification Matrix

Read this before claiming a module, provider, ticker, workflow, or report is working.

## Verification Layers

| Gate | Scope | Required evidence |
|---|---|---|
| V0 | Static integrity | Imports compile; no missing symbols or syntax errors |
| V1 | Deterministic contracts | Unit tests with mocked SDK/yfinance boundaries and failure injection |
| V2 | Adapter parity | The same mocked tool-call scenarios pass for all four adapters |
| V3 | Live data | Each data tool smokes against one KRX and one US ticker with bounded output |
| V4 | One-provider workflow | One ticker completes visible tool round trips, grounded synthesis, and report write |
| V5 | Provider qualification | V4 is repeated independently for each configured hosted provider and local Ollama |
| V6 | Roster qualification | The six default tickers complete without hiding partial failures |

Passing a lower gate does not imply a higher one. Missing credentials make a provider `NOT RUN`, not failed and not passed.

## Deterministic Tests First

Tests should cover behavior, not wording:

- config precedence, selected-provider key validation, and local no-key behavior;
- schema serialization and rejection of invalid tool arguments;
- full assistant tool-call turn retained before correlated results;
- multiple calls in one turn and a failed call alongside a successful call;
- unknown tools, handler exceptions, empty data, malformed provider output, refusal, truncation, no-progress, and max turns;
- yfinance exceptions and representative missing/different shapes for KRX and US instruments;
- indicator warm-up and known small fixtures;
- risk calculations with known returns, zero variance, and too few rows;
- backtest no-look-ahead, closed-trade accounting, open final position, and JSON-finite output;
- report currency, timestamp, filename safety, missing-evidence rendering, disclaimer, and no `nan`/`inf` leakage;
- CLI default roster and ticker overrides without invoking live services.

Mock the network/SDK boundary, not the code under test. Avoid assertions that merely match a prompt sentence or Markdown heading.

## Commands

As files land, use the repository's Windows environment and run only applicable commands:

```powershell
.\env\Scripts\Activate.ps1
python -m py_compile config.py llm.py data_tools.py agents.py report.py main.py
python -m unittest discover -s tests -p "test_*.py"
```

Reconciliation note (2026-08-20): the documented root `env/` directory was not present during creation of this pack. Treat activation as an unverified prerequisite until the environment is located or recreated; do not report setup as working merely because the command is documented.

If the project deliberately adopts another test runner, record that decision and dependency; do not assume `pytest` is installed because the tests are named `test_*.py`.

Live checks come after deterministic checks:

```powershell
python main.py 005930.KS
python main.py GOOGL
```

Do not run the default six-ticker roster merely to discover that one-ticker orchestration is broken.

## Provider Qualification Record

Record each provider separately:

```text
provider | model/tag | mocked contract | live single ticker | tool trace | report | limitation
```

For `local`, record `ollama list`, endpoint readiness, and the exact tag. For hosted providers, verify only the selected key is required and never print it. Model names and SDK semantics are time-sensitive; confirm them from current native documentation during implementation.

## Live-Data Reconciliation

For each live tool smoke, record ticker, returned currency, latest data date, row/observation count, missing fields, warnings, and whether values are finite and JSON-serializable. Compare one or two key values with the raw returned object in the test harness so normalization is not merely assumed.

Live yfinance success is not a permanent contract. Mocked shape tests preserve known cases; live smoke tests detect current upstream drift. Both are needed.

## Honest Handoff

Report every gate as `PASS`, `FAIL`, `NOT RUN`, or `BLOCKED`, with the exact command and relevant output. Never use “all tests pass” when only compilation ran, or “all providers supported” when only a fake client passed.
