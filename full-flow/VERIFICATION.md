# Verification record

Date: 2026-09-08. Scope: self-contained `full-flow/` teaching example.

| Check | Result | Evidence / limits |
|---|---|---|
| Python compilation | PASS | All example modules compile |
| Offline behavioral suite | PASS | 17 unittest cases; no live model or Yahoo calls |
| OpenAI adapter | PASS, mocked | Full assistant output retained; multiple correlated tool results |
| Claude adapter | PASS, mocked | Thinking signature retained; matching error tool result |
| Gemini adapter | PASS, mocked with real SDK types | Native content/thought signature and function ID preserved; retry config constructs |
| Market normalization | PASS, fixtures | Wrong equity identity rejected, annual missing fields preserved, daily-bar cutoff and adjusted-change calculation |
| Publication rules | PASS, fixtures | No-tool response, unknown tool/arguments, repeated calls, turn exhaustion, stale data, skeptical rejection, unknown citations, and invalid valuation blocked |
| Output | PASS, demo | Full configured roster produces a dated table and Markdown/HTML/JSON; context-only organizations receive no share price; fictional banner and sources retained |
| PowerShell scripts | PASS, syntax only | Parsed with PowerShell language parser; no scheduled task registered |
| Live yfinance | NOT RUN | Intentionally excluded from this task |
| Live OpenAI / Claude / Gemini | NOT RUN | No paid inference, account/model-access qualification, or claim of production readiness |
| Scheduler / delivery | NOT RUN | Optional registration code provided; output is local files, no email adapter |
| Investment performance | NOT ESTABLISHED | No backtest, forward track record, or evidence of outperformance |

Commands from repository root:

```powershell
.\full-flow\.venv\Scripts\python.exe -m unittest discover -s full-flow/tests -v
.\full-flow\.venv\Scripts\python.exe -m compileall -q -x '(\.venv|reports)' full-flow
.\full-flow\.venv\Scripts\python.exe full-flow/main.py --demo --quiet
```

The test suite's report-writing case initially hit a Windows sandbox permission issue with Python temporary directories. It passed outside that sandbox with the same assertions and no network calls. Ordinary demo output in the project reports directory succeeded within the sandbox.

Installed in the isolated `full-flow/.venv`: Python 3.13, openai 3.8.0, anthropic 1.4.0, google-genai 2.22.0, yfinance 1.7.0, pandas 3.0.5, tzdata 2026.3. These are observed test-environment versions, not an evergreen compatibility promise. `requirements.txt` is intentionally a short direct dependency list; a deployment should lock a qualified environment.

Structural validation cannot prove citation entailment or a model's financial judgment. The skeptical model also can be wrong. Before relying on live output, run one company/provider, inspect the tool trace and original company disclosures, review its valuation assumptions, and record the observed limitations.
