# Repository Guidelines

## Project Structure & Direction

This repository is rebuilding its experimental CrewAI stock-analysis scripts into a lean, tool-grounded application. Treat `.claude/NewStart.md` as the design reference. Existing `o3-final.py` and root-level `*.ipynb` files are legacy experiments and must remain untouched.

Keep the rebuild flat at the repository root:

- `config.py`: analysis and provider configuration.
- `llm.py`: provider-neutral interface plus Anthropic/OpenAI adapters.
- `data_tools.py`: defensive, JSON-serializable market-data tools.
- `agents.py`: bounded multi-round tool-calling loop and analyst roles.
- `report.py`: Markdown report generation.
- `main.py`: sequential CLI orchestration.

Generated files belong in `reports/` as `<TICKER>_analysis_<YYYYMMDD_HHMMSS>.md` and should not be newly committed.

## Setup, Run, and Development Commands

Use the Windows virtual environment:

```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
python -m py_compile config.py llm.py data_tools.py agents.py report.py main.py
python main.py 005930.KS
```

The rebuild is currently planned but not implemented, so the last two commands apply as files land. Use `python o3-final.py` only to inspect legacy behavior. The completed CLI should accept one or more ticker overrides and otherwise analyze the six-symbol default roster.

## Architecture and Coding Conventions

Follow PEP 8, four-space indentation, `snake_case` functions, `PascalCase` classes, and `UPPER_CASE` constants. Add type hints to public interfaces and centralize tunables in dataclasses. Do not add CrewAI, LangChain, vector databases, async execution, or memory/RAG layers. Agents must dynamically request real tools; do not prefetch data or let prompts invent unsupported facts. Keep tool results compact and JSON serializable, catch unstable `yfinance` failures, and return `{"error": ...}` payloads.

## Testing Guidelines

No automated suite or coverage threshold exists yet. Add tests under `tests/` as `test_*.py`, mocking network and LLM boundaries. Before integration, smoke-test every data tool with one KRX and one US ticker. Then run `python main.py 005930.KS` with both `LLM_PROVIDER=claude` and `LLM_PROVIDER=openai`; confirm tool-call traces, grounded synthesis, and report creation. Finally test the default roster.

## Commits and Pull Requests

History uses short but inconsistent subjects. Prefer imperative, scoped messages such as `Add Anthropic tool-call adapter`. Pull requests should describe the role or data flow changed, validation performed, provider(s) tested, and known data-source limitations. Include a report excerpt when output formatting or recommendation behavior changes.

## Security and Configuration

Keep `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in untracked `.env` files; only the selected provider's key is required at runtime. Never commit credentials or sensitive financial data. Treat recommendations as research output, not financial advice.
