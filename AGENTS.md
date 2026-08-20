# Repository Guidelines

## Project Structure & Direction

The active rebuild is planning-only. Treat `CLAUDE.md` and `.claude/NewStart.md` as its sources of truth. Seven CrewAI/LangChain experiments live in `archive/`; keep them untouched as design references. Legacy dependencies are absent from the trimmed `requirements.txt`.

New modules will stay at the repository root:

- `config.py`: analysis and provider dataclasses.
- `llm.py`: common client interface and four provider adapters.
- `data_tools.py`: compact, grounded market-data tools.
- `agents.py`: analyst roles and bounded tool-calling loop.
- `report.py`: Markdown rendering.
- `main.py`: sequential CLI orchestration.

Generated output belongs in ignored `reports/` as `<TICKER>_analysis_<YYYYMMDD_HHMMSS>.md`.

## Setup, Run, and Development Commands

```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
python -m py_compile config.py llm.py data_tools.py agents.py report.py main.py
python main.py 005930.KS
```

The first two commands apply now; the others apply after rebuild files land. There is no build step. The finished CLI should accept ticker overrides and default to `005930.KS`, `066570.KS`, `GOOGL`, `META`, `MSFT`, and `NVDA`. Do not use `archive/o3-final.py` as the active entry point.

## Architecture & Coding Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, `PascalCase` for classes, and `UPPER_CASE` for constants. Type public interfaces and centralize tunables in dataclasses. Support `claude`, `openai`, `google`, and `local` through native SDK adapters behind one `LLMClient`; do not introduce CrewAI, LangChain, LiteLLM, vector databases, async execution, or memory/RAG in v1.

Agents must request real tools dynamically before making claims. Keep tool responses small and JSON-serializable. Wrap unstable `yfinance` access and return `{"error": ...}` instead of crashing. Local models use Ollama's OpenAI-compatible endpoint.

## Testing Guidelines

No automated suite or coverage threshold exists yet. Add `tests/test_*.py` files and mock network/LLM boundaries. Smoke-test each data tool with one KRX and one US ticker. Run a single-ticker workflow through all four providers, verify visible tool-call round trips and grounded synthesis, then test the default roster and generated report. For `local`, confirm `ollama serve` and the exact installed model tag first.

## Commits & Pull Requests

Recent subjects are short but inconsistent. Use imperative, scoped messages such as `Add Google tool-call adapter`. Pull requests should describe data-flow changes, validation, providers tested, and data-source limitations. Include a report excerpt for recommendation or formatting changes.

## Security & Configuration

Keep `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` in untracked `.env`; only the selected hosted provider needs its key. Local Ollama needs no key. Never commit credentials or sensitive financial data, and label recommendations as research rather than financial advice.
