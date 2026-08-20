# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a personal research repo, not a packaged app or library. It explores building an AI multi-agent stock
analysis system (technical + fundamental + risk + sentiment analysis, synthesized into a BUY/HOLD/SELL call).
There is no build system, test suite, linter, or CI configured.

**The repo is mid-rebuild.** The original CrewAI/LangChain-based experiments have been moved into `archive/`
(content untouched, just relocated) and are kept only for reference. The active design — a hand-rolled,
multi-provider, tool-calling agent loop with no CrewAI/LangChain/litellm — is specified in
`.claude/NewStart.md` (durable design doc) and `AGENTS.md` (condensed contributor guide); treat both as the
source of truth for the rebuild. As of now the rebuild is **planning only** — `config.py`, `llm.py`,
`data_tools.py`, `agents.py`, `report.py`, `main.py` don't exist yet — so there is currently no runnable
entry point for the new design. `archive/o3-final.py` is the last known-working script, but see the
requirements-mismatch note below before trying to run it.

## Setup & running

- Python 3.13 venv already exists at `env/` (Windows: `env\Scripts\activate`).
- Install deps: `pip install -r requirements.txt`.
- `requirements.txt` was trimmed to what the *rebuild* needs — `yfinance`, `pandas`, `numpy`, `pandas_ta`,
  `anthropic`, `openai`, `google-genai`, `python-dotenv` — not what the archived scripts need. It no longer
  installs `crewai`, `langchain-openai`, `textblob`, or `plotly`, so **`archive/o3-final.py` and the archived
  notebooks will not run against the current `requirements.txt`** — `pip install crewai langchain-openai
  textblob plotly` separately first if you need to run legacy code.
- Planned providers for the rebuild: Anthropic, OpenAI, Google Gemini, and a local model (e.g. Gemma 4) served
  through Ollama's OpenAI-compatible endpoint — see `.claude/NewStart.md` for the `.env` keys each needs.
- To run the legacy reference script instead: `python archive/o3-final.py` (after installing its extra deps
  above). Requires a `.env` file with `OPENAI_API_KEY=...` — it raises `ValueError` at import time if missing
  (the qualitative/CrewAI pipeline calls `ChatOpenAI` directly). The ticker is hardcoded in the `if __name__ ==
  "__main__"` block (currently `"005930.KS"`); edit it there to analyze a different symbol. Tickers use Yahoo
  Finance format, e.g. `005930.KS` / `066570.KS` for KRX-listed Korean stocks.

## Architecture (archived designs, kept for reference)

Everything below describes the code now under `archive/` — none of it is wired into the active rebuild. All
files there (`o3-final.py` and every notebook) implement the same idea — a CrewAI crew that produces a
BUY/HOLD/SELL call for a ticker, written to `reports/` — but they split into two genuinely different designs,
not just cosmetic rewrites:

- **Tool-grounded design** (`archive/finance.ipynb`, `archive/finance3.ipynb`, `archive/o1-style.ipynb`,
  `archive/o1-english.ipynb`): CrewAI agents are given real `BaseTool` subclasses whose `_run()` calls
  `yfinance` directly (e.g. `ComprehensivePriceTool`, `FinancialMetricsTool` in finance3/o1-*; `StockPriceTool`,
  `BalanceSheetTool`, `IncomeStatementTool`, `InsiderTransactionsTool`, `StockNewsTool` + `ScrapeWebsiteTool` in
  finance.ipynb). Indicators (RSI/MACD/support-resistance) are hand-rolled with pandas `.ewm()`/`.rolling()`,
  not `pandas_ta`. The LLM only ever sees numbers it explicitly requested through a tool call — there's no
  separate quant/risk layer. `finance.ipynb` uses `Process.hierarchical` + `memory=True` with `gpt-4o-mini`;
  the other three use `Process.sequential` with 3 agents (financial/technical analyst + hedge-fund-manager).
- **Hybrid design** (`archive/claude-improved-o1.ipynb` → `archive/o3-final.py`): CrewAI agents all have
  `tools=[]` — they never fetch data themselves and generate prose purely from the task template + ticker
  string. Real numbers instead come from a fully separate `AdvancedAnalysis` pipeline (`TechnicalAnalyzer` via
  `pandas_ta`, `RiskManager`, `BacktestEngine`, a stubbed `MarketSentimentAnalyzer`) that runs independently.
  The two sides are **not cross-wired**: the LLM agents' prompts never see the computed indicators/risk
  metrics, and those quant numbers only get sewn together with the CrewAI prose at the very end, inside
  `save_analysis_to_markdown()`. Don't assume the "Qualitative Analysis" section of a generated report is
  grounded in the "Technical/Risk Analysis" numbers next to it in the same file — they were produced
  independently. (This disconnect — plus the fully-stubbed sentiment and the lack of any forward-looking role —
  is the main thing the rebuild in `.claude/NewStart.md` fixes.)

`archive/claude-improved-o1.ipynb` is architecturally identical to `archive/o3-final.py` but its own saved cell
output shows it crashing with `asyncio.run() cannot be called from a running event loop` (it calls
`asyncio.run()` inside a sync `create_analysis_crew` while already inside the Jupyter kernel's loop).
`o3-final.py` is the fixed evolution: `create_analysis_crew` was made `async` and awaits the analysis directly,
with `asyncio.run()` used only once, at the `if __name__ == "__main__"` guard of a plain script — that fix is
what makes it "final" and the one to treat as the legacy reference entry point.

`create_analysis_crew(ticker, config)` in `archive/o3-final.py` combines two independent pipelines and merges
their output:

1. **Quantitative pipeline** — `AdvancedAnalysis.run_complete_analysis()`: pulls price history via `yfinance`,
   then runs:
   - `TechnicalAnalyzer` — RSI, MACD, Bollinger Bands, moving averages, volume SMA via `pandas_ta`, then
     `generate_signals()` turns indicators into weighted -1/0/+1 signals.
   - `RiskManager` — VaR, Sharpe ratio, volatility, max drawdown.
   - `MarketSentimentAnalyzer` — **stubbed**: `analyze_news_sentiment`/`analyze_social_sentiment` return
     hardcoded constants, not real API calls.
   - `BacktestEngine` — simple signal-driven long/short backtest over the same price history.
   - `_make_final_decision()` combines all four (40/25/20/15% weights) into BUY/HOLD/SELL.
2. **Qualitative pipeline** — a CrewAI `Crew` (`AdvancedAgents`: financial/technical/risk/sentiment analysts
   plus a hedge-fund-manager synthesizer) running sequential `Tasks` against `o3-mini` via
   `langchain_openai.ChatOpenAI`, templated with `{company}` = ticker.

Both results are merged and rendered by `save_analysis_to_markdown()` into
`reports/<TICKER>_analysis_<YYYYMMDD_HHMMSS>.md`.

`AnalysisConfig` (dataclass) centralizes tunables (lookback period, MA windows, RSI period, VaR confidence,
cache TTL) and is threaded through nearly every class. `AnalysisCache` is a simple in-memory dict-based TTL
cache instantiated separately per class (e.g. `MarketSentimentAnalyzer` and `TechnicalAnalyzer` each get their
own instance) — it is not a shared/global cache.

`o3-final.py` was built up incrementally across an LLM chat session — Korean-language section comments (e.g.
"다음은 ...", "next is...") mark those chunk boundaries. One artifact of that process:
`TechnicalAnalyzer.generate_signals` is defined twice with identical bodies (the second silently wins); don't
assume single-definition behavior when tracing signal logic.

## Notebooks (all under `archive/`)

Each notebook is essentially one large code cell (a full script, not an exploratory analysis notebook).
Filenames indicate which model/style produced the draft — `o1-*` = written by o1, `claude-*` = written by
Claude — and the difference isn't purely stylistic (see Architecture above for the tool-grounded vs. hybrid
split). Per-file notes:

- `finance.ipynb` — tool-grounded, `Process.hierarchical`, `gpt-4o-mini`, `memory=True`, ticker `066570.KS`.
- `finance3.ipynb` — tool-grounded, `Process.sequential`, `o3-mini`, ticker `066570.KS`.
- `o1-style.ipynb` / `o1-english.ipynb` — same tool-grounded design as `finance3.ipynb` (down to the same
  `ComprehensivePriceTool`/`FinancialMetricsTool` classes) but with `gpt-4o-mini`, more defensive
  exception/`KeyError` handling around missing financial-statement rows, and ticker `005930.KS`. The two files
  are near-duplicates of each other — one commented/documented in Korean, the other translated to English —
  not independent designs.
- `claude-improved-o1.ipynb` — the hybrid design; predecessor of `o3-final.py` (see above for the bug it had).
- `claude-style.ipynb` — a more idiomatic/ambitious rewrite attempt (ABCs, `Protocol`, `TypedDict`, a pydantic
  `APIConfig` expecting `ALPHA_VANTAGE_KEY`/`FINNHUB_KEY` env vars that appear nowhere else in the repo,
  `tenacity` retries). **It is incomplete and not runnable**: `Agent`/`Task`/`Crew` are imported but never
  instantiated, `ReportGenerator._format_report_content` and `_generate_visualizations` are stub `pass`
  bodies, `_save_report` POSTs to a `github.com/markdown`-shaped endpoint rather than writing the file
  locally, and the file cuts off mid-`main()` with no `if __name__` guard.

`o3-final.py` is the most fleshed-out, actually-working version of the archived designs — but see the
Repository overview above: it's superseded in intent by the (not-yet-implemented) rebuild, not by any other
archived file.

## Reports

`reports/` holds generated markdown analysis output. It's listed in `.gitignore`, but several report files
generated before that rule was added are already committed — don't assume the directory is untracked.
