# Rebuild: lean, tool-grounded, multi-provider stock analysis agents

**Status:** planning only, no code written yet. This document is the durable reference for the rebuild;
implementation (`config.py`/`llm.py`/`data_tools.py`/`agents.py`/`report.py`/`main.py` below) has not
started.

## Context

The repo (`o3-final.py` + 6 notebooks) is a set of LLM-generated experiments comparing CrewAI-based
multi-agent stock analysis. Two problems this rebuild fixes:

1. **Dependency bloat.** `requirements.txt` has 255 packages; the code only uses ~10 of them. Much of the
   bloat isn't stray copy-paste — CrewAI itself transitively pulls in `embedchain`, `chromadb`,
   `qdrant-client`, `lancedb`, etc. for RAG/memory features this project never uses.
2. **The "AI agent" part is broken in the most-evolved file.** In `o3-final.py`, the CrewAI agents all have
   `tools=[]` — they write analysis prose from a template + ticker string alone, while a *separate*
   quant pipeline (`AdvancedAnalysis`/`TechnicalAnalyzer`/`RiskManager`) computes the real numbers. The two
   never talk to each other; they're just concatenated into the final markdown report. Sentiment is fully
   stubbed (hardcoded `0.7`/`0.6`). There's no forward-looking/forecast component anywhere in the repo —
   everything describes the present, nothing reasons about what's next.

Goal: replace CrewAI/langchain with a small hand-rolled agent loop using native tool-calling (Anthropic and
OpenAI, pluggable), where every analyst actually calls real data-fetching tools before writing anything, plus
a new forward-looking/forecast role. Target universe: Samsung Electronics (`005930.KS`), LG Electronics
(`066570.KS` — matches the ticker used in every existing `reports/*.md` file), Google (`GOOGL`), Meta
(`META`), Microsoft (`MSFT`), NVIDIA (`NVDA`).

Confirmed with user: dynamic multi-round tool-calling per agent (not prefetch-and-narrate), GOOGL over GOOG,
and the existing `o3-final.py`/notebooks are left untouched — this is new files alongside them, not a rewrite
in place.

## New files (flat, no package directory — matches the repo's existing flat layout)

- `config.py` — `AnalysisConfig` dataclass (period, interval, MA windows, RSI period, VaR confidence — same
  shape as the existing one in `o3-final.py`) + `LLMConfig` (provider: `"claude"` | `"openai"`, model name,
  read from `.env` / env vars, default provider `claude`).
- `llm.py` — provider-agnostic chat+tool-calling client. One small `LLMClient` interface with a single method
  `send(system: str, messages: list, tools: list[ToolSpec]) -> LLMResponse` (`LLMResponse` = either
  assistant text, or one-or-more tool calls to execute). Two implementations:
  - `AnthropicClient` — wraps `anthropic.Anthropic().messages.create(..., tools=[...])`, translates
    `tool_use` content blocks to the internal tool-call representation, appends `tool_result` blocks back in.
  - `OpenAIClient` — wraps `openai.OpenAI().chat.completions.create(..., tools=[...])`, translates
    `tool_calls` similarly.
  - A `get_llm_client(config: LLMConfig) -> LLMClient` factory picks the implementation.
- `data_tools.py` — the real, grounded data functions (plain Python, JSON-schema-described for tool-calling,
  not CrewAI `BaseTool` subclasses). Each returns a small JSON-serializable summary (not raw CSV dumps like
  the old `BaseTool._run()` methods — those wasted tokens):
  - `get_price_technicals(ticker, config)` — reuses the indicator math from `TechnicalAnalyzer` in
    `o3-final.py` (`pandas_ta`: RSI, MACD, Bollinger, MAs, volume SMA) but returns only the latest values +
    signal directions, not the full history.
  - `get_financial_statements(ticker)` — reuses the ratio-calculation pattern from `FinancialMetrics` in
    `finance3.ipynb`/`o1-style.ipynb`, generalized with the same defensive `KeyError`/empty-dataframe handling
    those files already have (needed because US vs KRX filers don't always populate the same yfinance line
    items).
  - `get_risk_metrics(ticker, config)` — reuses `RiskManager` (VaR, Sharpe, max drawdown, volatility) from
    `o3-final.py` verbatim.
  - `get_recent_news(ticker, limit=8)` — real headlines + links via `yf.Ticker(ticker).news` (the actual
    tool-grounded pattern `finance.ipynb`'s `StockNewsTool` already used) — replaces the fully-stubbed
    `MarketSentimentAnalyzer.analyze_news_sentiment`.
  - `get_analyst_outlook(ticker)` — **new**: real forward-looking data via
    `yf.Ticker(ticker).analyst_price_targets`, `.recommendations`/`.upgrades_downgrades`, and
    `.earnings_dates`/`.calendar` (upcoming earnings date, consensus target, recent rating changes). This is
    what grounds the new forecast role in real analyst data instead of the LLM inventing a scenario from
    nothing.
  - `run_backtest(ticker, config)` — reuses `BacktestEngine` from `o3-final.py` as an optional tool for the
    technical analyst.
  - Every function wraps its yfinance calls in try/except returning a `{"error": ...}` payload on failure
    (some of these fields are undocumented/scraped and do occasionally change shape) — same defensive style
    already used throughout `o1-style.ipynb`.
- `agents.py` — the CrewAI replacement:
  - `run_agent(llm, role_system_prompt, task_prompt, available_tools, max_turns=6) -> str` — the actual
    agent loop: send messages + tool specs, if the model responds with tool call(s) execute the matching
    local `data_tools` function(s) and feed results back as tool results, repeat until the model returns
    final text or `max_turns` is hit.
  - Five role definitions (system prompt + tool subset), replacing `AdvancedAgents`/`Tasks` in `o3-final.py`:
    **Technical Analyst** (`get_price_technicals`, `run_backtest`), **Fundamental Analyst**
    (`get_financial_statements`), **Risk Analyst** (`get_risk_metrics`, `get_price_technicals`),
    **Forecast Analyst** *(new)* (`get_analyst_outlook`, `get_recent_news`, `get_price_technicals`) — produces
    a bull/base/bear scenario with named catalysts, not just a snapshot, **Sentiment Analyst**
    (`get_recent_news`) — replaces the stub with a real read of actual headlines.
  - **Synthesizer** step: no tools, takes the five analysts' final text outputs as context, produces the
    final BUY/HOLD/SELL + target/stop/position-size call (same shape as `investment_recommendation` in
    `o3-final.py`'s `Tasks`, but now actually reading grounded output instead of disconnected quant JSON).
- `report.py` — markdown writer, adapted from `save_analysis_to_markdown()` in `o3-final.py`: same
  `reports/<TICKER>_analysis_<timestamp>.md` destination and overall section layout, plus a new **Forecast /
  Outlook** section for the new role's output.
- `main.py` — CLI entry point. `TICKERS` default list = the six above; accepts an optional ticker/list
  override as an argv. Runs each ticker's five analysts sequentially (kept simple for v1, no asyncio/CrewAI
  process model to reason about), then the synthesizer, then writes the report. Prints the final call + report
  path per ticker, matching the existing `__main__` block's console output style.
- `requirements.txt` — trimmed to what's actually imported: `yfinance`, `pandas`, `numpy`, `pandas_ta`,
  `anthropic`, `openai`, `python-dotenv`. (Drops `crewai`, `crewai-tools`, every `langchain*` package, and the
  ~240 unrelated packages — vector DBs, `auth0-python`, `kubernetes`, `snowflake-connector-python`, `docker`,
  `textblob`, `plotly`, `scrapegraph_py`, `spider-client`, `serpapi`, `selenium`, `pytube`, `pyvis`, etc. —
  none of which are referenced by any code in the repo, old or new. Verified `plotly`/`textblob` are imported
  but never actually used in `o3-final.py` either, so nothing currently working depends on them.)
- `.env` — needs both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` present (only the one matching the selected
  `LLM_PROVIDER` is required at runtime, but supporting both means either can be set).

## Not doing (kept out of scope for this pass)

- No CrewAI, no LangChain, no vector DB/memory layer.
- No async/concurrency — analysts run sequentially per ticker. Straightforward to parallelize later with
  `asyncio.gather` once the sequential version is proven correct; adding it now would be premature given the
  explicit goal of cutting complexity.
- Old files (`o3-final.py`, all 6 notebooks) are left completely untouched — nothing is deleted or moved.
- `CLAUDE.md` gets a short addition pointing at `main.py` as the new reference entry point, without rewriting
  the existing sections describing the legacy files (still accurate, still worth keeping for context).

## Verification (once implementation starts)

- Smoke-test each `data_tools.py` function directly against a live ticker from the target roster (one KRX
  symbol, one US symbol) before wiring it into the agent loop — some yfinance fields
  (`analyst_price_targets`, `recommendations`) are undocumented/scraped and worth confirming return real data
  for both markets rather than assuming from docs.
- Run `python main.py 005930.KS` end-to-end with each provider (`LLM_PROVIDER=claude` and
  `LLM_PROVIDER=openai`) and confirm: tool calls actually fire (visible via a printed trace of each tool
  call + args), the synthesizer's final call references specifics from the analysts' output (not generic
  boilerplate), and a report lands in `reports/`.
- Run the default multi-ticker roster end-to-end once single-ticker works.
