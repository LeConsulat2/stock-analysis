# Rebuild: lean, tool-grounded, multi-provider stock analysis agents

**Status:** planning only, no code written yet. This document is the durable reference for the rebuild;
implementation (`config.py`/`llm.py`/`data_tools.py`/`agents.py`/`report.py`/`main.py` below) has not
started.

## Context

The repo (`archive/o3-final.py` + 6 notebooks under `archive/`) is a set of LLM-generated experiments
comparing CrewAI-based multi-agent stock analysis. Two problems this rebuild fixes:

1. **Dependency bloat.** `requirements.txt` has 255 packages; the code only uses ~10 of them. Much of the
   bloat isn't stray copy-paste — CrewAI itself transitively pulls in `embedchain`, `chromadb`,
   `qdrant-client`, `lancedb`, etc. for RAG/memory features this project never uses.
2. **The "AI agent" part is broken in the most-evolved file.** In `o3-final.py`, the CrewAI agents all have
   `tools=[]` — they write analysis prose from a template + ticker string alone, while a *separate*
   quant pipeline (`AdvancedAnalysis`/`TechnicalAnalyzer`/`RiskManager`) computes the real numbers. The two
   never talk to each other; they're just concatenated into the final markdown report. Sentiment is fully
   stubbed (hardcoded `0.7`/`0.6`). There's no forward-looking/forecast component anywhere in the repo —
   everything describes the present, nothing reasons about what's next.

Goal: replace CrewAI/langchain with a small hand-rolled agent loop using native tool-calling — pluggable across
Anthropic, OpenAI, Google Gemini, and a local model (e.g. Gemma 4) served through Ollama — where every analyst
actually calls real data-fetching tools before writing anything, plus a new forward-looking/forecast role.
Target universe: Samsung Electronics (`005930.KS`), LG Electronics (`066570.KS` — matches the ticker used in
every existing `reports/*.md` file), Google (`GOOGL`), Meta (`META`), Microsoft (`MSFT`), NVIDIA (`NVDA`).

Confirmed with user: dynamic multi-round tool-calling per agent (not prefetch-and-narrate), GOOGL over GOOG,
no litellm (after researching the March 2026 PyPI supply-chain compromise of the `litellm` package plus its
separate proxy-server CVEs — native SDK per provider instead, dispatched by a plain `provider` string), and
four providers instead of two: Anthropic, OpenAI, Google Gemini, and a local model via Ollama. The existing
`o3-final.py`/notebooks have since been physically moved into `archive/` by the user (content untouched, just
relocated) — new rebuild files still land flat at the repo root, not inside `archive/`.

## New files (flat, no package directory — matches the repo's existing flat layout)

- `config.py` — `AnalysisConfig` dataclass (period, interval, MA windows, RSI period, VaR confidence — same
  shape as the existing one in `o3-final.py`) + `LLMConfig` (provider: `"claude"` | `"openai"` | `"google"` |
  `"local"`, model name, read from `.env` / env vars, default provider `claude`).
- `llm.py` — provider-agnostic chat+tool-calling client. One small `LLMClient` interface with a single method
  `send(system: str, messages: list, tools: list[ToolSpec]) -> LLMResponse` (`LLMResponse` = either
  assistant text, or one-or-more tool calls to execute). Four implementations:
  - `AnthropicClient` — wraps `anthropic.Anthropic().messages.create(..., tools=[...])`, translates
    `tool_use` content blocks to the internal tool-call representation, appends `tool_result` blocks back in.
  - `OpenAIClient` — wraps `openai.OpenAI().chat.completions.create(..., tools=[...])`, translates
    `tool_calls` similarly.
  - `GoogleClient` — wraps `google.genai.Client().models.generate_content(..., config=types.
    GenerateContentConfig(tools=[...]))` (the `google-genai` package — `google-generativeai` is deprecated,
    sunset Aug 2025). Auths via a plain `GOOGLE_API_KEY` (Gemini Developer API), not a GCP service
    account/Vertex AI setup. Default model a Gemini Flash variant.
  - `LocalClient` — **not a new SDK**: reuses `OpenAIClient`'s implementation with
    `base_url="http://localhost:11434/v1"` and a dummy `api_key`, pointed at a locally-running
    `ollama serve`. Confirmed Ollama's OpenAI-compatible endpoint returns proper `tool_calls`/
    `finish_reason: "tool_calls"` for Gemma 4, so no special-casing is needed in `agents.py`'s tool loop for
    this provider. The target model tag (e.g. a Gemma 4 tag) must already be pulled locally (`ollama pull
    <tag>`) before use — confirm the exact tag with `ollama list` rather than assuming one.
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
  `anthropic`, `openai`, `google-genai`, `python-dotenv`. (Drops `crewai`, `crewai-tools`, every `langchain*`
  package, `litellm` (never added — see the no-litellm decision above), and the ~240 unrelated packages —
  vector DBs, `auth0-python`, `kubernetes`, `snowflake-connector-python`, `docker`, `textblob`, `plotly`,
  `scrapegraph_py`, `spider-client`, `serpapi`, `selenium`, `pytube`, `pyvis`, etc. — none of which are
  referenced by any code in the repo, old or new. Verified `plotly`/`textblob` are imported but never actually
  used in `o3-final.py` either, so nothing currently working depends on them. The local/Ollama provider needs
  no additional package — it reuses `openai`.) This file has already been applied for real (not just
  planned) even though the code that will import these packages doesn't exist yet.
- `.env` — needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` present (only the one matching
  the selected `LLM_PROVIDER` is required at runtime, but supporting all three means any can be set). The
  local provider needs no API key, but does need `ollama serve` running with the target model already pulled.

## Not doing (kept out of scope for this pass)

- No CrewAI, no LangChain, no vector DB/memory layer, and no `litellm` (see the no-litellm decision above —
  four native SDK-backed clients instead of one routing layer).
- No async/concurrency — analysts run sequentially per ticker. Straightforward to parallelize later with
  `asyncio.gather` once the sequential version is proven correct; adding it now would be premature given the
  explicit goal of cutting complexity.
- Old files (`o3-final.py`, all 6 notebooks) are content-untouched — nothing in them was edited — but the user
  has since physically relocated them into `archive/` (this happened outside this rebuild's scope; noted here
  so this doc doesn't go stale against the actual repo layout).
- `CLAUDE.md` has been updated for real: the archive move changed every legacy file path it referenced, so it
  got more than the originally-planned "short addition" — its Architecture/Notebooks sections now describe
  the `archive/` designs explicitly as archived/reference-only, with paths corrected, plus a note that
  `requirements.txt` no longer installs what's needed to run them.

## Verification (once implementation starts)

- Smoke-test each `data_tools.py` function directly against a live ticker from the target roster (one KRX
  symbol, one US symbol) before wiring it into the agent loop — some yfinance fields
  (`analyst_price_targets`, `recommendations`) are undocumented/scraped and worth confirming return real data
  for both markets rather than assuming from docs.
- Run `python main.py 005930.KS` end-to-end with each provider (`LLM_PROVIDER=claude`, `LLM_PROVIDER=openai`,
  `LLM_PROVIDER=google`, `LLM_PROVIDER=local`) and confirm: tool calls actually fire (visible via a printed
  trace of each tool call + args), the synthesizer's final call references specifics from the analysts'
  output (not generic boilerplate), and a report lands in `reports/`. For `local`, confirm `ollama serve` is
  running and the target model tag is pulled before the run, and specifically check that tool-call turns
  actually round-trip (Gemma 4 tool-calling through Ollama is newer/less battle-tested than the three hosted
  APIs — worth confirming rather than assuming from the docs).
- Run the default multi-ticker roster end-to-end once single-ticker works.
