# Start Here

**Written:** 2026-10-29 · **Owner:** Jonathan · **Read this first in every new session.**

This is the owner-facing plan: what exists, how the pieces fit, and the next steps in order. It links the
engineering plans rather than replacing them. Update the **Status** column when a step finishes, and add a
dated line to the log at the bottom.

## Where things stand

| Piece | What it is | Status (2026-10-29) |
|---|---|---|
| `brief.py` + `watchlist.csv` | Daily morning brief, no AI, free. Flags unusual moves, earnings within 7 days, NZD returns, FX, concept of the day. | **Works on live data.** Email not configured yet. Not scheduled. |
| `full-flow/` | AI research desk: five model roles, valuation gate, BUY / HOLD / SELL / ABSTAIN with "why it could be wrong". | Demo and offline tests pass (2026-09-08). **No live data or paid model run yet.** Not scheduled. |
| Six-module rebuild (`config.py`, `llm.py`, ...) | The approved production architecture in `.claude/NewStart.md`. | Not started. Gates in [rebuild-execution.md](rebuild-execution.md). |
| `claude-main.py`, `codex-main.py` | Learning experiments. | Reference only. |
| `archive/` | Old CrewAI experiments. | Reference only. |
| Google Sheet (OPPD template copy) | Korean household-finance tracker. | Contains the template's example trades; its stock total is broken. Borrow ideas only. |
| 꿀직장TV weekly live | 김현준 (더퍼블릭자산운용), Wednesdays 12:00 KST = 16:00 NZ (summer). | A source of questions to test, not to follow wholesale. |

## How the pieces fit

Recommendation, not yet a ruling:

```text
Daily (free)       brief.py email      what moved, what's coming, one concept
                         │ flags a stock, or earnings are due
                         ▼
Weekly (paid AI)   full-flow           a deeper note on 1-3 companies, with the case against
                         │
                         ▼
You                decision journal    thesis, what would prove you wrong, review date
                         │
                         ▼
You                broker app          you place any trade yourself; no code here ever trades
```

- The **brief** is for daily awareness. It stays free and needs no API keys.
- **full-flow** is for depth, run weekly or after earnings, not daily. Running it daily on 10+ companies
  would mostly repeat itself and cost money.
- The **six-module rebuild** waits. full-flow already covers most of what it was meant to do. Decide its
  future at step 12, once full-flow has run live.

## The steps

Numbered in order, one idea each. **Learn** names the concept the step teaches.

### Now

| # | Step | Done when | Learn | Status |
|---|---|---|---|---|
| 1 | **Turn on the email.** Turn on Google 2-Step Verification, create an app password at <https://myaccount.google.com/apppasswords>, copy `.env.example` to `.env`, fill in `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD`, then run `.\env\Scripts\python.exe brief.py --send`. | A brief arrives in your inbox. | Reading the brief | Waiting on you |
| 2 | **Decide whether your watchlist stays private.** The GitHub repo is **public**. Your written reasons would be public on the next push. Option: git ignores `watchlist.csv` and an example copy is committed. Same question for live `full-flow/reports/`, which are not ignored today. | Decision recorded below (D1). | Privacy | Open |
| 3 | **Write your "why" and "wrong if"** for each stock in `watchlist.csv`. Give a reason, not a price. | The brief no longer says "not written yet". | A thesis | Open |
| 4 | **Schedule the brief.** Windows Task Scheduler, Tuesday to Saturday at 10:30 NZ. That is after the US close (08:00-10:00 NZ depending on the season) and includes the previous day's Korean close. Claude creates it on your OK. The PC must be on. | It arrives on its own for a week. | | After 1 |

### Next: weekly habits

| # | Step | Done when | Learn | Status |
|---|---|---|---|---|
| 5 | **Read one earnings report a week**, starting with whatever the brief flags. Upcoming as of 2026-10-29: Alphabet (29 Oct, US), Apple (3 Nov*), Acushnet (5 Nov*), Misto (12 Nov*). *Estimated dates. | You can say whether the quarter was good, and why the stock moved anyway. | Revenue, margin, guidance, beat or miss | Open |
| 6 | **Keep a live ledger by hand.** After each Wednesday live, write `notes/lives/YYYY-MM-DD.md` with three claims about your stocks, each labelled fact, opinion or forecast, plus how and when to check it. Keep it private: the content is his work. | Three weeks of ledgers, with one claim checked. | Fact, opinion and forecast | Open |
| 7 | **First live full-flow run: one company, one provider.** Example: `--provider claude --only GOLF`. Needs that provider's API key in `full-flow/.env` and a spending limit set in the provider's billing settings. Read the note, then check three of its facts against the company's own earnings release. | One AI note read and checked. Findings recorded in `full-flow/VERIFICATION.md`. | Price versus value | Needs D3 |

### Later

| # | Step | Done when | Learn | Status |
|---|---|---|---|---|
| 8 | **Merge to one watchlist.** `watchlist.csv` becomes the single list, and full-flow reads from it. Today there are two lists (see D2). | Editing one file changes both tools. | | Needs D2 |
| 9 | **Automate the live ledger.** YouTube Korean captions, then a transcript, then Claude extracts claims about your stocks into `notes/lives/`. | A ledger appears without manual note-taking. | | After 6 |
| 10 | **Track what you actually own.** Borrow the sheet's best idea: one list of real trades (`transactions.csv`), with positions and weights by stock, sector and currency computed from it. | You know your % in each stock, sector and currency (NZD, USD, KRW). | Cost basis, concentration, currency exposure | Open |
| 11 | **Decision journal.** Add a `journal/` template. Write an entry before any buy or sell, and review old entries monthly against what happened. | One entry written before a real decision. | Writing before acting | Open |
| 12 | **Decide the rebuild's future.** Choose one: build the six modules, promote full-flow to be the product, or retire the plan. | Ruling recorded in [rebuild-execution.md](rebuild-execution.md). | | After 7 |

## Open decisions for the owner

| ID | Decision | Needed for |
|---|---|---|
| D1 | Keep `watchlist.csv` and live `full-flow/reports/` private (git-ignored) or public? | Step 2, before step 3 |
| D2 | Which stocks make the single watchlist? `brief.py` has KO, Misto (081660.KS), VOO, SK Hynix (000660.KS). full-flow has Callaway (CALY), Adobe (ADBE), SpaceX (SPCX), plus PGA TOUR / LIV / TGL as news-only context. Both have GOOGL, 005930.KS, AAPL, GOLF, TSLA, NVDA. | Step 8 |
| D3 | Monthly AI budget and default provider for full-flow. | Step 7 |
| D4 | Holding horizon (months or years?) and US versus Korea priority. These shape what full-flow should value. | Step 7 |

## Known issues to fix when the step comes up

- **full-flow's schedule time.** Its README says the US close lands 06:00-08:00 Auckland. It is actually
  08:00 NZST (April-September), 09:00 NZDT (late September to early November, and mid-March to early April),
  and 10:00 NZDT (early November to mid-March). The suggested 08:30 run would usually report the previous
  session. Use 10:30 or later. Relevant at step 7 and beyond.
- **Docs drift.** `CLAUDE.md` says `reports/` is git-ignored. It isn't (only `reports/briefs/` is), and
  full-flow DEMO reports are committed. Matters for D1.
- **Model IDs** in `claude-main.py` and `full-flow/.env.example` were chosen on earlier dates. Re-check
  them against current provider docs before a paid run.

## Rules that stay true

- Nothing in this repo trades, and nothing ever will without a new explicit decision. You act in your broker app.
- Everything here is research and learning, not financial advice. full-flow's BUY or SELL is a gated research
  stance, not an instruction.
- Secrets live only in `.env` (root, for `brief.py`) and `full-flow/.env`. Both are git-ignored and must never
  be committed. The repo is public.
- Generated briefs and reports are written in **English**.
- Learn one concept at a time, in order. Explain it with a stock on the watchlist.

## Related documents

- [rebuild-execution.md](rebuild-execution.md): engineering gates for the six-module rebuild.
- [personal-research-roadmap.md](personal-research-roadmap.md): product recommendations from 2026-09-08.
- [`full-flow/README.md`](../../full-flow/README.md) and [`VERIFICATION.md`](../../full-flow/VERIFICATION.md): how full-flow works and what has been tested.
- `CLAUDE.md`: repository history and the archived designs.
- "Six Stocks, Eight Steps" (private Claude artifact, 2026-10-29): the learning path behind steps 3-11,
  with the Misto/Acushnet look-through example.

## Log

- **2026-10-29:** Plan written. `brief.py` built and qualified on live data (10 stocks, 3 FX pairs). Committed
  `477af61` on `new-start`. Email (step 1) is waiting on the owner's Gmail app password.
