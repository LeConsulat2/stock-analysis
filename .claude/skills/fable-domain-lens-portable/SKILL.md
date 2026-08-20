---
name: fable-domain-lens-portable
description: Portable, product-agnostic version of fable-domain-lens - copy this folder into any new project's .claude/skills/. In this repo, prefer fable-domain-lens; activate this one explicitly via "portable domain lens" or in a non-PRIVATE codebase. Encodes the method for converting lived experience into falsifiable design requirements - dated behavior facts over stereotypes, the request-behind-the-request, pressure-level (not surface-level) transfer between domains, and letting fresh user observations overrule old instincts.
---

# Fable Domain Lens (Portable)

## Purpose

Technical skill tells you *how* to build. Domain knowledge tells you *what will
actually happen when a human meets the thing* — and that knowledge is the maker's real
moat, because it can't be scraped or guessed. Years behind any counter — a pro shop, a
university office, an insurance desk, a clinic, a tournament tee — teach things no
requirements doc contains: what pressure does to routine, when people actually arrive,
what they ask first, what matters after something goes wrong.

This skill is the method for **cashing that knowledge in**: turning lived experience
into concrete, falsifiable design requirements — and keeping it honest, so a decade-old
hunch doesn't override what this month's real user shows you.

## Activation

Explicit: "portable domain lens" / "domain lens pass" / "behavior ledger".

Self-activate when (in a project without an app-specific sibling):
- scoping a new feature or product area from scratch
- a feature request arrives as a solution ("add a button that...") rather than a problem
- prioritizing a backlog where items serve different user moments
- writing UI copy, defaults, empty states, or tool/endpoint names — where domain
  knowledge shows most and costs least

## The Method

### Step 1 — Write behavior facts, not personas

A behavior fact is one sentence: **[who] does [observable action] because
[pressure/incentive]** — something personally watched, with a date/source.

```markdown
- Visitors arrive exactly at their booked time and ask every question then, because
  arriving early feels like wasted time. (observed: front-desk operations, years)
- Professionals write records in an end-of-day batch, not per-event, because the
  next person is already at the door. (observed: practitioner interviews)
- Institutional usage is tidal: near-zero, then everyone at once in the same two
  weeks. (observed: 7 years inside a university)
- After an incident, only what was recorded AT THE TIME counts; reconstruction is
  worth nothing. (observed: insurance)
- Under pressure, people abandon trained routine and grab the shortest path.
  (observed: competitive sport — and every professional at 5pm)
```

"Busy professionals value efficiency" is a stereotype — it forbids nothing and
predicts nothing. A real behavior fact **predicts a specific user action** you must
design for. That prediction is what makes it falsifiable.

### Step 2 — Convert each fact into a design consequence

One fact → one or more requirements, each checkable in the built product:

| Behavior fact | Design consequence (examples across product types) |
|---|---|
| End-of-day batching | Batch processing is first-class; drafts are a real status; a catch-up surface lists what's unfinished |
| Arrive on time, want everything | Cold start fast to the primary screen; nothing heavy blocks reading existing data |
| Tidal usage | Peak-day capacity is THE performance target; a durable queue beats a 20% faster single run |
| The record-at-the-time is everything | Autosave as crash recovery; append-only logs; automatic backups on lifecycle events — never only on demand |
| Pressure kills routine | The shortest path must also be the safe path; safety requiring discipline gets skipped exactly when it matters |

If a fact produces no consequence you can point to in the product, either the fact is
too vague (rewrite it) or the product has a gap (log it).

### Step 3 — Find the request behind the request

A feature request is a **moment of friction** wrapped in the requester's guess at a
solution. Before building the guess, recover the moment:

```markdown
Request: "[verbatim]"
Moment: [where they were, what they were doing, what blocked them — ask if unknown]
Behavior fact it reveals: [add to the ledger]
Solution space: [their guess + at least one alternative serving the same moment]
```

Classic shape: a user asks for automation ("generate it automatically when I finish"),
and the moment behind it reveals the opposite requirement — the record is theirs
professionally, so they need *control over when*, not automation. The request behind
the request is often control, visibility, or reassurance — not the mechanism named.

### Step 4 — Date it, test it, let real users overrule it

- Every fact carries provenance (`observed: front desk, 2015` vs `observed: pilot
  user #2, this month`).
- A fresh observation from a real user of THIS product **outranks** an old fact from
  an adjacent domain. The old fact bought you a good first guess; the new data grades
  the guess.
- When contradicted, update the ledger **in writing** — never keep designing from a
  disproven fact.
- Adjacent-domain facts transfer at the level of **human pressure** (deadlines,
  arrival behavior, incident aftermath, status anxiety), never surface detail.
  "Golfers like X" doesn't transfer; "people under time pressure do X" might.

## Where Domain Knowledge Pays Most

Spend it here first, in order:

1. **Defaults** — what 90% of users live with forever; the expert knows which default
   matches the real workflow.
2. **What NOT to build** — knowing users won't manually tag/categorize/maintain
   something kills a whole feature class before it wastes a month.
3. **Empty states and first-run** — the last-minute arriver judges the product in 90
   seconds.
4. **Failure moments** — what exists as a record when things go wrong outranks any
   feature that works when things go right.
5. **Copy and naming** — the profession's own words; name by benefit, never by
   technology.

## Verification Checklist

```markdown
[ ] Every design decision traces to a dated behavior fact, not a stereotype.
[ ] Every fact predicts a specific action, names its source, and lives in a ledger
    the next session can find.
[ ] Every feature request was decomposed to its moment before its solution was
    accepted.
[ ] Cross-domain transfers happen at the pressure level, not the surface level.
[ ] Contradicting observations updated the ledger in writing.
[ ] Defaults, empty states, and failure moments got domain attention before features.
```

## Anti-Patterns

```markdown
- Persona documents full of adjectives ("busy", "tech-savvy") that forbid nothing.
- Building the requested solution without recovering the moment behind it.
- Treating your instinct as permanent truth after real users start disagreeing.
- Transferring surface details across domains.
- Spending domain insight on feature ideas while defaults and empty states stay
  generic.
- An undated behavior claim — if you can't say where it came from, you can't retire
  it when it expires.
```

## Key Principle

Your biography is a dataset nobody else has. The discipline is to treat it *as* a
dataset — extract facts, date them, design from them, and let newer data overrule
older data — instead of as a vibe. Products built this way feel like they were made by
someone who has stood behind the counter, because they were.
