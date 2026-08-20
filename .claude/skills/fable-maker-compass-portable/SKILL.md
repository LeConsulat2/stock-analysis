---
name: fable-maker-compass-portable
description: Portable, product-agnostic version of fable-maker-compass - copy this folder into any new project's .claude/skills/. In this repo, prefer fable-maker-compass; activate this one explicitly via "portable maker compass" or in a non-PRIVATE codebase. Encodes big-picture-first discipline for any product (app, web service, MCP server, community site): the one-sentence wedge, the picture-of-a-person-in-a-moment, the feature admission test with kill criteria, conviction-vs-stubbornness ledger, and the single sort key - distance from a stranger paying or using.
---

# Fable Maker Compass (Portable)

## Purpose

The sibling skills answer *how users behave* (user-reality) and *where behavior
knowledge comes from* (domain-lens). This one answers the question that comes first
and gets skipped most: **does this deserve to be built at all, and does it strengthen
the one reason this product wins?**

It encodes a maker's stance:

> "If you don't use it, you are at a loss. I don't mind if you don't use it — I know
> the benefits clearly."

That is not arrogance. It is the position *earned* by knowing precisely what the
product does, for whom, at what price, against what alternative — so precisely that a
non-buyer is making a legible mistake rather than a mystery to chase. A maker without
this clarity chases every objection with a feature; a maker with it ships a coherent
product and lets the wrong customers walk.

## Activation

Explicit: "portable maker compass" / "compass check" / "does this serve the wedge".

Self-activate when (in a project without an app-specific sibling):
- a new feature is proposed (by anyone, including your own brainstorming)
- scope grows mid-build ("while we're here, we could also...")
- pricing, positioning, or competitor comparison comes up
- a plan has 5+ parallel workstreams and needs a ruthless ordering
- the user asks "should I build X or Y first"

## The Wedge

Every surviving product has ONE sentence explaining why a rational buyer/user picks it
over the alternative. Three required components:

- **The moat** — what's structurally hard to copy (a data source, a privacy posture,
  a lived domain expertise, a distribution position)
- **The price/effort story** — the number or comparison the buyer can defend in one
  sentence to whoever they answer to
- **The named alternative** — what they'd otherwise do (a competitor, a spreadsheet,
  a generic LLM, doing nothing)

Example shapes:
- Desktop tool: *"Fully offline [records], the data never leaves the laptop, at $20/mo
  where the cloud alternative is $120/mo and requires trusting someone else's server."*
- MCP server: *"Current [jurisdiction]-specific answers with citations to the actual
  source document, inside the tools people already use — vs generic answers that
  hallucinate the rules."*
- Analysis product: *"The trained eye of someone who did this professionally, applied
  to your own footage, rewindable, at a fraction of one lesson's price."*

Rules:
- Written down, dated, updated deliberately — never drifted.
- Every feature, copy decision, and price is tested against it.
- **Price on the wedge, not on cost or fear.** A low "safe" price deletes the
  one-sentence story and signals a toy.

## Big Picture First

1. **The picture is a person in a moment**, not an architecture diagram: one specific
   human, at one specific moment of pressure, on realistic hardware, inside their real
   constraints. Every subsystem you build should be legible from that one image.
2. **Features are admitted into the picture; the picture is not stretched to fit
   features.** When something doesn't fit — even something already built — it is
   repositioned or cut.
3. **The picture has a recitable boundary**: 3–5 things the product will NEVER do.
   The boundary is as load-bearing as the features; it's part of what the buyer buys.

## The Feature Admission Test

Before any feature enters a plan, answer in writing:

```markdown
Feature: [name]
1. WEDGE: Does it strengthen the moat, the price story, or the named-alternative
   contrast? Which, in one sentence? (None → decoration; park it.)
2. MOMENT: Which dated behavior fact / user moment does it serve? ("A user asked"
   is a moment to decompose; "it would be cool" is not a moment.)
3. COST OF COHERENCE: What does it complicate — settings surface, support burden,
   the trust story, the interruption-tolerance matrix?
4. THE FIVE-CUSTOMER TEST: With only 5 paying customers/users, would this still be
   the best use of the week it costs? (Validation > features.)
5. KILL CRITERIA: What observation, by when, would prove it failed and should be
   removed? (A feature without kill criteria can never be removed, only
   accumulated.)
```

Passes 1, 2, and 4 → build. Fails 1 but passes 2 → candidate for a *different
product*; write it to the backlog with that label rather than diluting this one.

## Conviction Without Stubbornness

"I don't mind if you don't use it" is only healthy while it stays falsifiable:

- **Conviction** = holding price and boundary steady when ONE prospect objects,
  because the wedge math is written down and still true.
- **Stubbornness** = holding them when a *pattern* of the RIGHT users — people who
  match the picture — bounces off the same point. Three picture-matching users
  independently rejecting the same thing is data, not noise.
- The dividing line is a ledger: objections recorded with who said them and whether
  the objector matches the picture. Objections from outside the picture are priced
  at zero — that is what "I don't mind" means.

Corollary for spend: money and weeks go to what the *buyer of the wedge* needs to say
yes — trust items (signing, provenance, reliability drills, transparency) — before
they go to features existing users would merely enjoy.

## Ordering Many Workstreams

When the plan holds more than the next month can, sort by exactly one key:
**distance from a stranger paying (or, for free products, a stranger returning
unprompted).** Everything is either on that path or it isn't. Ties broken by
risk-of-embarrassment: an untested recovery/restore/accuracy path is a trust bomb,
not a task. Re-derive and obey this ordering every time scope tempts you sideways.

## Verification Checklist

```markdown
[ ] The wedge sentence exists in writing, is dated, and was consulted (not assumed).
[ ] Every feature in the plan has a written admission test; failures were parked
    with a label, not smuggled in.
[ ] Every admitted feature has kill criteria.
[ ] Price/effort-ask was tested against the wedge story, not against fear.
[ ] Objections are logged with source and picture-match; conviction was checked
    against the pattern, not the loudest voice.
[ ] The plan is ordered by distance-from-a-stranger-paying, and the ordering is
    stated, not implied.
```

## Anti-Patterns

```markdown
- Chasing a non-buyer's objection with a feature instead of checking picture-match.
- Pricing low "to be safe" — deleting the story the buyer needs to tell.
- Admitting a feature because it's already half-built (sunk-cost admission).
- Product ideas and payment-path blockers in the same list, unranked.
- Stretching the picture to justify a feature you're excited about.
- Conviction with no ledger behind it — that's just mood.
- Building for the 100th user's edge case before the 1st user exists.
```

## Key Principle

The compass is one image (the person in the moment), one sentence (the wedge), and
one sort key (distance from a stranger paying). Everything else — features, pricing,
objections, scope — is tested against those three and admitted, parked, or priced at
zero. Calm conviction is the *output* of that discipline, not a personality trait:
you can say "if you don't use it, that's your loss" only because you can show the
working.
