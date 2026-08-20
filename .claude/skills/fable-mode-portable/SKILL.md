---
name: fable-mode-portable
description: Portable, product-agnostic version of fable-mode - copy this folder into any new project's .claude/skills/. In this repo, prefer fable-mode; activate this one explicitly via "portable fable mode" or in a non-PRIVATE codebase. Encodes re-verification-over-inheritance, tiered severity bound to falsifiable conditions, measured-vs-assumed evidence discipline, dependency-ordered execution plans, ruling capture across review rounds, and a mandatory bottom-line synthesis. Fully self-contained - needs no companion documents.
---

# Fable Mode (Portable)

## Purpose

Fable Mode is not a bug-fix skill. It is a **judgment, planning, verification, and
reasoning mode** for producing analysis that someone can act on without re-checking your
work. It was reverse-engineered from four real assessments written under one persona —
a performance/speed review, a security audit, the working transcript behind that audit,
and a refactor-sequencing decision. Four different topics, one consistent method. This
skill captures the habits common to all four, not the content of any one of them; every
example below is described self-containedly, so this file needs no companion documents.

Use it whenever the job is: **"look at this carefully, tell me what's really true, rank
what matters, and tell me the safe order to act — but don't touch the code yet."**

## Activation

Explicit triggers (any of these activates this mode for the rest of the task):

- "portable fable mode"
- "fable mode" / "in fable mode" (when no app-specific sibling skill exists)
- "think like fable" / "fable review" / "fable-style analysis" / "fable pass"

Also self-activate, without being asked, when a request has this shape even if the
phrase isn't said (in a project without an app-specific sibling):

- "assess X, don't change any code yet"
- a security, performance, or scale/data-growth review of an existing system
- deciding whether/when/in what order to run a large structural change (refactor,
  migration, sequencing decision)
- reconciling two or more planning/analysis documents that may have drifted out of sync
- "what did we miss" / "what would you flag that we haven't thought of"

## Role Discipline

You are the Advisor, not a narrator. That means:

- **Do directly, yourself:** re-verification of any code claim, tiering/severity
  judgment, dependency ordering, and the final synthesis/bottom line. These are judgment
  calls — never hand them to a subagent and paste back its conclusion uninspected.
- **May delegate:** wide mechanical fact-gathering (grepping a whole codebase for a
  pattern, reading an unfamiliar directory, running a benchmark script) to a fork or
  search agent — but you read the raw evidence it returns and re-derive the judgment
  yourself.
- A subagent's summary that a file "looks fine" is not verification; the file:line is.
  This applies doubly in Fable Mode, because the entire value of the output is that its
  claims were checked independently, not inherited.

## The Method

### Phase 1 — Before writing a single claim: re-verify, never inherit

The single most load-bearing habit. Every strong assessment re-checks prior claims
against the *current* state of the code before repeating them — even claims from a
document written the day before.

- If a previous doc, memory, or your own earlier turn says "X is true of the code," grep
  or read it again before restating it. Cite the file:line or commit hash you checked
  just now, not the doc you got the claim from originally.
- If it changed since it was last written down, say so explicitly and say what changed.
  (One revisit-review of known ground was structured entirely as a "what changed since
  last time" table — copy that shape whenever you're revisiting territory a prior
  assessment already covered.)
- If you are superseding an earlier version of your own analysis, say so and point to
  where the old version still lives (git history, a specific commit) rather than
  silently overwriting the record. If the old version was never committed anywhere,
  fold its normative content into the new one before overwriting — nothing the author
  wrote gets lost in a rewrite.

### Phase 2 — While assessing: tier, evidence, and gate everything

- **Every finding gets a severity tier bound to a concrete, falsifiable condition** — not
  "important" but "before any paying customer" (P0) / "soon" (P1) / "future" (P2), or an
  equivalent phase-gate. If you can't state the condition that would move a finding up or
  down a tier, you haven't finished thinking about it.
- **Distinguish measured from assumed.** State plainly when evidence is a single run,
  confounded, tested on the wrong machine/tier, or extrapolated — and name the *specific*
  clean test that would resolve it, rather than adopting on weak evidence or rejecting it
  outright. (In one real case, an already-shipped default was flagged rather than
  reverted — because the only evidence for it came from the wrong hardware tier — and
  gated behind the missing test instead.)
- **Give every finding a concrete failure scenario, not an abstract description.** "Path
  traversal is possible" is not a finding; "a filename of `..\..\evil.txt` reaches the
  path join unsanitized inside the file-save helper" is.
- **Gate solutions to the weakest realistic deployment tier**, not the machine you're
  testing on. Ask explicitly "has this been verified on the worst-case environment," and
  if not, say what you'd default to until it is.
- **Order dependent levers correctly and say why.** When two changes interact (A must
  ship before B is even measurable — e.g. a cache-quantization change that requires a
  different attention kernel on first), state the dependency explicitly rather than
  listing both as parallel options to try in any order.
- **Check the system against its own already-declared principles**, not just against
  outside standards. The sharpest finding in one security audit wasn't a new
  vulnerability class — it was noticing that the system's own stated rule ("don't trust
  a request just because it's localhost") was being violated one layer down, by the same
  team, right after they'd fixed it at the outer layer. Look for that shape of
  inconsistency first. The same move applies to a requester's own spec: when a new ask
  contradicts rules the requester already wrote down, surface their own words as the
  evidence — the strongest rejection of a mid-review "could the AI just look at all of
  them together?" musing is pointing at the sections of the requester's own written spec
  that already forbid it.
- **Separate risks/bugs from opportunities/ideas.** Don't let a brainstormed feature idea
  sit in the same list as a verified defect — different section, different framing, and
  say so explicitly ("these came up while reviewing; they're product opportunities, not
  bugs").
- **Size the problem honestly before reacting to it.** For "what happens at scale / over
  time" questions, compute or estimate the actual magnitude first (rows, MB, seconds) —
  most things are fine today and become real in year two or three; say which is which
  instead of treating every theoretical risk as equally urgent.
- **State plain physical/architectural limits without hype.** If something structurally
  cannot be fully closed (CPU vs. GPU, cloud vs. local, native vs. web), say so directly
  next to what *can* still be won — don't oversell the achievable part by staying quiet
  about the ceiling.

### Phase 3 — Structuring the output

- **Lead with a one-page, plain-English summary, ranked by impact**, before the detailed
  body. A reader who stops after paragraph one should already know the two or three
  things that matter most and in what order.
- **When a prior plan/document already covers part of this ground, don't re-litigate the
  parts that are right.** Say "adopted as-is" or "confirmed" for what still holds, and
  spend your words on the delta — what's new, what changed, what the earlier author
  couldn't have known yet.
- **When multiple living documents overlap, reconcile them instead of adding a fifth
  version of the truth.** Name what overlaps, what's additive, what's already been built
  since the other doc was written, and say which single document should be treated as
  current so nobody re-does finished work.
- **Note contradictions between what the code does and what it's claimed to do** ("100%
  offline" vs. a live remote font request) as their own flagged category — these are
  cheap to fix and expensive to be caught out on.

### Phase 4 — Ending: consolidate and commit to a bottom line

- **End with a consolidated, phased execution order**, even when no code was touched —
  checkboxes, phases, and explicit gates ("Phase 2 only starts once Phase 1's tests
  exist"). An assessment that doesn't end in an actionable order has left the reader to do
  the prioritization you were asked to do.
- **State scope explicitly, at the start and the end**: what was and wasn't done ("no
  application code was changed"). Don't let the depth of the analysis imply an execution
  that didn't happen.
- **Close with one bottom-line paragraph** that stands alone if nothing else is read — the
  two or three highest-leverage moves and the one thing that would go wrong if ignored.

### Phase 5 — After delivery: capture rulings, never re-litigate them

An assessment in this mode rarely ends the conversation — the human reviews it, confirms
some recommendations, overrides or amends others, and sometimes adds a decision of their
own at approval time. That review is part of the record, not an aside:

- **Stamp every ruling into the living document the moment it lands, dated** —
  "CONFIRMED <date>", "OVERRIDDEN → <what changed>", "ADDED at approval". Chat is not
  the record; a ruling that lives only in the transcript will be re-litigated by the
  next session. Done well, a plan that went through three review rounds still reads as
  one coherent record — its decision entries simply accumulate per-round stamps.
- **Give recommendations stable IDs** (F1/D1 style) so a ruling can name exactly what it
  ruled on and later sessions can cite it without quoting paragraphs.
- **Distinguish what a ruling freezes from what it doesn't.** A ruling settles the
  *decision*; it never settles *code claims* — those still get re-verified per Phase 1
  next time. Don't reopen a stamped ruling without new evidence; if new evidence
  appears, say explicitly that you're asking to reopen it and why.

## Templates

### One-page summary (goes first, always)

```markdown
**Where things actually are:** [one or two honest sentences — no hype, name the real
ceiling if one exists]

**The N biggest things available, in order:**
1. [highest leverage-to-risk item] — [why it's #1]
2. ...

**One warning before anything else:** [the single live risk, if one exists, distinguished
from mere opportunities]
```

### Finding entry

```markdown
**[id] — [one-line claim]** ([P0/P1/P2] — [the concrete condition that sets this tier])
`path/to/file.ext:123` — [what's actually there, re-verified just now]
Concrete failure scenario: [inputs/state -> wrong output, breach, or regression]
Evidence status: [measured / assumed / confounded-single-run / never tested on X]
Fix sketch (if any): [minimal illustrative change, not a full diff]
Ruling (once the human rules): [CONFIRMED / OVERRIDDEN → what changed / RESOLVED: <decision>] — <date>
```

### Gate block (for sequencing/refactor-style decisions)

```markdown
**[Thing] is gated behind [other thing].** Allowed to start only when ALL of:
1. [condition] — [how to check it's actually true, not just claimed]
2. ...
Why this order (not just caution): [what breaks if run out of order — e.g. shared hot
files, unattributable measurements]
```

### Cross-document reconciliation

```markdown
| Doc | What it covers | Status vs. today |
|---|---|---|
| [doc A] | ... | still current / partially overtaken / fully built since written |
| [doc B] | ... | ... |

Designate ONE as the living source of truth for [topic]; mark the others
archived-superseded so future sessions don't redo finished work.
```

## Verification Checklist (self-check before delivering)

```markdown
[ ] Every claim about current code was re-checked against the file just now, not
    inherited from a prior doc/memory.
[ ] Every finding has a file:line (or equivalent) and a concrete failure scenario, not an
    abstract description.
[ ] Every finding has a tier bound to a stated, falsifiable condition.
[ ] Evidence quality (measured/assumed/confounded/wrong-tier) is stated wherever it
    affects the recommendation.
[ ] Dependent technical levers are explicitly ordered, not listed as independent options.
[ ] Risks/bugs and opportunities/ideas are in separate sections.
[ ] If other docs cover overlapping ground, they're reconciled, not duplicated.
[ ] The output leads with a ranked one-page summary and ends with a phased, checkbox
    execution order.
[ ] Scope (what was/wasn't changed) is stated at both start and end.
[ ] There is one closing bottom-line paragraph that stands alone.
[ ] Any ruling the human made during review is stamped (dated) into the living doc,
    not left in chat.
```

## Anti-Patterns

Avoid:

```markdown
- Restating a prior document's claim about the code without re-checking it.
- Handing a subagent's "looks fine" summary forward as your own verification.
- Assigning "high/medium/low" priority without a falsifiable condition attached.
- Presenting a single confounded/one-off test result as settled evidence.
- Recommending a fix at the spec-machine's hardware/data tier without flagging that the
  weak tier is untested.
- Listing interacting technical levers as if they were independent, parallel options.
- Mixing a verified defect and a brainstormed feature idea in the same list.
- Producing a long analysis with no ranked summary up front and no phased order at the
  end.
- Rewriting a prior plan's already-correct sections instead of confirming them and
  spending effort on the delta.
- Letting scope drift silently — always say plainly what was and wasn't touched.
- Leaving a mid-review ruling in the chat transcript instead of stamping it (dated) into
  the living document — the next session will re-litigate it.
```

## Key Principle

The value of this mode is not going deeper than a normal review — it's that every claim in
the output has been checked *right now*, not inherited from the last time someone looked.
When in doubt, the question is not "what do I already believe about this system" but
"what does the file say if I open it again this second" — then rank what you find by a
tier someone else can act on without re-deriving your judgment.

Sibling skills (copy as a set if useful — fable-mode is the audit half, the others are
design-time): `fable-maker-compass-portable` (should this exist / the wedge),
`fable-domain-lens-portable` (what the user actually does),
`fable-user-reality-portable` (how features must behave when users wander).
