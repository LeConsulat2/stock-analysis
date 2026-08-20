---
name: fable-user-reality-portable
description: Portable, product-agnostic version of fable-user-reality - copy this folder into any new project's .claude/skills/. In this repo, prefer fable-user-reality (it cites local code); activate this one explicitly via "portable user reality" or when working on a codebase that is not PRIVATE. Encodes the law that users treat software like a room they walk around in, not a script - the Four Fates walk (Wander/Parallel/Switch/Vanish) for every long-running operation, and the Allow/Absorb/Guard/Disable guardrail taxonomy where Disable must name a concrete corruption and show its reason.
---

# Fable User Reality (Portable)

## Purpose

A **design-time skill** built on one law, observed across desktop apps, web apps,
mobile apps, and API/MCP servers alike:

> **Users treat your product like a room, not a script.** The moment something is
> allowed, someone will do it — mid-operation, in parallel, out of order, at the last
> minute, and while asking about something else.

The developer imagines a happy path: *start the operation → wait → read the result*.
The real user: *starts it → navigates elsewhere → starts a second one → switches to a
different record → closes the lid*. Every step is reasonable **to them**. The product
feels smooth when none of those steps loses work, and broken when any does.

The goal is never to trap the user in your designed flow. It is to make every flow
**interruption-tolerant**, and to restrict only the minimal surface where a real,
nameable corruption exists.

## Activation

Explicit: "portable user reality" / "user reality pass" / "wandering user check" /
"run the four fates".

Self-activate when (in a project without an app-specific sibling skill):
- designing any feature with an operation longer than ~2 seconds (upload, generation,
  analysis, sync, export, transcription, long query)
- writing `disabled`, a blocking modal, a lock screen, or a "please wait" state
- reviewing a bug of the form "the user did X while Y was running"
- a plan contains "the user will then..." — the tell of an assumed happy path

## The Law's Corollaries

1. **State must outlive the screen.** Whoever navigates away and returns expects work
   still there and still running. Long-running state lives **above** the component/
   page/request that started it — in a parent, a store, a backend job, a queue —
   never in the leaf that dies on navigation. (Web: not in the route component.
   Mobile: survives the OS killing the activity. MCP/API: survives the client
   disconnecting — job id, not held socket.)
2. **Parallel intent is normal, not abuse.** Starting a second heavy operation while
   the first runs is a Tuesday. Either the system genuinely runs both (Absorb —
   queue it) or it can't (Guard — say so, with the reason and the alternative).
   Silent failure or a mystery-dead button is never an option.
3. **Switching entities mid-flow is the #1 wander.** Mid-operation on record A, the
   user opens record B. Answer at design time: what does B's screen show, what
   happens to A's in-flight work, and how does the user get back to A's result?
4. **Closing the app is a user action, not a crash.** Every long operation defines
   its lid-close behavior: finish, persist-and-resume, or confirm-before-close.
   "Undefined" is the only wrong answer. On next launch, anything stuck
   `in_progress` is crash evidence — reset it visibly, don't let it wedge forever.
5. **A disabled control is a question the user can't ask.** Every restriction shows
   its reason (tooltip, inline text, badge) and covers the smallest surface —
   disable the conflicting action, never the whole page.

## The Guardrail Taxonomy

For every action that could collide with system state, choose one — in this order of
preference:

| Level | Meaning | When |
|---|---|---|
| **Allow** | Just works; state survives | No real conflict exists |
| **Absorb** | Accepted and queued/backgrounded | Can't run it *now*, can run it *later* |
| **Guard** | Allowed after an informed confirm / warned | Destructive to in-flight work but legitimate (e.g. a Stop button that rolls back the current item and keeps finished ones) |
| **Disable** | Blocked; minimal surface; reason shown | A corruption/safety failure you can NAME |

**The Disable test:** complete this sentence with something concrete — *"If the user
does this while X is running, then ___ is corrupted / lost / double-charged."* Can't
complete it → you don't get to disable; pick Allow, Absorb, or Guard. Can complete
it → disable the smallest thing that prevents it, and show why.

## The Four Fates Walk (before building)

```markdown
Feature: [name]  Operation: [the long-running thing]

Fate 1 — WANDER: user navigates away mid-operation, returns 3 minutes later.
  → What survives? Where does state live? How do they find the result?

Fate 2 — PARALLEL: user starts a second instance (same op, or another heavy op).
  → Allow / Absorb / Guard / Disable — which, and why (the named failure)?

Fate 3 — SWITCH: user changes the active entity (record/client/project) mid-op.
  → What does the new entity's screen show? Is the old work findable?
    Can they work on the new entity immediately?

Fate 4 — VANISH: user closes the app / OS kills it / client disconnects mid-op.
  → Finish / persist-and-resume / confirm-before-close — which, and what do they
    see on next launch?
```

A feature plan that can't answer all four isn't finished planning. Write the answers
into the plan so the next session inherits them.

## The Last-Minute User

People arrive exactly on time and want everything at once — the golfer walking onto
the tee, the practitioner opening the laptop 90 seconds before a session, the migrant
querying visa rules at 11pm mid-application. Consequences:

- Cold start degrades gracefully; never tune a startup wait below the reality of slow
  disks, cold caches, and antivirus scans.
- The primary need is reachable in one action from launch.
- Anything that must warm up (models, servers, indexes) warms lazily and never blocks
  reading existing data.

## Verification Checklist

```markdown
[ ] Every operation >2s has all Four Fates answered in writing.
[ ] Every restriction passes the Disable test (named corruption) and shows its
    reason to the user.
[ ] No long-running state lives in a component/request that can die on navigation
    or disconnect.
[ ] Parallel attempts are Absorbed or Guarded — never silently ignored.
[ ] Entity-switch mid-operation has a defined screen state and a way back.
[ ] Close/kill mid-operation has a defined behavior and a defined next-launch state
    (stuck in_progress rows get reset, visibly).
[ ] The last-minute user reaches their primary need in one action from launch.
```

## Anti-Patterns

```markdown
- "The user will then click..." with no branch for them not doing it.
- Disabling a whole page because one operation runs somewhere on it.
- A disabled control with no visible reason (reads as "broken").
- Long-running state in a leaf that unmounts / a request that disconnects.
- Treating a second parallel request as impossible instead of inevitable.
- Blocking close with no explanation, or allowing it with silent loss.
- Testing only the happy-path order; the wandering order IS the common order.
```

## Key Principle

Smoothness is not animation — it is **never punishing a reasonable human sequence**.
You can't enumerate every sequence, but you can make every operation
interruption-tolerant and every restriction minimal and explained. When in doubt, ask
not "why would they do that?" but "when they do that, what do they lose?" — and the
answer must be "nothing they made."
