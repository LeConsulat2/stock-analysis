---
name: fable-user-reality
description: Activate explicitly on "fable user reality" / "user reality pass" / "wandering user check". Also fits, unprompted, whenever designing or reviewing ANY feature with a long-running operation (transcription, AI generation, download, export, backup), a multi-step flow, or a disabled control. Encodes the core product law - users treat the app like a room they walk around in, not a script they follow - and the guardrail taxonomy (Allow / Absorb / Guard / Disable) with the rule that Disable is the last resort and must name what would actually break.
---

# Fable User Reality

## Purpose

This is a **design-time skill**, not a review skill. It exists because of one observed
truth that every good feature in this codebase eventually had to learn the hard way:

> **Users treat your app like a room, not a script.** The moment something is allowed,
> someone will do it — mid-operation, in parallel, out of order, at the last minute,
> and while asking a question about something else.

The developer's mental model is a happy path: *start beautify → wait → read result*.
The real user's behavior is: *start beautify → switch to another student → start writing
that student's note → remember an audio file → try to transcribe it too → close the
laptop lid*. Every one of those steps is reasonable **to them**. A product feels smooth
when none of those steps loses work, and feels broken when any of them does — even if
the developer never imagined the sequence.

The goal is never to trap the user in the flow you designed. The goal is to make every
flow **interruption-tolerant**, and to disable only the minimal surface where a real,
nameable corruption or safety problem exists.

## Activation

Explicit triggers:
- "fable user reality" / "user reality pass" / "wandering user check"
- "run the four fates" / "walk the wandering user script"

Self-activate, without being asked, when:
- designing or planning any feature with a long-running operation (AI call, transcribe,
  download, export, backup, upload)
- writing `disabled`, a blocking modal, a full-screen lock, or a "please wait" state
- reviewing a bug where "the user did X while Y was running"
- a plan says "the user will then..." — that phrase is the tell that a happy path is
  being assumed

## The Law and Its Corollaries

**The Law:** if the UI physically allows an action, some user will take it at the worst
possible moment. Design for that user, not the obedient one.

Corollaries, each proven in this codebase:

1. **State must outlive the screen.** A user who navigates away and comes back expects
   their work to still be there and still be running.
   - Built example: `SessionsPage`/`MeetingDashboard` are **always mounted**, hidden via
     CSS — so beautify/summarise streaming survives tab switches (`App.tsx`).
   - Broken example (found 2026-07-06): model download progress lost on Settings
     nav-away, because state lived in the unmounted component.
   - Rule: any long-running operation's state lives **above** the component that
     started it (parent state, backend, or queue) — never in the leaf.

2. **Parallel intent is normal, not abuse.** "Transcribe another audio while beautify is
   running" is not a user error; it's a Tuesday. Either the system can genuinely run
   both (Absorb — queue it), or it cannot (Guard — say so with the reason and what to do
   instead). Silent failure or a mystery-disabled button is never an option.
   - Built example: the `audio_queue` (max-10, durable, per-item "+ Here") exists
     precisely because users record faster than they transcribe.

3. **Switching entities mid-flow is the #1 wander.** A counsellor mid-beautify on
   student A **will** open student B. The question to answer at design time: what is on
   screen for B, what happens to A's in-flight work, and how does the user get back to
   A's result? (See also the known gap in memory: no in-view affordance to switch
   students — the wander exists whether or not you gave it a button.)

4. **Closing the app is a user action, not a crash.** Every long-running operation needs
   a defined answer to "lid closed mid-way": finish, persist-and-resume, or
   confirm-before-close. "Undefined" is the only wrong answer.
   - Built examples: `before-quit` backup interceptor; startup reset of stuck
     `status='transcribing'` queue items; autosave-as-crash-recovery.

5. **A disabled control is a question the user can't ask.** Every `disabled` must carry
   its reason to the user (tooltip, inline line, badge) and must be scoped to the
   smallest surface — disable the *conflicting action*, never the *whole page*.

## The Guardrail Taxonomy

For every user action that could collide with system state, choose ONE deliberately —
in this order of preference:

| Level | Meaning | When | Example in this app |
|---|---|---|---|
| **Allow** | Action just works; state survives | No real conflict exists | Tab-switching during beautify (always-mounted pages) |
| **Absorb** | Action is accepted and queued/backgrounded | System can't do it *now* but can do it *later* | Audio recorded during transcription → lands in `audio_queue` |
| **Guard** | Action allowed after an informed confirm, or warned | Action is destructive to in-flight work but legitimate | Stop button on batch beautify (aborts stream, rolls back current, keeps finished) |
| **Disable** | Action blocked, minimal surface, reason shown | Real corruption/safety issue you can NAME | Read-only mode while device-transfer is `pending` |

**The Disable test:** before writing `disabled`, complete this sentence with a concrete
failure — *"If the user does this while X is running, then ___ is corrupted/lost/
double-charged."* If you can't complete it, you don't get to disable; pick Allow,
Absorb, or Guard. If you can complete it, disable the smallest thing that prevents it,
and show the reason.

## The Four Fates Walk (design-time checklist)

For every feature with any operation longer than ~2 seconds, script and answer all four
fates **before building**:

```markdown
Feature: [name]  Operation: [the long-running thing]

Fate 1 — WANDER: user navigates away mid-operation, returns 3 minutes later.
  → What survives? Where does state live? How do they find the result?

Fate 2 — PARALLEL: user starts a second instance (same op, or a different heavy op).
  → Allow / Absorb / Guard / Disable — which, and why (the named failure)?

Fate 3 — SWITCH: user changes the active entity (student/client/meeting) mid-operation.
  → What does the new entity's screen show? Is the old work findable? Can they work
    on the new entity immediately?

Fate 4 — VANISH: user closes the app (or Windows kills it) mid-operation.
  → Finish / persist-and-resume / confirm-before-close — which, and what does the
    user see on next launch?
```

A feature plan that cannot answer all four is not finished planning. Write the answers
into the plan doc so the next session inherits them.

## The Last-Minute User

A special persona learned from golf operations, true everywhere: **people arrive exactly
on time and want everything at once.** The counsellor equivalent: opens the app 90
seconds before a session, needs the last note, the new note, and the schedule —
simultaneously — on a laptop that just woke up.

Design consequences:
- Cold-start paths must degrade gracefully (the 60s auto-login fallback exists because
  post-update AV scans are real; never "optimize" a startup wait below reality).
- The most-needed information must be reachable in one action from launch.
- Anything that must warm up (models, servers) warms lazily and never blocks reading
  existing data.

## Verification Checklist (self-check before delivering a design or review)

```markdown
[ ] Every operation >2s has all Four Fates answered in writing.
[ ] Every `disabled` in the diff/plan passes the Disable test (named corruption) and
    shows its reason to the user.
[ ] No long-running state lives in a component that can unmount.
[ ] Parallel attempts are Absorbed or Guarded — never silently ignored.
[ ] Entity-switch mid-operation has a defined screen state and a way back.
[ ] App-close mid-operation has a defined behavior and a defined next-launch state.
[ ] The last-minute user can reach their primary need in one action from launch.
```

## Anti-Patterns

```markdown
- "The user will then click..." in a plan, with no branch for them not doing it.
- Disabling a whole page/tab because one operation is running somewhere on it.
- A disabled button with no visible reason (the user reads it as "broken").
- Long-running state in a leaf component (dies on nav-away).
- Treating a second parallel request as impossible instead of inevitable.
- Blocking app close with no explanation, or allowing it with silent data loss.
- Testing only the happy path order; the wandering order IS the common order.
```

## Key Principle

Smoothness is not animation — it is **never punishing a reasonable human sequence**.
You cannot enumerate every sequence, but you can make every operation
interruption-tolerant and every restriction minimal and explained. When in doubt, the
question is not "why would they do that?" but "when they do that, what do they lose?"
— and the answer must be "nothing they typed."

Sibling skills: `fable-domain-lens` (where the behavior knowledge comes from),
`fable-maker-compass` (whether the feature deserves to exist), `fable-mode` (how to
audit what's already built).
