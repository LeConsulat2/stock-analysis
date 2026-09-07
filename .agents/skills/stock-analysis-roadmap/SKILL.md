---
name: stock-analysis-roadmap
description: Shape this personal stock-research product, choose practical milestones, and explain architecture or usability tradeoffs to its owner. Use for direction, Python versus TypeScript, folder growth, research habits, and scope decisions; use the rebuild skill for an implementation slice.
---

# Stock Analysis Roadmap

Help the owner choose a useful next capability and understand why it matters. The owner has professional golf experience and wants the essential starting ground, not a programming or finance curriculum. Use a golf analogy only when it clarifies a decision; professional background does not establish investment suitability.

## Ground the Recommendation

Read root `AGENTS.md`, `.claude/NewStart.md`, and the current file inventory before describing implementation status. Consult [personal-research-roadmap.md](../../plans/personal-research-roadmap.md) for the proposed product path and [rebuild-execution.md](../../plans/rebuild-execution.md) for engineering gates. Plans, code existence, and demonstrated behavior are different evidence.

Separate the user's desired outcome, approved implementation scope, and recommended changes when they diverge. Present proposed changes with reasons and acceptance conditions. Existing user authorization persists; do not ask again for routine work already requested. Recommendations alone do not silently rewrite application scope.

## Choose the Smallest Useful Milestone

Start with the recurring decision: understanding a company, checking a thesis, reviewing changes, or understanding portfolio exposure. Define what the owner will read and use before adding infrastructure. A demo can qualify its tool loop while remaining inadequate for an investment decision.

Recommend Python for this repository's research core unless new constraints justify a change. Keep the approved six root modules during the rebuild. Move related modules into packages when growth makes ownership difficult; add a database when dated records need querying, a UI when repeated interaction is cumbersome, and concurrency when measured waiting is a bottleneck. Explain technology choices as tradeoffs, not universal scalability laws.

Keep four-provider compatibility as the existing goal while qualifying one complete path first. Verify model versions at implementation time. More providers or analyst roles do not by themselves provide stronger financial evidence.

## Communicate at the Owner's Level

Explain the business question, observable outcome, structure, and proof in that order. Teach only concepts needed to assess the result: business economics, price versus value, cash versus accounting profit, uncertainty, and portfolio exposure.

Make progress under stated assumptions. Ask only for missing inputs that materially change the plan, such as horizon, markets, weekly time, or interface preference. Ask about financial circumstances when suitability or sizing is in scope. Open questions are not automatic permission gates.

Deliver a prioritized recommendation with a concrete first milestone, acceptance checks, and reasons later capabilities should wait. When asked to save a plan, date it, label recommendations versus decisions, and link the existing engineering plan instead of creating a competing implementation authority.
