# 2026-02-16 Agentic Near-Open-Ended Auto-Decider (Conservative)

## Goal
Replace fixed-space auto-HPO with a conservative agentic controller that can propose broader changes while preserving the EBM-onlineRL problem definition.

## Architecture
1. Hypothesis agent
- Produces proposal objects: rationale, trial config, and optional common-arg overrides.
- Starts from the incumbent and recent evidence, then explores nearby and budget-promoted variants.

2. Constraint checker
- Enforces invariants so proposals stay in scope:
  - online self-improvement loop must remain enabled.
  - evaluation protocol remains prefix-horizon based.
  - rollout and collection horizon consistency checks.
  - success threshold and key numerical ranges are sane.

3. Experiment runner/monitor
- Reuses the existing trial runner from `overnight_maze2d_autodecider.py`.
- Keeps monitor snapshots and timeout behavior compatible with current tooling.

4. Critic
- Updates simple beliefs from observed score deltas.
- Beliefs are persisted for auditability (`beliefs.json`).

5. Promotion policy (conservative)
- Proposal must clear pilot gain threshold.
- Proposal must pass confirmation run (second seed) before becoming incumbent.
- No confirmation => no promotion.

## Conservative Defaults
- `accept_delta=0.02`
- `require_confirmation=true`
- `proposals_per_round=4`
- `per_trial_timeout_min=35`

## Takeover Plan
1. Start in smoke mode and verify artifacts:
- proposals log, beliefs, results CSV, summary.
2. Run short conservative live budget (2-4 hours).
3. Compare against current auto-decider baseline on equal wall-clock and eval protocol.
4. If robust, use the agentic launcher as primary overnight entry point.

## Reusable Skillization Criteria
Promote to a Codex skill when:
1. At least two independent runs finish without integrity failures.
2. Promotion decisions are reproducible from logged artifacts.
3. No invariant-violation proposal is executed.
4. Mean objective is not worse than baseline by > 0.02 on matched budget.
