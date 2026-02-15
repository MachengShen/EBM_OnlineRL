# Maze2D Overnight Auto-Decider (Adaptive OnlineRL Loop)

## Goal
Run Maze2D online Diffuser experiments overnight **without manual prompting**, while remaining auditable:
- keep launching trials
- monitor intermediate results
- adaptively shift attention toward impactful hyperparameters
- optionally increase budgets for promising configurations

This is not intended to be a statistically perfect hyperparameter optimizer. It is an engineering tool to
avoid wasting overnight wall-clock on obviously unpromising settings and to discover high-leverage knobs.

## Why This Exists (Problem)
The chat interface is user-turn-driven; the agent cannot reliably "keep sending updates" without re-prompting.
Additionally, intermediate monitoring was blocked because `progress_metrics.csv` and `online_collection.csv`
were only written at the end of a run.

## Implementation Summary
### 1) Stream intermediate metrics to disk (enables monitoring)
In `scripts/synthetic_maze2d_diffuser_probe.py`:
- flush `progress_metrics.csv` after each evaluation event (`--eval_goal_every`).
- flush `online_collection.csv` after each online collection round.

This makes it possible for monitor/decider loops to observe `succ@{64,128,192,256}` while the run is active.

### 2) Adaptive controller (explore -> rank -> promote)
In `scripts/overnight_maze2d_autodecider.py`:
- maintain a results table over completed trials (`autodecider_results.csv`).
- compute a crude "hyperparameter influence" score as range-of-means across values (`autodecider_importance.csv`).
- choose next trial via:
  - **promotion** with probability `promote_prob`: take best observed base config and increase its budget
    (move to a larger `(train_steps, online_rounds)` level), if untried.
  - otherwise **bandit-weighted sampling** over key knobs with a softmax distribution over mean objective.

### Objective
Primary: `objective_succ_h256` = best observed `rollout_goal_success_rate_h256` within the run.

Budgets are allowed to vary (user requested freedom). Because comparisons become less "fair", we record:
- budgets (`train_steps`, `online_rounds`) alongside outcomes
- elapsed seconds
- replay transitions from `online_collection.csv` when available

## Tuned Knobs (Default Scope)
- `online_collect_episodes_per_round`
- `online_train_steps_per_round`
- `online_replan_every_n_steps`
- `online_goal_geom_p`
- budget knobs: `train_steps`, `online_rounds`

Other parameters are held fixed for comparability and to keep search space sane.

## Guardrails / Auditability
- Every trial stores:
  - command (`cmd.json`)
  - trial params (`trial_config.json`)
  - stdout/stderr (`run.log`)
- Controller stores:
  - config snapshot (`autodecider_config.json`)
  - consolidated results (`autodecider_results.csv`)
  - influence ranking (`autodecider_importance.csv`)

The intent is that a human can reproduce or post-hoc analyze every decision.

## How To Run
Preferred: in `tmux`.

```bash
/root/ebm-online-rl-prototype/scripts/launch_overnight_maze2d_autodecider_tmux.sh
```

Environment knobs:
- `BUDGET_HOURS` (default 12)
- `MAX_TRIALS` (default 12)
- `MONITOR_EVERY_SEC` (default 300)
- `SEED` (default 0)

## Open Questions / Next Iterations
- Add resume-from-checkpoint for budget promotion (currently promotion reruns from scratch).
- Add multi-objective scoring (success vs compute) for Pareto-style decision making.
- Add replication (re-run top configs) to reduce noise-driven overfitting.

