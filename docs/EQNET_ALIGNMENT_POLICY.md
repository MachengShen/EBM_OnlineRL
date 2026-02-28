# EqNet Self-Improvement Alignment Policy

This repository now defaults Maze2D synthetic training to a single policy profile:

- `alignment_profile=eqnet_self_improve_v1`

## Required settings in this profile

The following are enforced at runtime in training scripts unless you explicitly opt out:

- `n_episodes == 0`
- `train_steps == 0`
- `online_self_improve == true`
- `online_rounds > 0`
- `online_collect_transition_budget_per_round > 0`
- `disable_online_collection == false` (when available)
- `num_eval_queries * query_batch_size == 16`

## How to intentionally run legacy/offline settings

Use:

- `--alignment_profile legacy_offline`

This opt-out is explicit by design, so old offline-first configs are not used accidentally.

## Scripts covered

- `scripts/synthetic_maze2d_diffuser_probe.py`
- `scripts/synthetic_maze2d_gcbc_her_probe.py`
- `scripts/synthetic_maze2d_sac_her_probe.py`

## Launcher alignment

`exp_swap_matrix_maze2d.py` now emits aligned defaults by default (`n_episodes=0`, `train_steps=0`, online collection budget enabled, 16 eval trajectories/checkpoint).
