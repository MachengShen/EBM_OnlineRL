# Diffuser Key Deltas Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the highest-impact Diffuser behaviors (weighted loss, EMA inference model, waypoint control) to the PointMass online prototype and verify impact with an A/B run.

**Architecture:** Keep existing online loop and model architecture, but augment optimization objective and inference policy. Use EMA-smoothed model for sampling and optionally compute actions from predicted next-state waypoint instead of raw predicted action.

**Tech Stack:** Python, PyTorch, existing `ebm_online_rl` modules, existing training/eval script.

---

### Task 1: Weighted loss in diffusion objective

**Files:**
- Modify: `/root/ebm-online-rl-prototype/ebm_online_rl/online/diffusion.py`
- Modify: `/root/ebm-online-rl-prototype/scripts/online_pointmass_goal_diffuser.py`

**Steps:**
1. Add diffusion config knobs for `action_weight`, `loss_discount`, and optional per-observation weights.
2. Build a `[H+1, transition_dim]` weight tensor similar to Diffuser and apply it to elementwise L2 loss.
3. Keep existing API compatibility (loss still returns scalar tensor).

### Task 2: EMA model for planning/evaluation

**Files:**
- Modify: `/root/ebm-online-rl-prototype/scripts/online_pointmass_goal_diffuser.py`

**Steps:**
1. Add lightweight EMA helper (`ema_decay`, `ema_start_step`, `ema_update_every`).
2. Maintain `ema_diffusion` copy and update it during training bursts.
3. Use EMA model for planner rollouts/eval/checkpoint sampling; keep raw model for optimization.
4. Save both raw and EMA state dicts in checkpoints.

### Task 3: Waypoint-based control option

**Files:**
- Modify: `/root/ebm-online-rl-prototype/ebm_online_rl/online/planner.py`
- Modify: `/root/ebm-online-rl-prototype/scripts/online_pointmass_goal_diffuser.py`

**Steps:**
1. Add `control_mode` option (`action` vs `waypoint`).
2. For `waypoint`, sample trajectory and use predicted next state as waypoint to compute action `next_state - obs`.
3. Keep existing conditioning checks and action scaling behavior consistent.

### Task 4: Verification experiments

**Files:**
- Create outputs under: `/root/ebm-online-rl-prototype/runs/analysis/`

**Steps:**
1. Run baseline short experiment (old behavior settings).
2. Run improved short experiment (weighted+EMA+waypoint enabled).
3. Compare `eval_success_rate`, `eval_min_dist_mean`, `eval_final_dist_mean`, and training/validation loss trends.
4. Save machine-readable JSON summary and concise markdown report.
