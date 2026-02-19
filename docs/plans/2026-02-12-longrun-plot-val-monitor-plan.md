# Long-Run Loss Plot + Validation Monitoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Visualize training-loss evolution, add validation-loss monitoring, and run treatment/control experiments to test behavioral gains beyond random.

**Architecture:** Keep existing online training loop and add lightweight held-out replay episodes for periodic validation-loss probes. Run one treatment arm (trained planner with frequent eval/validation checks) and one strict control arm (random policy baseline) under matched evaluation conditions.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, existing `online_pointmass_goal_diffuser.py`.

---

### Task 1: Plot current training-loss trajectory

**Files:**
- Read: `runs/online_pointmass_goal_diffuser/maze2d_style_light_x0_20k_seed0_20260212/metrics.jsonl`
- Create: `runs/analysis/loss_curve_maze2d_style_light_x0_20k_seed0_20260212.png`

1. Parse finite `train_loss` points from `metrics.jsonl`.
2. Plot loss vs env steps with a simple moving-average overlay.
3. Save plot and a compact JSON summary of slope/drop stats.

### Task 2: Add validation-loss logging

**Files:**
- Modify: `scripts/online_pointmass_goal_diffuser.py`

1. Add config args for holdout episodes and validation-check cadence.
2. Route some warmup episodes into a holdout replay buffer.
3. Add `compute_validation_loss(...)` without optimizer steps.
4. Log `val_loss` and optional train/val ratio in each metrics row when evaluated.

### Task 3: Run treatment experiment with frequent eval + val monitoring

**Files:**
- Create run dir under `runs/online_pointmass_goal_diffuser/`

1. Launch long treatment run with moderate eval cadence and validation cadence.
2. Monitor loss/val-loss slopes and overfitting signals (val rising while train falls).
3. Stop at target budget or earlier only if collapse/instability criteria trigger.

### Task 4: Run strict control baseline and compare

**Files:**
- Create: analysis outputs under `runs/analysis/`

1. Generate random-policy baseline evaluation using the same env and episode horizon.
2. Compare treatment eval success/min-dist/final-dist against random baseline.
3. Produce concise technical note with hypotheses and conclusions.
