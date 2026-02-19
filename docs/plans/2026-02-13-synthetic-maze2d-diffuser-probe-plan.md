# Synthetic Maze2D Diffuser Probe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a synthetic Maze2D experiment in the original Diffuser implementation: random-policy data collection, diffusion training, and start-goal trajectory straightness probing.

**Architecture:** Add one standalone experiment script that uses the original `diffuser-maze2d` model/diffusion/dataset stack. The script will collect trajectories from `maze2d-umaze-v1`, convert them to D4RL-style arrays, train on train split with validation monitoring, then sample imagined trajectories for user-provided start/goal pairs and compute straightness metrics.

**Tech Stack:** Python, Gym/D4RL Maze2D, PyTorch, original `third_party/diffuser-maze2d` modules, Matplotlib, Pandas.

---

### Task 1: Add synthetic experiment script

**Files:**
- Create: `scripts/synthetic_maze2d_diffuser_probe.py`

**Step 1: Write script skeleton and CLI**
- Include args for env, episode count, episode length, horizon, diffusion/model hyperparameters, train/val split, query pairs, output dir, seed.

**Step 2: Implement random-policy dataset collection**
- Roll out random actions in `maze2d-umaze-v1`.
- Save a D4RL-style dict (`observations`, `actions`, `rewards`, `terminals`, `timeouts`).
- Mark trajectory boundaries using `timeouts`.

**Step 3: Implement synthetic env wrapper + dataset creation**
- Build a minimal env-like object exposing `get_dataset()`, `_max_episode_steps`, `name`.
- Instantiate `GoalDataset` (original diffuser dataset class) with `preprocess_fns=[]`.

**Step 4: Implement train/val training loop**
- Instantiate original `TemporalUnet` + `GaussianDiffusion`.
- Split dataset indices train/val.
- Train for configured steps with periodic validation.
- Save checkpoint + metrics CSV/JSON.

**Step 5: Implement start-goal query + straightness diagnostics**
- Parse query start/goal pairs.
- Sample imagined trajectories using conditional diffusion.
- Compute per-trajectory line-deviation and final-goal error.
- Save trajectory plot and query table.

### Task 2: Run and validate experiment

**Files:**
- Generate: `runs/analysis/synth_maze2d_diffuser_probe/<run_id>/...`

**Step 1: Execute script with pragmatic defaults**
Run:
```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin \
MUJOCO_GL=egl D4RL_SUPPRESS_IMPORT_ERROR=1 \
PYTHONPATH=third_party/diffuser-maze2d \
third_party/diffuser/.venv38/bin/python scripts/synthetic_maze2d_diffuser_probe.py
```

**Step 2: Check artifacts**
- Confirm files exist: `metrics.csv`, `train_val_loss.png`, `query_trajectories.png`, `query_metrics.csv`.

**Step 3: Summarize**
- Report train/val trend, query metrics, and whether queried trajectories appear straighter.
