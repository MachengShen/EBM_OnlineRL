# PointMass Compute-Saving + Success Metric Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply compute-saving and metric updates to the PointMass diffusion planner training script while preserving existing behavior where not explicitly changed.

**Architecture:** Remove multi-sample planning so inference always samples one trajectory; add optional conditioning checks default-off; run immediate burn-in training after warmup; switch success definition to ever-reached; relax CUDA hard requirement to allow CPU with warning; update defaults/readme and validate with compile + smoke run.

**Tech Stack:** Python 3, PyTorch, argparse, project scripts in `scripts/` and `ebm_online_rl/online/`.

---

### Task 1: Planner API simplification
- Modify `ebm_online_rl/online/planner.py`
- Remove `n_samples` argument.
- Force `model.sample(batch_size=1, ...)`.
- Keep action extraction from `traj[0,0,...]`.
- Make `check_conditioning` default `False` and validate only the single sample when enabled.

### Task 2: Training script API threading and defaults
- Modify `scripts/online_pointmass_goal_diffuser.py`
- Remove `n_plan_samples` from config, CLI, function signatures, and all call sites.
- Add `check_conditioning: bool = False` to config.
- Add CLI flag `--check_conditioning` (`action='store_true'`).
- Pass `check_conditioning` through rollout to planner.
- Update defaults:
  - `total_env_steps=30000`
  - `warmup_steps=5000`
  - `train_every=500`
  - `gradient_steps=200`
  - `batch_size=64`
  - `horizon=32`
  - `n_diffusion_steps=16`
  - `model_base_dim=32`
  - `model_dim_mults='1,2,4'`
  - `eval_every=10000`
  - `n_eval_episodes=20`

### Task 3: Burn-in training and success metric
- Modify `scripts/online_pointmass_goal_diffuser.py`
- In `rollout_episode`, set success as `min_dist <= threshold`.
- Immediately after warmup print, run one `train_burst(...)` burn-in when replay can sample.
- Set `next_train = env_steps + cfg.train_every` after burn-in.

### Task 4: Device behavior changes
- Modify `scripts/online_pointmass_goal_diffuser.py`
- Remove GPU-only hard error.
- Keep CUDA checks/settings when CUDA requested.
- For non-CUDA device, print explicit warning about slow CPU diffusion MPC.

### Task 5: README updates
- Modify `README.md`
- Remove GPU-only note.
- Add recommended run command line with `--device cuda:0`.
- State CPU execution is allowed but slow.

### Task 6: Verification
- Run `python -m py_compile` over project python files.
- Run GPU smoke command per requested fast settings.
- Optionally run a quick debug-conditioning smoke with `--check_conditioning`.
