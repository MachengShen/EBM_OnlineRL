# Diffuser Collector Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement GPT Pro's TOP-3-A intervention plan to close the Diffuser-vs-SAC collector gap in Maze2D, then run the minimal ablation grid to identify the best config.

**Architecture:** Three complementary execution-time changes to `synthetic_maze2d_diffuser_probe.py` (action scaling, EMA smoothing, adaptive replanning, prefix-progress plan scoring), an extended evaluation harness in `analyze_collector_stochasticity.py`, and a new ablation-grid runner script. No new training required for the first three interventions.

**Tech Stack:** Python 3.8, NumPy, PyTorch, existing diffuser probe module, existing stochasticity analyzer harness.

---

## Background: Exact Code Locations

The five key locations in the codebase:

| What | File | Lines |
|---|---|---|
| `Config` dataclass | `scripts/synthetic_maze2d_diffuser_probe.py` | 183–275 |
| CLI arg parser | `scripts/synthetic_maze2d_diffuser_probe.py` | 298–599 |
| `sample_best_plan_from_obs` | `scripts/synthetic_maze2d_diffuser_probe.py` | 1309–1352 |
| `rollout_to_goal` action-step | `scripts/synthetic_maze2d_diffuser_probe.py` | 1730–1824 |
| `_rollout_diffuser_stochastic` | `scripts/analyze_collector_stochasticity.py` | 244–300 |

The CLI pattern (store_true/store_false pair with `set_defaults`) is used for booleans; floats and ints use `type=float/int, default=Config.<field>`.

---

## Task 1: Add new Config fields and CLI args

**Files:**
- Modify: `scripts/synthetic_maze2d_diffuser_probe.py`

### Step 1: Add 9 new Config fields after line 231

In the `Config` dataclass, after the `wall_aware_plan_samples: int = 1` field (line 231), insert:

```python
    # --- Execution-time action transform (RANK 2) ---
    # Multiply planned actions by this scalar before clipping (counters conservative magnitude).
    diffuser_action_scale_mult: float = 1.0
    # EMA smoothing on executed actions: a_exec = (1-beta)*a_raw + beta*a_prev (0.0 = off).
    diffuser_action_ema_beta: float = 0.0

    # --- Adaptive replanning (RANK 4) ---
    # If True, replan early when goal progress stalls.
    adaptive_replan: bool = False
    # Minimum steps between replans when adaptive mode is on.
    adaptive_replan_min: int = 4
    # Maximum steps between replans (clamps upward when progress is steady).
    adaptive_replan_max: int = 16
    # Progress threshold: replan if dist reduction over last step < this.
    adaptive_replan_progress_eps: float = 0.01

    # --- Prefix-progress plan scoring / best-of-K (RANK 1) ---
    # Number of imagined plans to sample per replan step (1 = no selection).
    plan_samples: int = 1
    # Scoring mode: "none" | "min_dist_prefix" | "dist_at_L"
    plan_score_mode: str = "none"
    # Length of prefix used for scoring (-1 => use replan_every steps).
    plan_score_prefix_len: int = -1
```

### Step 2: Add CLI args for the new fields

After the `--wall_aware_plan_samples` argument (line 434), insert:

```python
    # --- Execution-time action transform ---
    parser.add_argument(
        "--diffuser_action_scale_mult",
        type=float,
        default=Config.diffuser_action_scale_mult,
        help="Multiply planned Diffuser actions by this scalar before clipping (default 1.0 = off).",
    )
    parser.add_argument(
        "--diffuser_action_ema_beta",
        type=float,
        default=Config.diffuser_action_ema_beta,
        help="EMA smoothing on Diffuser executed actions: a_exec=(1-b)*a_raw+b*a_prev (0.0=off).",
    )
    # --- Adaptive replanning ---
    parser.add_argument(
        "--adaptive_replan",
        dest="adaptive_replan",
        action="store_true",
        help="Enable adaptive replanning: replan early when goal progress stalls.",
    )
    parser.add_argument(
        "--no_adaptive_replan",
        dest="adaptive_replan",
        action="store_false",
        help="Disable adaptive replanning (default).",
    )
    parser.set_defaults(adaptive_replan=Config.adaptive_replan)
    parser.add_argument(
        "--adaptive_replan_min",
        type=int,
        default=Config.adaptive_replan_min,
        help="Min steps between adaptive replans.",
    )
    parser.add_argument(
        "--adaptive_replan_max",
        type=int,
        default=Config.adaptive_replan_max,
        help="Max steps between adaptive replans.",
    )
    parser.add_argument(
        "--adaptive_replan_progress_eps",
        type=float,
        default=Config.adaptive_replan_progress_eps,
        help="Replan early if per-step distance reduction < this value.",
    )
    # --- Prefix-progress plan scoring ---
    parser.add_argument(
        "--plan_samples",
        type=int,
        default=Config.plan_samples,
        help="Number of imagined plans sampled per replan; best selected by prefix score.",
    )
    parser.add_argument(
        "--plan_score_mode",
        type=str,
        default=Config.plan_score_mode,
        choices=["none", "min_dist_prefix", "dist_at_L"],
        help="Scoring mode for plan selection: none|min_dist_prefix|dist_at_L.",
    )
    parser.add_argument(
        "--plan_score_prefix_len",
        type=int,
        default=Config.plan_score_prefix_len,
        help="Prefix length for scoring (-1 = use replan_every steps).",
    )
```

### Step 3: Verify

```bash
python3 -m py_compile scripts/synthetic_maze2d_diffuser_probe.py
python3 scripts/synthetic_maze2d_diffuser_probe.py --help | grep -E "scale_mult|ema_beta|adaptive_replan|plan_samples|plan_score"
```

Expected: no compile errors; all new flags visible in `--help`.

### Step 4: Commit

```bash
git add scripts/synthetic_maze2d_diffuser_probe.py
git commit -m "feat: add Config fields and CLI args for action transform, adaptive replan, prefix scoring"
```

---

## Task 2: Implement prefix-progress plan scoring in `sample_best_plan_from_obs`

**Files:**
- Modify: `scripts/synthetic_maze2d_diffuser_probe.py:1309–1352`

### Step 1: Update function signature

Replace lines 1309–1321:

```python
def sample_best_plan_from_obs(
    model: GaussianDiffusion,
    dataset: GoalDataset,
    start_obs: np.ndarray,
    goal_xy: np.ndarray,
    horizon: int,
    device: torch.device,
    maze_arr: np.ndarray | None,
    wall_aware_planning: bool,
    wall_aware_plan_samples: int,
    waypoint_xy: np.ndarray | None = None,
    waypoint_t: int | None = None,
    plan_samples: int = 1,
    plan_score_mode: str = "none",
    plan_score_prefix_len: int = -1,
    replan_every: int = 8,
) -> Tuple[np.ndarray, np.ndarray, int]:
```

### Step 2: Replace the body (lines 1322–1352)

```python
    # Determine total candidates: max of wall_aware_plan_samples (legacy) and plan_samples (new).
    n_wall = max(1, int(wall_aware_plan_samples)) if bool(wall_aware_planning) else 1
    n_prefix = max(1, int(plan_samples))
    n_candidates = max(n_wall, n_prefix)

    observations, actions = sample_imagined_trajectory_from_obs(
        model=model,
        dataset=dataset,
        start_obs=start_obs,
        goal_xy=goal_xy,
        horizon=horizon,
        device=device,
        n_samples=n_candidates,
        waypoint_xy=waypoint_xy,
        waypoint_t=waypoint_t,
    )

    # Determine prefix length for scoring
    prefix_L = int(plan_score_prefix_len) if plan_score_prefix_len > 0 else max(1, int(replan_every))
    prefix_L = min(prefix_L, observations.shape[1])

    goal_xy_arr = np.asarray(goal_xy, dtype=np.float32)

    def _score_plan(i: int) -> float:
        xy = np.asarray(observations[i, :, :2], dtype=np.float32)
        if plan_score_mode == "min_dist_prefix":
            dists = np.linalg.norm(xy[:prefix_L] - goal_xy_arr, axis=-1)
            return float(np.min(dists))
        elif plan_score_mode == "dist_at_L":
            idx = min(prefix_L - 1, xy.shape[0] - 1)
            return float(np.linalg.norm(xy[idx] - goal_xy_arr))
        else:
            # "none" — fall back to final goal error (original behavior)
            return float(np.linalg.norm(xy[-1] - goal_xy_arr))

    best_idx = int(np.argmin([_score_plan(i) for i in range(n_candidates)]))

    selected_wall_hits = int(count_wall_hits_qpos_frame(maze_arr, observations[best_idx, :, :2]))
    return (
        np.asarray(observations[best_idx], dtype=np.float32),
        np.asarray(actions[best_idx], dtype=np.float32),
        selected_wall_hits,
    )
```

### Step 3: Update the call site in `rollout_to_goal` (lines 1785–1797)

The existing call at lines 1785–1797 needs three new kwargs passed through. The rollout function will be refactored in Task 3, but for now confirm the old call still works (new params have defaults so existing callers are unaffected).

### Step 4: Verify

```bash
python3 -m py_compile scripts/synthetic_maze2d_diffuser_probe.py
```

Expected: clean compile.

### Step 5: Commit

```bash
git add scripts/synthetic_maze2d_diffuser_probe.py
git commit -m "feat: prefix-progress plan scoring (best-of-K) in sample_best_plan_from_obs"
```

---

## Task 3: Implement action scaling + EMA smoothing in `rollout_to_goal`

**Files:**
- Modify: `scripts/synthetic_maze2d_diffuser_probe.py:1730–1824`

This is the highest-priority mechanical change (RANK 2, essentially free at eval time).

### Step 1: Update `rollout_to_goal` signature

After `wall_aware_plan_samples: int,` (line 1745), add:

```python
    action_scale_mult: float = 1.0,
    action_ema_beta: float = 0.0,
```

### Step 2: Add action transform in `open_loop` branch

Replace line 1771:
```python
            action = np.clip(act, act_low, act_high).astype(np.float32)
```
With:
```python
            # Action scaling + EMA transform
            act_scaled = np.clip(float(action_scale_mult) * act, act_low, act_high)
            if t == 0 or float(action_ema_beta) == 0.0:
                action = act_scaled.astype(np.float32)
            else:
                action = ((1.0 - float(action_ema_beta)) * act_scaled
                          + float(action_ema_beta) * _prev_exec_action)
                action = np.clip(action, act_low, act_high).astype(np.float32)
            _prev_exec_action = action.copy()
```

Also insert before the `open_loop` loop (line 1766):
```python
        _prev_exec_action = np.zeros(dataset.action_dim, dtype=np.float32)
```

### Step 3: Add action transform in `receding_horizon` branch

Replace line 1806:
```python
            action = np.clip(action, act_low, act_high).astype(np.float32)
```
With:
```python
            # Action scaling + EMA transform (RANK 2)
            act_scaled = np.clip(float(action_scale_mult) * action, act_low, act_high)
            if t == 0 or float(action_ema_beta) == 0.0:
                action = act_scaled.astype(np.float32)
            else:
                action = ((1.0 - float(action_ema_beta)) * act_scaled
                          + float(action_ema_beta) * _prev_exec_action)
                action = np.clip(action, act_low, act_high).astype(np.float32)
            _prev_exec_action = action.copy()
```

Insert before the `receding_horizon` loop (line 1780):
```python
        _prev_exec_action = np.zeros(dataset.action_dim, dtype=np.float32)
```

### Step 4: Verify

```bash
python3 -m py_compile scripts/synthetic_maze2d_diffuser_probe.py
```

### Step 5: Commit

```bash
git add scripts/synthetic_maze2d_diffuser_probe.py
git commit -m "feat: execution-time action scaling and EMA smoothing in rollout_to_goal"
```

---

## Task 4: Implement adaptive replanning in `rollout_to_goal` (`receding_horizon` branch)

**Files:**
- Modify: `scripts/synthetic_maze2d_diffuser_probe.py:1779–1814`

### Step 1: Add adaptive replanning params to signature

After `action_ema_beta: float = 0.0,` add:

```python
    adaptive_replan: bool = False,
    adaptive_replan_min: int = 4,
    adaptive_replan_max: int = 16,
    adaptive_replan_progress_eps: float = 0.01,
    plan_samples: int = 1,
    plan_score_mode: str = "none",
    plan_score_prefix_len: int = -1,
```

### Step 2: Replace the `receding_horizon` replan logic

Replace the `elif rollout_mode == "receding_horizon":` block's loop structure:

```python
    elif rollout_mode == "receding_horizon":
        planned_actions = np.zeros((0, dataset.action_dim), dtype=np.float32)
        plan_offset = 0
        _prev_exec_action = np.zeros(dataset.action_dim, dtype=np.float32)
        _steps_since_replan = 0
        _prev_dist = float("inf")
        _adaptive_stall_steps = 0
        _replan_triggers = 0

        for t in range(rollout_horizon):
            # Determine if we should replan
            _regular_replan = (t % replan_stride == 0) or (plan_offset >= len(planned_actions))

            if adaptive_replan and t > 0:
                _dist_now = float(np.linalg.norm(obs[:2] - goal_xy))
                _progress = _prev_dist - _dist_now
                if _progress < float(adaptive_replan_progress_eps):
                    _adaptive_stall_steps += 1
                else:
                    _adaptive_stall_steps = 0
                _prev_dist = _dist_now
                _adaptive_trigger = (
                    _adaptive_stall_steps >= 1
                    and _steps_since_replan >= int(adaptive_replan_min)
                    and plan_offset < len(planned_actions)  # avoid re-triggering on empty plan
                )
            else:
                _adaptive_trigger = False
                if t == 0:
                    _prev_dist = float(np.linalg.norm(obs[:2] - goal_xy))

            should_replan = _regular_replan or _adaptive_trigger

            if should_replan:
                _, best_actions, _ = sample_best_plan_from_obs(
                    model=model,
                    dataset=dataset,
                    start_obs=obs,
                    goal_xy=goal_xy,
                    horizon=planning_horizon,
                    device=device,
                    maze_arr=maze_arr,
                    wall_aware_planning=wall_aware_planning,
                    wall_aware_plan_samples=wall_aware_plan_samples,
                    plan_samples=int(plan_samples),
                    plan_score_mode=str(plan_score_mode),
                    plan_score_prefix_len=int(plan_score_prefix_len),
                    replan_every=replan_stride,
                )
                planned_actions = np.asarray(best_actions, dtype=np.float32)
                plan_offset = 0
                _steps_since_replan = 0
                if _adaptive_trigger:
                    _replan_triggers += 1
                    _adaptive_stall_steps = 0

            _steps_since_replan += 1

            if plan_offset >= len(planned_actions):
                action = np.zeros(dataset.action_dim, dtype=np.float32)
            else:
                action = planned_actions[plan_offset].astype(np.float32)
                plan_offset += 1

            # Action scaling + EMA transform (RANK 2)
            act_scaled = np.clip(float(action_scale_mult) * action, act_low, act_high)
            if t == 0 or float(action_ema_beta) == 0.0:
                action = act_scaled.astype(np.float32)
            else:
                action = ((1.0 - float(action_ema_beta)) * act_scaled
                          + float(action_ema_beta) * _prev_exec_action)
                action = np.clip(action, act_low, act_high).astype(np.float32)
            _prev_exec_action = action.copy()

            obs, _, _, _ = safe_step(rollout_env, action)
            dist = float(np.linalg.norm(obs[:2] - goal_xy))
            min_goal_dist = min(min_goal_dist, dist)
            final_goal_dist = dist
            rollout_wall_hits += count_wall_hits_qpos_frame(maze_arr, obs[:2])
            rollout_actions.append(action.copy())
            rollout_xy.append(obs[:2].copy())
```

> Note: The `open_loop` branch from Task 3 stays as-is (no adaptive replan there).

### Step 3: Verify

```bash
python3 -m py_compile scripts/synthetic_maze2d_diffuser_probe.py
```

### Step 4: Smoke-run the probe help to confirm no arg conflicts

```bash
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/synthetic_maze2d_diffuser_probe.py --help \
  | grep -E "scale_mult|ema_beta|adaptive|plan_sample|plan_score"
```

Expected: All new flags printed.

### Step 5: Commit

```bash
git add scripts/synthetic_maze2d_diffuser_probe.py
git commit -m "feat: adaptive replanning and prefix scoring wired into rollout_to_goal"
```

---

## Task 5: Extend `analyze_collector_stochasticity.py` evaluation harness

**Files:**
- Modify: `scripts/analyze_collector_stochasticity.py`

This adds the new Diffuser config flags to the analyzer, applies the action transform inside `_rollout_diffuser_stochastic`, and tracks four new metrics: `mean_action_l2`, `clip_fraction`, `steps_to_threshold_0p1`, `steps_to_threshold_0p2`.

### Step 1: Update `_rollout_diffuser_stochastic` signature (line 244)

Add parameters after `wall_aware_plan_samples: int,`:

```python
    action_scale_mult: float = 1.0,
    action_ema_beta: float = 0.0,
    adaptive_replan: bool = False,
    adaptive_replan_min: int = 4,
    adaptive_replan_max: int = 16,
    adaptive_replan_progress_eps: float = 0.01,
    plan_samples: int = 1,
    plan_score_mode: str = "none",
    plan_score_prefix_len: int = -1,
    goal_thresholds: Tuple[float, ...] = (0.1, 0.2),
```

Change return type to `Tuple[np.ndarray, float, float, dict]` — the dict carries the new metrics.

### Step 2: Add metrics tracking inside the rollout loop

After `act_low`/`act_high` init (line 265), add:

```python
    _prev_exec_action = np.zeros(dataset.action_dim, dtype=np.float32)
    _raw_action_norms: List[float] = []
    _exec_action_norms: List[float] = []
    _clip_flags: List[int] = []
    _first_hit_steps: Dict[float, Optional[int]] = {thr: None for thr in goal_thresholds}
    _steps_since_replan = 0
    _prev_dist = float("inf")
    _adaptive_stall_steps = 0
    stride = max(1, int(replan_every_n_steps))
```

Inside the `for t in range(int(rollout_horizon)):` loop, replace the existing clip and step block:

```python
        # Adaptive replan check
        if adaptive_replan and t > 0:
            _progress = _prev_dist - dist  # dist set below from previous iter
            if _progress < float(adaptive_replan_progress_eps):
                _adaptive_stall_steps += 1
            else:
                _adaptive_stall_steps = 0
        _adaptive_trigger = (
            adaptive_replan
            and _adaptive_stall_steps >= 1
            and _steps_since_replan >= int(adaptive_replan_min)
        )
        if (t % stride == 0) or (plan_offset >= len(planned_actions)) or _adaptive_trigger:
            _plan_obs, best_actions, _hits = dprobe.sample_best_plan_from_obs(
                model=model,
                dataset=dataset,
                start_obs=obs,
                goal_xy=np.asarray(goal_xy, dtype=np.float32),
                horizon=int(planning_horizon),
                device=device,
                maze_arr=None,
                wall_aware_planning=bool(wall_aware_planning),
                wall_aware_plan_samples=int(wall_aware_plan_samples),
                plan_samples=int(plan_samples),
                plan_score_mode=str(plan_score_mode),
                plan_score_prefix_len=int(plan_score_prefix_len),
                replan_every=stride,
            )
            planned_actions = np.asarray(best_actions, dtype=np.float32)
            plan_offset = 0
            _steps_since_replan = 0
            _adaptive_stall_steps = 0

        _steps_since_replan += 1
        action_raw = (planned_actions[plan_offset] if plan_offset < len(planned_actions)
                      else np.zeros(dataset.action_dim, dtype=np.float32)).astype(np.float32)
        plan_offset += 1

        # Action scaling + EMA
        act_scaled = np.clip(float(action_scale_mult) * action_raw, act_low, act_high)
        if t == 0 or float(action_ema_beta) == 0.0:
            action = act_scaled
        else:
            action = np.clip(
                (1.0 - float(action_ema_beta)) * act_scaled + float(action_ema_beta) * _prev_exec_action,
                act_low, act_high,
            ).astype(np.float32)
        _prev_exec_action = action.copy()

        # Log raw vs exec norms
        _raw_action_norms.append(float(np.linalg.norm(action_raw)))
        _exec_action_norms.append(float(np.linalg.norm(action)))
        clipped = int(np.any(action_raw * float(action_scale_mult) < act_low)
                      or np.any(action_raw * float(action_scale_mult) > act_high))
        _clip_flags.append(clipped)

        obs, _r, _d, _info = dprobe.safe_step(env, action)
        traj_xy.append(np.asarray(obs[:2], dtype=np.float32))
        dist = float(np.linalg.norm(obs[:2] - goal_xy))
        _prev_dist = dist
        min_goal_dist = min(min_goal_dist, dist)
        final_goal_dist = dist
        for thr in goal_thresholds:
            if _first_hit_steps[thr] is None and dist <= thr:
                _first_hit_steps[thr] = t

    extra_metrics = {
        "mean_action_l2_raw": float(np.mean(_raw_action_norms)) if _raw_action_norms else float("nan"),
        "mean_action_l2_exec": float(np.mean(_exec_action_norms)) if _exec_action_norms else float("nan"),
        "clip_fraction": float(np.mean(_clip_flags)) if _clip_flags else float("nan"),
    }
    for thr in goal_thresholds:
        key = f"steps_to_threshold_{str(thr).replace('.', 'p')}"
        extra_metrics[key] = _first_hit_steps[thr]

    return np.stack(traj_xy, axis=0), float(min_goal_dist), float(final_goal_dist), extra_metrics
```

### Step 3: Add CLI flags to the analyzer's `build_parser()`

Find where `--wall_aware_plan_samples` is defined in `analyze_collector_stochasticity.py` and insert the same 9 new args after it (same pattern as Task 1, Step 2).

### Step 4: Thread the new flags through to every `_rollout_diffuser_stochastic` call site in the analyzer

Search for all calls with:
```bash
grep -n "_rollout_diffuser_stochastic" scripts/analyze_collector_stochasticity.py
```
Add the new kwargs at each call site (they all default to the new CLI arg values).

### Step 5: Update the per-query result dict / CSV / JSON schema

Wherever the function returns are unpacked (3-tuple → 4-tuple now), add the `extra_metrics` fields. Add them to the CSV columns and to the per-condition JSON.

### Step 6: Verify

```bash
python3 -m py_compile scripts/analyze_collector_stochasticity.py
python3 scripts/analyze_collector_stochasticity.py --help | grep -E "scale_mult|ema_beta|adaptive|plan_sample"
```

### Step 7: Commit

```bash
git add scripts/analyze_collector_stochasticity.py
git commit -m "feat: extend stochasticity analyzer with action transform flags and new metrics"
```

---

## Task 6: Write the ablation grid runner script

**Files:**
- Create: `scripts/exp_diffuser_ablation_grid.py`

### Step 1: Write the script

The script drives `analyze_collector_stochasticity.py` as a subprocess across a 3-factor grid. Use subprocess + existing analyzer CLI pattern (same as `exp_swap_matrix_maze2d.py` drives probe scripts).

Grid (18 conditions total = 3 × 2 × 3, but first sweep is smaller):
- `alpha` (diffuser_action_scale_mult): `[1.0, 1.2, 1.4]`
- `beta` (diffuser_action_ema_beta): `[0.0, 0.5]`
- `adaptive_replan`: `[False, True]`
- Fixed: `q=6, samples_per_query=20, rollouts=6, horizon=192`

Script structure:
```python
#!/usr/bin/env python3
"""
Diffuser execution-time ablation grid.

Sweeps (alpha, beta, adaptive_replan) using analyze_collector_stochasticity.py
as the evaluation harness. Saves per-condition JSON summaries and a merged CSV.

Usage:
  python3 scripts/exp_diffuser_ablation_grid.py \
    --run_dir <diffuser_run_dir> \
    --sac_run_dir <sac_run_dir> \
    --base_dir runs/analysis/ablation_grid/grid_TIMESTAMP
"""
import argparse, json, os, subprocess, sys
from datetime import datetime
from itertools import product
from pathlib import Path

ALPHA_GRID = [1.0, 1.2, 1.4]
BETA_GRID  = [0.0, 0.5]
ADAPTIVE_GRID = [False, True]
DEFAULT_Q = 6
DEFAULT_SAMPLES = 20
DEFAULT_ROLLOUTS = 6
DEFAULT_HORIZON = 192

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sac_run_dir", default="")
    parser.add_argument("--base_dir", default=f"runs/analysis/ablation_grid/grid_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_queries", type=int, default=DEFAULT_Q)
    parser.add_argument("--samples_per_query", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--rollouts_per_query", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--alpha_grid", default=",".join(str(a) for a in ALPHA_GRID))
    parser.add_argument("--beta_grid",  default=",".join(str(b) for b in BETA_GRID))
    parser.add_argument("--no_adaptive", action="store_true", help="Skip adaptive=True conditions.")
    parser.add_argument("--plan_samples", type=int, default=1)
    parser.add_argument("--plan_score_mode", default="none")
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in args.alpha_grid.split(",")]
    betas  = [float(b) for b in args.beta_grid.split(",")]
    adapt_vals = [False] if args.no_adaptive else [False, True]

    PYTHON = "third_party/diffuser/.venv38/bin/python3.8"
    env_prefix = (
        "D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl "
        "LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH "
        "PYTHONPATH=third_party/diffuser-maze2d"
    )

    results = []
    for alpha, beta, adaptive in product(alphas, betas, adapt_vals):
        cond_name = f"alpha{alpha}_beta{beta}_adapt{int(adaptive)}"
        out_dir = base / cond_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"{env_prefix} {PYTHON} scripts/analyze_collector_stochasticity.py "
            f"--diffuser_run_dir {args.run_dir} "
            f"--num_queries {args.num_queries} "
            f"--samples_per_query {args.samples_per_query} "
            f"--rollouts_per_query {args.rollouts_per_query} "
            f"--rollout_horizon {args.horizon} "
            f"--device {args.device} "
            f"--diffuser_action_scale_mult {alpha} "
            f"--diffuser_action_ema_beta {beta} "
            f"{'--adaptive_replan' if adaptive else '--no_adaptive_replan'} "
            f"--plan_samples {args.plan_samples} "
            f"--plan_score_mode {args.plan_score_mode} "
            f"--outdir {out_dir}"
        )
        if args.sac_run_dir:
            cmd += f" --sac_run_dir {args.sac_run_dir}"

        print(f"\n[GRID] Running condition: {cond_name}")
        ret = subprocess.run(cmd, shell=True, check=False)
        summary_path = out_dir / "collector_stochasticity_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            data["condition"] = cond_name
            data["alpha"] = alpha
            data["beta"] = beta
            data["adaptive_replan"] = adaptive
            results.append(data)
            print(f"  => success={data.get('diffuser_rollout_success_rate_mean', 'N/A')}")
        else:
            print(f"  => FAILED (no summary)")

    # Save merged CSV
    import csv
    merged = base / "ablation_grid_results.csv"
    if results:
        keys = list(results[0].keys())
        with open(merged, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
    print(f"\n[GRID] Done. Results: {merged}")
    print("[GRID] Top conditions by diffuser success rate:")
    for r in sorted(results, key=lambda x: x.get("diffuser_rollout_success_rate_mean", 0), reverse=True)[:3]:
        print(f"  {r['condition']}: success={r.get('diffuser_rollout_success_rate_mean')}, "
              f"min_dist={r.get('diffuser_rollout_min_goal_dist_mean')}")

if __name__ == "__main__":
    main()
```

### Step 2: Make executable and verify

```bash
chmod +x scripts/exp_diffuser_ablation_grid.py
python3 -m py_compile scripts/exp_diffuser_ablation_grid.py
python3 scripts/exp_diffuser_ablation_grid.py --help
```

### Step 3: Commit

```bash
git add scripts/exp_diffuser_ablation_grid.py
git commit -m "feat: ablation grid runner for Diffuser execution-time improvements"
```

---

## Task 7: Run the ablation grid (long-running job)

**Prerequisites:** Tasks 1–6 complete. You need an existing Diffuser run directory with a checkpoint. Use the swap-matrix phase1 Diffuser run.

### Step 1: Find a valid Diffuser run dir

```bash
ls runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0/
```

Look for a `config.json` and a checkpoint file (`.pt` or `checkpoints/` dir).

### Step 2: Launch the ablation grid as a background relay job

```bash
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_diffuser_ablation_grid.py \
  --run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0 \
  --num_queries 6 \
  --samples_per_query 20 \
  --rollouts_per_query 6 \
  --horizon 192 \
  --device cuda:0 \
  --base_dir runs/analysis/ablation_grid/grid_$(date +%Y%m%d-%H%M%S) \
  2>&1 | tee runs/analysis/ablation_grid_latest.log
```

Expected output: one JSON summary per condition, merged `ablation_grid_results.csv`.

### Step 3: Quick verification after completion

```bash
ls runs/analysis/ablation_grid/grid_*/ablation_grid_results.csv
python3 -c "
import json, glob
csvs = glob.glob('runs/analysis/ablation_grid/grid_*/ablation_grid_results.csv')
if csvs:
    import csv
    with open(csvs[-1]) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r.get('diffuser_rollout_success_rate_mean') or 0), reverse=True)
    for r in rows[:5]:
        print(r['condition'], r.get('diffuser_rollout_success_rate_mean'), r.get('diffuser_rollout_min_goal_dist_mean'))
"
```

---

## Task 8: Analyze results and update WORKING_MEMORY

After the ablation grid completes:

1. Read `ablation_grid_results.csv` and identify:
   - Best condition by `diffuser_rollout_success_rate_mean`
   - Best condition by `diffuser_rollout_min_goal_dist_mean`
   - Whether `adaptive_replan=True` consistently helps/hurts

2. Update `docs/WORKING_MEMORY.md`:
   - Update "Current Best Result" if ablation beats baseline
   - Update H3 (SAC control-aware advantage) status based on whether alpha/beta closes the gap
   - Update "Next Experiment" with best config for 3-seed confirmation

3. Append to `HANDOFF_LOG.md`.

4. Commit working memory updates:
   ```bash
   git add docs/WORKING_MEMORY.md HANDOFF_LOG.md
   git commit -m "docs: ablation grid results and updated hypothesis status"
   ```

---

## Execution order summary

| Task | Est. time | Blocking? |
|---|---|---|
| T1: Config + CLI | 5 min | Yes (needed by all others) |
| T2: Prefix scoring | 10 min | No |
| T3: Action scaling + EMA | 10 min | No (after T2) |
| T4: Adaptive replan | 15 min | No (after T3) |
| T5: Extend analyzer | 20 min | Needed before T7 |
| T6: Ablation grid script | 10 min | Needed before T7 |
| T7: Run ablation grid | ~30-60 min runtime | Long job |
| T8: Analyze + update docs | 5 min | After T7 |

Tasks T2–T4 touch the same file and should run sequentially; T5–T6 are independent of each other and of T2–T4 (different files), so can be done in parallel with T2–T4.
