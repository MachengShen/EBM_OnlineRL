# Claude's Architecture Audit + Open Questions
## EqM Research Validation Plan — Maze2D First
Date: 2026-02-25

---

## 0. Summary

I audited the proposal (CODEX_EQM_RESEARCH_VALIDATION_PLAN_MAZE2D_FIRST_20260225.txt)
against the actual codebase and found **several implementation caveats** that would
cause subtle wrong-but-not-crashing bugs if implemented as written.

The core issue: **the codebase has two completely separate model pipelines**, and the
proposal's code snippets silently mix up their conventions.

---

## 1. Two Parallel Implementations

The repo contains two distinct implementations:

### Pipeline A: Maze2D probe (used for H2, H3, H1 analysis)
- **Entry point:** `scripts/synthetic_maze2d_diffuser_probe.py`
- **Model:** Janner's diffuser library — `diffuser.models.temporal.TemporalUnet`
- **EqM wrapper:** `EquilibriumMatchingDiffusion` (defined inline in the probe script)
- **Data:** `diffuser.datasets.sequence.GoalDataset` — D4RL offline data
- **Trajectory shape:** `[B, H, D]` — goal at index `H-1`
- **Packing order:** `[actions | observations]` — actions first in dims `[:action_dim]`, obs in `[action_dim:]`
- **Conditioning:** `apply_conditioning(x, cond_dict, action_dim)` from diffuser library
  - writes `x[:, t, action_dim:] = val` (obs dims only, actions left free)
  - `cond` is a plain Python dict `{int: tensor}` — **already dict-based**
- **Denoiser signature:** `model(x, cond, t)` — 3 args
  - **CRITICAL:** `cond` is accepted by TemporalUnet.forward but **completely ignored internally**
  - The model only uses `x` and `time`
  - You can safely call `model(x, {}, t)` for raw Jacobian measurements

### Pipeline B: PointMass online (NOT used for the Maze2D analysis)
- **Entry point:** `scripts/online_pointmass_goal_diffuser.py`
- **Model:** Custom `ebm_online_rl/online/temporal_unet.py`
- **EqM wrapper:** `EquilibriumMatching1D` in `ebm_online_rl/online/eqm.py`
- **Data:** `EpisodeReplayBuffer` — online experience
- **Trajectory shape:** `[B, H+1, D]` — goal at index `H`
- **Packing order:** `[observations | actions]` — obs first in dims `[:obs_dim]`, actions in `[obs_dim:]`
- **Conditioning:** `apply_inpainting(x, obs0, goal, obs_dim, act_dim)`
- **Denoiser signature:** `model(x, t)` — 2 args

---

## 2. Specific Caveats in the Proposal

### Caveat A: Wrong unpack_sa for Maze2D (Section 4.2)

The proposal writes:
```python
def unpack_sa(x: torch.Tensor, obs_dim: int, act_dim: int):
    s = x[:, :, :obs_dim]                          # WRONG for Maze2D
    a = x[:, :-1, obs_dim:obs_dim+act_dim]         # WRONG for Maze2D
    return s, a
```

Maze2D uses **[act | obs]** packing (actions first). Correct version:
```python
def unpack_sa(x: torch.Tensor, obs_dim: int, act_dim: int):
    a = x[:, :-1, :act_dim]                        # actions in first act_dim dims
    s = x[:, :, act_dim:act_dim+obs_dim]           # obs in remaining dims
    return s, a
```

Also note: Maze2D trajectories have shape `[B, H, D]` not `[B, H+1, D]`, so the `[:-1]`
slice for actions gives `[B, H-1, act_dim]` — both valid (start to second-to-last timestep).

### Caveat B: Locality profile denoiser signature (Section 2.1)

Proposal writes:
```python
y = denoiser(x, t0)   # WRONG for Maze2D pipeline
```

Maze2D denoiser needs 3 args:
```python
y = denoiser(x, {}, t0)  # empty cond dict, safe since cond is unused internally
```

### Caveat C: H3 edits targeting wrong files (Section 3.1, 3.2)

The proposal says to edit:
- `ebm_online_rl/online/conditioning.py` — add `apply_inpainting_dict`
- `ebm_online_rl/online/eqm.py` — add constraints dict to `sample()`

**These are PointMass-only files.** The Maze2D pipeline already has:
- `apply_conditioning(x, cond_dict, action_dim)` — already dict-based
- `EquilibriumMatchingDiffusion.conditional_sample(cond, ...)` — already takes a dict

For H3, we just need to add waypoint entries to the existing `cond` dict:
```python
cond = {
    0: normalize_condition(dataset, start_obs, device),
    horizon - 1: normalize_condition(dataset, goal_obs, device),
    t_wp: normalize_condition(dataset, waypoint_obs, device),   # add this
}
```

And `EquilibriumMatchingDiffusion.conditional_sample` already calls
`apply_conditioning(x, cond, self.action_dim)` every iteration — it will
automatically enforce all constraints including the waypoint.

### Caveat D: Waypoint velocity constraint behavior

The existing `sample_imagined_trajectory_from_obs()` function already supports waypoints:
```python
waypoint_obs = np.zeros(dataset.observation_dim, dtype=np.float32)
waypoint_obs[:2] = np.asarray(waypoint_xy, dtype=np.float32)
cond[int(waypoint_t)] = normalize_condition(dataset, waypoint_obs, device)
```

This writes `[x, y, 0.0, 0.0]` to the waypoint timestep — it constrains position AND
zeros the velocity. This is NOT the same as "leave velocity dims unchanged."

After normalization, `normalized(0)` for velocity is some nonzero value (depends on the
D4RL dataset's velocity statistics). So the constraint implicitly says "stop here with
near-zero velocity" — a stronger constraint than position-only.

**This is open question #1.** See Section 4 below.

### Caveat E: Sequence length off-by-one

GoalDataset produces trajectories of length `H` (not `H+1`).
The goal is at `horizon - 1`, not `horizon`.

```python
# GoalDataset.get_conditions:
{0: observations[0], self.horizon - 1: observations[-1]}
```

So in any analysis script for Maze2D, the last valid index is `H-1`, not `H`.

### Caveat F: H3 eval is already partially implemented!

The probe script already contains:
- `eval_waypoint_mode: str = "none"` config with `{"none", "feasible", "infeasible"}`
- `eval_waypoint_t: int` config
- `resolve_waypoint_t(planning_horizon, raw_waypoint_t)` function
- `sample_eval_waypoint(mode, replay_observations, start_xy, goal_xy, ...)` function
- `sample_imagined_trajectory_from_obs(..., waypoint_xy=..., waypoint_t=...)` support

So H3 only needs a **new standalone eval script** (`eval_maze2d_waypoint.py`) that
loads a trained checkpoint and runs the evaluation — it does NOT need new conditioning
infrastructure.

---

## 3. Corrected Implementation Plan

### Step A — H2 Locality Map (fastest, no environment needed)
Create: `scripts/analysis_eqm_locality_map_maze2d.py`

Key corrections vs proposal:
- Load model checkpoint from a trained probe run dir
- Use `model(x, {}, t0)` not `model(x, t0)`
- Trajectories are `[1, H, D]` with `[act | obs]` layout
- Compute gradient of `(y[0, t, :] * u).sum()` w.r.t. `x` → `[1, H, D]`
- `infl = grad[0].norm(dim=1)` gives `[H]` — influence vs position in sequence
- Report: offset from pivot t, mean/std influence vs offset

### Step B — H3 Waypoint Eval
Create: `scripts/eval_maze2d_waypoint.py`

Key approach:
- Load trained checkpoint + dataset (same as probe)
- For each eval episode: sample (start, goal), sample waypoint from replay
- Build `cond = {0: start, t_wp: waypoint, H-1: goal}`
- Call `model.conditional_sample(cond, horizon=H)` — existing code handles it
- Measure: waypoint_hit, goal_hit, joint_success
- Compare EqM+waypoint vs EqM-no-waypoint vs Diffuser+waypoint

### Step C — H1 Surrogate MPC Alignment
Create: `ebm_online_rl/models/forward_dynamics.py` + `scripts/analysis_eqm_maze2d_dyn_residual_alignment.py`

Key approach:
- Train ForwardDynamics in **normalized** trajectory space (same as EqM training)
- Training data: sample from GoalDataset, unpack with CORRECT `[act | obs]` convention:
  ```python
  a = x[:, :-1, :act_dim]               # shape [B, H-1, act_dim]
  s = x[:, :, act_dim:act_dim+obs_dim]  # shape [B, H, obs_dim]
  s_t = s[:, :-1, :]                    # [B, H-1, obs_dim]
  s_tp1 = s[:, 1:, :]                   # [B, H-1, obs_dim]
  ```
- J_dyn objective — same formula as proposal but with corrected unpack
- Alignment: cosine sim between `model(x, {}, t0)` and `grad_J`
- Also test: does one EqM step reduce J_dyn?

---

## 4. Open Questions for GPT-Pro

### Q1: H3 Waypoint Velocity Constraint

**Context:** In the existing code, waypoints are encoded as `[x, y, 0, 0]` then
normalized. This constrains ALL 4 obs dims at the waypoint timestep including velocity.
After normalization, this is NOT a "zero velocity" constraint in raw space — it's
`normalized([x, y, 0, 0])` which includes the mean/std correction for the D4RL dataset.

**The proposal says:** "clamp only position dims for waypoint/goal" and "implement by
writing only x,y into s_val and leaving other dims unchanged."

But `apply_conditioning` writes ALL obs dims: `x[:, t, action_dim:] = val`. To implement
position-only, we'd need either:
- A modified conditioning function that writes only `x[:, t, action_dim:action_dim+2]`
- Or pass a full obs vector where velocity dims are copied from the current trajectory
  state (but that's chicken-and-egg during iterative refinement)

**Question:** For H3, is constraining velocity to normalized(0) at the waypoint
acceptable? Or does this fundamentally change the experiment so that we're testing
"stop at waypoint" rather than "pass through position constraint"?

My intuition: for H3, we WANT to test whether EqM can satisfy position constraints
at test time. The velocity constraint is secondary. Zero-velocity waypoints are still
position constraints — the question is whether the model can compose them. But if
GPT-Pro believes position-only is important for the research validity, we can add a
thin wrapper.

### Q2: H1 Forward Dynamics Training Data Source

**Context:** The D4RL Maze2D dataset goes through `dataset.normalizer` before training.
We need to train ForwardDynamics in the same normalized space.

**Option A (GoalDataset, normalized):**
```python
# Use existing DataLoader from probe
for batch in dataloader:
    trajectories, cond = batch
    # trajectories: [B, H, act_dim + obs_dim] (already normalized)
    a = trajectories[:, :-1, :act_dim]
    s = trajectories[:, :, act_dim:]
    s_t, s_tp1 = s[:, :-1], s[:, 1:]
    # train ForwardDynamics on (s_t, a, s_tp1) — all in normalized space
```

**Option B (Raw + normalize):**
Load the raw D4RL dataset and apply `normalizer.normalize()` manually.

Option A seems strictly better (same code path as EqM training, no risk of off-by-one
in normalization). Is there any reason to prefer Option B?

---

## 5. Key File Reference

| Purpose | File |
|---|---|
| Maze2D EqM class (inline) | `scripts/synthetic_maze2d_diffuser_probe.py` lines 31-122 |
| Waypoint sampling | `scripts/synthetic_maze2d_diffuser_probe.py` lines 1359-1413 |
| Waypoint imagined traj | `scripts/synthetic_maze2d_diffuser_probe.py` lines 1440-1479 |
| apply_conditioning | `third_party/diffuser/diffuser/models/helpers.py` |
| GoalDataset | `third_party/diffuser/diffuser/datasets/sequence.py` |
| TemporalUnet forward | `third_party/diffuser/diffuser/models/temporal.py` |
| PointMass EqM (NOT for Maze2D analysis) | `ebm_online_rl/online/eqm.py` |
| PointMass conditioning (NOT for Maze2D analysis) | `ebm_online_rl/online/conditioning.py` |

---

END
