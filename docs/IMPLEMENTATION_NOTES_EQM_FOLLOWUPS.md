# Implementation Notes — EqM Mechanistic Follow-ups (Maze2D)

**Date:** 2026-02-26
**Agent:** Claude Opus 4.6
**Proposal:** `CODEX_EQM_MECHANISTIC_FOLLOWUPS_MAZE2D_20260226.md`
**Branch:** `analysis/results-2026-02-24`

---

## Task A — H2 Blockwise Locality + Regime Sweeps

### Script
`scripts/analysis_eqm_locality_map_maze2d_blockwise.py`

### Error-prone areas
- **Packing indices**: The `[act|obs]` Maze2D convention required careful index handling in the `blockwise_locality_probe()` function. The `proj_scalar` splits `y[0, t, :ACT_DIM]` for act_out and `y[0, t, ACT_DIM:]` for obs_out. Similarly, input influence is `grad[0, :, :ACT_DIM].norm(dim=1)` for act_in and `grad[0, :, ACT_DIM:].norm(dim=1)` for obs_in.
- **Denoiser call**: 3-arg form `model(x, {}, t0)` — easy to forget the empty cond dict.

### Sanity checks
- Shape asserts: verified x shape `[1, H, D]` with D=6, H=64.
- Normalization: each probe normalizes influence by `max + 1e-12` to ensure `[0, 1]` range.
- Cross-verified dataset regime results against the original H2 locality script — the obs_out__obs_in block (which dominates the mixed-norm profile) shows comparable offset-0 dominance.

### Key observation
The blockwise decomposition reveals something the original mixed-norm H2 probe hid: **obs→obs is extremely local** (ratio ~27x on dataset) while **obs→act has much weaker locality** (ratio ~1.4x). This makes physical sense: observations influence nearby actions less sharply because the U-Net's skip connections propagate information laterally in the action channel.

---

## Task C — H1 Alignment Sweeps

### Script
`scripts/analysis_eqm_maze2d_dyn_alignment_sweeps.py`

### Error-prone areas
- **Tensor contiguity**: The denoiser output `f_field` was non-contiguous after the U-Net forward pass. Using `.view(-1)` caused a `RuntimeError`. Fixed by switching to `.reshape(-1)`.
- **Packing in `unpack_sa`**: States are `x[:, :, ACT_DIM:ACT_DIM + OBS_DIM]` and actions are `x[:, :-1, :ACT_DIM]`. Getting this wrong would silently compute meaningless dynamics residuals.

### Sanity checks
- J_dyn on dataset trajectories ≈ 2e-6 (consistent with previous H1 results).
- J_dyn on pure noise (gamma=0) ≈ 8.6 — reasonable for random trajectories.
- Dot-positive fraction = 1.0 for all corrupted/EqM-iter regimes — strong signal.
- Dataset baseline with lam_u=0: dot+ = 0.43, cos = -0.003, descent = 0.027. This confirms the known operating-point artifact: near equilibrium, J_dyn gradient is near-zero and alignment is noisy.

### Key observation
The sweeps cleanly resolve the "low single-step descent" confusion from the original H1 results. With lam_u=0:
- **Corrupted regime**: dot+ = 1.0 and small-step descent = 1.0 for ALL gamma values (0.0 to 0.99).
- **EqM iterate regime**: dot+ = 1.0 and descent = 1.0 for ALL k values (0 to 25).
- **Dataset baseline**: dot+ = 0.43, descent = 0.027 — confirming it's an operating-point artifact.

The cosine similarity increases as samples get cleaner: cos goes from 0.18 (gamma=0) to 0.56 (gamma=0.9), then drops back at gamma=0.99 where J_dyn is again tiny.

---

## Task B — H3 Execution-Level Waypoint Eval

### Script
`scripts/eval_maze2d_waypoint_exec.py`

### Error-prone areas
- **`sample_eval_waypoint` requires `start_xy`**: In the execution-level test, the agent starts from a random env.reset() position, not a controlled position. Initial implementation passed `start_xy=None` which caused a reshape error. Fixed by sampling start from replay observations (used for waypoint feasibility checking only — the actual env start is still random).
- **Replanning speed**: With `replan_every=1` and 300-step episodes, each episode requires 300 × 25 EqM steps = 7,500 U-Net forward passes. For 50 episodes × 3 modes, this would take ~9 hours. Switched to `replan_every=8` (matching the existing receding horizon default).
- **Action normalization**: Planned trajectories are in normalized space. Actions must be unnormalized via `dataset.normalizer.unnormalize()` and clipped to `env.action_space` bounds before stepping.
- **Shifting waypoint index**: Correctly computing `t_wp_local = t_wp_global - k_exec` and checking `1 <= t_wp_local <= H-2` before including waypoint in conditioning.

### Sanity checks
- Goal-only baseline (no waypoint) reaches the goal reliably, confirming the planning + replanning loop works.
- With replan_every=8, ~38 replans per episode — a reasonable MPC stride.

### Results

| Mode | WP Hit | Goal Hit | Joint | Mean Min WP Dist | Mean Min Goal Dist |
|---|---|---|---|---|---|
| pos_only | 0.260 | 0.880 | 0.240 | 1.082 | 0.294 |
| pos_and_zero_vel | 0.240 | 0.880 | 0.220 | 1.013 | 0.298 |
| no_waypoint | 0.000 | 0.900 | 0.000 | inf | 0.256 |

### Key observations
1. **Waypoint constraints do work at execution level** — 24% joint success vs 0% baseline.
2. **Goal hit rate is high** (~88-90%) regardless of waypoint, confirming the planning+replanning pipeline works.
3. **pos_only and pos_and_zero_vel are nearly identical** (26% vs 24% wp hit), consistent with H2 locality findings.
4. **The gap vs imagination-level is large**: imagination-level showed 100% waypoint hit, but execution-level shows only ~25%. This is the key finding — conditioning perfectly clamps waypoints in the planned trajectory, but execution drift and replanning cause the agent to miss waypoints that are off its natural path.
5. **No-waypoint baseline never hits the waypoint** (0%), confirming that waypoint visits are not accidental.
6. Settings: replan_every=8, n_episodes=50, n_plan_samples=1, max_steps=300, eps_wp=0.5, eps_goal=0.5.

---

## General Notes

- All scripts share `scripts/maze2d_eqm_utils.py` for model loading, dataset construction, normalization, and waypoint sampling.
- Environment conventions: `[act(2)|obs(4)]` packing, `obs = [x, y, vx, vy]`, `act = [force_x, force_y]`.
- Denoiser always called with 3 args: `model(x, cond_dict, t0)`.
- EqM sampling: `x = x - step_size * model(x, cond, t0)` for K=25 steps at t0=0.
