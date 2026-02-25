# GPT-PRO Handoff: PointMass DI vs Maze2D Debug (Focused Bundle)

**Date:** 2026-02-25
**Repo:** /root/ebm-online-rl-prototype
**Branch:** analysis/results-2026-02-24

## 1. Objective
Diagnose why PointMass with double-integrator (DI) under EqM/diffusion-style planning underperforms relative to Maze2D, and identify the most likely causes via controlled ablations.

## 2. Relevant Scripts (only)
- `scripts/online_pointmass_goal_diffuser.py`
- `ebm_online_rl/envs/pointmass2d.py`
- `ebm_online_rl/online/planner.py`

## 3. Key Result Snapshot (PointMass DI)
Primary run root:
- `runs/analysis/pointmass_di_ranked_ablation_20260225-150732/`

Consolidated summary JSONs:
- `runs/analysis/pointmass_di_ranked_ablation_20260225-150732/ranked_ablation_summary_20260225-final.json`
- `runs/analysis/pointmass_di_ranked_ablation_20260225-150732/ranked_ablation_summary_20260225-extended.json`

### 3.1 Ablation table (same protocol unless noted)
| Run | Change vs baseline | eval_success@3k | eval_success@6k | Notes |
|---|---|---:|---:|---|
| r1 | DI + action-control baseline (`action_limit=0.1`) | 0.0333 | 0.0333 | reference |
| r2 | r1 + `replay_goal_position_only` | 0.0333 | 0.0333 | no measurable effect |
| r3 | r2 + `action_limit=1.0` (no damping) | 0.0333 | N/A (stopped) | severe overshoot (`eval_final_dist_mean@3k=7.0017`) |
| r4 | r3 + damping/clip (`damping=1.5`, `vclip=2.0`) | 0.1000 | 0.0333 | temporary gain at 3k, not sustained |
| r5 | long horizon only (`h=96`, `ep_len=192`, `action_limit=0.1`) | N/A (partial) | N/A | partial to `env_steps=2880`, persistent drift |

### 3.2 Current ranked likely causes
1. Terminal stability/rebound mismatch in DI rollouts (min-distance hits without stable final settling).
2. Horizon/reachability coupling mismatch vs Maze2D evaluation behavior.
3. Dynamics/actuation scaling mismatch (authority alone destabilizes; damping helps but did not sustain gains).
4. Goal-velocity semantics mismatch appears secondary (r2 ~= r1).

## 4. Maze2D reference used for comparison
- Diffuser reference success (`h256`): `0.8333`
  - source: `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_vs_existing_baselines_20260225.json`
- Horizon dependence from Maze2D run:
  - `h64=0.0`, `h128=0.5`, `h192=0.75`, `h256=0.8333`
  - source: `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/progress_metrics.csv`

## 5. Recommended next discriminating run
```bash
cd /root/ebm-online-rl-prototype && .venv/bin/python scripts/online_pointmass_goal_diffuser.py \
  --algo eqm --device cuda:0 --seed 0 --total_env_steps 6000 --warmup_steps 500 \
  --train_every 500 --gradient_steps 20 --batch_size 32 --horizon 96 \
  --eqm_steps 25 --eqm_step_size 0.1 --eqm_c_scale 1.0 --model_base_dim 16 --model_dim_mults 1,2 \
  --val_source replay --val_every 500 --val_batches 8 --val_batch_size 64 \
  --eval_every 3000 --n_eval_episodes 12 --episode_len 192 \
  --state_limit 1.0 --eval_goal_mode random --eval_min_start_goal_dist 0.5 --eval_max_start_goal_dist 1.5 \
  --success_threshold 0.2 --dynamics_model double_integrator --double_integrator_dt 0.1 \
  --initial_velocity_std 0.0 --double_integrator_velocity_damping 1.5 --double_integrator_velocity_clip 2.0 \
  --planner_control_mode action --replay_goal_position_only --action_limit 1.0 \
  --logdir runs/analysis/pointmass_di_ranked_ablation_20260225-150732/r6_di_longh96_ep192_action1p0_damp1p5_vclip2p0
```

## 6. Bundle contents
This handoff intentionally excludes checkpoints (`.pt`) and unrelated Maze2D/Pipeline artifacts. It includes only PointMass-debug scripts, run configs/metrics/summaries, and the two small Maze2D reference files required for contrast.
