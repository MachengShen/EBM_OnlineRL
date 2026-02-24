# GPT Pro Handoff: EBM OnlineRL — Experiment Results & Next Steps

**Date:** 2026-02-24
**Branch:** analysis/results-2026-02-24
**Repo:** https://github.com/MachengShen/EBM_OnlineRL

## 1. Project Overview
This cycle tested whether diffusion-based planning can match or outperform SAC-style baselines under online or online-like protocols, with focus on Maze2D and follow-up architecture diagnostics (UNet vs EqNet).

From the last GPT-PRO proposal onward, work covered: (a) Maze2D action/replanning ablations, (b) locomotion collector/learner condition comparisons, (c) EqNet-vs-UNet three-seed online-RL comparison, and (d) author-repo mechanistic diagnostics around terminal-goal inpainting width K.

## 2. Key Scripts
| Purpose | Script |
|---|---|
| Maze2D swap-matrix experiment driver | `scripts/exp_swap_matrix_maze2d.py` |
| Goal-suffix checkpoint evaluator | `scripts/eval_synth_maze2d_checkpoint_goal_suffix.py` |
| Relay smoke-gate runner | `scripts/stage0_smoke_gate.py` |
| Discord progress poster (single-log) | `scripts/discord_score_poster.py` |
| Discord progress poster (matrix) | `scripts/discord_swap_matrix_monitor.py` |
| Batch launch orchestrator | `scripts/launch_all_experiments.sh` |
| Resume pipeline after EqNet stage | `scripts/resume_after_eqnet.sh` |

### Required environment
```bash
source /root/.codex-discord-relay.env
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export PYTHONPATH=third_party/diffuser-maze2d
```

## 3. Experiment: Maze2D Action/Adaptive-Replan Ablation Grid
### Design
12-condition grid over action scaling and adaptive replanning controls (`alpha x beta x adaptive_replan`) with matched probe setup against SAC reference.
### How to run
```bash
third_party/diffuser/.venv38/bin/python3.8 scripts/exp_diffuser_ablation_grid.py \
  --diffuser-run-dir <diffuser_run_dir> --sac-run-dir <sac_run_dir> \
  --num-queries 6 --samples-per-query 20 --rollouts-per-query 6 --rollout-horizon 192 \
  --alpha-grid 1.0,1.2,1.4 --beta-grid 0.0,0.5 --adaptive-grid 0,1 --plan-samples-grid 1 \
  --base-dir runs/analysis/ablation_grid/grid_YYYYMMDD-HHMMSS
```
### Artifacts
- `runs/analysis/ablation_grid/grid_20260221-134801/ablation_grid_results.csv`
### Results (top 5 by Diffuser success@h256)
| alpha | beta | adaptive | Diffuser@256 | SAC@256 | Delta(D-S) | clip_frac |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.5 | 1 | 0.6389 | 0.6667 | -0.0278 | 0.0360 |
| 1.4 | 0.5 | 1 | 0.5833 | 0.6667 | -0.0833 | 0.5272 |
| 1.2 | 0.5 | 1 | 0.5556 | 0.6667 | -0.1111 | 0.3627 |
| 1.0 | 0.0 | 0 | 0.5278 | 0.7222 | -0.1944 | 0.0378 |
| 1.0 | 0.0 | 1 | 0.5278 | 0.6667 | -0.1389 | 0.0356 |
### Key findings
- Best observed Diffuser condition: `alpha=1.0, beta=0.5, adaptive=1` with success@256=`0.6389`.
- Diffuser remained below SAC in this grid (`delta_diffuser_minus_sac` negative for top rows).

## 4. Experiment: Locomotion Collector/Learner Comparison
### Design
Compared `diffuser_warmstart_sac`, `sac_scratch`, and `gcbc_diffuser` on hopper/walker2d medium-expert, summarizing final normalized scores per seed.
### Artifacts
- `runs/analysis/locomotion_collector/grid_20260221-200301/locomotion_collector_results.csv`
### Results (final normalized score; mean ± std across seeds)
| Env | Condition | n | Final score mean | std |
|---|---|---:|---:|---:|
| hopper-medium-expert-v2 | diffuser_warmstart_sac | 3 | 0.1412 | 0.0495 |
| hopper-medium-expert-v2 | gcbc_diffuser | 3 | 0.2515 | 0.1326 |
| hopper-medium-expert-v2 | sac_scratch | 3 | 0.0678 | 0.0049 |
| walker2d-medium-expert-v2 | diffuser_warmstart_sac | 3 | 0.0617 | 0.0097 |
| walker2d-medium-expert-v2 | gcbc_diffuser | 3 | 0.0695 | 0.0203 |
| walker2d-medium-expert-v2 | sac_scratch | 3 | 0.0890 | 0.0168 |
### Key findings
- Hopper: `gcbc_diffuser` highest mean (`0.2515`), then `diffuser_warmstart_sac` (`0.1412`), then `sac_scratch` (`0.0678`).
- Walker2d: `sac_scratch` highest mean (`0.0890`), with `gcbc_diffuser` (`0.0695`) and `diffuser_warmstart_sac` (`0.0617`) lower.

## 5. Experiment: EqNet vs UNet (3-seed Online-RL Ablation)
### Artifacts
- `.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet_vs_unet_summary.json`
### Results
| Arch | n | success_mean | min_goal_dist_mean | final_goal_dist_mean | wall_hits_mean |
|---|---:|---:|---:|---:|---:|
| unet | 3 | 0.7778 | 0.3838 | 0.4397 | 84.5556 |
| eqnet | 3 | 0.2778 | 0.8307 | 0.9634 | 89.9167 |

EqNet-minus-UNet deltas: success `-0.5000`, min_goal_dist `+0.4469`, final_goal_dist `+0.5237`, wall_hits `+5.3611`.

## 6. Experiment: Author-Repo Continuation Checkpoints
### Artifacts
- `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/continuation_metrics.csv`
- `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/summary.json`
### Results
- Checkpoint comparison count: 9 (`eqnet_better=7`, `unet_better=1`, `tie=1`).
- Per-step success deltas are in `tmp/gpt_pro_summary_metrics_20260224.json` under `author_gap_continuation.per_step`.

## 7. Experiment: Goal Inpainting Width (K) Diagnostics
### Artifacts
- Step65000 k50: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step65000_k50.json`
- Step80000 dense n32: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step80000_dense_kgrid_n32.json`
- Step80000 highconf n128: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step80000_highconf_kgrid_n128.json`
### Results highlights
- Step65000 @K=50: UNet `0.7188` (92/128), EqNet `0.8750` (112/128).
- Step80000 dense n32 focus: K=1 UNet `0.9062` vs EqNet `0.0000`; K=2 UNet `0.9375` vs EqNet `0.0000`; K=5 UNet `0.9688` vs EqNet `0.5938`.
- Step80000 highconf n128 (all tested K):
| K | UNet mean | EqNet mean | Delta(E-U) |
|---:|---:|---:|---:|
| 1 | 0.9609 | 0.0000 | -0.9609 |
| 2 | 0.9609 | 0.0000 | -0.9609 |
| 5 | 1.0000 | 0.5781 | -0.4219 |
| 8 | 0.9922 | 0.7891 | -0.2031 |
| 10 | 0.9922 | 0.9453 | -0.0469 |
| 20 | 0.9375 | 0.8672 | -0.0703 |

## 8. Ongoing: Online-RL Goal-Suffix Pilot (Inference-only K sweep on checkpoints)
### Artifacts
- `.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/goal_suffix_online_pilot_seed0_20260224-144846/results.csv`
### Current status
- Rows complete: `15/20` (pending `5`).
- Completed matched comparisons so far (`success_h256`):
| checkpoint | K | UNet | EqNet | Delta(E-U) |
|---|---:|---:|---:|---:|
| step5000 | 1 | 0.5000 | 0.0000 | -0.5000 |
| step5000 | 25 | 0.5000 | 0.0000 | -0.5000 |
| step10000 | 1 | 0.8333 | 0.0833 | -0.7500 |
| step10000 | 25 | 0.9167 | 0.0833 | -0.8333 |
| step15000 | 1 | 0.8333 | 0.4167 | -0.4167 |

## 9. Open Questions (ranked)
| Priority | Question | Blocking? | Resolves with |
|---:|---|---|---|
| 1 | In the ongoing goal-suffix pilot, does EqNet recover at higher checkpoints/K once all cells finish? | yes | Complete remaining 5 cells and recompute full UNet-vs-EqNet table |
| 2 | Are K-sensitivity conclusions checkpoint-dependent or stable across steps 55k/65k/80k? | no | Cross-checkpoint high-confidence sweep on shared K grid |
| 3 | For online RL proper, does widening terminal-goal anchoring improve EqNet without retraining? | yes | Run online evaluation sweep with goal suffix widths and fixed checkpoints |

## 10. Recommended Next Experiments
### PRIORITY 1: Finish ongoing 20-cell goal-suffix pilot and produce final paired table
```bash
cd /root/ebm-online-rl-prototype
bash .worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/goal_suffix_online_pilot_seed0_20260224-144846/rerun_remaining.sh
```
Expected outputs:
- `.../goal_suffix_online_pilot_seed0_20260224-144846/results.csv` (20 rows)
- `.../goal_suffix_online_pilot_seed0_20260224-144846/summary.json`

## 11. Implementation Notes for GPT Pro
- Do not retrain models for the immediate next step; all current diagnostics are inference-time protocol probes.
- Keep online-RL-vs-author-repo protocol differences explicit: terminal conditioning width (`K`) and replanning cadence are high-impact factors.
- Use file-backed metrics only; avoid relying on prose snapshots in `docs/WORKING_MEMORY.md` when making final claims.
