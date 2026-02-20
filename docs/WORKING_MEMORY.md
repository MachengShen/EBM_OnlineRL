# EBM Online RL Working Memory (living snapshot)

Last updated: 2026-02-20T22:15:28+08:00
Repo: /root/ebm-online-rl-prototype
Branch: master
Commit: b9cdd15  (dirty: yes — HANDOFF_LOG.md, docs/WORKING_MEMORY.md modified; research_finding.txt untracked)

## Objective
Validate and improve online Maze2D performance by determining whether gains are driven by data collection policy (collector), learner updates, or both. Primary focus: characterize and explain SAC's collector advantage over Diffuser and design targeted Diffuser improvements.

## Active Hypotheses (max 3)
- H1 (falsified): Diffuser collector dominates — generates better replay for any learner.
  - Status: falsified (3-seed swap matrix, h256)
  - Evidence AGAINST: diffuser replay main effect `0.8194` vs SAC replay `0.9097` (n=12/collector, h256) — `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
  - Evidence AGAINST: SAC endpoint pairwise L2 > Diffuser in `11/12` consolidation cells — `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_overall_summary.json`
  - Next discriminating test: N/A — already falsified; 5-seed matrix will confirm stability.

- H2 (partially supported): Diffuser learner/planner advantage with shared replay.
  - Status: partially supported (3-seed swap matrix, h256)
  - Evidence FOR: best cell `warmstart|sac_her_sparse→diffuser` = `0.9722 ± 0.0481` — `swap_matrix_results.csv`
  - Evidence FOR: learner main effect diffuser `0.8819` vs SAC `0.8472` (n=12/learner, h256)
  - Next discriminating test: promote to 5 seeds; run waypoint/diversity diagnostics.

- H3 (supported): SAC collector advantage is control-aware (better goal-directed coverage), not raw action noise.
  - Status: supported (multi-seed consolidation 12 cells + visual falsification)
  - Evidence FOR: Diffuser action pairwise L2 > SAC in `11/12` cells; SAC endpoint pairwise L2 > Diffuser in `11/12` cells — `consolidated_overall_summary.json`
  - Evidence FOR: SAC reaches near-goal (`<=0.1`) ~61 steps faster on average where both succeed (seed-0, 4/6 queries) — `visual_check_phase1_seed0_q6_s20_r6_h192/steps_to_goal_threshold_0p1_summary.json`
  - Next discriminating test: replanning-cadence/horizon ablations on Diffuser to isolate control-frequency effects.

## Required Environment (minimal, no secrets)
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1
MUJOCO_GL=egl
LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
PYTHON=third_party/diffuser/.venv38/bin/python3.8
# Must be set before process start — script-level env mutation is insufficient for mujoco_py dynamic libs
```

## Current Best Result (non-smoke only)
- Metric: success@h256 = `0.9722 ± 0.0481`  (n=3 seeds, condition=warmstart+SAC-replay→diffuser-learner)
- Artifact: `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
- Commit: `b9cdd15`
- Notes: 3-seed matrix; promote to 5 seeds for stable confidence intervals

## Running / Active Jobs
- None currently active.

## Next Experiment (runnable)
```bash
# Intent: promote swap matrix from 3→5 seeds to strengthen H2 confidence intervals
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  $PYTHON scripts/exp_swap_matrix_maze2d.py \
  --seeds 0 1 2 3 4 \
  --collectors diffuser sac_her_sparse \
  --learners diffuser sac_her_sparse \
  --modes warmstart frozen \
  --device cuda:0 \
  --base-dir runs/analysis/swap_matrix/5seed_$(date +%Y%m%d-%H%M%S)
```
Expected outputs:
- `runs/analysis/swap_matrix/5seed_.../swap_matrix_results.csv`
- `runs/analysis/swap_matrix/5seed_.../swap_matrix_results.md`

Quick verification:
```bash
ls -lh runs/analysis/swap_matrix/5seed_*/swap_matrix_results.{csv,md}
```

## Open Questions (ranked by blocking priority)
1. Is 3-seed matrix sufficient for paper CI, or must 5-seed run complete? — blocking: yes — resolves with: 5-seed matrix + CI comparison
2. Does replanning cadence/horizon explain Diffuser controllability gap? — blocking: no — resolves with: `scripts/exp_replan_horizon_sweep.py`
3. Is `eval_samples_per_query>1` needed to decouple query-coverage from single-trajectory success? — blocking: no — resolves with: re-run stochasticity diagnostics with `--eval_samples_per_query 4+`
4. What concrete Diffuser modification targets the endpoint-diversity gap? — blocking: no — resolves with: mechanism analysis post-ablation

## Key Artifact Pointers
- LAST_FULL_RUN_PATH: `runs/analysis/swap_matrix/LAST_FULL_RUN_PATH.txt` → `runs/analysis/swap_matrix/swap_matrix_20260219-231605/`
- CONSOLIDATION_ROOT: `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/`
- VISUAL_CHECK_ROOT: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/`
- PAPER_NOTES: `research_finding.txt`, `research_finding_paper_outline.md`
- QUERY0_DIAG: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/trajectory_plots/query_00_first_hit_diagnostics.png`
- QUERY0_ACTION_AUDIT: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/query_00_action_magnitude_audit.json`
- QUERY0_SAC_CADENCE: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/query_00_sac_cadence_sensitivity.json`
- GPT_PRO_HANDOFF_PROMPT: `gpt_pro_diffuser_improvement_question_2026-02-20.txt`

## Debug note: query-0 visual vs numeric mismatch (validated)
- User concern:
  - query-0 looks geometrically cleaner for Diffuser, but steps-to-hit (`<=0.1`) showed SAC faster.
- Verification status:
  - Recomputed first-hit steps directly from `query_00_trajectories.npz`; matched prior result exactly.
  - Therefore implementation is correct for "first timestep reaching threshold".
- Query-0 reconciled facts:
  - First-hit steps:
    - Diffuser hits: `[131, 140, 134, 161, 150, 155]`
    - SAC hits: `[157, 121, 92, 84, None, None]`
  - Geometric efficiency to first-hit (successful rollouts):
    - Diffuser path-to-hit length mean: `2.21`
    - SAC path-to-hit length mean: `3.13`
  - Temporal speed to first-hit:
    - Diffuser mean step displacement to hit: `0.0153`
    - SAC mean step displacement to hit: `0.0285`
  - Post-hit wandering (full-horizon rollout continues after hit):
    - Diffuser after-hit path length mean: `0.248`
    - SAC after-hit path length mean: `1.472`
- Interpretation:
  - Diffuser is often more path-efficient/geometrically cleaner on q0, but SAC can reach `<=0.1` earlier by moving faster per timestep.
  - Visual plots use full-horizon trajectories (not stop-at-hit), so SAC can look worse due to post-hit wandering despite early hit in some rollouts.

## Debug note: query-0 conservative-action hypothesis (validated)
- User question:
  - Is Diffuser's smaller per-step displacement mainly caused by action clipping?
- Verification status:
  - Checked code path: both methods are clipped before `env.step`, but SAC actor is already bounded by tanh.
  - Query-0 audit confirms clipping is minor and not the primary cause.
- Query-0 action-magnitude audit facts:
  - t0 stochastic action samples (`n=20`):
    - Diffuser raw action L2 mean `0.9314`, clip rate `0.000`.
    - SAC raw action L2 mean `1.0791`, clip rate `0.000`.
  - Rollout actions (`6 rollouts x 192 steps`):
    - Diffuser raw action L2 mean `0.7726`, clip fraction `0.0165`.
    - SAC raw action L2 mean `1.2191`, clip fraction `0.0000`.
  - Replay action-magnitude context (seed0 exports):
    - Diffuser replay L2 mean `0.7512` (close to uniform-[-1,1]^2 baseline `0.7653`).
    - SAC replay L2 mean `0.8451` (higher-magnitude distribution).
- Additional control-cadence caveat (SAC query-0 sensitivity):
  - `decision_every=1`: hit-step mean `100.5` over `6/6` successes, displacement mean `0.0118`.
  - `decision_every=16`: hit-step mean `111.4` over `5/6` successes, displacement mean `0.0240`.
- Interpretation:
  - Conservative Diffuser magnitude is not a manual-clipping artifact in this setup.
  - More plausible drivers are method-level behavior: diffusion denoising + plan selection objective (wall hits, final-goal error) and collector control cadence choices.

## Debug note: planner selection semantics (clarified)
- User concern:
  - Does Diffuser do post-hoc selection across multiple environment rollouts, which would be invalid at test time?
- Clarified behavior:
  - Diffuser executes one environment trajectory at a time (no parallel env rollout selection).
  - At each replan step, Diffuser may sample multiple imagined plans from the diffusion model and select one before acting.
  - Selection objective in code is lexicographic `(wall_hits, final_goal_error)`.
- Important context split:
  - Main Diffuser probe/eval path uses `maze_arr` and defaults `wall_aware_planning=True`, `wall_aware_plan_samples=8`, so multi-candidate imagined-plan selection is enabled.
  - `scripts/analyze_collector_stochasticity.py` explicitly calls planner with `maze_arr=None`, forcing effective single-candidate planning (`n_candidates=1`) in that analyzer.
- Interpretation:
  - The prior visual/stochasticity analyzer comparison does not use multi-candidate wall-aware selection in rollout execution.
  - Fairness discussions should distinguish "multiple model samples per decision" from "multiple environment rollout attempts."

## Update: non-privileged planner selection patch (2026-02-20)
- Change applied:
  - Removed wall-hit term from Diffuser plan-selection objective.
  - Plan selection now uses final-goal-error only across imagined candidates.
  - Defaults changed to non-privileged single-candidate planning:
    - `wall_aware_planning=False`
    - `wall_aware_plan_samples=1`
- Files changed:
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - `scripts/exp_swap_matrix_maze2d.py` (explicitly pins `wall_aware_planning=False`, `wall_aware_plan_samples=1` for swap-matrix launches)
- Rationale:
  - Align collector behavior with inference-time assumptions that do not require privileged maze-layout logic for action selection.

## Logging policy
- Append-only history: `HANDOFF_LOG.md`
- Living snapshot: `docs/WORKING_MEMORY.md` (this file — overwrite/compact, do NOT append)
- Record commit hash + subject + scope in HANDOFF_LOG when committing; update this header accordingly.
- Nested `EBM_OnlineRL/` follows the same pattern with its own files.

## Archive note
Pre-cleanup verbose logs: `/root/.log-archive/memory-cleanup-20260219-172245/ebm/`
Pre-migration WM backup: `memory/archive/WORKING_MEMORY_pre_migration_20260220.md`
