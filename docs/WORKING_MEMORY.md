# EBM Online RL Working Memory (living snapshot)

Last updated: 2026-02-21T18:30:00+08:00
Repo: /root/ebm-online-rl-prototype
Branch: master
Commit: b8bd16b
GitHub: https://github.com/MachengShen/EBM_OnlineRL/tree/master

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

- H3 (partially supported, revised): SAC collector advantage is control-aware (better goal-directed coverage), not raw action noise.
  - Status: partially supported — consolidation + visual evidence holds; ablation grid synergy was threshold-artefact
  - Evidence FOR: Diffuser action pairwise L2 > SAC in `11/12` cells; SAC endpoint pairwise L2 > Diffuser in `11/12` cells — `consolidated_overall_summary.json`
  - Evidence FOR: SAC reaches near-goal (`<=0.1`) ~61 steps faster on average where both succeed (seed-0, 4/6 queries) — `visual_check_phase1_seed0_q6_s20_r6_h192/steps_to_goal_threshold_0p1_summary.json`
  - Evidence REVISED (grid v1 vs v2 correction): Grid v1 (threshold=0.5) showed EMA+adaptive synergy of +11pp and best Diffuser 0.639 vs SAC 0.667 (2.8pp gap). Grid v2 (threshold=0.2, corrected) shows NO synergy — beta×adaptive interaction disappears (Δ≈0), best Diffuser 0.500 vs SAC 0.639 (13.9pp gap). The interventions help Diffuser reach within-0.5 of goal but NOT within-0.2.
  - Evidence FOR (alpha scaling): alpha>1.0 still neutral-to-harmful at threshold=0.2 (alpha=1.4 clips 53%), disconfirming raw-magnitude deficit.
  - SAC baseline at threshold=0.2: mean 0.646 (range 0.556–0.694).
  - Next discriminating test: understand WHY EMA+adaptive helps at 0.5 but not 0.2. Hypothesis: smoothing helps gross approach but Diffuser still fails final-meter precision. Test: visualize last-N-step trajectories near goal for best vs baseline condition.

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
- None currently running.

## Ablation Grid Results

### Grid v1 (COMPLETE — grid_20260221-134801, threshold=0.5 — HISTORICAL ONLY)
- **WARNING**: threshold=0.5 was wrong; results not comparable to swap matrix (threshold=0.2)
- Best Diffuser: alpha=1.0, beta=0.5, adapt=True → success@0.5=0.639 (SAC 0.667, gap 2.8pp)
- Appeared to show strong EMA+adaptive synergy (+11pp) — artefact of loose threshold

### Grid v2 (COMPLETE — grid_v2_20260221-171631, threshold=0.2 — CANONICAL)
- Design: 3×2×2 factorial — alpha∈{1.0,1.2,1.4} × beta∈{0.0,0.5} × adaptive∈{0,1}, seed_0 diagnostic (6q×6r, h192)
- Pooled SAC baseline: `success@0.2=0.646` (range 0.556–0.694 across conditions)
- Diffuser baseline (alpha=1.0, beta=0.0, adapt=False): `success@0.2=0.444`, SAC gap = 25.0pp
- **Best Diffuser**: `alpha=1.4, beta=0.5, adapt=True` → `success@0.2=0.500` (+5.6pp vs baseline, SAC gap = **13.9pp**)
- Artifact: `runs/analysis/ablation_grid/grid_v2_20260221-171631/ablation_grid_results.csv`

### Main effects (avg Diffuser success@0.2)
| Factor | Level | Avg |
|--------|-------|-----|
| alpha | 1.0 | 0.444 |
| alpha | 1.2 | 0.431 |
| alpha | **1.4** | **0.465** |
| beta | 0.0 | 0.440 |
| beta | **0.5** | **0.454** |
| adaptive | **False** | **0.454** |
| adaptive | True | 0.440 |

Note: adaptive=False is now *slightly better* at threshold=0.2 (reversed vs threshold=0.5 finding).

### Key interaction: beta × adaptive (threshold=0.2)
| | adapt=F | adapt=T | Δ |
|---|---|---|---|
| beta=0.0 | 0.444 | 0.435 | -0.009 |
| beta=0.5 | 0.463 | 0.444 | -0.019 |

**No synergy at threshold=0.2.** The EMA+adaptive interaction seen in grid v1 was a threshold=0.5 artefact.

### Full ranking (success@0.2)
| Condition | D_succ | SAC_succ | gap | D_hit@0.2 | SAC_hit@0.2 | clip |
|---|---|---|---|---|---|---|
| alpha=1.4, beta=0.5, adapt=T | **0.500** | 0.639 | 13.9pp | 0.500 | 0.639 | 0.534 |
| alpha=1.0, beta=0.0, adapt=T | 0.472 | 0.611 | 13.9pp | 0.472 | 0.611 | 0.034 |
| alpha=1.0, beta=0.5, adapt=F | 0.472 | 0.694 | 22.2pp | 0.472 | 0.694 | 0.037 |
| alpha=1.4, beta=0.5, adapt=F | 0.472 | 0.667 | 19.4pp | 0.472 | 0.667 | 0.538 |
| alpha=1.2, beta=0.5, adapt=T | 0.444 | 0.556 | 11.1pp | 0.444 | 0.556 | 0.365 |
| alpha=1.0, beta=0.0, adapt=F (baseline) | 0.444 | 0.694 | 25.0pp | 0.444 | 0.694 | 0.037 |
| alpha=1.2, beta=0.0, adapt=F | 0.444 | 0.694 | 25.0pp | 0.444 | 0.694 | 0.368 |
| alpha=1.2, beta=0.5, adapt=F | 0.444 | 0.583 | 13.9pp | 0.444 | 0.583 | 0.371 |
| alpha=1.4, beta=0.0, adapt=F | 0.444 | 0.667 | 22.2pp | 0.444 | 0.667 | 0.537 |
| alpha=1.4, beta=0.0, adapt=T | 0.444 | 0.694 | 25.0pp | 0.444 | 0.694 | 0.529 |
| alpha=1.2, beta=0.0, adapt=T | 0.389 | 0.611 | 22.2pp | 0.389 | 0.611 | 0.361 |
| alpha=1.0, beta=0.5, adapt=T | 0.389 | 0.639 | 25.0pp | 0.389 | 0.639 | 0.036 |

### Key interpretation
- Execution-time interventions (EMA, adaptive replan) help at coarse threshold (0.5) but **do not transfer to precise goal-reaching (0.2)**.
- SAC gap at threshold=0.2 remains ~14–25pp; EMA+adaptive only recovered ~5.6pp (not the previously claimed 11pp).
- New hypothesis: Diffuser fails at final-meter precision. The model may approach the goal but oscillates or diverges in the last 0.2 units. This is a planner objective / denoising precision issue, not a gross control-cadence issue.

## Next Experiment — Priority A (blocking — paper CI)
```bash
# Intent: promote swap matrix from 3→5 seeds to strengthen H2 confidence intervals
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_swap_matrix_maze2d.py \
  --seeds 0 1 2 3 4 --collectors diffuser sac_her_sparse \
  --learners diffuser sac_her_sparse --modes warmstart frozen \
  --device cuda:0 --base-dir runs/analysis/swap_matrix/5seed_$(date +%Y%m%d-%H%M%S) \
  2>&1 | tee runs/analysis/swap_matrix/5seed_latest.log
```

## Next Experiment — Priority B (mechanism understanding)
```bash
# Intent: visualize last-N-step trajectories for best (alpha=1.4,beta=0.5,adapt=T) vs baseline
# to understand why EMA+adaptive helps at 0.5 but not 0.2 (final-meter precision hypothesis)
# Use existing visual_check infra: add --out-last-n-steps flag or post-process trajectory NPZ
# (no new script needed — read alpha1.40_beta0.50_adapt1_K1/collector_stochasticity_summary.json
#  and cross-check trajectories for near-miss (0.2–0.5 band) vs success (<=0.2))
```

## Open Questions (ranked by blocking priority)
1. Is 3-seed matrix sufficient for paper CI, or must 5-seed run complete? — blocking: yes — resolves with: 5-seed matrix + CI comparison
2. Does replanning cadence/horizon explain Diffuser controllability gap? — blocking: no — resolves with: `scripts/exp_replan_horizon_sweep.py`
3. Is `eval_samples_per_query>1` needed to decouple query-coverage from single-trajectory success? — blocking: no — resolves with: re-run stochasticity diagnostics with `--eval_samples_per_query 4+`
4. What concrete Diffuser modification targets the endpoint-diversity gap? — blocking: no — resolves with: mechanism analysis post-ablation
5. How much of Diffuser collector quality depended on privileged multi-candidate selection defaults? — blocking: no — resolves with: 1-seed and 3-seed reruns under non-privileged defaults vs old defaults

## Key Artifact Pointers
- LAST_FULL_RUN_PATH: `runs/analysis/swap_matrix/LAST_FULL_RUN_PATH.txt` → `runs/analysis/swap_matrix/swap_matrix_20260219-231605/`
- CONSOLIDATION_ROOT: `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/`
- VISUAL_CHECK_ROOT: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/`
- PAPER_NOTES: `research_finding.txt`, `research_finding_paper_outline.md`
- QUERY0_DIAG: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/trajectory_plots/query_00_first_hit_diagnostics.png`
- QUERY0_ACTION_AUDIT: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/query_00_action_magnitude_audit.json`
- QUERY0_SAC_CADENCE: `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/query_00_sac_cadence_sensitivity.json`
- GPT_PRO_HANDOFF_PROMPT: `gpt_pro_diffuser_improvement_question_2026-02-20.txt`
- SMOKE_OLDVSNEW_LOG: `runs/analysis/smoke_old_vs_new_defaults_latest.log`
- SMOKE_OLDVSNEW_ROOT: `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/`
- SMOKE_OLDVSNEW_SUMMARY: `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/old_vs_new_smoke_summary.json`
- SMOKE_OLDVSNEW_ROWS: `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/old_vs_new_smoke_rows.csv`

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

## Update: old-vs-new default smoke (2026-02-20)
- Scope:
  - Compared old default behavior (`wall_aware_planning=True`, `wall_aware_plan_samples=8`) vs new non-privileged defaults (`wall_aware_planning=False`, `wall_aware_plan_samples=1`) using the same trained checkpoint + replay artifact.
  - Budget: `6 queries x 3 rollouts`, rollout horizon `128`, success threshold `0.2`, replan every `16`.
- Readout (new minus old):
  - success rate: `-0.0556` (`0.2778` vs `0.3333`)
  - min goal distance mean: `+0.0568` (`0.7802` vs `0.7234`)
  - final goal distance mean: `+0.0567` (`0.7824` vs `0.7257`)
  - wall hits mean: `+18.7778` (`47.5` vs `28.72`)
  - first-hit (`<=0.1`) counts: `1` vs `1` (not enough paired hits for robust speed inference)
- Caveat:
  - This is a small smoke result, not a stable estimate; per-query effects were mixed (some queries improved under new defaults).
  - Next step is to promote to at least 1-seed full eval, then multi-seed confirmation.

## Logging policy
- Append-only history: `HANDOFF_LOG.md`
- Living snapshot: `docs/WORKING_MEMORY.md` (this file — overwrite/compact, do NOT append)
- Record commit hash + subject + scope in HANDOFF_LOG when committing; update this header accordingly.
- Nested `EBM_OnlineRL/` follows the same pattern with its own files.

## Archive note
Pre-cleanup verbose logs: `/root/.log-archive/memory-cleanup-20260219-172245/ebm/`
Pre-migration WM backup: `memory/archive/WORKING_MEMORY_pre_migration_20260220.md`


## 2026-02-20T14:48:18.435Z
### Objective
- Preserve the current validation-cycle state and hand off remaining callback-driven Maze2D experiment work for future agents.

### Changes
- Updated `HANDOFF_LOG.md` with new handoff content (`+70` lines).
- Updated `docs/WORKING_MEMORY.md` with current progress/context (`+25/-2`).
- Added untracked handoff artifacts: `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, and `memory/`.
- Captured current execution state: `pending=0`, `running=0`, `done=1`, `failed=0`, `blocked=1`, `canceled=0`.
- Preserved active plan tail (tasks 7-15) covering script alignment, smoke checks, callback mini-pipeline, mismatch fixes, and full validation launches.

### Evidence
- Repo context: `/root/ebm-online-rl-prototype` on branch `master`.
- Command: `git status --porcelain=v1`
- Key output: `M HANDOFF_LOG.md`, ` M docs/WORKING_MEMORY.md`, `?? gpt_pro_handoff_bundle_20260220.zip`, `?? gpt_pro_handoff_bundle_20260220/`, `?? memory/`.
- Command: `git diff --stat`
- Key output: `HANDOFF_LOG.md | 70 +...`, `docs/WORKING_MEMORY.md | 25 +...--`, `2 files changed, 93 insertions(+), 2 deletions(-)`.

### Next steps
- Provide the exact attached plan text/path so tasks can be mapped precisely (instead of inferred from filenames).
- Confirm the exact `relay-long-task-callback` command/interface expected in this repo.
- Confirm whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` should be recreated in this validation cycle.
- Continue remaining plan tail: run smoke verification, harden aggregation/experiment scripts, execute mini end-to-end callback pipeline, fix any schema/analysis mismatches, then launch full validation runs with ongoing `docs/WORKING_MEMORY.md` and `HANDOFF_LOG.md` updates.


## 2026-02-21T07:41:43.578Z
### Objective
- Capture a handoff snapshot for the completed documentation/memory update cycle in `/root/ebm-online-rl-prototype`.
- Preserve current repo state and concrete follow-ups for the next agent.

### Changes
- Updated `HANDOFF_LOG.md` (50 inserted lines).
- Updated `docs/WORKING_MEMORY.md` (net repo diff: 118 insertions, 25 deletions across 2 files).
- Left untracked artifacts present: `gpt_pro_bundle_20260221.zip`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`.
- Task state at handoff: `pending=0`, `running=0`, `done=1`, `failed=0`, `blocked=0`, `canceled=0`.

### Evidence
- Path/context: `/root/ebm-online-rl-prototype` on branch `master`.
- Command: `git status --porcelain=v1`
- Output paths:
- `HANDOFF_LOG.md` (modified)
- `docs/WORKING_MEMORY.md` (modified)
- `gpt_pro_bundle_20260221.zip` (untracked)
- `gpt_pro_handoff_bundle_20260220.zip` (untracked)
- `gpt_pro_handoff_bundle_20260220/` (untracked)
- `memory/` (untracked)
- Command: `git diff --stat`
- Output summary:
- `HANDOFF_LOG.md | 50 +++++++++++++++++++++++++++`
- `docs/WORKING_MEMORY.md | 93 ++++++++++++++++++++++++++++++++++++--------------`
- `2 files changed, 118 insertions(+), 25 deletions(-)`

### Next steps
- Review and commit `HANDOFF_LOG.md` + `docs/WORKING_MEMORY.md` together if content is final.
- Decide retention policy for bundle artifacts and `memory/` (commit vs ignore vs external storage).
- Continue the next run from `docs/WORKING_MEMORY.md` and append the subsequent result to `HANDOFF_LOG.md`.
