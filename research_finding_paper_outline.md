# Paper-Oriented Finding Outline (SAC vs Diffuser Exploration)
Date: 2026-02-20
Project: `/root/ebm-online-rl-prototype`

## 1) Core Claim
- In the current Maze2D setting, SAC's exploration advantage over diffuser is better explained by **control-aware exploration** (better conversion of behavior into goal progress/coverage), not by larger raw action randomness.

## 2) Quantitative Evidence
- Phase1 matched-query aggregate (`n=36` per method; success=`rollout_min_goal_distance<=0.2`):
  - Diffuser: success `0.7222`, min-goal-distance `0.4718`, final-goal-error `0.5405`.
  - SAC: success `0.8333`, min-goal-distance `0.3013`, final-goal-error `0.4863`.
  - Delta (SAC - Diffuser): success `+0.1111`, min-goal-distance `-0.1704`, final-goal-error `-0.0542`.
- Coverage at `h256` in completed swap-matrix learning rows (`n=12` rows per collector):
  - Diffuser replay: success/query-coverage `0.8194 ± 0.1056`, cell-coverage `0.8308 ± 0.1311`.
  - SAC replay: success/query-coverage `0.9097 ± 0.0750`, cell-coverage `0.9014 ± 0.0700`.
- Stochasticity pilot (seed 0, multiple contexts):
  - Diffuser action stochasticity (`action_pairwise_l2`) > SAC in all tested contexts.
  - SAC still shows equal-or-higher endpoint diversity/success in several contexts.

## 3) Mechanistic Interpretation (Careful Wording)
- Supported interpretation:
  - SAC policies are more effective at **directing** exploration toward goal-reaching outcomes.
  - Diffuser can produce higher action variability, but that does not reliably translate to better goal progress.
- Wording to avoid:
  - Do not claim SAC is globally better in all tasks.
  - Do not claim a proven single root cause from this dataset alone.

## 4) Important Caveats
- This is a limited-seed result (`3` seeds in the current swap matrix).
- In this setup, `eval_samples_per_query=1`; query-coverage numerically equals success-rate at each horizon.
- Therefore, cell-coverage and distance/error metrics are the more independent supporting signals.
- Some metrics are not perfectly symmetric across methods (method-specific internals differ), so only directly comparable metrics should be emphasized in paper text.

## 5) Figure Mapping (Suggested)
- Figure A: Phase1 matched-query comparison (Diffuser vs SAC) for:
  - success rate, min-goal-distance, final-goal-error.
- Figure B: Stochasticity vs outcome:
  - action stochasticity (`action_pairwise_l2`) and endpoint diversity/success side-by-side.
- Figure C: Coverage summary at h256:
  - success/query-coverage and cell-coverage by collector replay source.
- Figure D: Geometry/control proxies:
  - `path_over_direct`, `start_jump_ratio`, `end_jump_ratio`.

## 6) Reproducibility Metadata
- Scripts:
  - `scripts/exp_swap_matrix_maze2d.py`
  - `scripts/analyze_collector_stochasticity.py`
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - `scripts/synthetic_maze2d_sac_her_probe.py`
- Primary artifacts:
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/*/seed_*/query_metrics.csv`
  - `runs/analysis/collector_stochasticity/*pilot_q10_s40_r8_h192/collector_stochasticity_summary.json`
- Commit context:
  - `b9cdd15` (`fix: exp_swap_matrix episode_len 64->256; add causal ablation scripts`)
  - Note: workspace has additional uncommitted documentation/tracking files after that commit.

## 7) Ready-to-Use Paper Text (Draft)
- "Across matched Maze2D query sets, SAC achieved higher goal-reaching success than diffuser while exhibiting lower action-level stochasticity, indicating that SAC's advantage is unlikely to stem from larger raw action noise. Instead, the evidence is more consistent with better control-aware exploration dynamics, where policy behavior more effectively converts exploration into goal progress and coverage. This conclusion is bounded by a 3-seed matrix and should be validated with larger-seed follow-up studies."
