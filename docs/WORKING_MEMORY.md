# EBM Online RL Working Memory (living snapshot)

Last updated: 2026-02-19T23:20:00+08:00

## Objective
Validate and improve online Maze2D performance while testing whether gains are driven by data collection policy, learner updates, or both.

## Current best-known concise result
Single-seed matched-budget snapshot (`maze2d-umaze-v1`, seed=0, h256, 12 queries):
- Diffuser: success `0.8333`
- SAC+HER (sparse): success `0.8333`
- SAC+HER (shaped): success `0.7500`
- GCBC+HER: success `0.5833`

## What is clear vs unclear
- Clear (current evidence): GCBC+HER is weaker than Diffuser/SAC-sparse in this setting.
- Not yet established: Diffuser > SAC-sparse in general.
- Not yet established: collector-vs-learner causal mechanism (needs dedicated ablations).

## Active run (as of 2026-02-19T23:20+08:00)
- **Collector–learner swap matrix** restarted via relay-actions callback after bug fix.
  - Script: `scripts/exp_swap_matrix_maze2d.py`
  - Seeds: 0, 1, 2 | Collectors: diffuser, sac_her_sparse | Learners: diffuser, sac_her_sparse
  - Modes: warmstart + frozen | Device: cuda:0
  - Launch wrapper: `/tmp/run_swap_matrix.sh` | Live log: `/tmp/swap_matrix_run.log`
  - Results will land in: `runs/analysis/swap_matrix/full_<STAMP>/swap_matrix_results.{csv,md}`
- **Bug fixed**: `exp_swap_matrix_maze2d.py` line 288: `episode_len` was `64` (GoalDataset yielded 0 samples for diffuser with horizon=64); corrected to `256` (matches prior working runs).
- When complete: read `swap_matrix_results.md` and adjudicate H1 (collector-driven) vs H2 (learner-driven).

## Priority next experiments
1. ~~Fixed-replay and collector/learner swap experiments~~ → **IN FLIGHT** (see above).
2. Multi-seed reruns (at least 3 seeds) for Diffuser, SAC+HER sparse, GCBC+HER under identical protocol.
3. Increase evaluation set size beyond 12 trajectories for stability.

## Latest implementation status (2026-02-19T22:54:13+08:00)
- Implemented replay interoperability for cross-learner transfer:
  - Diffuser probe: `--replay_import_path`, `--replay_export_path`, replay metadata + fingerprint logging.
  - SAC+HER probe: `--replay_import_path`, `--replay_export_path`, replay metadata + fingerprint logging.
- Added fixed-replay switch to both primary probes:
  - `--disable_online_collection` (keeps training rounds, skips new env collection).
- Added waypoint evaluation mode to Diffuser probe:
  - `--eval_waypoint_mode {none,feasible,infeasible}`
  - `--eval_waypoint_t`, `--eval_waypoint_eps`
  - progress metrics now include waypoint hit/distance fields.
- Added experiment orchestration scripts:
  - `scripts/exp_swap_matrix_maze2d.py`
  - `scripts/exp_replan_horizon_sweep.py`
  - `scripts/analyze_posterior_diversity.py`
- Updated eval utility:
  - `scripts/eval_synth_maze2d_checkpoint_prefix.py` now supports `--planning-horizon`.

## Verification snapshot
- Pass:
  - `python3 -m compileall -q scripts`
  - help checks for all modified/new scripts (using `third_party/diffuser/.venv38/bin/python` + Mujoco env vars for Maze2D scripts).
  - replay export/import smoke for Diffuser and SAC+HER sparse (import logs show transitions/episodes + fingerprint).
  - smoke artifact generation:
    - `output/smoke/swap_matrix_sac_smoke4/swap_matrix_results.csv`
    - `output/smoke/swap_matrix_sac_smoke4/swap_matrix_results.md`
    - `output/smoke/replan_sweep_smoke/replan_horizon_sweep.csv`
    - `output/smoke/replan_sweep_smoke/replan_horizon_sweep.md`
  - swap-matrix interpreter fallback fix:
    - `scripts/exp_swap_matrix_maze2d.py` now defaults `--python` to `third_party/diffuser/.venv38/bin/python3` when available (fallbacks: repo `.venv` then `sys.executable`).
    - verified by launching with system Python:
      - `/usr/bin/python3 scripts/exp_swap_matrix_maze2d.py --smoke ...`
      - child-run command in `output/smoke/swap_matrix_python_default_check/phase1_collectors/diffuser/seed_0/stdout_stderr.log` starts with `/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3`.
- Not fully completed in this turn:
  - long-duration swap-matrix/sweep runs
  - full runtime confirmation for waypoint feasible-vs-infeasible gap
  - full runtime confirmation for posterior-diversity analyzer (help/syntax only)

## Canonical paths
- Main comparison artifacts:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356`
- Probe/launcher scripts:
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - `scripts/synthetic_maze2d_gcbc_her_probe.py`
  - `scripts/synthetic_maze2d_sac_her_probe.py`

## Logging policy (for this repo)
- Append-only history: `HANDOFF_LOG.md`
- Living snapshot: `docs/WORKING_MEMORY.md`
- If committing code/docs, record commit hash + subject + scope in handoff and update this file's significance summary.
- Nested `EBM_OnlineRL/` now follows the same pattern with its own `HANDOFF_LOG.md` + `docs/WORKING_MEMORY.md`.

## Archive note
Verbose prior records were preserved at:
`/root/.log-archive/memory-cleanup-20260219-172245/ebm/`
