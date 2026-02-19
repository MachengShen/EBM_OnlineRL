# EBM Online RL Handoff Log (append-only)

## 2026-02-19T17:22:45+08:00
### Scope
- Consolidated repository handoff history.
- Replaced `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` with this canonical `HANDOFF_LOG.md`.

### Current repository state
- Path: `/root/ebm-online-rl-prototype`
- Branch: `master` tracking `origin/main`
- No active training run tracked at handoff time.

### Most relevant experimental outcomes (latest concise view)
- Environment/protocol snapshot: `maze2d-umaze-v1`, seed=0, matched budget, eval horizon 256, 12 diverse queries.
- Latest reported success@h256:
  - Diffuser: `0.8333`
  - SAC+HER (sparse): `0.8333`
  - SAC+HER (shaped): `0.7500`
  - GCBC+HER: `0.5833`
- Interpretation: GCBC+HER underperformed; Diffuser ties SAC+HER sparse in this single-seed setting.

### Key evidence paths
- Comparison run root:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356`
- Summary files:
  - `.../diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
  - `.../sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
  - `.../sac_her_shaped_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
  - `.../gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0_rerun_20260217-203446/summary.json`

### Open technical question
- Main unresolved hypothesis remains causal attribution:
  - Is the observed advantage primarily collector-driven, learner-driven, or both?
- Required next experiment family:
  - fixed-replay and collector/learner swap ablations across multiple seeds.

### Commit note
- Latest major repo head seen in prior handoff chain: `99f4705` (plus earlier `8f8d425`, `b3c0b6f`).
- Future commits must be recorded here with hash + subject + scope.

## 2026-02-19T17:58:00+08:00
### Scope
- Completed memory-file consolidation for the nested prototype repo at `EBM_OnlineRL/`.

### Actions
- Added:
  - `/root/ebm-online-rl-prototype/EBM_OnlineRL/HANDOFF_LOG.md`
  - `/root/ebm-online-rl-prototype/EBM_OnlineRL/docs/WORKING_MEMORY.md`
- Removed:
  - `/root/ebm-online-rl-prototype/EBM_OnlineRL/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`

### Outcome
- EBM prototype now consistently uses:
  - append-only handoff logs
  - compact living working-memory snapshots

## 2026-02-19T22:54:13+08:00
### Scope
- Implemented the GPT-Pro validation plan TODOs in code (replay interoperability, swap/sweep runners, waypoint eval mode, posterior-diversity analyzer), with smoke-level verification.

### Changed
- `scripts/synthetic_maze2d_diffuser_probe.py`
  - Added replay artifact standardization and metadata hash support:
    - new helpers: `save_replay_artifact`, `load_replay_artifact`, `replay_dataset_fingerprint`
    - backward-compatible aliases preserved: `--replay_load_npz/--replay_save_npz`
    - new preferred aliases: `--replay_import_path/--replay_export_path`
  - Added fixed-replay control flag: `--disable_online_collection`.
  - Added waypoint evaluation mode:
    - `--eval_waypoint_mode {none,feasible,infeasible}`
    - `--eval_waypoint_t`
    - `--eval_waypoint_eps`
  - Evaluation now logs waypoint metrics (`waypoint_hit_rate`, `waypoint_min_distance_mean`) in progress/summary.
- `scripts/synthetic_maze2d_sac_her_probe.py`
  - Added replay import/export support (`--replay_import_path`, `--replay_export_path`) with import/export fingerprint logging.
  - Added `--disable_online_collection` for frozen-replay learner runs.
- `scripts/eval_synth_maze2d_checkpoint_prefix.py`
  - Added `--planning-horizon` override.
  - Updated to call `evaluate_goal_progress` with replay observations + explicit waypoint-off params.
- Added new orchestrator/analyzer scripts:
  - `scripts/exp_swap_matrix_maze2d.py`
  - `scripts/exp_replan_horizon_sweep.py`
  - `scripts/analyze_posterior_diversity.py`

### Evidence
- TODO1 existence/tracking:
  - `git ls-files scripts/synthetic_maze2d_diffuser_probe.py` (present and tracked).
- Syntax verification:
  - `python3 -m compileall -q scripts` (pass).
- CLI verification (venv + Mujoco env):
  - `scripts/synthetic_maze2d_diffuser_probe.py --help` includes replay import/export + waypoint + disable collection flags.
  - `scripts/synthetic_maze2d_sac_her_probe.py --help` includes replay import/export + disable collection flags.
  - `scripts/eval_synth_maze2d_checkpoint_prefix.py --help` includes `--planning-horizon`.
  - New scripts help passed:
    - `scripts/exp_swap_matrix_maze2d.py --help`
    - `scripts/exp_replan_horizon_sweep.py --help`
    - `scripts/analyze_posterior_diversity.py --help`
- Replay export/import smoke (Diffuser):
  - Export log: `output/smoke/plan_impl_diffuser_replay/export/stdout_stderr.log`
  - Import log: `output/smoke/plan_impl_diffuser_replay/import/stdout_stderr.log`
  - Evidence line: `[replay] imported transitions=48 episodes=2 fingerprint=bb2297b5497ac4e1`
- Replay export/import smoke (SAC+HER sparse):
  - Export log: `output/smoke/plan_impl_sac_replay/export/stdout_stderr.log`
  - Import log: `output/smoke/plan_impl_sac_replay/import/stdout_stderr.log`
  - Evidence line: `[replay] imported ... fingerprint=bb2297b5497ac4e1`
- Swap-matrix runner smoke artifact generation:
  - `output/smoke/swap_matrix_sac_smoke4/swap_matrix_results.csv`
  - `output/smoke/swap_matrix_sac_smoke4/swap_matrix_results.md`
- Replan×horizon sweep runner smoke artifact generation:
  - `output/smoke/replan_sweep_smoke/replan_horizon_sweep.csv`
  - `output/smoke/replan_sweep_smoke/replan_horizon_sweep.md`

### Open items / limitations
- Full-duration swap matrix and sweep were not executed in this turn (only smoke-level artifact checks).
- Waypoint feasible vs infeasible runtime sanity (`infeasible` hit-rate lower) was implemented but not fully validated end-to-end due long-running probe jobs.
- `scripts/analyze_posterior_diversity.py` was added and help-verified; full runtime smoke did not complete within short timeout budget in this turn.

### Next step
- Launch long-running matrix/sweep with relay auto-callback (`relay-long-task-callback`) and follow-up analysis task on completion.

## 2026-02-19T23:08:38+08:00
### Scope
- Fixed a run-blocking orchestration issue discovered during real preflight of the swap-matrix runner.

### Changed
- `scripts/exp_swap_matrix_maze2d.py`
  - Added `_default_python(root)` selection logic and wired argparse default:
    - prefer `third_party/diffuser/.venv38/bin/python3`
    - fallback to repo `.venv/bin/python3`
    - final fallback to `sys.executable`
  - Goal: make `exp_swap_matrix_maze2d.py` robust when invoked from `/usr/bin/python3` (which lacks `gym/d4rl`).

### Evidence
- Real preflight failure root cause before patch:
  - `output/smoke/swap_matrix_preflight_real/phase2_learners/warmstart/diffuser_to_diffuser/seed_0/stdout_stderr.log`
  - Error: `ModuleNotFoundError: No module named 'gym'` from `/usr/bin/python3`.
- Post-patch verification:
  - Command: `/usr/bin/python3 scripts/exp_swap_matrix_maze2d.py --smoke --seeds 0 --collectors diffuser --learners diffuser --modes warmstart --base-dir output/smoke/swap_matrix_python_default_check --timeout-min 0.03 --device cpu`
  - Child command line in:
    - `output/smoke/swap_matrix_python_default_check/phase1_collectors/diffuser/seed_0/stdout_stderr.log`
  - Now starts with:
    - `/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3 .../synthetic_maze2d_diffuser_probe.py`

### Open items
- Long-running matrix/sweep execution is still pending; next action should use relay auto-callback/watch with this patched runner to avoid immediate environment failures.

## 2026-02-19T15:08:34Z
### Scope
- Launched full collector–learner swap matrix experiment (H1 vs H2 causal ablation).

### Actions
- Fixed wrapper script `/tmp/run_swap_matrix.sh` to use correct Python:
  `third_party/diffuser/.venv38/bin/python3.8`
- Validated pipeline via live smoke run:
  - env loads, 20 train steps complete, eval rollout fires, replay export path wired.
  - Slowness on CPU confirmed (diffusion inference ~1175% CPU for smoke); full run uses `cuda:0`.
- Launched full run in background (Claude Bash bg task `b3d16cd`):
  - seeds: 0, 1, 2
  - collectors: diffuser, sac_her_sparse
  - learners: diffuser, sac_her_sparse
  - modes: warmstart + frozen
  - device: cuda:0

### Active run artifacts
- Live log: `/tmp/swap_matrix_run.log`
- Results will be written to: `runs/analysis/swap_matrix/swap_matrix_<STAMP>/`
  - `swap_matrix_results.csv`
  - `swap_matrix_results.md`

### Next step
- When run completes, read `swap_matrix_results.md` and report H1 vs H2 verdict.
- Update this log with commit hash if results are committed.

## 2026-02-19T23:20:00+08:00
### Scope
- Plan execution: preflight checks, bug fix, and relaunch of collector-learner swap matrix.

### Actions
- Identified correct Python env: `third_party/diffuser/.venv38/bin/python3.8` (has gym 0.23.1, torch, d4rl).
- Confirmed all 6 experiment scripts pass `--help` checks with that env + correct PYTHONPATH.
- Found and fixed bug in `scripts/exp_swap_matrix_maze2d.py`:
  - Line 288: `episode_len` was `64` (full mode), caused `GoalDataset produced zero samples` for diffuser (horizon=64 requires episode_len > 64; prior working runs used 256).
  - Fix: changed to `256`.
- Killed failed full run (PIDs 30794, 57233) that had already been launched with the buggy config.
- Updated `/tmp/run_swap_matrix.sh` wrapper to use `--python $VENV_PY` explicitly.
- Restarted full swap matrix run via `relay-actions` job_start callback (see below).

### Run configuration (corrected)
- Seeds: 0, 1, 2 | Collectors: diffuser, sac_her_sparse | Learners: diffuser, sac_her_sparse
- Modes: warmstart + frozen | Device: cuda:0
- `episode_len=256, horizon=64, n_episodes=400` (matches prior working runs)
- Log: `/tmp/swap_matrix_run.log`
- Results: `runs/analysis/swap_matrix/full_<STAMP>/swap_matrix_results.{csv,md}`

### Commit to record
- Commit will follow this entry; hash to be appended once made.
