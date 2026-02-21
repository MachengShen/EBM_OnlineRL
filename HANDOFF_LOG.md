# EBM Online RL Handoff Log (append-only)

## 2026-02-21T15:35:00+08:00
<!-- meta: {"type":"experiment_complete","artifact":"runs/analysis/ablation_grid/grid_20260221-134801","dirty":false} -->

### Scope
Ablation grid complete (12/12 conditions). Key findings confirm EMA+adaptive_replan combination as the winning intervention.

### Results
- **Best**: `alpha=1.0, beta=0.5, adaptive=True` → success=63.9% (baseline 52.8%, +11pp; SAC 66.7-72.2% on same eval)
- Diffuser-SAC collector gap reduced from ~19pp → ~8pp with best params
- Key interaction: EMA alone (50.0%) and adaptive alone (52.8%) do NOT help; **together** → 63.9%
- Action scaling (alpha>1.0): neutral to mildly counterproductive; alpha=1.4 causes ~53% clip fraction (saturation)
- Second best: `alpha=1.4, beta=0.5, adaptive=True` → 58.3% (min_dist 0.558, slightly better path quality than #1)
- Artifact: `runs/analysis/ablation_grid/grid_20260221-134801/ablation_grid_results.{csv,json}`

### H3 update
- H3 (SAC control-aware advantage) now partially targeted: best Diffuser params reduce the gap by ~57% (19→8pp)
- Remaining gap is structural (replanning cadence, horizon, goal-directed plan selection) — not just action magnitude

### Next step
- 5-seed swap matrix to strengthen H2 confidence intervals (see WORKING_MEMORY "Next Experiment")
- Then: re-run 3-seed swap matrix WITH best ablation params (beta=0.5, adaptive=True) to test end-to-end gap closure

---

## 2026-02-21T14:30:00+08:00
<!-- meta: {"type":"bugfix+launch","commit":"33d2484","dirty":false} -->

### Scope
Fixed CLI arg underscore→hyphen mismatch in `exp_diffuser_ablation_grid.py` (all conditions were failing with argparse error). Ablation grid is now running.

### Actions
- `33d2484`: Fixed `exp_diffuser_ablation_grid.py`: all args now use hyphens (`--diffuser-run-dir`, `--num-queries`, etc.), removed `--no-adaptive-replan` (not a valid argparse flag), fixed `--outdir`→`--out-dir`, `--sac_run_dir`→`--sac-run-dir`.
- Pushed fix to master + main.
- Relaunched ablation grid: `runs/analysis/ablation_grid/grid_20260221-134801` — confirmed PID 31439 at 102% CPU on condition `alpha1.00_beta0.00_adapt0_K1`.

### Grid spec
- 12 conditions: alpha=[1.0,1.2,1.4] × beta=[0.0,0.5] × adaptive=[0,1] × plan_samples=[1]
- Budget per condition: 6 queries × 20 samples × 6 rollouts × horizon=192
- Output: `runs/analysis/ablation_grid/grid_20260221-134801/ablation_grid_results.{csv,json}`

---

## 2026-02-21T14:00:00+08:00
<!-- meta: {"type":"implementation","commits":["f9cbca3","f1e40e1","f3abe66","3a0e512","52506d6"],"dirty":false} -->

### Scope
Implemented GPT Pro's Diffuser improvement plan (TOP-3-A + RANK 1 prefix scoring). All changes are in execution-time; no retraining required. Also created ablation grid runner. Pushed all commits to master + main.

### Actions
- T1 `f9cbca3`: Added 9 new Config fields + CLI args to `synthetic_maze2d_diffuser_probe.py`:
  - `diffuser_action_scale_mult`, `diffuser_action_ema_beta` (action transform, RANK 2)
  - `adaptive_replan`, `adaptive_replan_min`, `adaptive_replan_max`, `adaptive_replan_progress_eps` (adaptive replan, RANK 4)
  - `plan_samples`, `plan_score_mode`, `plan_score_prefix_len` (prefix-progress scoring, RANK 1)
- T2 `f1e40e1`: Extended `sample_best_plan_from_obs` with best-of-K prefix scoring (min_dist_prefix / dist_at_L modes).
- T3+T4 `f3abe66`: Extended `rollout_to_goal` with action scaling+EMA and adaptive replanning logic.
- T5 `3a0e512`: Extended `analyze_collector_stochasticity.py`:
  - New flags: `--diffuser-action-scale-mult`, `--diffuser-action-ema-beta`, `--adaptive-replan`, `--plan-samples`, `--plan-score-mode`
  - New output metrics: `mean_action_l2_raw`, `clip_fraction`, `hit_rate_0p1`, `hit_rate_0p2`, `steps_to_0p1_mean`, `steps_to_0p2_mean`
  - SAC is now optional (`--sac-run-dir` can be omitted for Diffuser-only ablations)
- T6 `52506d6`: New `scripts/exp_diffuser_ablation_grid.py` — sweeps alpha × beta × adaptive × plan_samples.
- Implementation plan: `docs/plans/2026-02-21-diffuser-collector-improvements.md`
- Pushed all commits to: `https://github.com/MachengShen/EBM_OnlineRL/tree/master` and `main`

### Changed files
- `scripts/synthetic_maze2d_diffuser_probe.py` (+175 lines)
- `scripts/analyze_collector_stochasticity.py` (+265 lines)
- `scripts/exp_diffuser_ablation_grid.py` (new)
- `docs/plans/2026-02-21-diffuser-collector-improvements.md` (new)

### Next step (runnable — ablation grid)
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_diffuser_ablation_grid.py \
  --diffuser_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0 \
  --sac_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_0 \
  --num_queries 6 --samples_per_query 20 --rollouts_per_query 6 --rollout_horizon 192 \
  --alpha_grid 1.0,1.2,1.4 --beta_grid 0.0,0.5 --adaptive_grid 0,1 --plan_samples_grid 1 \
  --base_dir runs/analysis/ablation_grid/grid_YYYYMMDD-HHMMSS \
  2>&1 | tee runs/analysis/ablation_grid_latest.log
```
Expected outputs:
- `runs/analysis/ablation_grid/grid_.../ablation_grid_results.csv`
- `runs/analysis/ablation_grid/grid_.../ablation_grid_results.json`

## 2026-02-21T14:30:00+08:00
<!-- meta: {"type":"fix+launch","commit":"33d2484","dirty":false} -->

### Scope
Fixed ablation grid arg names (underscore→hyphen, `--outdir`→`--out-dir`, removed `--no-adaptive-replan`). Launched corrected 12-condition ablation grid.

### Actions
- Bug: `exp_diffuser_ablation_grid.py` passed underscore-style args to analyzer → argparse error exit_code=2 on all conditions.
- Fix commit `33d2484`: corrected all arg names, removed non-existent `--no-adaptive-replan` flag.
- Two failed runs: `grid_20260221-134600`, `grid_20260221-134630` (both usable as negative control).
- Corrected grid launched at 13:48: `runs/analysis/ablation_grid/grid_20260221-134801` (12 conditions, PID 31435).
  - First condition `alpha1.00_beta0.00_adapt0_K1` running (13:48–, analyzer PID 31439, 103% CPU).

### Next step
Monitor `grid_20260221-134801`. When `ablation_grid_results.csv` contains >1 row with metrics, analyze top conditions.

## 2026-02-20T23:00:00+08:00
<!-- meta: {"type":"commit+bundle","commit":"5c9bcb6","dirty":false} -->

### Scope
- Committed all experiment scripts and research notes to GitHub remote (master branch).
- Created GPT-Pro handoff zip bundle with all referenced artifacts.

### Actions
- Staged and committed: `scripts/analyze_collector_stochasticity.py`, `scripts/exp_swap_matrix_maze2d.py`, `scripts/synthetic_maze2d_diffuser_probe.py`, `gpt_pro_diffuser_improvement_question_2026-02-20.txt`, `research_finding.txt`, `research_finding_paper_outline.md`, `docs/WORKING_MEMORY.md`, `HANDOFF_LOG.md`; deleted `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.
- Pushed to: `https://github.com/MachengShen/EBM_OnlineRL/tree/master`
- Commit: `5c9bcb6` — "feat: add collector-stochasticity analyzer, GPT-Pro handoff, and non-privileged Diffuser defaults"
- Created bundle zip:
  - Path: `/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip` (0.18 MB)
  - Structure:
    ```
    gpt_pro_handoff_bundle_20260220/
      gpt_pro_diffuser_improvement_question_2026-02-20.txt  ← updated with repo/ paths
      repo/
        scripts/{synthetic_maze2d_diffuser_probe.py, analyze_collector_stochasticity.py,
                 exp_swap_matrix_maze2d.py, exp_replan_horizon_sweep.py}
        runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv
        runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_overall_summary.json
        runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/{...json, ...npz, ...png}
    ```

### GitHub link
- https://github.com/MachengShen/EBM_OnlineRL/blob/master/gpt_pro_diffuser_improvement_question_2026-02-20.txt

### Next step
- Send `gpt_pro_handoff_bundle_20260220.zip` + GitHub link to GPT Pro for ranked intervention suggestions.

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

### Commit
- `b9cdd15` fix: exp_swap_matrix episode_len 64->256; add causal ablation scripts
  - Scope: scripts/exp_swap_matrix_maze2d.py, scripts/exp_replan_horizon_sweep.py, scripts/analyze_posterior_diversity.py, probe scripts, docs/WORKING_MEMORY.md

## 2026-02-19T23:16:37+08:00
### Scope
- Attempted to execute callback analysis task `t-0001` (summarize full swap-matrix results and update memory artifacts).

### Evidence inspected
- Path pointer:
  - `runs/analysis/swap_matrix/LAST_FULL_RUN_PATH.txt` -> `runs/analysis/swap_matrix/full_20260219-230952`
- Output presence check:
  - Missing:
    - `runs/analysis/swap_matrix/full_20260219-230952/swap_matrix_results.csv`
    - `runs/analysis/swap_matrix/full_20260219-230952/swap_matrix_results.md`
- Active process check:
  - `ps` shows running matrix orchestration and child collector job:
    - `scripts/exp_swap_matrix_maze2d.py --python ... --seeds 0,1,2 ... --device cuda:0`
    - child `scripts/synthetic_maze2d_diffuser_probe.py ... --logdir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0`
- Wrapper log:
  - `/tmp/swap_matrix_run.log` contains launch header only; no completion footer yet.

### Conclusion
- Task is blocked by run state: the long swap-matrix job is still in progress, so final CSV/MD aggregates do not exist yet.

### Next step
- Re-run this analysis callback after run completion and after `swap_matrix_results.csv/.md` appear in the active run directory.


## 2026-02-19T15:17:10.490Z
### Objective
- Preserve continuity for the validation-cycle handoff on `master`, including current repo state, blocked status, and unresolved operator questions needed to proceed.

### Changes
- `HANDOFF_LOG.md` modified (`+28` lines).
- `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` deleted (`-1458` lines).
- `docs/WORKING_MEMORY.md` modified (`15` line-level changes per diff stat).
- Current task counters: `pending=0`, `running=0`, `done=0`, `failed=0`, `blocked=1`, `canceled=0`.
- Last recorded plan tail indicates remaining work centers on script/schema alignment, mini end-to-end callback validation, mismatch fixes, and then full validation launches with ongoing memory/handoff updates.

### Evidence
- Repo root: `/root/ebm-online-rl-prototype`
- Branch: `master`
- Command: `git status --porcelain=v1`
- Output paths: `HANDOFF_LOG.md`, `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`, `docs/WORKING_MEMORY.md`
- Command: `git diff --stat`
- Diff summary: `3 files changed, 38 insertions(+), 1463 deletions(-)`
- Plan snapshot source includes remaining steps 8-15 and the three unresolved clarification questions.

### Next steps
- Provide the exact attached plan content or file path to map remaining steps precisely.
- Confirm the exact `relay-long-task-callback` command/interface expected in this repo.
- Decide whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` must be recreated during this validation cycle.
- After clarifications, resume from step 8 (SAC/HER probe contract checks) and continue through mini-pipeline verification before full callback-driven validation runs.

## 2026-02-19T15:16:04Z
### Scope
- Diagnosed and fixed GoalDataset zero-samples bug blocking diffuser collector cells.
- Restarted clean full swap matrix run.

### Bug
- Root cause: prior Codex run (`full_20260219-230952`) launched diffuser probe with `--episode_len 64 --horizon 64`.
- GoalDataset requires `episode_len > horizon`; equal values produce zero valid samples → immediate crash.
- SAC cells were unaffected (SAC does not use GoalDataset).
- Fix: `exp_swap_matrix_maze2d.py` already had correct default (`episode_len=256`, `horizon=64`); stale Codex config was the issue.

### Active run (clean restart)
- PID: 14848 (bash wrapper), child 14858 (python3.8 orchestrator)
- Wrapper: `/tmp/run_swap_matrix.sh` (updated with fix annotation)
- Log: `/tmp/swap_matrix_run.log` (wrapper headers) + per-cell logs in run dir
- Run dir: `runs/analysis/swap_matrix/swap_matrix_20260219-231605/`
- Confirmed running: diffuser collector seed_0 at step ~2400/6000 (no GoalDataset error)

### Next
- Wait for all 12 phase1 collector cells (2 methods × 3 seeds) + 24 phase2 learner cells.
- On completion, read `swap_matrix_results.md` for H1 vs H2 verdict.


## 2026-02-19T15:19:24.507Z
### Objective
- Prepare a clean handoff for continuing the Maze2D experiment workflow (preflight/smoke-test sequence, callback launch, evaluation, and documentation closure).

### Changes
- Updated `HANDOFF_LOG.md`.
- Updated `docs/WORKING_MEMORY.md`.
- Deleted `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.
- Net repo delta: 3 files changed, 114 insertions, 1463 deletions.
- Task snapshot at handoff: pending=0, running=0, done=3, failed=0, blocked=0, canceled=0.

### Evidence
- Context: `workdir=/root/ebm-online-rl-prototype`, `repo_root=/root/ebm-online-rl-prototype`, branch `master`.
- Command: `git status --porcelain=v1`.
- Output paths: `M HANDOFF_LOG.md`, `D HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`, `M docs/WORKING_MEMORY.md`.
- Command: `git diff --stat`.
- Output summary: `HANDOFF_LOG.md | 77 ++`, `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt | 1458 ------------------------------------`, `docs/WORKING_MEMORY.md | 42 +-`.

### Next steps
- Decide which driver script is “this experiment”: `scripts/exp_replan_horizon_sweep.py` or `scripts/exp_swap_matrix_maze2d.py`.
- Choose and document the success gate metric (return, success rate, diversity, or another metric).
- Confirm whether “rally auto callback” means `relay-long-task-callback`.
- Continue from the listed plan tail: preflight compile/`--help`, smoke tests for touched scripts, fix/rerun loop, short integrated mini-run, callback validation, then full launch/eval/diversity analysis.
- Append concrete run/eval outcomes back into `docs/WORKING_MEMORY.md` and `HANDOFF_LOG.md`.

## 2026-02-20T12:01:41+08:00
### Scope
- Validated overnight callback behavior for swap-matrix run from user-supplied relay transcript.

### Evidence
- Relay events (default instance) show expected callback chain:
  - `job.then_task.queued` for `j-20260219-230952-9ab7` -> `t-0001`
  - `task.started` then `task.finished status=blocked`
- Session file state matches transcript:
  - `j-20260219-230952-9ab7`: `failed`, `exitCode=143`
  - `t-0001`: `blocked`
- Artifact check still shows no `swap_matrix_results.csv/.md` under `full_20260219-230952` at validation time.

### Conclusion
- RallyAutoCallback itself worked.
- Blocked status is expected from missing final result artifacts at callback execution time.

### Paths
- `/root/.codex-discord-relay/relay.log`
- `/root/.codex-discord-relay/sessions.json`
- `/root/.codex-discord-relay/jobs/discord:1472061022239195304:thread:1473203408256368795/j-20260219-230952-9ab7/job.log`
- `/root/ebm-online-rl-prototype/runs/analysis/swap_matrix/`

## 2026-02-20T14:25:54+08:00
### Scope
- Completed post-run swap-matrix analysis task (`t-0001`) after artifacts became available.

### Changed
- Updated run pointer:
  - `runs/analysis/swap_matrix/LAST_FULL_RUN_PATH.txt` -> `runs/analysis/swap_matrix/swap_matrix_20260219-231605`
- Refreshed living summary:
  - `docs/WORKING_MEMORY.md`

### Evidence
- Result files found:
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.md`
- Run health:
  - 30 rows total (`phase`: 6 collection + 24 learning), all `rc=0`
  - No `status=timeout` rows
- Per-cell mean/std success by horizon (n=3 each):
  - `frozen | diffuser -> diffuser`: h64 `0.0000±0.0000`, h128 `0.3056±0.0962`, h192 `0.8056±0.1273`, h256 `0.8611±0.0481`
  - `frozen | diffuser -> sac_her_sparse`: h64 `0.1389±0.0481`, h128 `0.5556±0.2927`, h192 `0.7222±0.2679`, h256 `0.8056±0.1925`
  - `frozen | sac_her_sparse -> diffuser`: h64 `0.0000±0.0000`, h128 `0.4722±0.1273`, h192 `0.8611±0.0962`, h256 `0.9444±0.0481`
  - `frozen | sac_her_sparse -> sac_her_sparse`: h64 `0.0556±0.0481`, h128 `0.7778±0.0481`, h192 `0.8889±0.0481`, h256 `0.9167±0.0000`
  - `warmstart | diffuser -> diffuser`: h64 `0.0000±0.0000`, h128 `0.3889±0.0481`, h192 `0.7222±0.0962`, h256 `0.7500±0.0833`
  - `warmstart | diffuser -> sac_her_sparse`: h64 `0.0000±0.0000`, h128 `0.6667±0.0833`, h192 `0.8333±0.0833`, h256 `0.8611±0.0481`
  - `warmstart | sac_her_sparse -> diffuser`: h64 `0.0278±0.0481`, h128 `0.6389±0.0962`, h192 `0.9722±0.0481`, h256 `0.9722±0.0481`
  - `warmstart | sac_her_sparse -> sac_her_sparse`: h64 `0.0556±0.0962`, h128 `0.7500±0.0833`, h192 `0.7778±0.0962`, h256 `0.8056±0.0481`
- Best/Worst by h256 mean (from `swap_matrix_results.md`):
  - Best: `warmstart`, `sac_her_sparse -> diffuser`, `0.9722`
  - Worst: `warmstart`, `diffuser -> diffuser`, `0.7500`
- Main-effect readout at h256 (learning rows):
  - By collector: diffuser replay `0.8194`, SAC replay `0.9097`
  - By learner: diffuser learner `0.8819`, SAC learner `0.8472`

### Conclusion
- H1 (diffuser collector-driven advantage) is not supported by this 3-seed matrix.
- H2 (diffuser learner/planner advantage under shared replay) is partially supported.
- No infrastructure failures were observed in this completed matrix (no rc!=0, no timeouts).

### Next step
- Promote this matrix to 5 seeds for confidence bounds and stability.
- Then run replan-horizon sweep + waypoint/diversity diagnostics for mechanistic follow-up.

## 2026-02-20T15:24:59+08:00
### Scope
- Added an isolated collector-stochasticity diagnostic script to compare diffuser denoising stochasticity against SAC policy stochasticity on matched Maze2D start-goal queries.

### Changed
- Added:
  - `scripts/analyze_collector_stochasticity.py`
- Updated:
  - `docs/WORKING_MEMORY.md`

### Evidence
- Script checks:
  - `python3 -m py_compile scripts/analyze_collector_stochasticity.py`
  - `third_party/diffuser/.venv38/bin/python3.8 scripts/analyze_collector_stochasticity.py --help`
- Smoke run (completed):
  - command uses env bootstrap:
    - `D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH third_party/diffuser/.venv38/bin/python3.8 scripts/analyze_collector_stochasticity.py ...`
  - output dir:
    - `output/smoke/collector_stochasticity_smoke/`
  - produced:
    - `collector_stochasticity.csv`
    - `collector_stochasticity_summary.json`
    - `collector_stochasticity_report.md`

### Notes
- This host requires loader-path bootstrap before process start for `mujoco_py` dynamic libs; script-level env mutation alone is insufficient in this runtime.

### Next step
- Run the new script on the top 2-4 swap-matrix cells (best/worst h256) with larger `--num-queries`, `--samples-per-query`, and `--rollouts-per-query` to quantify whether SAC’s collector advantage is primarily higher exploration breadth or higher action-level controllability.

## 2026-02-20T16:20:52+08:00
### Scope
- Ran collector-stochasticity diagnostics to answer whether SAC's collector advantage is mainly from stronger exploration noise and to contextualize warmstart vs frozen behavior.

### Actions
- Attempted a high-budget phase1 run:
  - `--num-queries 20 --samples-per-query 64 --rollouts-per-query 16 --rollout-horizon 256`
  - Aborted due in-turn runtime constraints.
- Executed four pilot runs (seed 0) with matched reduced-cost settings:
  - `--num-queries 10 --samples-per-query 40 --rollouts-per-query 8 --rollout-horizon 192 --diffuser-replan-every 32 --sac-decision-every 16`

### Evidence
- Run artifacts:
  - `runs/analysis/collector_stochasticity/phase1_seed0_pilot_q10_s40_r8_h192/`
  - `runs/analysis/collector_stochasticity/warmstart_diffreplay_seed0_pilot_q10_s40_r8_h192/`
  - `runs/analysis/collector_stochasticity/frozen_diffreplay_seed0_pilot_q10_s40_r8_h192/`
  - `runs/analysis/collector_stochasticity/warmstart_sacreplay_seed0_pilot_q10_s40_r8_h192/`
- Aggregate readout (`collector_stochasticity_summary.json`):
  - phase1 baseline:
    - action pairwise L2: diffuser `0.8131` vs sac `0.2731`
    - endpoint pairwise L2: diffuser `0.4585` vs sac `0.4840`
    - rollout success: diffuser `0.6375` vs sac `0.8375`
  - warmstart + diffuser replay:
    - action pairwise L2: diffuser `0.7469` vs sac `0.2838`
    - endpoint pairwise L2: diffuser `0.1201` vs sac `0.3582`
    - rollout success: diffuser `0.9000` vs sac `0.9625`
  - frozen + diffuser replay:
    - action pairwise L2: diffuser `0.7783` vs sac `0.1568`
    - endpoint pairwise L2: diffuser `0.1329` vs sac `0.6171`
    - rollout success: diffuser `0.9000` vs sac `0.8250`
  - warmstart + SAC replay (best-cell context):
    - action pairwise L2: diffuser `0.5323` vs sac `0.2979`
    - endpoint pairwise L2: diffuser `0.2780` vs sac `0.3499`
    - rollout success: diffuser `0.9250` vs sac `0.9375`

### Conclusion
- In all pilot contexts, diffuser has higher per-action stochasticity than SAC.
- SAC can still show higher endpoint diversity and often higher success, so any SAC collector edge is not explained by larger raw action-noise magnitude alone.
- Warmstart vs frozen behavior is context-dependent; a warmstart-underperforming cell does not imply monotonic degradation from continued training.

### Next step
- Promote this diagnostic to multi-seed (0/1/2 at minimum) and larger query/rollout budget, then cross-reference with swap-matrix h256 rankings for confidence intervals.

## 2026-02-20T18:33:30+08:00
### Scope
- Answered user follow-up on whether SAC-vs-diffuser exploration interpretation is supported by goal-coverage evidence.

### Actions
- Extracted goal coverage metrics from completed swap-matrix artifacts at h256.
- Computed both:
  - phase1 collector-only comparison (from per-run `summary.json -> progress_last`)
  - phase2 learning-row main effects by collector (from `swap_matrix_results.csv`)

### Evidence
- Paths:
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0/summary.json`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_1/summary.json`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_2/summary.json`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_0/summary.json`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_1/summary.json`
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_2/summary.json`
- Phase1 collector-only @h256 (n=3/collector):
  - diffuser: success/query-cov `0.7500±0.2205`, cell-cov `0.8413±0.1672`
  - sac_her_sparse: success/query-cov `0.8056±0.1925`, cell-cov `0.8783±0.0183`
- Phase2 learning rows @h256 (n=12/collector):
  - diffuser replay: success/query-cov `0.8194±0.1056`, cell-cov `0.8308±0.1311`
  - sac_her_sparse replay: success/query-cov `0.9097±0.0750`, cell-cov `0.9014±0.0700`

### Interpretation note
- In this protocol `eval_samples_per_query=1`; therefore query-coverage rate at horizon h equals success-rate at horizon h numerically.
- Goal-cell coverage provides the more independent coverage signal and still trends higher for SAC in the aggregate here.

### Next step
- If needed, rerun coverage diagnostics with `eval_samples_per_query>1` (and multi-seed) so query-coverage becomes independent from single-trajectory success.

## 2026-02-20T18:43:20+08:00
### Scope
- Performed root-cause analysis for user question: why diffuser appears inferior to SAC in exploration.

### Actions
- Recomputed phase1 matched-query comparisons from collector `query_metrics.csv` across seeds 0/1/2.
- Extracted geometry/control proxies (`path_over_direct`, `start_jump_ratio`, `end_jump_ratio`) and compared method means.
- Cross-checked completed swap-matrix h256 coverage main effects and pilot stochasticity summaries.

### Evidence
- `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_{0,1,2}/query_metrics.csv`
- `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_{0,1,2}/query_metrics.csv`
- `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/*/seed_*/summary.json`
- `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
- `runs/analysis/collector_stochasticity/*pilot_q10_s40_r8_h192/collector_stochasticity_summary.json`

### Key results
- Phase1 matched-query aggregate (n=36/method):
  - Diffuser success `0.7222`, min-goal-dist `0.4718`.
  - SAC success `0.8333`, min-goal-dist `0.3013`.
  - SAC-Diff deltas: success `+0.1111`, min-goal-dist `-0.1704`.
- Control/geometry proxies:
  - `path_over_direct`: Diffuser `1.1095` vs SAC `2.3092`.
  - `start_jump_ratio`: Diffuser `3.4616` vs SAC `0.2314`.
  - `end_jump_ratio`: Diffuser `3.5042` vs SAC `1.0153`.
- Matrix h256 collector main effect still favors SAC replay in both success/query-coverage and cell-coverage.
- Pilot stochasticity continues to show Diffuser action noise > SAC action noise in all tested contexts.

### Conclusion
- Evidence supports a control-aware exploration advantage for SAC (better conversion of exploration into goal progress/coverage), not a simple "more random action noise" explanation.

### Next step
- Promote stochasticity and coverage diagnostics to multi-seed + larger budget and run replanning cadence/horizon ablations to isolate control-frequency effects.

## 2026-02-20T18:46:20+08:00
### Scope
- Created persistent paper-reference note for SAC-vs-diffuser exploration finding.

### Changed
- Added:
  - `research_finding.txt`
- Updated:
  - `docs/WORKING_MEMORY.md`
  - `HANDOFF_LOG.md`

### Evidence
- Finding/metrics source paths are listed inside `research_finding.txt`.
- Commit context captured from:
  - `git log -1`
  - script-scoped `git log` for relevant experiment scripts

### Next step
- If requested, promote `research_finding.txt` into a manuscript-ready structured summary (claim/evidence/limitation blocks).

## 2026-02-20T18:57:09+08:00
### Scope
- Added paper-ready structured note for SAC-vs-diffuser exploration result.

### Changed
- Added:
  - `research_finding_paper_outline.md`
- Updated:
  - `docs/WORKING_MEMORY.md`
  - `HANDOFF_LOG.md`

### Evidence
- Structured note is grounded in metrics and artifact paths already captured in:
  - `research_finding.txt`
  - swap-matrix and collector-stochasticity artifacts referenced in the new outline.

### Next step
- If requested, produce a reproducible plotting script/notebook to generate the proposed figure panels.

## 2026-02-20T20:13:48+08:00
### Scope
- Completed multi-seed consolidation of collector-stochasticity diagnostics for the SAC-vs-Diffuser exploration hypothesis.

### Actions
- Finished all missing seed-2 cells in the consolidation suite (after correcting phase2 path naming):
  - `phase1_seed2`
  - `warmstart_diffreplay_seed2`
  - `frozen_diffreplay_seed2`
  - `warmstart_sacreplay_seed2`
- Aggregated all 12 summary files into reproducible consolidated artifacts.

### Evidence
- Consolidation root:
  - `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/`
- Per-cell summaries (all present):
  - `*/collector_stochasticity_summary.json` for `4 contexts x 3 seeds`
- New aggregate artifacts:
  - `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_seed_metrics.csv`
  - `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_context_summary.csv`
  - `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_overall_summary.json`
  - `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_summary.md`

### Key readout
- Cell count: `12`
- Diffuser action stochasticity > SAC:
  - pairwise L2 in `11/12` cells
  - action std in `11/12` cells
- SAC endpoint diversity > Diffuser in `11/12` cells.
- SAC rollout success > Diffuser in `7/12` cells.
- Overall means (cell-average):
  - action pairwise L2: Diffuser `0.6929` vs SAC `0.3111`
  - endpoint pairwise L2: Diffuser `0.2921` vs SAC `0.4614`
  - rollout success: Diffuser `0.8177` vs SAC `0.8229`
- Exception:
  - `warmstart_sacreplay_seed2` is the only cell where SAC action stochasticity exceeded Diffuser.

### Conclusion
- Consolidated multi-seed evidence still supports a control-aware SAC exploration advantage rather than a raw-action-noise explanation.

### Next step
- Run replanning-cadence/horizon and `eval_samples_per_query>1` follow-ups, then design/control modifications to Diffuser targeting controllability and endpoint diversity.

## 2026-02-20T21:03:36+08:00
### Scope
- Added direct visual falsification workflow for SAC-vs-Diffuser control/trajectory-effectiveness conjecture and executed a phase1 visual check.

### Changed
- Updated:
  - `scripts/analyze_collector_stochasticity.py`
    - added optional trajectory plotting mode (`--save-trajectory-plots`, `--plot-rollouts-per-query`, `--plot-max-queries`)
    - per-query SAC-vs-Diffuser trajectory figure output
    - summary overlay grid output
    - per-query trajectory `.npz` export for reproducibility
  - `research_finding.txt` (mechanism clarification + direct visual test section)
  - `docs/WORKING_MEMORY.md`

### Evidence
- Visual-check run:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/`
- Key outputs:
  - `collector_stochasticity_summary.json`
  - `collector_stochasticity_report.md`
  - `trajectory_plots/trajectory_overlay_grid.png`
  - `trajectory_plots/query_00_sac_vs_diffuser.png` ... `query_05_sac_vs_diffuser.png`
  - `trajectory_plots/query_0*_trajectories.npz`

### Key readout
- (`q=6`, `samples=20`, `rollouts=6`, `h=192`, phase1 seed0)
  - action pairwise L2: Diffuser `0.7517` vs SAC `0.3269`
  - endpoint pairwise L2: Diffuser `0.3322` vs SAC `0.3629`
  - rollout success: Diffuser `0.5556` vs SAC `0.6389`

### Conclusion
- The direct visual check is consistent with (but not alone conclusive for) the control-aware hypothesis that SAC converts stochasticity into more effective trajectory branching than Diffuser in this setup.

### Next step
- Repeat the same visual protocol on additional seeds/cells and pair with replanning-cadence ablations before deciding concrete Diffuser control modifications.

## 2026-02-20T22:00:00+08:00
<!-- meta: {"type":"infra","commit":"b9cdd15","dirty":true} -->

### Scope
Migration: converted docs/WORKING_MEMORY.md from append-style to v2.2 living-snapshot format; applied v2.2 skill update.

### Repo state
- Path: /root/ebm-online-rl-prototype
- Branch: master
- Commit: b9cdd15 (dirty: yes)

### Hypothesis tested
- N/A — infra/documentation

### Actions
- Backed up pre-migration files to `memory/archive/`:
  - `memory/archive/WORKING_MEMORY_pre_migration_20260220.md`
  - `memory/archive/HANDOFF_LOG_pre_migration_20260220.md`
- Removed appended timestamped log blocks from docs/WORKING_MEMORY.md (old lines 207–340).
  - All removed blocks were already present in HANDOFF_LOG — no unique content lost.
- Restructured docs/WORKING_MEMORY.md into v2.2 schema:
  - header (Repo/Branch/Commit/dirty), Objective, Active Hypotheses (H1–H3), Required Environment, Current Best Result, Next Experiment (runnable), Open Questions, Key Artifact Pointers.
- Updated skill file to v2.2:
  - `/root/.agents/skills/experiment-working-memory-handoff/SKILL.md`

### Output artifacts
- `docs/WORKING_MEMORY.md` (rewritten, 97 lines)
- `memory/archive/WORKING_MEMORY_pre_migration_20260220.md`
- `memory/archive/HANDOFF_LOG_pre_migration_20260220.md`
- `/root/.agents/skills/experiment-working-memory-handoff/SKILL.md` (v2.2)

### Next step (runnable)
```bash
# 5-seed swap matrix to strengthen H2 confidence intervals
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_swap_matrix_maze2d.py \
  --seeds 0 1 2 3 4 \
  --collectors diffuser sac_her_sparse \
  --learners diffuser sac_her_sparse \
  --modes warmstart frozen \
  --device cuda:0 \
  --base-dir runs/analysis/swap_matrix/5seed_$(date +%Y%m%d-%H%M%S)
```

## 2026-02-20T21:26:34+08:00
### Scope
- Computed direct steps-to-near-goal (`<=0.1`) metric from saved SAC-vs-Diffuser trajectory visual-check rollouts.

### Changed
- Updated:
  - `research_finding.txt`
  - `docs/WORKING_MEMORY.md`
- Added metrics artifacts:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/steps_to_goal_threshold_0p1.json`
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/steps_to_goal_threshold_0p1_summary.json`

### Evidence
- Source trajectory files:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/trajectory_plots/query_0*_trajectories.npz`
- Metric definition:
  - first timestep index where `||xy_t - goal|| <= 0.1`; `None` if never reached.

### Key readout
- Queries where both methods reached threshold: `4/6`.
- Best-hit step delta (Diffuser - SAC): mean `+61.25`, median `+59.0`.
- Mean-hit step delta (Diffuser - SAC): mean `+35.55`, median `+31.33`.
- Faster method among comparable queries (`4/4`): SAC.
- Per-query min-hit steps (Diffuser vs SAC):
  - q00 `131` vs `84`
  - q01 `None` vs `None`
  - q02 `None` vs `None`
  - q03 `159` vs `76`
  - q04 `126` vs `82`
  - q05 `152` vs `81`

### Conclusion
- On this seed-0 visual subset, SAC reaches very-near-goal faster where both methods succeed, despite some Diffuser trajectories appearing geometrically shorter by eye.

### Next step
- Run the same steps-to-threshold metric on more seeds/cells to test whether this result is stable beyond the small visual subset.

## 2026-02-20T21:40:10+08:00
### Scope
- Investigated apparent contradiction between query-0 visual geometry and steps-to-threshold numerics.

### Actions
- Recomputed query-0 first-hit (`<=0.1`) directly from saved rollout arrays (`query_00_trajectories.npz`).
- Computed additional diagnostics for query-0:
  - path length to first-hit
  - mean per-step displacement to first-hit
  - post-hit trajectory length (continued rollout after first-hit)
- Generated focused diagnostic figure showing full vs first-hit-truncated trajectories.

### Evidence
- Source data:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/trajectory_plots/query_00_trajectories.npz`
- Existing metrics artifacts:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/steps_to_goal_threshold_0p1.json`
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/steps_to_goal_threshold_0p1_summary.json`
- New visual diagnostic:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/trajectory_plots/query_00_first_hit_diagnostics.png`

### Key readout (query 0)
- First-hit steps (`<=0.1`):
  - Diffuser: `[131, 140, 134, 161, 150, 155]`
  - SAC: `[157, 121, 92, 84, None, None]`
- Path length to first-hit (successful rollouts):
  - Diffuser mean: `2.21`
  - SAC mean: `3.13`
- Mean step displacement to first-hit:
  - Diffuser: `0.0153`
  - SAC: `0.0285`
- Post-hit path length (full rollout continues after hit):
  - Diffuser mean: `0.248`
  - SAC mean: `1.472`

### Conclusion
- No implementation bug found in first-hit computation.
- Visual-vs-numeric mismatch is explained by metric semantics:
  - first-hit step captures temporal speed;
  - plot geometry emphasizes trajectory shape and includes post-hit wandering (SAC wanders more), which can make SAC look worse while still hitting threshold earlier in some rollouts.

### Next step
- If desired, extend this decomposition (path-to-hit vs time-to-hit vs post-hit wander) to all queries/seeds and report paired statistics to separate geometry quality from temporal control speed.

## 2026-02-20T22:35:00+08:00
### Scope
- Investigated whether Diffuser's smaller per-step displacement is caused by clipping artifacts or by method-level behavior.

### Actions
- Audited clipping and action-magnitude paths in code:
  - `scripts/analyze_collector_stochasticity.py`
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - `scripts/synthetic_maze2d_sac_her_probe.py`
  - `third_party/diffuser-maze2d/diffuser/models/diffusion.py`
- Reproduced query-0 with instrumentation and exported action-magnitude diagnostics.
- Added a SAC decision-cadence sensitivity check for query-0 (`decision_every=1` vs `16`).
- Added replay-level action-norm context stats (Diffuser vs SAC replay exports + uniform baseline).

### Changed
- Updated:
  - `docs/WORKING_MEMORY.md`
- Added artifacts:
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/query_00_action_magnitude_audit.json`
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/query_00_sac_cadence_sensitivity.json`
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/collector_replay_action_norm_stats_seed0.json`

### Evidence
- Query-0 action audit (`query_00_action_magnitude_audit.json`):
  - t0 sample clip rate: Diffuser `0.000`, SAC `0.000`
  - rollout clip fraction: Diffuser `0.0165`, SAC `0.0000`
  - rollout raw action L2 mean: Diffuser `0.7726`, SAC `1.2191`
  - rollout step-displacement mean: Diffuser `0.0126`, SAC `0.0232`
- Replay action norms (`collector_replay_action_norm_stats_seed0.json`):
  - Diffuser replay L2 mean `0.7512`
  - SAC replay L2 mean `0.8451`
  - Uniform `[-1,1]^2` baseline L2 mean `0.7653`
- SAC cadence sensitivity (`query_00_sac_cadence_sensitivity.json`):
  - `decision_every=1`: hit-step mean `100.5` over `6/6` successes, displacement mean `0.0118`
  - `decision_every=16`: hit-step mean `111.4` over `5/6` successes, displacement mean `0.0240`

### Conclusion
- Diffuser conservativeness is not primarily a manual clipping artifact in this setup.
- The gap is better explained by method/control behavior: SAC emits larger bounded actions in this query and cadence choices materially affect observed displacement/success.
- This supports a mechanism view centered on control strategy and planner objective, not clipping bugs.

### Next step
- Extend this same action-magnitude + clip-rate audit to all 6 visual-check queries and 3 seeds, then pair with replan/decision cadence sweep for a stable mechanism claim.

## 2026-02-20T22:36:00+08:00
### Scope
- Persisted conservative-action root-cause audit into paper-note artifact.

### Changed
- Updated:
  - `research_finding.txt`

### Evidence
- Added section "6) Conservative-action root cause check (query 0)" with:
  - clip-rate audit results
  - SAC cadence sensitivity results
  - replay action-norm context stats
  - code-path references for clipping/tanh/denoising clamp

### Conclusion
- Paper note now explicitly records that observed Diffuser conservativeness is not primarily from manual clipping in the current implementation.

## 2026-02-20T22:08:57+08:00
### Scope
- Clarified Diffuser plan-selection semantics after user concern about potential post-hoc multi-rollout selection unfairness.

### Actions
- Read planner/eval/analyzer code paths to separate:
  - imagined-plan sampling/selection
  - executed environment rollout behavior
- Cross-checked phase1 swap-matrix launcher defaults and run config.

### Evidence
- Diffuser plan selection key and candidate count gate:
  - `scripts/synthetic_maze2d_diffuser_probe.py:1318`
  - `scripts/synthetic_maze2d_diffuser_probe.py:1340`
- Diffuser rollout loop executes one env trajectory using selected planned action:
  - `scripts/synthetic_maze2d_diffuser_probe.py:1730`
  - `scripts/synthetic_maze2d_diffuser_probe.py:1783`
- Analyzer path forces `maze_arr=None` (disables wall-aware multi-candidate selection):
  - `scripts/analyze_collector_stochasticity.py:280`
- Phase1 launcher defaults include single eval sample/query and standard replan cadence:
  - `scripts/exp_swap_matrix_maze2d.py:308`
  - `scripts/exp_swap_matrix_maze2d.py:313`

### Conclusion
- There is no post-hoc selection across multiple environment rollouts in the Diffuser rollout path.
- Multi-candidate selection, when enabled, happens over imagined model plans before acting (MPC-style), not over multiple env attempts.
- In the collector-stochasticity visual analyzer, this multi-candidate wall-aware selection is effectively off because `maze_arr=None`.

### Next step
- For strict fairness checks, run explicit ablation with `--no_wall_aware_planning` (or `--wall_aware_plan_samples 1`) and compare against SAC under matched decision cadence.

## 2026-02-20T22:10:39+08:00
### Scope
- Created a dedicated GPT-Pro handoff question document for "simple, effective Diffuser improvements" using validated project findings and fairness caveats.

### Changed
- Added:
  - `gpt_pro_diffuser_improvement_question_2026-02-20.txt`
- Updated:
  - `docs/WORKING_MEMORY.md`

### Evidence
- New document includes:
  - current swap-matrix and stochasticity findings
  - query-0 implementation checks and clipping audit summary
  - planner-selection fairness clarification
  - explicit ask for ranked interventions + ablation plan + fairness checklist
- Paths referenced inside prompt:
  - `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
  - `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_overall_summary.json`
  - `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/*`

### Conclusion
- Handoff artifact is ready to send to GPT Pro for idea generation without losing experimental context or repeating known pitfalls.

### Next step
- Send `gpt_pro_diffuser_improvement_question_2026-02-20.txt` to GPT Pro and request ranked low-effort interventions + week-1 ablation plan.

## 2026-02-20T22:15:28+08:00
### Scope
- Implemented non-privileged Diffuser plan-selection behavior per user fairness request.

### Actions
- Updated Diffuser planner selection logic to remove maze-wall-hit scoring from decision objective.
- Switched default planner behavior to single-candidate (no multi-candidate wall-aware selection).
- Updated swap-matrix launcher defaults to explicitly disable wall-aware candidate selection for future matrix runs.
- Ran syntax/CLI verification checks.

### Changed
- Updated:
  - `scripts/synthetic_maze2d_diffuser_probe.py`
    - plan selection now minimizes final-goal-error only (no wall-hit in ranking)
    - defaults: `wall_aware_planning=False`, `wall_aware_plan_samples=1`
    - CLI help text updated to reflect new semantics
  - `scripts/exp_swap_matrix_maze2d.py`
    - common launcher config now pins:
      - `wall_aware_planning=False`
      - `wall_aware_plan_samples=1`
  - `docs/WORKING_MEMORY.md`

### Evidence
- Previous selection key removed from decision path:
  - `scripts/synthetic_maze2d_diffuser_probe.py` (`sample_best_plan_from_obs`)
- Explicit non-privileged launcher defaults:
  - `scripts/exp_swap_matrix_maze2d.py` common config block
- Verification commands run:
```bash
python3 -m py_compile scripts/exp_swap_matrix_maze2d.py scripts/synthetic_maze2d_diffuser_probe.py
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/synthetic_maze2d_diffuser_probe.py --help
python3 scripts/exp_swap_matrix_maze2d.py --help
```

### Conclusion
- The main Diffuser probe/collector path no longer relies on wall-hit scoring for plan selection, and future swap-matrix runs default to inference-aligned, non-privileged single-candidate planning.

### Next step
- Re-run at least a smoke + one seed of swap-matrix/collector diagnostics under new defaults to quantify impact of removing privileged selection logic.
## 2026-02-20T22:47:26+08:00
### Scope
- Completed follow-up analysis task for old-vs-new Diffuser default smoke comparison and updated working memory with evidence.

### Actions
- Parsed job log for the relay-launched smoke run:
  - `runs/analysis/smoke_old_vs_new_defaults_latest.log`
- Read newest smoke artifacts:
  - `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/old_vs_new_smoke_summary.json`
  - `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/old_vs_new_smoke_rows.csv`
- Computed per-query and rollout-level deltas for success, goal distance, wall hits, and first-hit counts.
- Updated living snapshot:
  - `docs/WORKING_MEMORY.md`

### Evidence
- Log:
  - `runs/analysis/smoke_old_vs_new_defaults_latest.log`
- Summary artifact:
  - `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/old_vs_new_smoke_summary.json`
- Row-level artifact:
  - `runs/analysis/smoke_old_vs_new_defaults_20260220-224338/old_vs_new_smoke_rows.csv`

### Key readout (new defaults minus old defaults)
- Setup:
  - old: `wall_aware_planning=True`, `wall_aware_plan_samples=8`
  - new: `wall_aware_planning=False`, `wall_aware_plan_samples=1`
  - matched checkpoint/replay, `6 queries x 3 rollouts`, horizon `128`, success threshold `0.2`
- Deltas:
  - success rate: `-0.0556` (`0.2778` vs `0.3333`)
  - min goal distance mean: `+0.0568` (`0.7802` vs `0.7234`)
  - final goal distance mean: `+0.0567` (`0.7824` vs `0.7257`)
  - wall hits mean: `+18.7778` (`47.5` vs `28.72`)
  - first-hit (`<=0.1`) count: `1` vs `1`; no paired-hit rows for robust speed delta.

### Conclusion
- This smoke sample suggests the non-privileged default change can reduce collector effectiveness on average under this small budget, but effects are mixed per query and not yet statistically stable.

### Next step
- Promote this to at least one full seed (same metric suite as phase1) and then multi-seed confirmation before drawing a hard mechanism claim.


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

## 2026-02-21T16:00:00+08:00
### Scope
- Ablation grid `grid_20260221-134801` completed. Analyzed results and updated WORKING_MEMORY.

### Evidence
- Artifact: `runs/analysis/ablation_grid/grid_20260221-134801/ablation_grid_results.csv`
- 12 conditions: alpha∈{1.0,1.2,1.4} × beta∈{0.0,0.5} × adaptive∈{0,1}

### Key readout
- SAC baseline: `success=0.7222`
- Diffuser baseline (alpha=1.0, beta=0.0, adapt=False): `success=0.5278`
- Best Diffuser: alpha=1.0, beta=0.5, adapt=True → `success=0.6389` (+11pp, -8pp gap to SAC remains)
- Key finding: EMA smoothing (beta=0.5) alone does NOT help; adaptive replanning alone does NOT help; **both together produce a clear +11pp gain** (interaction effect)
- Alpha scaling: neutral or counterproductive; alpha=1.4 causes 53% clip fraction

### Conclusion
- EMA + adaptive replanning is the strongest execution-time intervention found so far.
- An 8pp gap to SAC remains. Next: promote 5-seed swap matrix, then consider baking best ablation params into a new full swap-matrix run.

### Next step
- 5-seed swap matrix (blocking open Q#1 — paper CI)
