# Working Memory (Bootstrapping Ablations)

This repo already has a detailed working memory file for Diffuser Maze2D investigations:
- `docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md`

This file exists because the current plan explicitly required `docs/WORKING_MEMORY.md`.
Entries here are append-only and focus on the bootstrapping ablation plan execution status.

## 2026-02-18 19:39 CST+0800
### User question / goal
- Execute plan `p-20260218-174710-8905` (bootstrapping hypothesis ablation matrix):
  - isolate collector vs learner weight usage (EMA/online + teacher collector)
  - isolate stochasticity (multi-seed)
  - isolate replay nonstationarity (fixed replay snapshot / replay load)
- Requirements: minimal diffs, run verification commands, update `docs/WORKING_MEMORY.md` + `HANDOFF_LOG.md` (append-only), write run artifacts under `output/`.

### Non-trivial hypothesis (record)
- "Diffuser's edge (vs goal-conditioned baselines) is primarily collector-driven via a planner-driven bootstrapping loop: early planning success yields higher-quality online replay, which then compounds."
  - Prior evidence + discussion is recorded in `docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md` (see 2026-02-18 17:14 and 17:49 blocks).

### Evidence inspected (verification steps)
- Help/CLI verification:
  - `python3 scripts/agentic_maze2d_autodecider.py --help` (works)
  - `python3 scripts/overnight_maze2d_autodecider.py --help` (works)
  - `bash scripts/overnight_online_maze2d_driver.sh --help` (now prints usage and exits)
  - `scripts/synthetic_maze2d_diffuser_probe.py --help` requires Mujoco env vars; confirmed usable with:
    - `LD_LIBRARY_PATH+=/root/.mujoco/mujoco210/bin`
    - `MUJOCO_GL=egl`
    - `D4RL_SUPPRESS_IMPORT_ERROR=1`
    - `PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d`

### Changes made (minimal, to enable ablations)
- Added explicit ablation knobs + replay snapshot IO in `scripts/synthetic_maze2d_diffuser_probe.py`:
  - `--collector_weights {ema,online}`
  - `--eval_weights {ema,online}`
  - `--collector_ckpt_path <checkpoint.pt>` + `--collector_ckpt_weights {ema,online}`
  - `--replay_load_npz <replay.npz>` + `--replay_save_npz <replay.npz>`
  - `--fixed_replay_snapshot_round N` + `--fixed_replay_snapshot_npz <snap.npz>`
- Hardened `scripts/overnight_online_maze2d_driver.sh` to support `--help/-h` without launching runs.

### Run artifacts produced (smoke baseline)
- Smoke run directory:
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_baseline/seed_0/`
- Key artifacts:
  - `.../stdout_stderr.log`
  - `.../summary.json`
  - `.../progress_metrics.csv`, `.../online_collection.csv`
  - `.../checkpoint_last.pt`, plots (`query_trajectories.png`, `train_val_loss.png`, etc.)
- Stop-criteria scan:
  - `rg -n -i "(nan|inf|overflow|diverg|assert|traceback)" output/bootstrapping/2026-02-18_bootstrap -S`
  - No NaN/traceback indicators found in the smoke run (only warnings).

### Conclusions (current status)
- Plan is **not complete yet**:
  - verification steps completed
  - minimal code knobs implemented
  - baseline smoke run completed and artifacts validated
- The actual ablation matrix runs (multi-seed + fixed replay + teacher collector) are still pending.

### Open items / next steps
- Launch the ablations under `output/bootstrapping/2026-02-18_bootstrap/maze2d/`:
  - baseline vs `--collector_weights online`
  - fixed replay snapshot (`--fixed_replay_snapshot_round 1`) and replay load runs
  - teacher-collector run using an existing checkpoint for `--collector_ckpt_path`
- Promote best condition(s) from 1-seed to 3 seeds, then 5 seeds as time allows.
- Summarize outcomes as a small table (condition x seeds x success metrics x divergence count) and append here + to `HANDOFF_LOG.md`.

## 2026-02-18 20:39 CST+0800
### Evidence inspected / runs executed
- Collector/learner weights smoke:
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_collect_online/seed_0/`
  - Condition: `--collector_weights online` (eval kept at default `ema`)
  - Artifacts include: `summary.json`, `progress_metrics.csv`, `online_collection.csv`, `checkpoint_last.pt`.
- Fixed replay smoke:
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze/seed_0/`
  - Condition: `--fixed_replay_snapshot_round 1` (freeze replay after round 1; rounds 2..N skip collection)
  - Artifacts include: `summary.json`, `online_collection.csv` (rounds 2/3 have `did_collect=0` and `replay_frozen=1`), `checkpoint_last.pt`.
- Replay load smoke (failed first attempt):
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_replay_load/seed_1/`
  - Error: `ValueError: GoalDataset produced zero samples...` (caused by mismatch between loaded replay episode length vs default CLI values).
- Replay load smoke (fixed config; partial artifacts):
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_replay_load_fixedcfg/seed_1/`
  - Ran with `--replay_load_npz <snapshot> --episode_len 64 --max_path_length 64 --horizon 32`.
  - Produced `metrics.csv`, `progress_metrics.csv`, plots, `checkpoint_last.pt`.
  - Missing expected end-of-run artifacts (`summary.json`, `query_metrics.csv`, `query_trajectories.png`) and the `stdout_stderr.log` contains only the shell command + pybullet line, suggesting the process likely terminated before finalization (needs re-run with unbuffered stdout and/or investigation of kill cause).

### Problems encountered (and workarounds)
- Replay snapshot path semantics footgun:
  - Passing a relative `--fixed_replay_snapshot_npz` that already includes the run dir prefix causes the code to prefix it again with `logdir/`, creating nested paths under the run dir.
  - Evidence: snapshot written at:
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze/seed_0/output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze/seed_0/replay_snapshot_round1.npz`
  - Workaround: pass either an absolute path or a basename like `--fixed_replay_snapshot_npz replay_snapshot_round1.npz` (so it lands under `--logdir`).
- Replay load requires config alignment:
  - When using `--replay_load_npz`, you must also pass consistent `--episode_len`, `--max_path_length`, and `--horizon` to avoid `GoalDataset produced zero samples`.
- CLI/help ergonomics:
  - `python3 scripts/synthetic_maze2d_diffuser_probe.py --help` can fail without Mujoco env vars and the correct venv; use the env block recorded in `docs/plans/2026-02-18-maze2d-bootstrapping-ablation-plan.md`.

### Open items
- Fix the snapshot-path semantics (small patch) to match typical CLI expectations and avoid nested paths.
- Make replay-load runs robust/reproducible:
  - store/load meta (episode_len, horizon, max_path_length) in the replay npz, or document required flags in the plan.
  - force unbuffered stdout for runs launched via pipes (`PYTHONUNBUFFERED=1`) so errors are always captured.
- Run 3-seed ablation matrix at the intended budget after smoke validation completes.

## 2026-02-18 21:31 CST+0800
### Goal (stability/hygiene pass)
- Reduce repo noise from generated artifacts, and make probe runs more reproducible/debuggable (logs always captured; clearer replay-load failures).

### Changes made
- `.gitignore` expanded to ignore `output/` and other artifact dirs.
- `scripts/synthetic_maze2d_diffuser_probe.py`:
  - force line-buffered stdout/stderr (helps `tee` capture; avoids empty logs when killed)
  - better replay-load validation: if loaded replay has `episode_len_max <= horizon`, fail fast with a clear error
  - snapshot-path fix: if user passes a relative snapshot path that already includes `--logdir` prefix, do not double-prefix.

### Evidence inspected (smoke validation)
- Replay-load run now completes cleanly with end-of-run artifacts:
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_replay_load_fixedcfg2/seed_1/summary.json`
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_replay_load_fixedcfg2/seed_1/stdout_stderr.log` (now contains progress + final summary)
- Snapshot path now lands where expected (no nested-prefix):
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze2/seed_0/replay_snapshot_round1.npz`

### Open items
- Decide repo-level objective invariants to preserve (env IDs, primary metric, success threshold, eval query protocol).
- Decide canonical Maze2D entrypoint (`scripts/overnight_maze2d_autodecider.py` vs `scripts/agentic_maze2d_autodecider.py`) before doing any refactor/wrapper work.

## 2026-02-18 22:34 CST+0800
### User direction (decision)
- Canonical controller should be the **agentic** one (`scripts/agentic_maze2d_autodecider.py`), not the older rule-based overnight selector.

### Changes made (stability + handoff readiness)
- Updated `.gitignore` to ignore `third_party/` in addition to existing artifact ignores, so repo status/push scope stays tractable.
- Updated `README.md` with Maze2D section:
  - canonical entrypoint = `scripts/agentic_maze2d_autodecider.py`
  - protocol/objective invariants (metrics/eval semantics to preserve under refactor)
  - canonical env var block for Maze2D runs
  - canonical launcher command
- Updated `requirements.txt` with explicit scope note (point-mass minimal deps vs Maze2D environment managed via `third_party/diffuser/.venv38`).
- Updated `scripts/agentic_maze2d_autodecider.py`:
  - new optional `--base-dir`
  - safety guard: refuse to run if `base_dir` already exists and is non-empty
  - config metadata now records `base_dir`, `root`, `python`, and `main_script`
- Updated `scripts/launch_agentic_maze2d_autodecider_tmux.sh`:
  - optional `BASE_DIR` passthrough (only forwarded when non-empty)
- Updated `docs/AGENTIC_AUTODECIDER_EXTERNAL_PROPOSAL_IMPLEMENTATION.md` to document `--base-dir` + launcher passthrough.

### Verification executed
- `python3 -m compileall -q ebm_online_rl scripts` (pass)
- `python3 -c "import ebm_online_rl; print('import ok')"` (pass)
- `python3 scripts/agentic_maze2d_autodecider.py --help` (pass; includes `--base-dir`)
- `python3 scripts/overnight_maze2d_autodecider.py --help` (pass)
- `bash -n scripts/launch_agentic_maze2d_autodecider_tmux.sh` (pass)
- `bash -n scripts/launch_overnight_maze2d_autodecider_tmux.sh` (pass)
- `bash -n scripts/overnight_online_maze2d_driver.sh` (pass)
- `bash -n scripts/overnight_five_step_monitor_driver.sh` (pass)
- `shellcheck` unavailable (`shellcheck: MISSING`)
- Maze2D probe help (with Mujoco + PYTHONPATH env): pass

### Additional smoke evidence
- Agentic smoke run completed (rc=0):
  - `output/smoke/agentic_smoke_20260218-222646/agentic.log`
  - `output/smoke/agentic_smoke_20260218-222646/summary.json`
- Run produced expected progress rows through online round:
  - `output/smoke/agentic_smoke_20260218-222646/20260218-222646_ts20_or1_ep1_t10_rp4_gp080/progress_metrics.csv`

### Push status / blocker
- Git remote for this repo is currently unset (`git remote -v` empty).
- Cannot push until remote URL is provided/configured.

### Brief experiment status snapshot
- Strongest full run in current logs remains:
  - `runs/analysis/synth_maze2d_diffuser_probe/20260215-205611`
  - success@256 trend in `progress_metrics.csv`: peaked at `0.75` (steps 12000-16000), ended `0.625` at step 18000.
- Bootstrapping ablation matrix execution is still incomplete:
  - smoke baselines/variants ran and tooling issues were fixed
  - multi-seed + teacher-collector matrix remains pending.

## 2026-02-19 14:31 CST+0800
### User question / goal
- Push the latest local code to remote.
- Provide a brief, handoff-ready summary of experiments run so far, results, and latest algorithm understanding for GPT Pro continuation.

### Evidence inspected
- Git state:
  - `git status --short --branch` -> `## master` (clean)
  - `git log --oneline -n 8` -> HEAD `8f8d425 Add Maze2D agentic workflow, probes, docs, and reproducibility fixes`
  - `.git/config` and `git remote -v` -> no remotes configured (push destination missing)
- Main longer diffuser run:
  - `runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/summary.json`
  - `progress_last.rollout_goal_success_rate_h256 = 0.625` at step 18000; peak around 0.75 earlier in run.
- Matched method comparison (maze2d-umaze-v1, seed=0, 12 diverse eval queries, h256):
  - `.../compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json` -> success@h256=0.8333
  - `.../gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0_rerun_20260217-203446/summary.json` -> success@h256=0.5833
  - `.../sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json` -> success@h256=0.8333
  - `.../sac_her_shaped_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json` -> success@h256=0.7500
- Bootstrapping smoke matrix:
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/*/seed_*/summary.json`
  - All smoke variants currently show rollout success near 0 at h64 in these tiny-budget checks; these runs validate pipeline mechanics rather than final performance ranking.

### Conclusions
- Push is currently blocked only by missing remote configuration; local branch is clean and committed at `8f8d425`.
- Current best concise method comparison (single seed) is:
  - Diffuser ~= SAC+HER sparse on success@h256 (both 0.8333)
  - Diffuser > GCBC+HER (0.8333 vs 0.5833)
  - SAC+HER shaped did not beat sparse in this run (0.7500 vs 0.8333)
- Latest understanding remains: there is signal that diffuser planning helps, but collector-vs-learner causality is still unproven; fixed-replay / collector-swap ablations are the key unresolved test.

### Open items
- Configure a git remote (name + URL) to complete push.
- Run multi-seed fixed-replay / collector-learner swap ablations to directly test the bootstrapping hypothesis.
