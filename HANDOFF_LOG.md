## 2026-02-18 19:39 CST+0800
- **Changed**:
  - Implemented bootstrapping-ablation knobs (collector weights, teacher collector, replay snapshot/load/save) in `scripts/synthetic_maze2d_diffuser_probe.py`.
  - Made `scripts/overnight_online_maze2d_driver.sh --help/-h` print usage and exit (avoid accidental run launch).
  - Created required append-only artifacts: `docs/WORKING_MEMORY.md`, `HANDOFF_LOG.md`.
- **Evidence**:
  - Smoke baseline run artifacts:
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_baseline/seed_0/summary.json`
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_baseline/seed_0/stdout_stderr.log`
  - Stop scan: `rg -n -i "(nan|inf|overflow|diverg|assert|traceback)" output/bootstrapping/2026-02-18_bootstrap -S`
- **Next step**:
  - Launch multi-seed ablation matrix runs (baseline, `--collector_weights online`, fixed replay snapshot/load, teacher collector) and summarize results into a small table.

## 2026-02-18 20:39 CST+0800
- **Changed**:
  - Ran additional smoke ablations under `output/bootstrapping/2026-02-18_bootstrap/maze2d/`:
    - `smoke_collect_online/seed_0` (`--collector_weights online`)
    - `smoke_fixed_replay_freeze/seed_0` (`--fixed_replay_snapshot_round 1`)
    - `smoke_replay_load/seed_1` (failed; replay-load mismatch)
    - `smoke_replay_load_fixedcfg/seed_1` (partial artifacts; missing `summary.json`/query artifacts)
- **Evidence**:
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_collect_online/seed_0/summary.json`
  - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze/seed_0/summary.json`
  - Replay-load failure message (in tool output): `ValueError: GoalDataset produced zero samples...`
  - Snapshot path duplication evidence:
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze/seed_0/output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze/seed_0/replay_snapshot_round1.npz`
- **Next step**:
  - Patch snapshot path resolution to avoid nested paths when a relative path contains directories.
  - Re-run replay-load experiment with unbuffered stdout and confirm `summary.json` is produced.
  - Then scale to 3 seeds per condition at the real budget.

## 2026-02-18 21:31 CST+0800
- **Changed**:
  - Repo hygiene: expanded `.gitignore` to ignore `output/` and other common artifact dirs.
  - Probe reproducibility: updated `scripts/synthetic_maze2d_diffuser_probe.py` to:
    - line-buffer stdout/stderr (so `tee` captures logs reliably)
    - fail fast on replay-load horizon/episode-length mismatch
    - avoid double-prefixing `--fixed_replay_snapshot_npz` when it already includes `--logdir`
- **Evidence**:
  - Replay-load now completes with `summary.json` and query artifacts:
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_replay_load_fixedcfg2/seed_1/summary.json`
  - Snapshot now written at the intended path (no nested prefix):
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/smoke_fixed_replay_freeze2/seed_0/replay_snapshot_round1.npz`
- **Next step**:
  - Decide canonical Maze2D entrypoint + objective invariants, then do any autodecider/README refactor without changing metrics/decision logic.

## 2026-02-18 22:34 CST+0800
- **Changed**:
  - Adopted user decision that Maze2D canonical controller is `scripts/agentic_maze2d_autodecider.py`.
  - Added handoff/stability updates:
    - `.gitignore`: ignore `third_party/` (plus prior artifact ignores)
    - `README.md`: Maze2D canonical run path + objective invariants
    - `requirements.txt`: scope notes (point-mass minimal deps vs Maze2D env)
    - `scripts/agentic_maze2d_autodecider.py`: optional `--base-dir` + non-empty-dir guard + richer config metadata
    - `scripts/launch_agentic_maze2d_autodecider_tmux.sh`: optional `BASE_DIR` passthrough
    - `docs/AGENTIC_AUTODECIDER_EXTERNAL_PROPOSAL_IMPLEMENTATION.md`: documented `--base-dir`
- **Evidence**:
  - Verification passes:
    - `python3 -m compileall -q ebm_online_rl scripts`
    - `python3 -c "import ebm_online_rl; print('import ok')"`
    - `python3 scripts/agentic_maze2d_autodecider.py --help`
    - `bash -n scripts/launch_agentic_maze2d_autodecider_tmux.sh`
    - `bash -n scripts/launch_overnight_maze2d_autodecider_tmux.sh`
    - `bash -n scripts/overnight_online_maze2d_driver.sh`
  - Additional smoke run completed:
    - `output/smoke/agentic_smoke_20260218-222646/agentic.log` (rc=0)
  - Latest strongest full run reference unchanged:
    - `runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/progress_metrics.csv` (success@256 peak 0.75; final 0.625)
- **Blocker**:
  - Repo has no configured remote (`git remote -v` empty), so push cannot proceed until remote URL is provided.
- **Next step**:
  - Configure `origin`, push current branch, then hand GPT Pro this repo + concise experiment summary from the working-memory block above.

## 2026-02-19 14:31 CST+0800
- **Changed**:
  - Collected a GPT-Pro handoff-ready brief from current experiment artifacts and git state.
  - Verified local HEAD and push readiness; identified push blocker.
- **Evidence**:
  - Git: `git status --short --branch` (clean `master`), HEAD `8f8d425`, `.git/config` has no remote.
  - Comparison summaries:
    - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
    - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
    - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_shaped_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
    - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0_rerun_20260217-203446/summary.json`
  - Bootstrapping smoke summaries:
    - `output/bootstrapping/2026-02-18_bootstrap/maze2d/*/seed_*/summary.json`
- **Next step**:
  - User provides remote URL/name; then run `git remote add <name> <url>` (if needed) and `git push <name> master`.
  - Continue with multi-seed fixed-replay/collector-swap ablations for causal attribution.
