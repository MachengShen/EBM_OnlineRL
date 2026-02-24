# EBM Online RL Working Memory (living snapshot)

Last updated: 2026-02-24T16:10:00+08:00
Repo: /root/ebm-online-rl-prototype
Branch: analysis/results-2026-02-24
Commit: 5ac5e48 (dirty: yes)
GitHub: https://github.com/MachengShen/EBM_OnlineRL/tree/master

## Objective
Investigate whether Diffuser can learn purely from online self-play (random init → collect → retrain → improve) without any pretrained checkpoints or expert data. Test this across maze2d (umaze/medium/large) and locomotion (hopper/walker2d) environments using a 2×2 swap matrix: {Diffuser, SAC} × {collector, learner}.

## GPT-Pro Bundle Snapshot (2026-02-24)
- Bundle report: `GPT_PRO_HANDOFF_20260224.md`
- Metrics snapshot (file-backed): `GPT_PRO_SUMMARY_METRICS_20260224.json`
- Implementation manifest: `GPT_PRO_IMPLEMENTATION_FILES_20260224.txt`
- Branch commit carrying implementation/report files: `5ac5e48`
- Goal-suffix online pilot status at snapshot time: `15/20` rows complete (`.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/goal_suffix_online_pilot_seed0_20260224-144846/results.csv`)

## Research Question
Can a diffusion-based planner (Diffuser) do "reinforcement learning" — starting from random initialization, collecting its own experience, and iteratively improving — to reach any goal from any initial state?

## Automation Hardening (v1.1 implemented in repo docs/tools)
- Purpose: make long-run callback behavior default-safe with a smoke-first gate and bounded iterative debugging.
- Contract doc (updated):
  - `docs/plans/2026-02-24-iterative-debug-supervisor-contract.md`
- New runnable helper:
  - `scripts/stage0_smoke_gate.py`
- Current policy direction:
  - enforce smoke-first launch (`1/5/10` epoch-equivalent) before full budget
  - keep bounded supervisor loop for full-stage failures
  - keep explicit blocked terminal statuses and policy guardrails

## Task t-0006 Run Audit (author continuation overlapfix)
- Target run:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/`
- Completion gate:
  - `summary.json` and `continuation_metrics.csv` are present; `run.log` ends with `Finished overnight driver`.
  - No nonzero-exit evidence found, so retry policy did not trigger and no `_retry1` relaunch was executed.
- Final H1-H4 (from `summary.json`):
  - H1 `task_too_hard_only`: `weakened`
  - H2 `planner_eval_mismatch`: `inconclusive`
  - H3 `undertraining`: `supported`
  - H4 `eqnet_should_outperform_unet`: `supported`
- EqNet vs UNet milestone comparison (`continuation_metrics.csv`):
  - EqNet exceeded UNet success at `7/9` checkpoints (`15000, 25000, 35000, 45000, 55000, 65000, 80000`), tied at `5000`, lower at `75000`.

## Goal Inpaint K=50 Check (2026-02-24, step65000, pair T0->T1)
- New artifact:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step65000_k50.json`
- Matched protocol (`sampling_steps=64`, `n_exec_steps=44`, `timeout=200`, `temperature=0.5`, `num_envs=16`, `num_episodes=8`, `n=128` per arch):
  - UNet @ `K=50`: `0.71875` (`92/128`)
  - EqNet @ `K=50`: `0.87500` (`112/128`)
- Comparison vs prior same-checkpoint ablation:
  - EqNet: `K=25 -> K=50`: `0.890625 -> 0.875000` (`-0.015625`)
  - UNet: `K=25 -> K=50`: `0.578125 -> 0.718750` (`+0.140625`)
  - EqNet remains far above `K=1` (`0.0000`), supporting narrow-anchor brittleness but not a monotonic larger-K gain.

## Goal Inpaint Dense Grid (2026-02-24, latest step80000 checkpoints)
- New artifacts:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step80000_dense_kgrid_n32.json`
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step80000_dense_kgrid_n32.csv`
- Protocol: same inference settings, quick dense scan `n=32` rollouts per model per `K`, `K={1,2,3,5,8,10,12,15,20,25,30,40,50,64}`.
- User-requested checkpoints (`K=1,2,5`) on latest training state:
  - `K=1`: UNet `0.9062` (`29/32`), EqNet `0.0000` (`0/32`)
  - `K=2`: UNet `0.9375` (`30/32`), EqNet `0.0000` (`0/32`)
  - `K=5`: UNet `0.9688` (`31/32`), EqNet `0.5938` (`19/32`)
- Trend summary:
  - EqNet is highly sensitive to `K` and collapses at very small anchors (`K<=3`) even with latest checkpoints.
  - EqNet recovery region appears around moderate anchors (`K≈8..20`), with best quick-scan point at `K=20` (`0.9062`).
  - UNet remains strong across the scanned range and peaks at `K=3/8/10/12` (`1.0000` in this quick scan).

## Active Hypotheses
- H1: Pure Diffuser online loop can learn goal-reaching from scratch (maze2d → locomotion).
  - Status: under test
  - Evidence: maze2d-umaze warmstart|diffuser→diffuser achieves 0.75 success@h256 (3-seed matrix). Medium/large mazes + locomotion now launched.

- H2: Performance gaps between Diffuser and SAC are attributable to collector quality vs learner quality.
  - Status: under test
  - Evidence: Prior umaze matrix showed SAC collector > Diffuser collector (0.91 vs 0.82 main effect). New swap conditions isolate this for locomotion.

- H3: EqNet denoiser improves Maze2D online self-improvement vs UNet under matched budget.
  - Status: weakened (current 3-seed ablation favors UNet)
  - Evidence: `eqnet_vs_unet_summary.json` reports success `0.2778 ± 0.1273` (EqNet) vs `0.7778 ± 0.2097` (UNet), with EqNet worse on min-goal-distance (`+0.4469`) and final-goal-distance (`+0.5237`), and slightly worse wall hits (`+5.36`), `n=3` per arch.

- H4: UNet implementation can fit offline expert Maze2D data without immediate overfitting.
  - Status: supported (diagnostic)
  - Evidence: expert-replay offline run (`600` steps, seed `0`) shows train/val dropping from `0.859/0.862` to `0.233/0.240`; final val-train gap `+0.0068`; best val step `550` gap `+0.0083`.
  - Caveat: single-seed and no online/self-improve loop in this diagnostic.

- H5: EqNet can fit expert replay, but under matched budget it converges to a worse loss regime and larger train-val gap than UNet.
  - Status: supported (diagnostic)
  - Evidence: corrected EqNet replay run (`600` steps, seed `0`) reached final train/val `0.3565/0.4533` with gap `+0.0968`, versus UNet `0.2333/0.2401` with gap `+0.0068` on the same replay and budget.
  - Caveat: this pass was train/val-focused; final fixed-query rollout evaluation was terminated on CPU due long runtime.

- H6: In the author repo quick zero-step sanity, EqNet should outperform UNet if protocol transfer is correct.
  - Status: not supported at this tiny budget (both failed equally)
  - Evidence: author-repo run `author_zero_step_20260223-180840` (`gridland-n5-gc`, `300` gradient steps each, no author-code modifications) yields `avg_completion_mean=0.0` for both UNet and EqNet; EqNet is slower (`train +718s`, `eval +286s` vs UNet).
  - Caveat: this is a fast sanity budget and not a full paper-scale run, so result is a diagnostic signal, not a final architecture ranking.

- H7: EqNet at 300 steps in author-repo quick sanity is undertrained, but still underperforms UNet at matched checkpoints.
  - Status: supported (diagnostic)
  - Evidence: continuation from the exact Step-1 checkpoints (`continuation_probe_20260223-195217`) shows EqNet val loss improves from `0.1266` at step300 to `0.1172` at step500 (`-7.43%`), while UNet stays near-flat (`0.0838 -> 0.0842`).
  - Caveat: this is still far below paper-scale budget (`500k` steps in `goal_stitching/paper_experiments.sh`), so it diagnoses early-training behavior, not final asymptotic ranking.

- H8: The author-repo failure at tiny budget is primarily undertraining; with GPU continuation to 5k steps, both models improve strongly and the EqNet-vs-UNet val gap narrows substantially.
  - Status: supported (diagnostic)
  - Evidence: GPU continuation run `continuation_probe_gpu_20260223-205021` (same step300 checkpoints, same dataset/config family) yields UNet val `0.0772 -> 0.0126` and EqNet val `0.1182 -> 0.0185`; EqNet-minus-UNet val gap shrinks `0.0410 -> 0.0059`.
  - Caveat: this is still far below paper-scale `500k` and uses checkpoint-resume probing (without original optimizer state), so it is a convergence-signal diagnostic, not final ranking.

## Bug Fixed (2026-02-22T17:xx)
- **Bug**: `wall_aware_planning` and `wall_aware_plan_samples` were in `common` config in `exp_swap_matrix_maze2d.py` but the SAC probe script doesn't have these args → SAC collector seeds 0 and 1 failed with "unrecognized arguments".
- **Fix**: Moved both keys from `common` to `diffuser_only` (commit pending).
- **Impact on current run**: SAC collector seeds 0 and 1 failed (no replay files); their Phase2 cells will also fail quickly. Diffuser collector cells (seeds 0,1,2) and Diffuser-replay-dependent Phase2 cells are unaffected.
- **Recovery**: After the full batch completes, restart maze2d-medium with the same `--base-dir`. `--resume` is the default, so it will skip completed cells, re-run SAC Phase1 (fixed), then re-run SAC-replay Phase2 cells.
- **maze2d large + locomotion**: Will use the fixed script from the start (no bug).

## Running / Active Jobs

### EqNet vs UNet 3-seed ablation — COMPLETED (started 2026-02-22T19:55:04+08:00, finished 2026-02-23T02:50+08:00)
- Relay job: `j-20260222-195504-8810`
- Worktree commit: `e99252d`
- Run root pointer: `.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/LAST_EQNET_3SEED_RUN.txt`
- Current run root (canonical): `.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/`
- Output artifacts now present:
  - `eqnet_vs_unet_summary.json`
  - `eqnet_vs_unet_rows.csv`
  - `eqnet_vs_unet_summary.md`
  - `eqnet_vs_unet_success_curve.png`
- Aggregate results (`n=3` each):
  - UNet: success `0.7778 ± 0.2097`, min-goal-dist `0.3838 ± 0.3258`, final-goal-dist `0.4397 ± 0.3242`, wall-hits `84.5556 ± 20.2581`
  - EqNet: success `0.2778 ± 0.1273`, min-goal-dist `0.8307 ± 0.0837`, final-goal-dist `0.9634 ± 0.0978`, wall-hits `89.9167 ± 27.7768`
  - EqNet minus UNet delta: success `-0.5000`, min-goal-dist `+0.4469`, final-goal-dist `+0.5237`, wall-hits `+5.3611`
  - Caveat: small sample (`n=3`) and high across-seed variance, so treat as directional evidence pending larger-seed confirmation.
- Per-seed success@h256 from `eqnet_vs_unet_rows.csv`:
  - UNet: seed0 `0.5833`, seed1 `0.7500`, seed2 `1.0000`
  - EqNet: seed0 `0.1667`, seed1 `0.4167`, seed2 `0.2500`
- Poster status: stopped by chained resume launcher after EqNet completion (PID `47454` no longer running).
- Reviewer handoff docs pushed on implementation branch:
  - branch: `feature/eqnet-maze2d` (remote: `origin/feature/eqnet-maze2d`)
  - commit: `2f91aed`
  - report: `.worktrees/eqnet-maze2d/docs/GPT_PRO_EQNET_UNET_REVIEW_20260223.md`
  - table CSV: `.worktrees/eqnet-maze2d/docs/GPT_PRO_EQNET_UNET_TABLE_20260223.csv`
- GPT-Pro bundle refresh (2026-02-23):
  - branch: `feature/eqnet-maze2d` (remote: `origin/feature/eqnet-maze2d`)
  - commit: `5e0d363` (`docs: add short EqNet diagnostic summary for GPT-Pro handoff`)
  - short interpretation doc: `.worktrees/eqnet-maze2d/docs/GPT_PRO_EQNET_DIAG_SHORT_SUMMARY_20260223.md`
  - handoff zip: `.worktrees/eqnet-maze2d/gpt_pro_eqnet_diag_bundle_20260223.zip`
  - bundle includes: scripts, main EqNet-vs-UNet results, expert replay EqNet/UNet diagnostics, and interpretation docs.
- Audit unblock (2026-02-23):
  - Probe script is present on the same remote branch and can be reviewed directly:
    - `scripts/synthetic_maze2d_diffuser_probe.py`
    - branch URL: `https://github.com/MachengShen/EBM_OnlineRL/blob/feature/eqnet-maze2d/scripts/synthetic_maze2d_diffuser_probe.py`
    - introducing commit on this branch: `e99252d` (`Add EqNet denoiser option and Maze2D ablation tooling`)
- Expert dataset UNet diagnostic (2026-02-23, CPU non-disruptive):
  - Replay exported from D4RL Maze2D dataset:
    - `.worktrees/eqnet-maze2d/runs/analysis/expert_dataset_unet_diag/maze2d_umaze_d4rl_replay_full.npz`
    - transitions=`1,000,000`, episodes=`12,459`
  - Offline UNet run (no online collection, val logging enabled):
    - `.worktrees/eqnet-maze2d/runs/analysis/expert_dataset_unet_diag/unet_offline_expert_noeval_seed0_20260223-140459/`
    - artifacts: `metrics.csv`, `train_val_loss.png`, `summary.json`, `overfit_summary.json`
    - key readout: final train/val=`0.2333/0.2401`, val-train gap=`+0.0068` (no immediate overfit signal), but query success@0.2 on fixed 3 queries=`0.0`
- Expert dataset EqNet diagnostic (2026-02-23, corrected from UNet misread; CPU non-disruptive):
  - Offline EqNet run (same replay, seed, and train budget as UNet fit check):
    - `.worktrees/eqnet-maze2d/runs/analysis/expert_dataset_unet_diag/eqnet_offline_expert_noeval_seed0_20260223-142032/`
    - artifacts: `metrics.csv`, `train_val_loss.png`, `checkpoint_last.pt`, `overfit_summary_eqnet_train_only.json`
    - key readout (step 600): train/val=`0.3565/0.4533`, val-train gap=`+0.0968`
    - paired comparison artifact: `.worktrees/eqnet-maze2d/runs/analysis/expert_dataset_unet_diag/eqnet_vs_unet_expert_diag_compare.csv`
    - note: final fixed-query rollout evaluation was terminated after train step 600 due CPU runtime; train/val conclusions remain valid for overfit diagnosis.

### Author repo Step-1 zero-step sanity — COMPLETED (started 2026-02-23T18:08+08:00, finished 2026-02-23T18:52+08:00)
- Author repo:
  - local path: `/tmp/diffusion-stitching` (commit `d27cf2a`)
  - env: `/root/miniconda3/envs/dstitch39`
- Data + run root:
  - dataset: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/datasets/gridland_n5_gc_author_notebookstyle.npy`
  - run root: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/author_zero_step_20260223-180840/`
  - pointer: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/LAST_STEP1_QUICK_RUN.txt`
- Output artifacts:
  - `unet/{train.log,train_wall_seconds.txt,eval.log,eval.json}`
  - `eqnet/{train.log,train_wall_seconds.txt,eval.log,eval.json}`
  - `step1_quick_table.csv`
  - `step1_quick_summary.json`
- Quick table summary (`train_steps=300`, `num_envs=1` eval):
  - UNet: `avg_completion_mean=0.0000`, `avg_completion_std=0.0000`, `train_wall_seconds=378`, `eval_wall_seconds=240.52`
  - EqNet: `avg_completion_mean=0.0000`, `avg_completion_std=0.0000`, `train_wall_seconds=1096`, `eval_wall_seconds=526.96`
  - EqNet minus UNet: completion `0.0000`, train wall `+718s`, eval wall `+286.44s`
- Implementation note:
  - Local helper `/tmp/eval_author_checkpoint.py` was patched to handle `eval_model()` tuple return in this author commit (`res[0]`), avoiding false `TypeError`; no author-repo source files were modified.

### Author repo continuation loss-signal check — COMPLETED (finished 2026-02-23T20:22+08:00)
- Objective:
  - Determine whether the step-300 failure signal is mostly implementation breakage or insufficient training.
- Method:
  - Loaded the exact Step-1 checkpoints (UNet/EqNet at step300) and continued each to step500 with trajectory-level train/val logging on the same dataset.
- Run root:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/continuation_probe_20260223-195217/`
- Key outputs:
  - `unet_continuation_300_to_500.csv`
  - `eqnet_continuation_300_to_500.csv`
  - `continuation_probe_summary.json`
- Key readout:
  - UNet val: `0.08379 @300 -> 0.08423 @500` (near-flat/noisy)
  - EqNet val: `0.12658 @300 -> 0.11718 @500` (continues improving)
  - EqNet gap vs UNet narrows but remains: `+0.04278 @300 -> +0.03295 @500`
- Budget context:
  - Author paper scripts use `gradient_steps=500000` (orders of magnitude above the 300-step quick sanity).

### Author repo GPU continuation probe — COMPLETED (finished 2026-02-23T21:01+08:00)
- Objective:
  - Continue both step300 checkpoints on GPU to test whether poor quick-sanity outcomes are mainly training-budget artifacts.
- Environment:
  - `dstitch39` now uses CUDA PyTorch (`torch 2.5.1+cu121`, `torch.cuda.is_available=True`, RTX 3090).
- Run root:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/continuation_probe_gpu_20260223-205021/`
- Key outputs:
  - `unet_continuation_300_to_5000.csv`
  - `eqnet_continuation_300_to_5000.csv`
  - `continuation_probe_gpu_summary.json`
  - pointer: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/LAST_CONTINUATION_GPU_RUN.txt`
- Key readout:
  - UNet val: `0.0772 @300 -> 0.0126 @5000` (`-83.7%`)
  - EqNet val: `0.1182 @300 -> 0.0185 @5000` (`-84.4%`)
  - EqNet minus UNet val gap: `+0.0410 @300 -> +0.0059 @5000`
  - Speed (300→5000 continuation):
    - UNet: `200.7s` total (`0.0427 s/step`)
    - EqNet: `469.1s` total (`0.0998 s/step`)

### Batch experiment runner — ACTIVE (resumed automatically after EqNet completion)
- **Chained launcher**: PID `13108`, script `scripts/resume_after_eqnet.sh`
- **Log**: `runs/analysis/resume_after_eqnet.log`
- **Trigger result**: detected EqNet job exit code `0` at `2026-02-23T02:51:45+08:00` and started Phase 1.
- **Sequence**: medium maze (resume) → large maze → locomotion (all 3 phases chained)
- **Current stage**: medium maze resumed (`scripts/exp_swap_matrix_maze2d.py`, PID `14590`)
- **Latest progress evidence**: `find .../maze2d_medium_20260222-145304 -name summary.json | wc -l` -> `26` while process `14590` is still alive.
- **Completed cells (7/24)** — safe, have summary.json, will be skipped on resume:
  - phase1/diffuser/seed_{0,1,2}
  - phase2/frozen/diffuser_to_diffuser/seed_{0,1}
  - phase2/warmstart/diffuser_to_diffuser/seed_{0,1}
- **Will re-run**: phase2/warmstart/diffuser_to_diffuser/seed_2 (was in-progress when paused)
- **Remaining (17 cells)**: SAC Phase1 (3 seeds, bug fixed), all SAC-dependent Phase2 cells, diff_to_diff seed_2

### Launcher guardrail (2026-02-22)
- Avoid `pgrep -f "<pattern>"` wait loops when the same pattern appears in the current launcher command; this caused a self-match infinite wait and blocked EqNet launch despite idle GPU.
- For any run expected to exceed ~30 minutes, launch the Discord poster immediately and verify one successful post before leaving the turn.
- For architecture ablations where strong baseline data already exists, prioritize novel architecture first (or interleave by seed) to maximize early information gain instead of front-loading baseline repeats.
- Clarification guardrail (local to this repo): if user instruction and attached plan target differ (e.g., UNet vs EqNet), ask one clarification question before launching new runs and record the answer in `HANDOFF_LOG.md`.
- GPU-first guardrail (2026-02-23): for training/long eval loops, use GPU by default when available; use CPU only for lightweight checks or explicit user request. If GPU exists but the active env is CPU-only, fix/switch to a CUDA-capable env before long runs.

### New code (commit 328ac6e)
- `scripts/exp_locomotion_collector_study.py`: Added 3 new conditions:
  - `diffuser_online`: Pure Diffuser→Diffuser from random init
  - `sac_collects_diffuser_learns`: SAC collects → Diffuser trains
  - `diffuser_collects_sac_learns`: Diffuser collects (w/ online improvement) → SAC trains
  - New functions: `build_diffuser_from_scratch()`, `episodes_to_sequence_dataset()`, `train_diffuser_steps()`, `evaluate_diffuser_loco()`, `collect_diffuser_episodes_online()`
- `scripts/exp_swap_matrix_maze2d.py`: Added `--env` flag for medium/large maze support

### Locomotion conditions explained
| Condition | Collector | Learner | Tests |
|---|---|---|---|
| `diffuser_online` | Diffuser (random init, improving) | Diffuser | Core question: can Diffuser self-improve? |
| `sac_scratch` | SAC | SAC | Baseline |
| `sac_collects_diffuser_learns` | SAC | Diffuser | Is Diffuser a good learner? |
| `diffuser_collects_sac_learns` | Diffuser (improving) | SAC | Is Diffuser a good collector? |

## Prior Maze2d Results (umaze, 3-seed)
- Best cell: warmstart|SAC→Diffuser = 0.9722 ± 0.0481 success@h256
- Worst cell: warmstart|Diffuser→Diffuser = 0.7500 ± 0.0833
- Collector main effect: SAC replay 0.91 vs Diffuser replay 0.82
- Learner main effect: Diffuser learner 0.88 vs SAC learner 0.85

## Prior Locomotion Results (pretrained Diffuser collector, 3-seed)
| Env | diffuser_warmstart_sac | sac_scratch | gcbc_diffuser |
|---|---|---|---|
| hopper | 0.141 ± 0.061 | 0.068 ± 0.006 | 0.252 ± 0.162 |
| walker2d | 0.062 ± 0.012 | 0.089 ± 0.021 | 0.069 ± 0.025 |
Note: These used frozen pretrained Diffuser. New experiments start from random init.

## Required Environment
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1
MUJOCO_GL=egl
LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
PYTHON=third_party/diffuser/.venv38/bin/python3
```

## Key Artifact Pointers
- Launch script: `scripts/launch_all_experiments.sh`
- Maze2d swap matrix: `scripts/exp_swap_matrix_maze2d.py`
- Locomotion study: `scripts/exp_locomotion_collector_study.py`
- Prior umaze results: `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
- Prior loco results: `runs/analysis/locomotion_collector/grid_20260221-200301/locomotion_collector_results.csv`
- EqNet plan attachment: `/root/.codex-discord-relay/instances/claude/uploads/discord_1472061022239195304_thread_1473203408256368795/attachments/1771752035647_f0083819_CODEX_TODO_EQNET_MAZE2D.txt`

## EqNet Integration Review (2026-02-22)
- Status: reviewed; not implemented yet.
- Plan quality: strong and feasible for this repo, with a few required interface and packaging corrections.
- Reviewed artifacts:
  - GPT-Pro plan attachment: `/root/.codex-discord-relay/instances/claude/uploads/discord_1472061022239195304_thread_1473203408256368795/attachments/1771752035647_f0083819_CODEX_TODO_EQNET_MAZE2D.txt`
  - Upstream repo: `https://github.com/rvl-lab-utoronto/diffusion-stitching` (local inspect: `/tmp/diffusion-stitching`, commit `d27cf2ab7bf760dc62742b34e7bacf4e83ea9562`)
  - EqNet file inspected: `/tmp/diffusion-stitching/eqnet.py`
- Required corrections before implementation:
  - Packaging/import caveat: upstream `eqnet.py` is a convenience copy with relative imports (`from . import BaseNNDiffusion`, `from ..utils import GroupNorm1d`), so it is not directly importable from repo root without package placement or adapter shim.
  - Signature caveat: our Maze2D diffusion stack calls denoisers as `model(x, cond, t)` (`third_party/diffuser-maze2d/diffuser/models/diffusion.py`), while EqNet expects `(x, noise, condition)` (`/tmp/diffusion-stitching/eqnet.py`); adapter must reorder/map arguments.
  - Model-selection caveat: `scripts/synthetic_maze2d_diffuser_probe.py` currently hardcodes `TemporalUnet`; add `--denoiser_arch {unet,eqnet}` and instantiate accordingly.
  - Horizon caveat: EqNet asserts horizon is power-of-two (`x.shape[1] & (x.shape[1] - 1) == 0`), so keep/validate compatible horizons in config.
  - Instrumentation caveat: current probe does not log planning runtime directly; add explicit timing metrics before compute-cost comparisons.
  - Safety caveat: active batch job remains live (`scripts/launch_all_experiments.sh`, PID 64887; child PID 64890), so EqNet work should be isolated in a branch/worktree to avoid perturbing in-flight experiments.
- Corrected execution sequence:
  - Isolate work (branch/worktree), then add denoiser switch + EqNet adapter wiring.
  - Run adapter sanity checks (forward/backward + finite grads + shift-smoke).
  - Run smoke (`unet` vs `eqnet`, seed 0), then 3-seed umaze ablation with identical budgets.
  - Only after stability: optional replanning-frequency sweep and medium/large promotion.

## Logging policy
- Append-only history: `HANDOFF_LOG.md`
- Living snapshot: `docs/WORKING_MEMORY.md` (this file — overwrite/compact)
- Record commit hash + subject in HANDOFF_LOG when committing
- Infra-only incident log: `/root/PIPELINE_FAILURE_LOG.md` is reserved for AutoML/relay automation failures; keep experiment sequencing/design reflections in `HANDOFF_LOG.md` / `docs/WORKING_MEMORY.md`.

### HANDOFF_LOG verbosity rules (mandatory)
- `### Exact command(s) run` entries MUST be ≤5 lines total. Use the form: `script_name.py --key_arg val [...]` → `run_dir/`. Never paste full inline Python scripts, full absolute paths to every arg, or raw shell wrapper output.
- Always use `<<'EOF'` (quoted heredoc) when appending markdown to the log from shell. Unquoted `<<EOF` causes backtick expansion and garbled entries.
- Do NOT record raw `ps` output, shell snapshot paths, or process-table lines in command sections.
- If a command block exceeds 5 lines, replace it with a reference: "see `<run_dir>/stdout_stderr.log`".


## 2026-02-22T12:03:29.162Z
### Objective
- Preserve cross-session continuity for the Maze2D validation workflow and callback-based experiment orchestration.
- Hand off exact repo state, completed progress, and active blockers so the next agent can resume without re-discovery.

### Changes
- Working tree on `master` has tracked edits in `.gitignore`, `HANDOFF_LOG.md`, `docs/WORKING_MEMORY.md`, `research_finding.txt`, and `scripts/exp_swap_matrix_maze2d.py`.
- Net tracked delta from `git diff --stat`: 5 files changed, 387 insertions, 216 deletions.
- New untracked artifacts/scripts present: `MUJOCO_LOG.TXT`, `gpt_pro_bundle_20260221.zip`, `gpt_pro_bundle_20260221b.zip`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`, `scripts/discord_score_poster.py`, `scripts/discord_swap_matrix_monitor.py`, `scripts/launch_all_experiments.sh`.
- Task snapshot indicates no active execution (`pending=0`, `running=0`) with partial progress (`done=1`, `blocked=2`).

### Evidence
- Workdir/repo root: `/root/ebm-online-rl-prototype`
- Branch check context: `master`
- Command: `git status --porcelain=v1`
- Command: `git diff --stat`
- Plan tail reference includes remaining pipeline tasks (steps 8-15): probe smoke verification, eval/replan/swap/diversity script alignment, mini end-to-end callback run, mismatch fixes, then full validation launches with memory/handoff updates.
- Open dependency questions captured:
- Exact attached plan content or path is still needed.
- Exact `relay-long-task-callback` command/interface expected in this repo is still needed.
- Confirmation needed on whether to recreate `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.

### Next steps
- Resolve the three open questions before further implementation to avoid interface drift and rework.
- Execute remaining plan steps 8-15 in order, starting with syntax/help/smoke checks and a mini callback-enabled E2E validation pass.
- For each completed experiment phase, append evidence-backed updates to `HANDOFF_LOG.md` and refresh `docs/WORKING_MEMORY.md`.


## 2026-02-22T18:54:29.885Z
### Objective
- Preserve the exact repo/experiment handoff state on `master` so the next agent can resume callback-based validation and launch flow without re-triage.

### Changes
- Branch: `master`; working tree is dirty.
- Tracked files modified: `.gitignore`, `HANDOFF_LOG.md`, `docs/WORKING_MEMORY.md`, `research_finding.txt`, `scripts/exp_swap_matrix_maze2d.py`.
- Diff summary: 5 files changed, 672 insertions, 205 deletions.
- Untracked items present: `MUJOCO_LOG.TXT`, `gpt_pro_bundle_20260221.zip`, `gpt_pro_bundle_20260221b.zip`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`, `scripts/discord_score_poster.py`, `scripts/discord_swap_matrix_monitor.py`, `scripts/launch_all_experiments.sh`, `scripts/resume_after_eqnet.sh`.
- Task counts snapshot: `pending=0`, `running=0`, `done=2`, `failed=0`, `blocked=2`, `canceled=0`.
- Latest plan tail indicates implementation reached script updates/smoke stages; remaining work is callback mini-pipeline verification, mismatch fixes, then full validation launches.

### Evidence
- Workdir/repo root: `/root/ebm-online-rl-prototype`.
- Command source: `git status --porcelain=v1`.
- Command source: `git diff --stat`.
- Path touched: `/root/ebm-online-rl-prototype/.gitignore`.
- Path touched: `/root/ebm-online-rl-prototype/HANDOFF_LOG.md`.
- Path touched: `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md`.
- Path touched: `/root/ebm-online-rl-prototype/research_finding.txt`.
- Path touched: `/root/ebm-online-rl-prototype/scripts/exp_swap_matrix_maze2d.py`.
- Open blockers captured in context: exact attached plan content/path; exact `relay-long-task-callback` command/interface for this repo; decision on recreating `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.

### Next steps
- Resolve the three open blocker questions before new long launches.
- Run a mini end-to-end callback smoke pass (short run per experiment family) and fix any schema/analysis mismatches.
- Launch full validation experiments one-by-one via callback workflow after smoke pass is clean.
- After each completion, append evidence/results to `HANDOFF_LOG.md` and refresh `docs/WORKING_MEMORY.md`.

## Author repo success-vs-step GPU check — COMPLETED (finished 2026-02-23T21:xx+08:00)
- Objective:
  - Measure trajectory success as a function of training steps for UNet vs EqNet on the author-repo setup.
- Run root:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/success_curve_gpu_20260223-210750/`
- Clean eval outputs:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/success_curve_gpu_20260223-210750/eval_success_clean_n2e1/success_vs_step_clean_n2e1.csv`
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/success_curve_gpu_20260223-210750/eval_success_clean_n2e1/success_vs_step_summary_clean_n2e1.json`
- Protocol:
  - Checkpoints: steps `500, 1000, 2000, 3000, 4000, 5000`
  - Matched eval settings for both arches: `device=cuda`, `num_envs=2`, `num_episodes=1`, `n_exec_steps=44`, `sampling_steps=64`
- Key readout:
  - UNet success mean: `0.0` at every evaluated step.
  - EqNet success mean: `0.0` at every evaluated step.
  - EqNet minus UNet: `0.0` at every evaluated step.
- Interpretation:
  - Under this protocol there is no trajectory-success gain yet for either model up to 5000 steps; no EqNet-over-UNet success advantage is observed.

## Author repo easiest-task sanity replication — COMPLETED (finished 2026-02-23T22:02+08:00)
- Objective:
  - Test whether zero completion is purely due to task difficulty by selecting explicit easiest GridLand start-goal pairs.
- Run dir:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/success_curve_gpu_20260223-210750/easiest_task_repl_gpu_20260223-220221/`
- Outputs:
  - `scan_unet_pairs.json`
  - `easiest_pair_final_eval.json`
- Protocol:
  - Scan pairs with UNet (`num_envs=2`, `num_episodes=1`, `timeout=200`), pick best pair, then evaluate both UNet/EqNet on that pair (`num_envs=8`, `num_episodes=4`, `timeout=200`).
- Key readout:
  - All scanned pairs remained `0.0` for UNet.
  - Best scanned pair (`T0->T1`) final eval: UNet=`0.0`, EqNet=`0.0`.
- Diagnostic implication:
  - Zero completion is not explained solely by “task too hard” in this current setup.
  - Also note: GridLand reset path does not use `task_id` options in this author commit, so task-id loops are not creating distinct tasks for this path.

### Author repo gap overnight loop — FAILED (run_id: overnight_gap_hypothesis_20260223-222451)
- Relay job: `j-20260223-222606-97d6` (exit `1`)
- Run root:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_20260223-222451/`
- Available artifacts:
  - `phase0_easy_scan.json`
  - `run.log`
- Missing (run terminated before continuation):
  - `continuation_metrics.csv`
  - `summary.json`
- Failure signature:
  - `IndexError` in `eval_pair` at phase1 knob (`n_exec_steps=128`) due action-index overflow.
- Patch applied:
  - `.worktrees/eqnet-maze2d/scripts/author_eqnet_gap_overnight.py`
  - guard: cap action execution by `min(n_exec_steps, acts.shape[1])`.
- Provisional hypothesis readout from partial run:
  - H1 (`task too hard only`): weakened
  - H2 (`planner/eval mismatch`): inconclusive
  - H3 (`undertraining to non-zero success`): inconclusive
  - H4 (`EqNet > UNet success`): weakened
  - EqNet exceeded UNet on trajectory success: **no** (observed data only)
- Next discriminative overnight command:
```bash
cd /root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d && source /root/miniconda3/bin/activate dstitch39 && python scripts/author_eqnet_gap_overnight.py --repo /tmp/diffusion-stitching/goal_stitching --dataset runs/analysis/author_repo_zero_step/datasets/gridland_n5_gc_author_notebookstyle.npy --unet_ckpt runs/analysis/author_repo_zero_step/success_curve_gpu_20260223-210750/unet/checkpoints/succcurve-unet-gridland-n5-gc-d51e8ab9diffusion_ckpt_5000.pt --eqnet_ckpt runs/analysis/author_repo_zero_step/success_curve_gpu_20260223-210750/eqnet/checkpoints/succcurve-eqnet-gridland-n5-gc-3546ab1ddiffusion_ckpt_5000.pt --run_dir runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-0639 --device cuda --start_step 5000 --end_step 80000 --milestone_every 10000 --metric_batches 20 --batch_size 256 --schedule_total_steps 500000 --seed 0 --wait_for_gpu --allow_overlap --gpu_wait_timeout_sec 60 --gpu_wait_poll_sec 15
```


## 2026-02-23T22:09:39.943Z
### Objective
- Hand off the current Maze2D validation/ablation state on `master`, including what changed, what is blocked, and what to run next.

### Changes
- Tracked edits exist in [`.gitignore`](/root/ebm-online-rl-prototype/.gitignore), [`HANDOFF_LOG.md`](/root/ebm-online-rl-prototype/HANDOFF_LOG.md), [`docs/WORKING_MEMORY.md`](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md), [`research_finding.txt`](/root/ebm-online-rl-prototype/research_finding.txt), and [`scripts/exp_swap_matrix_maze2d.py`](/root/ebm-online-rl-prototype/scripts/exp_swap_matrix_maze2d.py).
- Untracked artifacts/scripts are present: [`MUJOCO_LOG.TXT`](/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT), [`gpt_pro_bundle_20260221.zip`](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221.zip), [`gpt_pro_bundle_20260221b.zip`](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221b.zip), [`gpt_pro_handoff_bundle_20260220.zip`](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip), [`gpt_pro_handoff_bundle_20260220/`](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220/), [`memory/`](/root/ebm-online-rl-prototype/memory/), [`scripts/discord_score_poster.py`](/root/ebm-online-rl-prototype/scripts/discord_score_poster.py), [`scripts/discord_swap_matrix_monitor.py`](/root/ebm-online-rl-prototype/scripts/discord_swap_matrix_monitor.py), [`scripts/launch_all_experiments.sh`](/root/ebm-online-rl-prototype/scripts/launch_all_experiments.sh), [`scripts/resume_after_eqnet.sh`](/root/ebm-online-rl-prototype/scripts/resume_after_eqnet.sh).
- Diffstat snapshot: 5 tracked files changed, 1542 insertions, 204 deletions.
- Task tracker snapshot: `done=3`, `blocked=2`, `pending=0`, `running=0`, `failed=0`, `canceled=0`.

### Evidence
- Repo root: `/root/ebm-online-rl-prototype`
- Branch: `master` from `git branch --show-current`
- Dirty tree and untracked files from `git status --porcelain=v1`
- File-level change volume from `git diff --stat`
- Remaining execution context from `last_plan_tail` (steps 8-15) and blocker questions in the provided snapshot

### Next steps
- Confirm the exact attached plan text/path so remaining tasks can be mapped precisely.
- Confirm the required `relay-long-task-callback` interface/command for this repo.
- Confirm whether recreating `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` is required in this cycle.
- After confirmations, continue the remaining sequence: finish script/schema verification, run the mini callback pipeline, fix mismatches, then launch full validation runs one-by-one and keep [`docs/WORKING_MEMORY.md`](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md) and [`HANDOFF_LOG.md`](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) updated after each completion.

## 2026-02-24T10:51:16+08:00
### Smoke-gated supervisor v1.1
- Updated proposal doc to v1.1 with mandatory Stage-0 smoke gate, explicit terminal statuses, auto-fix guardrails, stable state path, and risk/caveat coverage.
- Added `scripts/stage0_smoke_gate.py` for smoke->full execution gating with required-artifact checks and JSON state output.
- Local runtime validation (no long experiment launched):
  - `tmp/stage0_gate_tests_1771901452/t1`: `rc=0`, state `success`.
  - `tmp/stage0_gate_tests_1771901452/t2`: `rc=20`, state `blocked_smoke_failed_exit`, full stage blocked.
  - `tmp/stage0_gate_tests_1771901452/t3`: `rc=31`, state `blocked_full_missing_artifacts`.
- Evidence:
  - `docs/plans/2026-02-24-iterative-debug-supervisor-contract.md`
  - `scripts/stage0_smoke_gate.py`
  - `tmp/stage0_gate_tests_1771901452/`

## 2026-02-24T10:52:22+08:00
### Smoke-gate runner hardening
- `scripts/stage0_smoke_gate.py` now resolves relative `--smoke-required-file`, `--full-required-file`, and `--smoke-run-dir` against `--cwd` for deterministic relay execution.
- Runtime check from non-project shell cwd (`/tmp`) passed:
  - `tmp/stage0_gate_tests_1771901452/t4_rel`: `rc=0`, state `success`, smoke cleanup action `deleted_smoke_run_dir_kept_manifest`.

## 2026-02-24T10:58:23+08:00
### Callback canary prep
- Prepared an in-thread relay callback canary using `job_start` watcher callback (`thenTask`, `runTasks=true`) to validate end-to-end behavior of `scripts/stage0_smoke_gate.py` under relay execution.
- Mapped relay-native integration hook points in `/root/codex-discord-relay/relay.js` for a future first-class supervisor contract implementation.

## 2026-02-24T11:00:08+08:00
### Callback E2E canary result (task t-0002)
- Validated run directory from `/tmp/relay_callback_e2e_last_path.txt`:
  - `/root/ebm-online-rl-prototype/tmp/relay_callback_e2e/relay_e2e_20260224-105924`
- Relay IDs:
  - `job_id=j-20260224-105924-2b5b`
  - `task_id=t-0002`
  - `run_id=relay_e2e_20260224-105924`
- Verification outcome:
  - `state.json` status is `success`.
  - smoke/full phases both `passed` with `exit_code=0` and no missing required files.
  - smoke cleanup behavior is correct: `deleted_smoke_run_dir_kept_manifest` and `smoke_manifest.json` present.
  - `gate.out.log` shows pass for smoke and full phases; `gate.err.log` is empty.
- Current status:
  - Callback canary E2E path is healthy for the stage-0 smoke-gate runner.

## High-confidence side-by-side checkpoint eval (2026-02-24, pair `T0->T1`)
- Parent run:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/`
- New artifacts:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/eval_highconf_n16e8_20260224/side_by_side.csv`
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/eval_highconf_n16e8_20260224/side_by_side.json`
- Eval protocol (same for UNet/EqNet):
  - `sampling_steps=64`, `n_exec_steps=44`, `timeout_steps=200`, `temperature=0.5`
  - `num_envs=16`, `num_episodes=8` -> `128` rollouts/model/checkpoint
- Matched-checkpoint results:
  - Step `55000`: UNet `0.6094` (`78/128`, CI95 `[0.5228,0.6895]`), EqNet `0.9531` (`122/128`, CI95 `[0.9015,0.9783]`), delta `+0.3438`
  - Step `65000`: UNet `0.6719` (`86/128`, CI95 `[0.5866,0.7472]`), EqNet `0.8906` (`114/128`, CI95 `[0.8248,0.9337]`), delta `+0.2188`
- Current readout:
  - EqNet > UNet on both high-confidence matched checkpoints; prior 8-rollout advantage is reinforced under higher sample count.

## Mechanistic diagnosis: horizon vs goal propagation (2026-02-24)
- Parent run root:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/`
- New diagnostic artifacts:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_influence_diag_20260224/goal_influence_summary.json`
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step65000.json`
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step65000_quicksweep_n32.json`

### Key implementation facts (author setup)
- `horizon=64`, `pad=True` => `gen_horizon=128`
- `inpaint=True`, `goal_inpaint_steps=25`
- `sampling_steps=64`, `n_exec_steps=44`, `memory=20`
- EqNet kernel schedule (`n_layers=25`, `kernel_expansion_rate=5`):
  - `[3x5, 5x5, 7x5, 9x5, 11x5, 13x2]`
  - theoretical receptive field = `349` positions (`> gen_horizon=128`)

### Goal-to-first-action sensitivity (fixed start `T0`, matched noise, goals `{T1,T5,B5,L2,R4}`)
- UNet step55k: mean `||Δa0(goal,ref)|| = 1.4295`, between/within `72.69`
- UNet step65k: mean `||Δa0(goal,ref)|| = 1.4419`, between/within `78.37`
- EqNet step55k: mean `||Δa0(goal,ref)|| = 1.3036`, between/within `40.97`
- EqNet step65k: mean `||Δa0(goal,ref)|| = 1.3088`, between/within `35.12`
- Readout: both models are strongly goal-sensitive at first action in this setup; “no goal effect on first action” is not supported.

### Goal inpainting-width ablation @ step65k (pair `T0->T1`, high-confidence `n=128`/arch)
- UNet: `goal_inpaint_steps 25 -> 1`: `0.5781 -> 0.9219` (`+0.3438`)
- EqNet: `goal_inpaint_steps 25 -> 1`: `0.8906 -> 0.0000` (`-0.8906`)
- Readout: EqNet performance is highly dependent on multi-step terminal goal anchoring in this setup.

### Mechanistic implication for prior online RL gap
- Prior online RL EqNet-vs-UNet result (3-seed umaze): EqNet underperformed UNet (`0.2778` vs `0.7778` success@h256).
- Likely key protocol/architecture interaction:
  - online RL stack conditions only `{t=0, t=horizon-1}` in `GoalDataset` + `apply_conditioning` (single terminal anchor);
  - author setup anchors a `25`-step terminal suffix via inpainting.
- Working hypothesis update:
  - EqNet is more brittle than UNet to narrow terminal anchoring; widening terminal goal anchoring can flip relative performance.


## 2026-02-24T04:19:23.553Z
### Objective
- Preserve the current Maze2D validation/debug state and hand off enough context to resume callback-driven experiment execution without re-discovery.
- Capture repo, task, and plan status at a point where implementation progress is ahead of verification and launch execution.

### Changes
- Modified tracked files: [.gitignore](/root/ebm-online-rl-prototype/.gitignore), [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md), [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md), [research_finding.txt](/root/ebm-online-rl-prototype/research_finding.txt), [scripts/exp_swap_matrix_maze2d.py](/root/ebm-online-rl-prototype/scripts/exp_swap_matrix_maze2d.py).
- Current diff footprint is large: 1,997 insertions and 204 deletions across 5 tracked files.
- Added new untracked orchestration/monitoring assets including [scripts/discord_score_poster.py](/root/ebm-online-rl-prototype/scripts/discord_score_poster.py), [scripts/discord_swap_matrix_monitor.py](/root/ebm-online-rl-prototype/scripts/discord_swap_matrix_monitor.py), [scripts/launch_all_experiments.sh](/root/ebm-online-rl-prototype/scripts/launch_all_experiments.sh), [scripts/resume_after_eqnet.sh](/root/ebm-online-rl-prototype/scripts/resume_after_eqnet.sh), [scripts/stage0_smoke_gate.py](/root/ebm-online-rl-prototype/scripts/stage0_smoke_gate.py), and planning/memory artifacts.
- Task snapshot: `pending=0`, `running=0`, `done=4`, `failed=0`, `blocked=2`, `canceled=0`.
- Plan tail indicates implementation focus through experiment/eval scripts, mini pipeline validation, then full validation launches; three blocker questions remain unresolved (attached plan source, callback interface contract, whether to recreate `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`).

### Evidence
- Command: `cd /root/ebm-online-rl-prototype && git status --porcelain=v1` (shows modified tracked files plus untracked run/bundle/script artifacts, including [docs/plans/2026-02-24-iterative-debug-supervisor-contract.md](/root/ebm-online-rl-prototype/docs/plans/2026-02-24-iterative-debug-supervisor-contract.md) and [memory/](/root/ebm-online-rl-prototype/memory/)).
- Command: `cd /root/ebm-online-rl-prototype && git diff --stat` (reports 5 files changed, 1997 insertions, 204 deletions).
- Key generated/untracked artifacts visible in status: [MUJOCO_LOG.TXT](/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT), [gpt_pro_bundle_20260221.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221.zip), [gpt_pro_bundle_20260221b.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221b.zip), [gpt_pro_handoff_bundle_20260220.zip](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip), [gpt_pro_handoff_bundle_20260220/](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220/).

### Next steps
- Resolve the three explicit blockers: exact attached-plan source, exact `relay-long-task-callback` command/interface for this repo, and whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` must be recreated.
- Run syntax/help/smoke verification for newly adjusted experiment/eval scripts before any long launch.
- Execute the end-to-end mini pipeline with callback-enabled monitoring (short run per family), then fix schema/analysis mismatches found.
- Launch full validation experiments one-by-one with callback workflow after mini pipeline passes.
- Append evidence-backed updates to [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md) and [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) after each run completion.

## 2026-02-24T14:34:08+08:00
### High-confidence K-grid follow-up at latest checkpoint (`step80000`)
- Run id: `goal_inpaint_ablation_step80000_highconf_kgrid_n128`
- Parent run: `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/`
- Artifacts:
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step80000_highconf_kgrid_n128.json`
  - `.worktrees/eqnet-maze2d/runs/analysis/author_repo_zero_step/overnight_gap_hypothesis_overlapfix_20260224-095558/goal_inpaint_ablation_20260224/goal_inpaint_ablation_step80000_highconf_kgrid_n128.csv`
- Protocol (matched with prior high-confidence eval family):
  - Pair `T0->T1`, `sampling_steps=64`, `n_exec_steps=44`, `timeout_steps=200`, `temperature=0.5`
  - `num_envs=16`, `num_episodes=8` => `128` rollouts/model/K
  - `K in {1,2,5,8,10,20}`
- Key results:
  - `K=1`: UNet `0.960938` (`123/128`), EqNet `0.000000` (`0/128`)
  - `K=2`: UNet `0.960938` (`123/128`), EqNet `0.000000` (`0/128`)
  - `K=5`: UNet `1.000000` (`128/128`), EqNet `0.578125` (`74/128`)
  - `K=8`: UNet `0.992188` (`127/128`), EqNet `0.789062` (`101/128`)
  - `K=10`: UNet `0.992188` (`127/128`), EqNet `0.945312` (`121/128`)
  - `K=20`: UNet `0.937500` (`120/128`), EqNet `0.867188` (`111/128`)
- Current interpretation update:
  - EqNet remains highly sensitive to low `K` (collapse at `K=1,2`) and recovers as `K` widens.
  - At latest checkpoint `step80000`, UNet is higher than EqNet for all tested `K` in this narrowed high-confidence sweep.
  - This is checkpoint-dependent, not a global dominance claim: prior high-confidence matched-checkpoint eval (`K=25`) still shows EqNet > UNet at `step55000` and `step65000`.
- Suggested next discriminative run:
  - Cross-checkpoint matched high-confidence sweep at a small common grid (`K={1,5,10,25}`) for `step in {55000,65000,80000}` to map EqNet-vs-UNet regime boundaries directly.

## 2026-02-24T15:13:51+08:00
### Callback E2E refresh (relay smoke-gate)
- Re-ran callback e2e smoke gate and confirmed the expected contract on a fresh run:
  - run dir: `/root/ebm-online-rl-prototype/tmp/relay_callback_e2e/relay_e2e_20260224-151240`
  - `state.json.status=success`
  - `smoke_cleanup.action=deleted_smoke_run_dir_kept_manifest`
  - `smoke_manifest.json` exists with two entries
  - smoke dir removed and full dir present
  - `gate.err.log` empty
- Validation summary check: `ALL_OK=1`.
