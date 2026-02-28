# EBM Online RL Working Memory (living snapshot)

Last updated: 2026-02-25T00:00:00Z
Repo: /root/ebm-online-rl-prototype
Branch: analysis/results-2026-02-24
Commit: d58e8eb (clean)
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


## 2026-02-24T08:24:41.760Z
### Objective
- Preserve a precise handoff snapshot so the next agent can continue the Maze2D validation workflow on `analysis/results-2026-02-24` without re-discovery.

### Changes
- No code or config files were modified in this handoff step; this entry is state capture only.
- Captured repo context: workdir/repo root `/root/ebm-online-rl-prototype`, branch `analysis/results-2026-02-24`.
- Captured untracked artifacts: `MUJOCO_LOG.TXT`, `gpt_pro_bundle_20260221.zip`, `gpt_pro_bundle_20260221b.zip`, `gpt_pro_bundle_20260224_full.zip`, `gpt_pro_bundle_20260224_full/`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`.
- Captured task state: `pending=0 running=0 done=5 failed=0 blocked=2 canceled=0`.
- Preserved remaining plan tail (steps 8-15): finish script contract/schema alignment, run mini callback pipeline, fix mismatches, then launch full validation runs and update memory/handoff docs.
- Preserved unresolved blockers/questions: exact attached plan source, exact `relay-long-task-callback` interface, and whether to recreate `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.

### Evidence
- Path context: `/root/ebm-online-rl-prototype`.
- Command snapshot: `git status --porcelain=v1`.
- Untracked paths observed:
- `/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT`
- `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221.zip`
- `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221b.zip`
- `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full.zip`
- `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full/`
- `/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip`
- `/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220/`
- `/root/ebm-online-rl-prototype/memory/`
- Plan tail references scripts:
- `/root/ebm-online-rl-prototype/scripts/train_synthetic_maze2d_sac_her_probe.py`
- `/root/ebm-online-rl-prototype/scripts/eval_synth_maze2d_checkpoint_prefix.py`
- `/root/ebm-online-rl-prototype/scripts/exp_replan_horizon_sweep.py`
- `/root/ebm-online-rl-prototype/scripts/exp_swap_matrix_maze2d.py`
- `/root/ebm-online-rl-prototype/scripts/analyze_posterior_diversity.py`

### Next steps
- Get the exact attached plan content/path and map each remaining item to concrete commands.
- Confirm the required `relay-long-task-callback` invocation contract for this repo.
- Decide whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` must be regenerated in this cycle.
- Execute remaining validation steps (8-15), starting with probe schema/checkpoint verification and mini callback pipeline.
- After each completed run, update `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md` and `/root/ebm-online-rl-prototype/HANDOFF_LOG.md` with evidence-backed outcomes.

## 2026-02-24T20:26:11+08:00
### CRL baseline anchor (official contrastive_rl on Maze2D umaze)
- Run id: `crl_contrastive_nce_ts18000_seed0_20260224-2019`
- Run dir:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_contrastive_nce_ts18000_seed0_20260224-2019/`
- Config summary:
  - `env_name=maze2d_umaze`, `alg=contrastive_nce`, `seed=0`, `num_actors=1`, `max_number_of_steps=18000`
  - start-goal extraction indices: `start_index=0`, `end_index=2`

### Comparable evaluation outcome (fixed 12-query baseline set, h256)
- Source query set:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/query_metrics.csv`
- CRL fixed-query eval artifacts:
  - `.../query_eval_summary.json`
  - `.../query_eval_records.csv`
- Metric (`threshold=0.2`, `horizon=256`):
  - CRL (contrastive_nce): `success@h256 = 0.0833` (`1/12`)

### Side-by-side snapshot (single-seed, preliminary)
- Comparison artifact:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_vs_existing_baselines_20260224.json`
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_vs_existing_baselines_20260224.csv`
- Values:
  - Diffuser: `0.8333`
  - SAC+HER sparse: `0.8333`
  - GCBC+HER: `0.5833`
  - CRL (official, nce, 18k-step): `0.0833`

### Readout and next discriminative run
- Readout: CRL is currently far below existing baselines under this first official-implementation run and single-seed budget.
- Caution: likely undertraining relative to transition volume used by the other methods.
- Next run (runnable):
```bash
cd /root/ebm-online-rl-prototype/third_party/google-research-contrastive_rl/contrastive_rl && source /root/miniconda3/bin/activate contrastive_rl && export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 && export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/root/miniconda3/envs/contrastive_rl/lib:$LD_LIBRARY_PATH && export D4RL_SUPPRESS_IMPORT_ERROR=1 && python lp_contrastive.py --env_name=maze2d_umaze --alg=contrastive_nce --start_index=0 --end_index=2 --seed=0 --num_actors=1 --max_number_of_steps=272000
```

## 2026-02-24T20:49:38+08:00
### CRL fair-budget controls (transition + gradient alignment)
- Added explicit CRL knobs in `lp_contrastive.py`:
  - `--step_limiter_key` (`auto|actor_steps|learner_steps`)
  - `--num_sgd_steps_per_step`
  - `--samples_per_insert`
  - `--samples_per_insert_tolerance_rate`
  - `--min_replay_size`
  - `--batch_size`
- Added `step_limiter_key` to `ContrastiveConfig` and wired it into `distributed_layout.py` coordinator selection.
- New launcher script:
  - `scripts/run_crl_maze2d_fair_budget.py`
  - derives fair CRL budget from baseline summaries and emits runnable `command.sh` + `fair_budget_config.json`.

### Dry-run aligned budget (current Maze2D comparison root)
- run dir:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-204814/`
- computed targets:
  - target transitions: `272338`
  - aligned actor steps: `272384` (episode length `256`)
  - target gradient descents: `18000`
  - `num_sgd_steps_per_step=1`
  - `learner_steps_target=18000`
  - `samples_per_insert=16.91729323`
  - `step_limiter_key=learner_steps`

### Runnable next command
```bash
cd /root/ebm-online-rl-prototype && bash runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-204814/command.sh
```
- Guardrail: when deriving CRL `samples_per_insert`, auto-enforce the Reverb tolerance bound (`error_buffer >= 2*max(1,samples_per_insert)`) before launching, otherwise startup can abort in replay table construction.
- Updated fair-budget launcher to auto-bump tolerance (`samples_per_insert_tolerance_rate`) to valid Reverb bounds; latest dry-run artifact: `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110/fair_budget_config.json`.
- Runtime sanity: `timeout 30s` startup on the aligned command (`.../crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110/command.sh`) entered normal actor/evaluator loop with no immediate Reverb limiter exception.

## ${NOW}
### CRL fair-budget fixed-query comparison (completed)
- Run id: `crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110`
- Status:
  - Training run reached limiter and terminated cleanly (`StepsLimiter: Max steps of 18000 was reached`) with final checkpoints saved under `/root/acme/b986f8ba-1180-11f1-bc6b-f638e7586226/checkpoints/`.
  - Fixed-query evaluation completed and artifacts were written.
- Comparable fixed-query result (`12` queries, `h256`, threshold `0.2`):
  - `contrastive_rl_nce_fair_budget`: `0.9166666666666666` (`11/12`)
- Comparison artifact now including fair CRL row:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_vs_existing_baselines_with_fair_20260224.csv`
- Snapshot comparable values:
  - Diffuser `0.8333`, SAC+HER sparse `0.8333`, GCBC+HER `0.5833`, CRL old `0.0833`, CRL fair-budget `0.9167`.
- Immediate next step:
  - Run at least `3` seeds with the same fair-budget contract to check if the CRL advantage holds beyond this single-seed result.

## 2026-02-24T21:26:15+08:00
### Correction
- The prior section header `## ${NOW}` is a literal placeholder from a quoted-heredoc append.
- Canonical timestamp for that section is `2026-02-24T21:26:15+08:00`.

## 2026-02-24T21:30:15+08:00
### Task t-0008: CRL fair-budget run-log parse + fixed-query refresh
- Run analyzed:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110/run.log`
- Parsed final terminal metrics artifact:
  - `.../crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110/run_terminal_metrics.json`
- Final metrics extracted from log:
  - `Actor Steps` last seen: `360192`
  - `Learner Steps` last seen: `18399`
  - `Success 1000` last seen: `0.924`
  - last line with all three fields: `Actor=360192`, `Learner=18321`, `Success1000=0.924`

### Fixed 12-query h256 evaluation rerun
- Re-ran:
  - `python /tmp/eval_crl_fixed_queries.py`
- Updated eval artifacts:
  - `.../crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110/query_eval_summary.json`
  - `.../crl_contrastive_nce_fair_t272384_g18000_sgd1_seed0_20260224-205110/query_eval_records.csv`
- Observed comparable outcome (`12` fixed queries, `horizon=256`, threshold `0.2`):
  - fair-budget CRL `success@h256 = 1.0000` (`12/12`)

### Comparison table refresh
- Updated canonical comparison files (requested):
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_vs_existing_baselines_20260224.json`
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/crl_vs_existing_baselines_20260224.csv`
- Current comparable rows:
  - Diffuser `0.8333`
  - SAC+HER sparse `0.8333`
  - GCBC+HER `0.5833`
  - CRL old (`18k-step`) `0.0833`
  - CRL fair-budget aligned run `1.0000`


## 2026-02-24T13:31:04.615Z
### Objective
- Preserve the current `analysis/results-2026-02-24` experiment state so the next agent can resume validation and callback-based full runs without re-triage.

### Changes
- Updated `/root/ebm-online-rl-prototype/HANDOFF_LOG.md` (`+261` lines per `git diff --stat`).
- Updated `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md` (`+161` lines per `git diff --stat`).
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221.zip`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221b.zip`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full.zip`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full/`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220/`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/memory/`.
- Left untracked artifacts in place for review/bundling: `/root/ebm-online-rl-prototype/scripts/run_crl_maze2d_fair_budget.py`.
- Captured task snapshot: `pending=0 running=0 done=6 failed=0 blocked=2 canceled=0`.

### Evidence
- Command: `git status --porcelain=v1` (branch `analysis/results-2026-02-24`, modified + untracked files listed above).
- Command: `git diff --stat` (`HANDOFF_LOG.md | 261`, `docs/WORKING_MEMORY.md | 161`, total `422 insertions`).
- Plan tail recorded with remaining items 8-15, ending in callback-based full validations and post-run memory/handoff updates.

### Next steps
- Provide the exact attached plan content or path so task mapping is exact (not inferred from filenames).
- Confirm the precise `relay-long-task-callback` command/interface expected in this repo.
- Confirm whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` should be recreated in this cycle.
- Resume from plan item 8: smoke verification for SAC/HER probe, then script hardening for eval/replan/swap/diversity analysis.
- Run mini end-to-end callback pipeline, fix schema/analysis mismatches, then launch full validation runs one-by-one.
- After each completion, append evidence-backed updates to `/root/ebm-online-rl-prototype/HANDOFF_LOG.md` and refresh `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md`.

## 2026-02-24T21:44:29+08:00
### CRL obsolete-run cleanup (user requested)
- Cleaned up obsolete old CRL baseline artifacts (`crl_contrastive_nce_ts18000_seed0_20260224-2019` + `_recheck_20260224`) by moving them under:
  - `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/archive_irrelevant_20260224/`
- Removed all `method=contrastive_rl_nce` rows from active comparison tables:
  - `.../crl_vs_existing_baselines_20260224.{json,csv}`
  - `.../crl_vs_existing_baselines_with_fair_20260224.{json,csv}`
- Active comparable rows now reflect only relevant methods:
  - Diffuser `0.8333`
  - SAC+HER sparse `0.8333`
  - GCBC+HER `0.5833`
  - CRL fair-budget (`1.0000` in direct table; `0.9167` in with_fair table)
- Note: old run data is archived (not deleted) for auditability.

## 2026-02-24T22:01:58+08:00
### CRL old-run hard delete
- Completed user-requested permanent deletion of obsolete archived CRL-old folders:
  - `.../archive_irrelevant_20260224/crl_contrastive_nce_ts18000_seed0_20260224-2019`
  - `.../archive_irrelevant_20260224/crl_contrastive_nce_ts18000_seed0_20260224-2019_recheck_20260224`
- Removed empty archive directory `.../archive_irrelevant_20260224`.
- Active comparison root now excludes old CRL run directories entirely.

## 2026-02-25T00:23:32+08:00
### EqM minimal-change pointmass implementation + tuning (new)
- Implemented EqM drop-in module and integrated online algo switch:
  - `ebm_online_rl/online/eqm.py`
  - `ebm_online_rl/online/__init__.py`
  - `scripts/online_pointmass_goal_diffuser.py` (`--algo {diffusion,eqm}`, EqM knobs, goal-influence probe)
- Run root:
  - `runs/analysis/pointmass_eqm_minchange_20260224-230943/`
- Key summaries:
  - `eqm_sweep_2k_summary.json`
  - `eqm_sweep_2k_summary.csv`
  - `eqm_vs_diffusion_compare_summary.json`

### Latest comparable readout (pointmass, matched budgets)
- 2k smoke:
  - Diffusion: `eval_success_rate=0.00`
  - EqM (`k=25`, `step=0.05`, `c=1.0`): `eval_success_rate=0.10`
- EqM 2k sweep best practical config:
  - `k=25`, `step=0.10`, `c=1.0` -> `eval_success_rate=0.15`
- 6k matched control (`eval_every=2000`):
  - EqM (`k=25`, `step=0.10`, `c=1.0`): `0.15 @2k`, `0.10 @4k`, `0.15 @6k`
  - Diffusion (`n_diffusion_steps=8`): `0.00 @2k`, `0.00 @4k`, `0.05 @6k`
- EqM action-mode ablation (`planner_control_mode=action`, 2k):
  - `eval_success_rate=0.00` (worse than waypoint for this setup)

### Current decision
- Keep EqM waypoint config as current pointmass best:
  - `--eqm_steps 25 --eqm_step_size 0.10 --eqm_c_scale 1.0 --planner_control_mode waypoint`
- Interpretation: EqM is viable and stronger than matched Diffusion in this minimal-change pointmass loop, but absolute success is still modest and below stronger Maze2D baselines.

### Next discriminating run
- Port same EqM module into Maze2D probe with minimal API swap and run matched-budget side-by-side eval against current Diffusion/SAC/GCBC tables.

## 2026-02-25T02:23:00+08:00
### EqM Maze2D budget-match sweep (completed)
- Run root:
  - `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/`
- Config family (both runs):
  - `train_steps=6000`, `online_rounds=4`, `online_train_steps_per_round=3000`, `online_collect_transition_budget_per_round=4096`
  - `eqm_steps=25`, `eqm_c_scale=1.0`, `eqm_step_size in {0.10,0.08}`
- Final `h256` results:
  - `eqm_k25_s010_budgetmatch`: `0.7500` (`9/12`)
  - `eqm_k25_s008_budgetmatch`: `0.7500` (`9/12`)
- Best observed checkpoints (from `progress_metrics.csv`):
  - both variants reached `0.9167` at step `8000`
- Comparator snapshot (`h256`):
  - Diffuser `0.8333`, SAC+HER sparse `0.8333`, GCBC+HER `0.5833`
- New summary artifacts:
  - `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_budgetmatch_summary.json`
  - `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_budgetmatch_summary.csv`
  - `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_vs_existing_baselines_20260225.json`
  - `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_vs_existing_baselines_20260225.csv`

### Matched-budget no-HER control pipeline (running)
- Run root:
  - `runs/analysis/synth_maze2d_diffuser_probe/noher_budgetmatch_20260225-022110/`
- Launcher:
  - `/tmp/run_noher_budgetmatch_overnight.sh`
  - PID file: `/tmp/run_noher_budgetmatch_overnight.pid` (`39949` at launch)
  - Driver log: `/tmp/run_noher_budgetmatch_overnight.out`
  - In-run log: `runs/.../driver.log`
- Execution order:
  1. `sac_noher_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0`
  2. `gcbc_noher_ts6000_or4_ep64_t3000_rp16_gp040_seed0`
- Planned output:
  - `runs/.../noher_vs_her_summary.json`
  - `runs/.../noher_vs_her_summary.csv`
- Purpose:
  - Quantify absolute HER gain (`HER - noHER`) for SAC and GCBC under the same budget/protocol as existing baseline rows.

### Correction: no-HER overnight launcher method
- The `nohup` chained launcher (`/tmp/run_noher_budgetmatch_overnight.sh`) is not reliable under the current exec wrapper for multi-stage sequencing.
- Observed behavior: supervisor shell disappeared; only first SAC subprocess persisted as orphan.
- Mitigation applied: killed orphan SAC subprocesses and switched plan to relay-managed `job_start` continuation for unattended SAC->GCBC->summary sequencing.

## 2026-02-25T02:45:36+08:00
### Task t-0009 complete: matched-budget HER gain readout (SAC vs GCBC)
- Final no-HER run root:
  - `runs/analysis/synth_maze2d_diffuser_probe/noher_budgetmatch_20260225-022453/`
- Completion status:
  - `sac_noher_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0`: `rc=0`
  - `gcbc_noher_ts6000_or4_ep64_t3000_rp16_gp040_seed0`: `rc=0`
- Comparable `h256` (`12` query protocol):
  - SAC+HER sparse: `0.8333` (`10/12`)
  - SAC no-HER sparse: `0.8333` (`10/12`)
  - SAC HER gain: `+0.0000`
  - GCBC+HER: `0.5833` (`7/12`)
  - GCBC no-HER: `0.0833` (`1/12`)
  - GCBC HER gain: `+0.5000` absolute (`+6/12`, `+600%` relative vs no-HER)
- Artifacts:
  - `runs/analysis/synth_maze2d_diffuser_probe/noher_budgetmatch_20260225-022453/noher_vs_her_summary.json`
  - `runs/analysis/synth_maze2d_diffuser_probe/noher_budgetmatch_20260225-022453/noher_vs_her_summary.csv`
- Current implication:
  - In this setup, HER is critical for GCBC performance but not for SAC sparse performance.


## 2026-02-24T18:46:21.686Z
### Objective
- Advance `analysis/results-2026-02-24` toward callback-ready Maze2D validation by aligning probe/training scripts with checkpoint/output contracts and preserving experiment continuity for the next agent.

### Changes
- Expanded handoff continuity docs with major updates to `HANDOFF_LOG.md` and `docs/WORKING_MEMORY.md`.
- Updated `ebm_online_rl/online/__init__.py` and added `ebm_online_rl/online/eqm.py`.
- Modified `scripts/online_pointmass_goal_diffuser.py` for updated online/diffuser behavior and run contract alignment.
- Modified `scripts/synthetic_maze2d_diffuser_probe.py` for schema/contract compatibility.
- Modified `scripts/synthetic_maze2d_gcbc_her_probe.py` for schema/contract compatibility.
- Modified `scripts/synthetic_maze2d_sac_her_probe.py` for schema/contract compatibility.
- Added `scripts/run_crl_maze2d_fair_budget.py`.
- Added/retained run artifacts and bundles: `MUJOCO_LOG.TXT`, `gpt_pro_bundle_20260221.zip`, `gpt_pro_bundle_20260221b.zip`, `gpt_pro_bundle_20260224_full.zip`, `gpt_pro_bundle_20260224_full/`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`.
- Current execution summary indicates `done=7`, `blocked=2`, `pending=0`, `running=0`.

### Evidence
- Workdir/repo root: `/root/ebm-online-rl-prototype`.
- Branch: `analysis/results-2026-02-24`.
- Command: `git status --porcelain=v1` (shows modified and untracked files listed above).
- Command: `git diff --stat` (reports `7 files changed, 1244 insertions(+), 70 deletions(-)` across tracked edits).
- Plan tail indicates remaining items around SAC/HER smoke verification, checkpoint-prefix eval aggregation, horizon-sweep and swap-matrix experiments, posterior-diversity analysis, mini end-to-end callback run, and full validation rollout with memory/handoff updates.

### Next steps
- Resolve blocker 1: obtain the exact attached plan content/path to map remaining tasks precisely.
- Resolve blocker 2: confirm the exact `relay-long-task-callback` command/interface expected in this repo.
- Confirm whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` must be recreated in this cycle.
- Run syntax/help/smoke verification for `scripts/synthetic_maze2d_sac_her_probe.py` under the final schema/checkpoint contract.
- Implement or finalize `scripts/eval_synth_maze2d_checkpoint_prefix.py` robust aggregation outputs.
- Implement or finalize `scripts/exp_replan_horizon_sweep.py` with callback-ready metadata/output.
- Implement or finalize `scripts/exp_swap_matrix_maze2d.py` with callback-ready metadata/output.
- Implement or finalize `scripts/analyze_posterior_diversity.py` to consume produced outputs and emit validation summaries.
- Execute a mini end-to-end callback pipeline, fix schema/analysis mismatches, then launch full validation runs sequentially and append evidence-backed updates to `docs/WORKING_MEMORY.md` and `HANDOFF_LOG.md`.

### Reporting guardrail (2026-02-25)
- Always begin experiment results with an algorithm glossary block before metrics: define symbol semantics and units explicitly (`K := eqm_steps = number of EqM refinement iterations`, `S := eqm_step_size = per-step descent size`) and only then use compact run tags (`k25_s010`, etc.).

### PointMass threshold knob + smoke (2026-02-25)
- Added `--success_threshold` to `scripts/online_pointmass_goal_diffuser.py` and wired it to both train/eval env constructors.
- Verified via smoke run (`1k` env steps, EqM `K=25`, `S=0.1`, threshold `0.2`): `eval_success_rate=0.6` (`n_eval_episodes=5`) at `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_rerun_20260225-1010/smoke_check/`.
- Pending long sanity run: EqM `50k` env steps with the same settings (`K=25`, `S=0.1`, `success_threshold=0.2`).

### Agentic Autodecider removed (2026-02-25, commit d58e8eb)
- 6 scripts deleted: `agentic_maze2d_autodecider.py`, `agentic_role_orchestrator.py`, `launch_agentic_maze2d_autodecider_tmux.sh`, `launch_agentic_role_orchestrator_tmux.sh`, `overnight_maze2d_autodecider.py`, `launch_overnight_maze2d_autodecider_tmux.sh`.
- Reason: relay auto-ML pipeline fully supersedes these. Git history preserves them.
- HEAD now: `d58e8eb` (clean), pushed to `origin/analysis/results-2026-02-24`.

## PointMass rollout visual debug (2026-02-25)
- New visual diagnostics from latest checkpoint:
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_step10000_rollout_trajectories_grid.png`
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_step10000_rollout_distance_traces.png`
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_step10000_rollout_debug_summary.json`
- Diagnostic readout (`step_10000`, 12 replayed eval episodes, threshold `0.2`):
  - `mean_min_dist=0.1232`, `mean_final_dist=0.9581`, `success_rate=0.8333`, `rebound_after_hit=8/12`.
- Current interpretation:
  - The large min-vs-final gap is due to trajectory rebound: the planner often reaches near-goal mid-episode but does not stay there through the episode end.

## PointMass loss-curve diagnostics (2026-02-25)
- Added active-run loss plot and summary:
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_loss_curve_step_latest.png`
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_loss_curve_summary.json`
- Current readout (`env_steps=17850`):
  - train loss decreases (`8.9446 -> 7.5094`) while validation loss increases (`0.5694 -> 4.1889`).
  - latest eval checkpoint in metrics: `eval_success_rate=0.62 @15000`.
- Success-definition mismatch evidence (`step_10000` rollout replay, `12` episodes, threshold `0.2`):
  - min-distance success `10/12` vs final-distance success `2/12`.
- Implication:
  - Current `eval_success_rate` (min-distance criterion) is optimistic for trajectory quality; final-distance-based metrics should be added for control-stability reporting.

## PointMass loss-plot root-cause check (2026-02-25)
- Added aligned plot to diagnose early val << train artifact:
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_loss_curve_aligned_paired_only.png`
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_loss_curve_aligned_summary.json`
- Key evidence:
  - burn-in already ran before online rows (`burnin_loss=0.5996`), so early low val is not "without training".
  - first finite val row at `550`; first finite train row at `1000` (timing mismatch in raw curve).
  - paired rows still show large negative gap, consistent with train/val distribution mismatch (dynamic replay vs tiny fixed holdout).

## 2026-02-25T11:41:53+08:00
### PointMass val-source alignment fix (user-requested bug fix)
- Code change in `scripts/online_pointmass_goal_diffuser.py`:
  - Added `--val_source {replay,warmup_holdout}` (default: `replay`).
  - Validation now defaults to sampling from the same online replay distribution as training.
  - Post-warmup scheduling now aligns first `val_loss` timestamp with the first post-warmup train cadence (`next_val = env_steps + val_every`).
  - Added logging fields: `val_source`, `num_episodes_in_val_replay`.

### Smoke evidence
- Replay default smoke run:
  - `runs/analysis/pointmass_val_source_align_smoke_20260225-113140/`
  - first finite `train_loss` and `val_loss` both at `env_steps=1020` (`8.1101`, `9.0637`), `val_source=replay`.
- Legacy compatibility smoke run:
  - `runs/analysis/pointmass_val_source_holdout_compat_smoke_20260225-113802/`
  - first finite pair at `env_steps=1020`, with classic negative gap retained (`val=0.6965`, `train=7.0313`, gap `-6.3348`) under explicit `--val_source warmup_holdout`.

### Current decision
- Treat replay-sourced validation as canonical default for ongoing pointmass online runs.
- Keep warmup-holdout mode only for explicit ablation/debug comparison.

## 2026-02-25T12:19:35+08:00
### PointMass unbounded-state + eval-distance filtered rerun (user /inject)
- Implemented experiment-setting knobs for the user suggestion:
  - `PointMass2D`: added `unbounded_state_space` + `state_sample_std` sampling mode.
  - `online_pointmass_goal_diffuser.py`: added `--state_limit`, `--unbounded_state_space`, `--state_sample_std`, `--eval_min_start_goal_dist`, `--eval_max_start_goal_dist`.
  - Evaluation (`eval_goal_mode=random`) now enforces start-goal distance in configured range.

### New run (updated setting)
- Run root:
  - `runs/analysis/pointmass_unbounded_evaldist_20260225-120517/eqm_k25_s010_unbounded_std1p0_evald05_15_seed0/`
- Config summary:
  - `unbounded_state_space=true`, `state_sample_std=1.0`
  - `eval_min_start_goal_dist=0.5`, `eval_max_start_goal_dist=1.5`
  - `val_source=replay`, `success_threshold=0.2`, EqM `K=25`, `S=0.1`

### Latest readout
- First completed eval checkpoint (`env_steps=3000`, `n_eval_episodes=30`):
  - `eval_success_rate=0.4333`
  - `eval_final_dist_mean=1.5202`
  - `eval_min_dist_mean=0.2780`
- Artifacts:
  - `.../config.json`
  - `.../metrics.jsonl`
  - `.../checkpoints/step_3000.pt`

### Operational note
- This run was stopped after the first completed eval checkpoint to return fast feedback and avoid waiting on long eval sweeps in-thread.

### Current decision
- Keep this protocol available as the new stricter debug setting (unbounded sampling + distance-filtered eval).
- If accepted, rerun full horizon and/or multi-seed under this exact config for stable comparison.

## 2026-02-25T12:22:41+08:00
### Task t-0010: parse long PointMass EqM run + baseline comparison

### Algorithm glossary
- `K := eqm_steps` = number of EqM refinement iterations per planning call.
- `S := eqm_step_size` = per-iteration EqM descent step size.
- `success_threshold := 0.2` = success distance criterion in this run.

### Parsed artifacts
- Target run:
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_20260225-1015/metrics.jsonl`
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_20260225-1015/config.json`
- Baseline reference:
  - `runs/analysis/pointmass_eqm_minchange_20260224-230943/eqm_vs_diffusion_compare_summary.json`
- New parse artifact:
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_20260225-1015/t0010_eval_summary.json`

### Checkpointed eval_success_rate trajectory
- `5000: 0.56`
- `10000: 0.68`
- `15000: 0.62`
- `20000: 0.60`
- `25000: 0.72` (best)
- `30000: 0.56`
- `35000: 0.54`
- `40000: 0.42` (latest eval)

### Final/terminal metrics (latest finite rows)
- Run target in config: `total_env_steps=50000`; last logged row: `env_steps=43000`.
- At `env_steps=43000`:
  - `train_loss=5.5720`, `val_loss=1.9780`, `train_val_gap=-3.5940`
  - `episode_final_dist=1.1407`, `episode_min_dist=0.1610`, `episode_success=1.0`
  - `num_episodes_in_replay=856`

### Comparison vs older EqM-vs-Diffusion summary
- Baseline (`eqm_vs_diffusion_compare_summary.json`):
  - EqM smoke-2k: `0.10`
  - EqM long-6k last: `0.15`
  - Diffusion long-6k last: `0.05`
- Current run deltas:
  - Latest eval `0.42`: `+0.32` vs EqM smoke-2k; `+0.27` vs EqM long-6k last; `+0.37` vs Diffusion long-6k last.
  - Best eval `0.72`: `+0.62` vs EqM smoke-2k; `+0.57` vs EqM long-6k last; `+0.67` vs Diffusion long-6k last.

### Current decision
- Treat this as a partial long-run readout with clear mid-run peak (`0.72 @ 25k`) and late-checkpoint drop (`0.42 @ 40k`), not a completed 50k endpoint.
- Continue comparisons using replay-sourced validation (`--val_source replay`) for aligned train/val distribution.


## 2026-02-25T04:27:37.608Z
### Objective
- Continue the `analysis/results-2026-02-24` validation cycle: finalize callback-ready experiment/eval/analyze scripts, run mini pipeline verification, then launch full validation runs with memory/handoff updates after each completion.

### Changes
- Tracked modified files: [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md), [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md), [ebm_online_rl/envs/pointmass2d.py](/root/ebm-online-rl-prototype/ebm_online_rl/envs/pointmass2d.py), [scripts/online_pointmass_goal_diffuser.py](/root/ebm-online-rl-prototype/scripts/online_pointmass_goal_diffuser.py).
- Diffstat snapshot: `4 files changed, 515 insertions(+), 13 deletions(-)`.
- Untracked artifacts/logs present: [MUJOCO_LOG.TXT](/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT), [gpt_pro_bundle_20260221.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221.zip), [gpt_pro_bundle_20260221b.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221b.zip), [gpt_pro_bundle_20260224_full.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full.zip), [gpt_pro_bundle_20260224_full/](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full/), [gpt_pro_handoff_bundle_20260220.zip](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip), [gpt_pro_handoff_bundle_20260220/](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220/), [memory/](/root/ebm-online-rl-prototype/memory/).
- Task counters: `pending=0`, `running=0`, `done=8`, `failed=0`, `blocked=2`, `canceled=0`.

### Evidence
- Command/state snapshot provided from `/root/ebm-online-rl-prototype` on branch `analysis/results-2026-02-24`.
- `git status --porcelain=v1` shows 4 tracked modified files and 8 untracked artifact/log entries.
- `git diff --stat` reports the 515/13 line delta with largest additions in handoff/memory docs plus pointmass/diffuser code edits.
- Plan tail indicates remaining flow around steps `8` to `15`: SAC/HER probe verification, eval/exp/analyze script alignment, mini pipeline run, mismatch fixes, and full callback-launched validations.
- Open blocker questions captured: exact attached plan content/path, exact `relay-long-task-callback` interface for this repo, and whether to recreate `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.

### Next steps
- Resolve the 3 open blocker questions first to clear the 2 blocked tasks.
- Execute remaining plan items in order starting with step 8 (`scripts/synthetic_maze2d_sac_her_probe.py` syntax/help/smoke verification), then steps 9-15.
- After each run completion, append evidence-backed results to [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) and refresh [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md).
- Decide whether untracked bundle/log artifacts should be retained, moved, ignored, or versioned before any commit/push.

## 2026-02-25T13:38:00+08:00
### PointMass dynamics ablation: first-order vs double-integrator (completed)

### Algorithm glossary
- `K := eqm_steps` = EqM refinement iterations per planning call.
- `S := eqm_step_size` = EqM per-iteration descent step size.
- `dt := double_integrator_dt` = integration timestep for second-order pointmass.

### Code updates
- `ebm_online_rl/online/planner.py`
  - waypoint mode now supports second-order state layout (`obs_dim == 2 * act_dim`) by deriving acceleration from predicted next velocity.
  - waypoint action is clipped to `[-action_scale, action_scale]`.
- `scripts/online_pointmass_goal_diffuser.py`
  - added `--dynamics_model`, `--double_integrator_dt`, `--initial_velocity_std`.
  - train/eval env constructors now receive dynamics settings.
  - eval random start-goal sampling now uses `sample_state` for start and env `goal_distance` filtering.
  - rollout min-distance initialization now uses env `goal_distance` (position-only semantics in DI).
- `ebm_online_rl/envs/pointmass2d.py`
  - default `double_integrator_dt` updated to `0.1` (from `1.0`) for stable scale under current episode/action limits.

### Run root
- `runs/analysis/pointmass_dynamics_ablation_20260225-131440/`

### Key artifacts
- Smokes:
  - `.../smoke/smoke_v3/first_order/`
  - `.../smoke/smoke_v3/double_integrator/`
- Matched-budget compare:
  - `.../smoke/compare_12k/first_order_k25_s010/` (stopped manually at `env_steps=6400`)
  - `.../smoke/compare_12k/double_integrator_k25_s010_dt010/` (`env_steps=6400` complete)
- Summary/plot:
  - `.../smoke/compare_12k/dynamics_ablation_summary.json`
  - `.../smoke/compare_12k/dynamics_ablation_compare.png`

### Shared-checkpoint comparison (`n_eval_episodes=30`)
- `env_steps=3000`:
  - first-order `eval_success_rate=0.1667`
  - double-integrator `eval_success_rate=0.0000`
- `env_steps=6000`:
  - first-order `eval_success_rate=0.2000`
  - double-integrator `eval_success_rate=0.0000`

### Current interpretation
- Under current EqM planner/training settings, adding velocity-state double-integrator dynamics did not improve pointmass success and underperformed first-order at matched checkpoints.
- The DI dynamics required stabilization fixes (`dt=0.1`, waypoint action clipping); after stabilization, control quality remained weaker than first-order in this run.

### Next discriminating step
- Keep everything fixed and test DI with non-zero initial velocity diversity:
  - `--initial_velocity_std 0.1` (and optionally `0.2`) at 12k+ budget, then compare against the same first-order baseline checkpoints.

## 2026-02-25T14:11:00+08:00
### Task t-0011: replay-val long-run parse vs t0010 reference (completed)

### Algorithm glossary
- `K := eqm_steps` = EqM refinement iterations per planning call.
- `S := eqm_step_size` = EqM per-iteration descent step size.

### Parsed artifacts
- Replay-val run:
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_replayval_20260225/metrics.jsonl`
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_replayval_20260225/config.json`
- Reference summary:
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_20260225-1015/t0010_eval_summary.json`
- New t-0011 output:
  - `runs/analysis/pointmass_eqm_minchange_20260225-1002/eqm_best_k25_s010_long50k_s020_run_replayval_20260225/t0011_replayval_analysis_summary.json`

### Current readout
- Replay-val run status:
  - `rows_total=389`, `max_env_steps=19950` (target `50000` not reached)
  - eval checkpoints only at `5000`, `10000`, `15000`
- Train/val trend at eval checkpoints:
  - `5000`: train `9.6899`, val `9.9320`
  - `10000`: train `8.3194`, val `8.8747`
  - `15000`: train `7.7966`, val `7.3637`
- Latest finite paired losses:
  - `train=7.1341`, `val=7.3922` at `env_steps=19500`
- Eval success:
  - latest `0.64 @15000`
  - best `0.82 @5000`

### Post-30k loss question
- Cannot determine from this run artifact: there are no finite train/val rows at or beyond `30000` (`max_env_steps=19950`).

### Comparison vs t0010
- Reference (`t0010`) had deeper horizon (`max_env_steps=43000`) with:
  - latest eval `0.42 @40000`
  - best eval `0.72 @25000`
- Replay-val deltas vs reference (non-step-matched, directional only):
  - latest eval delta: `+0.22` (`0.64 - 0.42`)
  - best eval delta: `+0.10` (`0.82 - 0.72`)

### Decision
- Treat replay-val run as a partial early/mid-horizon readout with favorable checkpoint success but insufficient depth to answer post-30k loss behavior.
- Next discriminating action is to continue/redo replay-val to `>=30000` and reassess loss trend there.

## ${ts}
### t-0011 correction note
- Replay-val parse correction: finite paired train/val rows are `38` (`t0011_replayval_analysis_summary.json`), superseding the earlier `39` handoff line.

## 2026-02-25T14:11:55+08:00
### t-0011 correction note (timestamp fix)
- Canonical correction time: `2026-02-25T14:11:55+08:00`; replay-val finite paired train/val rows are `38`.


## 2026-02-25T06:12:21.524Z
### Objective
- Advance the `analysis/results-2026-02-24` validation cycle for Maze2D/PointMass online RL experiments, with callback-ready experiment scripts and continuously updated handoff memory for future agents.

### Changes
- Repository remains on branch `analysis/results-2026-02-24` with active in-progress working tree.
- Expanded experiment continuity docs: [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) (`+414`) and [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md) (`+257`).
- Code changes were made in [ebm_online_rl/envs/pointmass2d.py](/root/ebm-online-rl-prototype/ebm_online_rl/envs/pointmass2d.py) (substantial), [ebm_online_rl/online/planner.py](/root/ebm-online-rl-prototype/ebm_online_rl/online/planner.py) (targeted), and [scripts/online_pointmass_goal_diffuser.py](/root/ebm-online-rl-prototype/scripts/online_pointmass_goal_diffuser.py) (substantial).
- New untracked artifacts/bundles are present: [MUJOCO_LOG.TXT](/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT), [gpt_pro_bundle_20260221.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221.zip), [gpt_pro_bundle_20260221b.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260221b.zip), [gpt_pro_bundle_20260224_full.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full.zip), [gpt_pro_bundle_20260224_full/](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full/), [gpt_pro_handoff_bundle_20260220.zip](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220.zip), [gpt_pro_handoff_bundle_20260220/](/root/ebm-online-rl-prototype/gpt_pro_handoff_bundle_20260220/), and [memory/](/root/ebm-online-rl-prototype/memory/).
- Task board snapshot: `pending=0 running=0 done=9 failed=0 blocked=2 canceled=0`.

### Evidence
- Workdir/repo root: `/root/ebm-online-rl-prototype`.
- Command context: `git status --porcelain=v1` reported modified tracked files and untracked artifacts exactly as listed above.
- Command context: `git diff --stat` reported:
- `HANDOFF_LOG.md | 414`
- `docs/WORKING_MEMORY.md | 257`
- `ebm_online_rl/envs/pointmass2d.py | 86`
- `ebm_online_rl/online/planner.py | 16`
- `scripts/online_pointmass_goal_diffuser.py | 171`
- `5 files changed, 919 insertions(+), 25 deletions(-)`
- Last recorded plan tail indicates remaining pipeline work on probe verification, experiment scripts, mini end-to-end callback run, mismatch fixes, and full validation launches.

### Next steps
- Execute remaining plan items 8-15 in order, starting with SAC/HER probe syntax/help/smoke verification and ending with full validation experiment launches plus per-run memory updates.
- Complete/validate these scripts for callback-ready outputs and robust aggregation: `eval_synthetic_maze2d_sac_her_probe.py`, `eval_synth_maze2d_checkpoint_prefix.py`, `exp_replan_horizon_sweep.py`, `exp_swap_matrix_maze2d.py`, `analyze_posterior_diversity.py`.
- Run one short end-to-end mini pipeline across experiment families, fix schema/analysis mismatches, then launch full runs one-by-one.
- Resolve blocked clarifications before full rollout: exact attached plan content/path, exact `relay-long-task-callback` command/interface expected in this repo, and whether to recreate `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.

## 2026-02-25T16:10:13+08:00
### PointMass DI vs Maze2D: ranked-cause ablation continuation

### What changed
- Added `--action_limit` to `scripts/online_pointmass_goal_diffuser.py` and wired it through train/eval env construction.
- Continued DI ablations in `runs/analysis/pointmass_di_ranked_ablation_20260225-150732/`:
  - `r3_di_action_posgoal_actionlimit1p0`
  - `r4_di_action_posgoal_actionlimit1p0_damp1p5_vclip2p0`
  - `r5_di_longh96_ep192_action01` (partial)
- Wrote consolidated summary artifacts:
  - `ranked_ablation_summary_20260225-extended.json`
  - `ranked_ablation_summary_20260225-final.json`

### Current ablation readout (DI)
- `r1` (action-mode baseline, action_limit=0.1): success `0.0333 @3k`, `0.0333 @6k`.
- `r2` (+ replay-goal position-only): no change (`0.0333 @3k`, `0.0333 @6k`).
- `r3` (+ action_limit=1.0, no damping): success unchanged at `3k` (`0.0333`), severe overshoot (`eval_final_dist_mean=7.0017`).
- `r4` (+ action_limit=1.0 + damping=1.5 + vclip=2.0): `0.10 @3k` then `0.0333 @6k` (temporary gain, no sustained improvement).
- `r5` (long horizon only: `h96`, `ep192`, action_limit=0.1): partial to `2880`; logged episodes remained `episode_success=0.0` and showed drift/rebound.

### Maze2D evidence used for causal ranking
- `.../diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/progress_metrics.csv` (last row):
  - `rollout_goal_success_rate_h64=0.0`
  - `rollout_goal_success_rate_h128=0.5`
  - `rollout_goal_success_rate_h192=0.75`
  - `rollout_goal_success_rate_h256=0.8333`
- `.../eqm_vs_existing_baselines_20260225.json`: diffuser reference `h256=0.8333`.

### Ranked likely causes (current evidence)
1. **Rebound/terminal-stability mismatch under DI control**: repeated min-distance hits with poor final-distance settling; increasing authority alone worsens this.
2. **Reachability + horizon coupling**: Maze2D success is strongly horizon-dependent (`h64` fails, `h192+` succeeds); current PointMass DI protocol is likely under-horizoned and under-structured.
3. **Actuator/physics mismatch vs Maze2D**: PointMass DI integration and limits do not yet reproduce Maze2D’s effective damping/friction and control smoothness.
4. **Goal semantics mismatch (velocity components)**: position-only replay goals alone had negligible effect (r2 ~= r1).

### Current decision
- Keep `r4` as the best-stabilized DI variant tested so far, but acknowledge no sustained gain at `6k`.
- Treat long-horizon-only (`r5`) as a negative partial; it needs paired stabilization (damping/authority) to be meaningful.

### Next discriminating run
- Run long-horizon **with** stabilized high-authority control (combine successful pieces rather than single-factor):
  - `horizon=96`, `episode_len=192`, `action_limit=1.0`, `double_integrator_velocity_damping=1.5`, `double_integrator_velocity_clip=2.0`, `planner_control_mode=action`, `replay_goal_position_only`.

## 2026-02-25T16:28:58+08:00
### PointMass GPT-PRO handoff bundle + commit push
- Added focused handoff report: `GPT_PRO_POINTMASS_DEBUG_HANDOFF_20260225.md`.
- Built upload-ready bundle (PointMass-only relevant artifacts):
  - `/root/.codex-discord-relay/uploads/discord_1472061022239195304_thread_1473203408256368795/pointmass_debug_bundle_20260225.zip` (`43K`).
- Pushed implementation/docs commit:
  - `3d3388c` on `origin/analysis/results-2026-02-24`.

### Bundle scope
- Included: `scripts/online_pointmass_goal_diffuser.py`, `ebm_online_rl/envs/pointmass2d.py`, `ebm_online_rl/online/planner.py`, PointMass DI ablation `r1..r5` configs/metrics/summaries, and minimal Maze2D reference files used in causal comparison.
- Excluded: model checkpoints (`.pt`) and unrelated large artifacts.


## 2026-02-26T09:05:29.951Z
### Objective
- Preserve a clear workspace and task-state snapshot for the next agent on branch `analysis/results-2026-02-24`.
- Record what changed in the log and which artifacts are present but untracked.

### Changes
- `HANDOFF_LOG.md` was modified with 13 inserted lines (`git diff --stat`).
- Current tracked change: `M HANDOFF_LOG.md`.
- Current untracked artifact: `MUJOCO_LOG.TXT`.
- Current untracked artifact: `docs/EQM_RESEARCH_FINDINGS.md`.
- Current untracked artifact: `eqm_maze2d_followups_20260226.zip`.
- Current untracked artifact: `gpt_pro_bundle_20260221.zip`.
- Current untracked artifact: `gpt_pro_bundle_20260221b.zip`.
- Current untracked artifact: `gpt_pro_bundle_20260224_full.zip`.
- Current untracked artifact: `gpt_pro_bundle_20260224_full/`.
- Current untracked artifact: `gpt_pro_handoff_bundle_20260220.zip`.
- Current untracked artifact: `gpt_pro_handoff_bundle_20260220/`.
- Current untracked artifact: `memory/`.
- Current untracked artifact: `scripts/viz_maze2d_waypoint_exec_trajectories.py`.
- Task snapshot: `pending=1`, `running=0`, `done=2`, `failed=0`, `blocked=0`, `canceled=0`.

### Evidence
- Command: `git status --porcelain=v1` (repo: `/root/ebm-online-rl-prototype`, branch: `analysis/results-2026-02-24`).
- Command: `git diff --stat` produced `HANDOFF_LOG.md | 13 +++++++++++++`.
- Path: `/root/ebm-online-rl-prototype/HANDOFF_LOG.md`.
- Path: `/root/ebm-online-rl-prototype/docs/EQM_RESEARCH_FINDINGS.md`.
- Path: `/root/ebm-online-rl-prototype/scripts/viz_maze2d_waypoint_exec_trajectories.py`.
- Path: `/root/ebm-online-rl-prototype/memory/`.

### Next steps
- Resolve the single remaining pending task and document the outcome in `HANDOFF_LOG.md`.
- Decide which untracked artifacts should be committed (likely `docs/`, `scripts/`, `memory/`) versus retained as external bundles/logs.
- Re-run `git status --porcelain=v1` after triage to confirm the intended working set before the next commit.


## 2026-02-26T09:06:55.604Z
### Objective
- Preserve continuity for future agents on `analysis/results-2026-02-24` by capturing current repo state, documentation updates, and task queue status.
- Record that the current run queue is drained: `pending=0`, `running=0`, `done=3`, `failed=0`, `blocked=0`, `canceled=0`.

### Changes
- Updated `HANDOFF_LOG.md` (`+48` lines).
- Updated `docs/WORKING_MEMORY.md` (`+35` lines).
- Untracked artifacts currently present: `MUJOCO_LOG.TXT`, `docs/EQM_RESEARCH_FINDINGS.md`, `eqm_maze2d_followups_20260226.zip`, `gpt_pro_bundle_20260221.zip`, `gpt_pro_bundle_20260221b.zip`, `gpt_pro_bundle_20260224_full.zip`, `gpt_pro_bundle_20260224_full/`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`, `scripts/viz_maze2d_waypoint_exec_trajectories.py`.

### Evidence
- Path: `/root/ebm-online-rl-prototype`
- Branch: `analysis/results-2026-02-24`
- Command: `git status --porcelain=v1`
- Command: `git diff --stat`
- Diff summary: `HANDOFF_LOG.md | 48 insertions`, `docs/WORKING_MEMORY.md | 35 insertions`, `2 files changed, 83 insertions(+)`

### Next steps
- Review `HANDOFF_LOG.md` and `docs/WORKING_MEMORY.md` for final consistency, then commit if accurate.
- Triage untracked zips/directories into keep/archive/delete (or update `.gitignore`) before the next commit.
- Continue follow-up analysis from `docs/WORKING_MEMORY.md`, with focus on `docs/EQM_RESEARCH_FINDINGS.md` and `scripts/viz_maze2d_waypoint_exec_trajectories.py`.


## 2026-02-26T09:07:57.368Z
### Objective
- Capture a clean handoff snapshot for future agents after completing the current analysis cycle on branch `analysis/results-2026-02-24`.
- Record documentation updates and newly generated result artifacts for continuity.

### Changes
- Updated `HANDOFF_LOG.md` (`+71` lines).
- Updated `docs/WORKING_MEMORY.md` (`+58` lines).
- Current worktree includes new untracked artifacts:
- `MUJOCO_LOG.TXT`
- `docs/EQM_RESEARCH_FINDINGS.md`
- `eqm_maze2d_followups_20260226.zip`
- `gpt_pro_bundle_20260221.zip`
- `gpt_pro_bundle_20260221b.zip`
- `gpt_pro_bundle_20260224_full.zip`
- `gpt_pro_bundle_20260224_full/`
- `gpt_pro_handoff_bundle_20260220.zip`
- `gpt_pro_handoff_bundle_20260220/`
- `memory/`
- `scripts/viz_maze2d_waypoint_exec_trajectories.py`
- Task tracker snapshot: `pending=0 running=0 done=4 failed=0 blocked=0 canceled=0`.

### Evidence
- Repo root: `/root/ebm-online-rl-prototype`
- Branch: `analysis/results-2026-02-24`
- Command: `git status --porcelain=v1`
- Result highlights:
- `M HANDOFF_LOG.md`
- `M docs/WORKING_MEMORY.md`
- `??` entries for artifacts listed above
- Command: `git diff --stat`
- Result: `HANDOFF_LOG.md | 71 +`, `docs/WORKING_MEMORY.md | 58 +`, total `129 insertions`.

### Next steps
- Review and commit `HANDOFF_LOG.md` and `docs/WORKING_MEMORY.md` together as the canonical continuity update.
- Decide which large zip/folder artifacts should be retained in-repo vs moved to external storage/releases.
- Validate whether `scripts/viz_maze2d_waypoint_exec_trajectories.py` should be promoted into the tracked experiment workflow with brief usage notes in docs.


## 2026-02-26T09:27:20.972Z
### Objective
- Capture current experiment/repo state on `analysis/results-2026-02-24` so the next agent can resume immediately with no active tasks.

### Changes
- Updated [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) (`+110` lines).
- Updated [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md) (`+97` lines).
- Left new untracked artifacts in repo root, including `MUJOCO_LOG.TXT`, multiple `gpt_pro*_bundle*.zip` files, `eqm_maze2d_followups_20260226.zip`, and `memory/`.
- Task queue snapshot: `pending=0`, `running=0`, `done=5`, `failed=0`, `blocked=0`, `canceled=0`.

### Evidence
- Repo root: `/root/ebm-online-rl-prototype`
- Branch check and workspace state from `git status --porcelain=v1`:
- Modified: [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md), [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md)
- Untracked: `MUJOCO_LOG.TXT`, `eqm_maze2d_followups_20260226.zip`, `gpt_pro_bundle_20260221.zip`, `gpt_pro_bundle_20260221b.zip`, `gpt_pro_bundle_20260224_full.zip`, `gpt_pro_bundle_20260224_full/`, `gpt_pro_handoff_bundle_20260220.zip`, `gpt_pro_handoff_bundle_20260220/`, `memory/`
- Change summary from `git diff --stat`:
- [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) `| 110 +++++++++++++++++++++++++++++++++++++++++++++++++`
- [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md) `| 97 +++++++++++++++++++++++++++++++++++++++++++`
- Total: `2 files changed, 207 insertions(+)`

### Next steps
- Review and commit the documentation updates in [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) and [docs/WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md).
- Decide which untracked bundles/logs are archival outputs vs. should be ignored/cleaned before the next coding cycle.
- If handoff packaging is complete, tag or note the canonical bundle to avoid ambiguity across multiple similarly named zip artifacts.

## 2026-02-27T10:59:55+08:00
### Maze2D scaffold probe implementation (EqM + Diffuser)
- Implemented shared scaffold helpers for `[act|obs]` trajectories:
  - `ebm_online_rl/online/scaffold.py` (`build_anchor_times`, `extract_anchor_xy`, `apply_pos_only_anchors_`).
- Integrated scaffold insertion into Maze2D checkpoint sampling path:
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - EqM: insert anchors after endpoint conditioning during refinement; optional eta anneal (`eqm_eta_start/end`).
  - Diffuser: custom reverse-loop wrapper for scaffold anchors + optional `diff_steps` override.
- Added unified evaluation script:
  - `scripts/maze2d_scaffold_probe.py`
  - supports `--algo {eqm,diffuser}`, `--scaffold {none,insert_mid}`, smoothness metrics, and rollout metrics.
- Added implementation-risk notes:
  - `docs/IMPLEMENTATION_NOTES_SCAFFOLD_EQM_VS_DIFFUSER.md`.

### Smoke evidence
- EqM scaffold smoke:
  - `runs/analysis/maze2d_scaffold_probe_smoke/eqm_scaffold_smoke/metrics.json`
- Diffuser scaffold smoke:
  - `runs/analysis/maze2d_scaffold_probe_smoke/diffuser_scaffold_smoke/metrics.json`
- Both paths executed end-to-end under diffuser venv; runtime import and cross-module class-detection bugs were fixed.

### Current next discriminating step
- Run umaze baseline vs scaffold (compute-matched) for both algorithms using `scripts/maze2d_scaffold_probe.py`, then escalate to medium/large only if smoke-level deltas and logs are coherent.
## 2026-02-27T15:50:00+08:00
### Maze2D scaffold-first training: live progress snapshot
- Run: `runs/analysis/maze2d_scaffold_training_20260227-153430/eqm_umaze_h64_scaffold_insertmid_fastmon_seed0/`
- Early trend is favorable:
  - `rollout_goal_success_rate_h256`: `0.00 @1k -> 0.0625 @2k -> 0.1875 @3k -> 0.5625 @4k -> 0.875 @6k`
  - train/val losses decreased from ~`0.55/0.56` to ~`0.067/0.069` by `step=6000`.
- Online transition accumulation so far:
  - round1 `2979`, round2 `2938`, cumulative `5917` (20K gate not yet reached).
- Runtime status at snapshot:
  - no active process for this run found; latest files are `progress_metrics.csv` + `online_collection.csv` through `step=6000`/round2 collection.

### Decision
- Treat as encouraging partial scaffold evidence, but not a full gate decision.
- Next action: relaunch/continue this exact run protocol under durable background logging to reach `>=20K` online transitions and apply the failure gate there.

## 2026-02-27T16:30:19+08:00
### Maze2D default-policy alignment update (self-improvement-first)

### What changed
- Updated `scripts/synthetic_maze2d_diffuser_probe.py` defaults to prefer online self-improvement over offline warmup:
  - `n_episodes=0`, `train_steps=0`
  - `online_self_improve=true`, `online_rounds=8`
  - `online_collect_transition_budget_per_round=3000`
  - `num_eval_queries=8`, `query_batch_size=2` (16 eval trajectories)
  - `eval_goal_every=1000`
- Added bootstrap replay path for `n_episodes=0`:
  - if no replay import is provided, initial replay is collected with current policy before first training phase.
  - geometric start-goal sampling now falls back to maze/env sampling when replay-based spans are unavailable.

### Verification evidence
- Runtime smoke (no offline init path):
  - `runs/analysis/tmp_self_improve_default_smoke_20260227/`
  - Observed sequence: bootstrap replay -> online collection -> online training -> progress/summary artifacts written.
- Parser/help verified under diffuser venv with project PYTHONPATH.

### Current implication
- New default behavior is now directly compatible with “self-improvement from scratch” experiments.
- To maintain strict compute matching, launcher scripts that hardcode old values still need explicit harmonization (same collection budget and eval sample settings).

## 2026-02-27T16:49:09+08:00
### EqNet self-improvement alignment policy (current default)
- Canonical profile now enforced in training scripts: `alignment_profile=eqnet_self_improve_v1`.
- Enforced defaults (unless explicit `--alignment_profile legacy_offline`):
  - `n_episodes=0`
  - `train_steps=0`
  - `online_self_improve=true`, `online_rounds>0`, `online_collect_transition_budget_per_round>0`
  - `num_eval_queries * query_batch_size = 16`
- Coverage:
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - `scripts/synthetic_maze2d_gcbc_her_probe.py`
  - `scripts/synthetic_maze2d_sac_her_probe.py`
- Stability updates for no-offline bootstrap path:
  - GCBC/SAC now support replay-empty start-goal fallback and bootstrap collection from current policy.
  - `online_min_accepted_episode_len` default set to `8` in GCBC/SAC to avoid short-episode rejection loops during bootstrap.
- Launcher alignment:
  - `scripts/exp_swap_matrix_maze2d.py` now emits aligned defaults (`n_episodes=0`, `train_steps=0`, budgeted online collection, 16 eval trajectories/checkpoint).
- Runtime evidence:
  - `runs/analysis/tmp_alignment_smoke_gcbc_20260227_v4/` (pass)
  - `runs/analysis/tmp_alignment_smoke_sac_20260227_v1/` (pass)
  - alignment guard fail-fast confirmed in diffuser script for legacy offline args.
- Policy note for future agents:
  - `docs/EQNET_ALIGNMENT_POLICY.md`

## 2026-02-27T17:08:47+08:00
### Task t-0012: latest scaffold run completion + 20K gate assessment

### Run analyzed
- `runs/analysis/maze2d_scaffold_training_20260227-155628/eqm_umaze_h64_scaffold_insertmid_fastmon_seed0/`
- Status: finished (`train_steps_total=20000`, `checkpoint_step20000.pt`, `checkpoint_last.pt`).

### Loss and success progression
- Loss (`metrics.csv`):
  - `step=1000`: train `0.0783`, val `0.0878`
  - `step=6000`: train `0.0703`, val `0.0720`
  - `step=14000`: train `0.0685`, val `0.0649`
  - `step=16000`: train `0.0666`, val `0.0648`
  - `step=20000`: train `0.0669`, val `0.0653`
- `rollout_goal_success_rate_h256` (`progress_metrics.csv`):
  - early rise: `0.125 @1k`, `0.4375 @4k`, `0.8125 @6k`
  - peak: `1.000 @14k`
  - gate point: `0.250 @16k`
  - final: `0.750 @20k`

### 20K online-transition gate
- `online_collection.csv` transitions per round:
  - `[2984, 3000, 2975, 3000, 2948, 3000, 2941, 2955]`
- Cumulative online transitions:
  - `20848` at round 7 (`step_before_retrain=16000`) -> first `>=20K` gate crossing
  - `23803` by run end
- Gate decision:
  - **Fail (recommend stop + diagnose)**: by >=20K online transitions, policy success did not show stable improvement (`0.25 @16k`, final `0.75 @20k` still below `0.8125 @6k`) while losses were mostly flattened in late training.

### Next discriminating step
- Run a compute-matched paired comparison under `alignment_profile=eqnet_self_improve_v1`:
  - EqM baseline `--scaffold none`
  - EqM scaffold `--scaffold insert_mid`
- Compare at `6k/12k/20k` on `rollout_goal_success_rate_h256`, with the same online transition budget and eval cardinality.


## 2026-02-27T09:09:28.839Z
### Objective
- Stabilize and prepare the Maze2D EQM validation workflow on branch `analysis/results-2026-02-24`, including callback-ready experiment scripts, checkpoint evaluators, and downstream analysis hooks.
- Preserve cross-session continuity with a clear state snapshot (10 tasks done, 2 blocked) and explicit unblock questions.

### Changes
- Updated core memory/research docs: [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md), [WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md), [EQM_RESEARCH_FINDINGS.md](/root/ebm-online-rl-prototype/docs/EQM_RESEARCH_FINDINGS.md).
- Added scaffold/alignment artifacts: [EQNET_ALIGNMENT_POLICY.md](/root/ebm-online-rl-prototype/docs/EQNET_ALIGNMENT_POLICY.md), [IMPLEMENTATION_NOTES_SCAFFOLD_EQM_VS_DIFFUSER.md](/root/ebm-online-rl-prototype/docs/IMPLEMENTATION_NOTES_SCAFFOLD_EQM_VS_DIFFUSER.md), [scaffold.py](/root/ebm-online-rl-prototype/ebm_online_rl/online/scaffold.py), [__init__.py](/root/ebm-online-rl-prototype/ebm_online_rl/online/__init__.py).
- Expanded probe/eval/experiment scripts for schema/checkpoint robustness and callback metadata: [synthetic_maze2d_diffuser_probe.py](/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py), [synthetic_maze2d_gcbc_her_probe.py](/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_gcbc_her_probe.py), [synthetic_maze2d_sac_her_probe.py](/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_sac_her_probe.py), [eval_synth_maze2d_checkpoint_prefix.py](/root/ebm-online-rl-prototype/scripts/eval_synth_maze2d_checkpoint_prefix.py), [eval_synth_maze2d_checkpoint_goal_suffix.py](/root/ebm-online-rl-prototype/scripts/eval_synth_maze2d_checkpoint_goal_suffix.py), [exp_replan_horizon_sweep.py](/root/ebm-online-rl-prototype/scripts/exp_replan_horizon_sweep.py), [exp_swap_matrix_maze2d.py](/root/ebm-online-rl-prototype/scripts/exp_swap_matrix_maze2d.py), [maze2d_eqm_utils.py](/root/ebm-online-rl-prototype/scripts/maze2d_eqm_utils.py).
- Added new analysis/viz/probe assets and bundles, including [analysis_blockwise_locality_compare.py](/root/ebm-online-rl-prototype/scripts/analysis_blockwise_locality_compare.py), [analysis_boundary_influence_maze2d.py](/root/ebm-online-rl-prototype/scripts/analysis_boundary_influence_maze2d.py), [analysis_locality_vs_noise_maze2d.py](/root/ebm-online-rl-prototype/scripts/analysis_locality_vs_noise_maze2d.py), [analysis_subblock_locality_compare.py](/root/ebm-online-rl-prototype/scripts/analysis_subblock_locality_compare.py), [maze2d_scaffold_probe.py](/root/ebm-online-rl-prototype/scripts/maze2d_scaffold_probe.py), [smoothness_maze2d_compare.py](/root/ebm-online-rl-prototype/scripts/smoothness_maze2d_compare.py), [viz_maze2d_diffuser_compare.py](/root/ebm-online-rl-prototype/scripts/viz_maze2d_diffuser_compare.py), [viz_scaffold_segments.py](/root/ebm-online-rl-prototype/scripts/viz_scaffold_segments.py), and [memory/](/root/ebm-online-rl-prototype/memory/).
- Diff snapshot from provided status: 12 tracked files changed, 2130 insertions, 146 deletions; multiple untracked artifacts/bundles present.

### Evidence
- Repo/workdir: `/root/ebm-online-rl-prototype`
- Branch: `analysis/results-2026-02-24`
- Commands used for state snapshot: `git status --porcelain=v1`, `git diff --stat`
- Additional artifacts observed: [MUJOCO_LOG.TXT](/root/ebm-online-rl-prototype/MUJOCO_LOG.TXT), [eqm_maze2d_followups_20260226.zip](/root/ebm-online-rl-prototype/eqm_maze2d_followups_20260226.zip), [gpt_pro_bundle_20260224_full.zip](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full.zip), [gpt_pro_bundle_20260224_full/](/root/ebm-online-rl-prototype/gpt_pro_bundle_20260224_full/).
- Task counts at snapshot: `pending=0 running=0 done=10 failed=0 blocked=2 canceled=0`
- Plan tail indicates remaining work: mini end-to-end callback pipeline, mismatch fixes, then full validation launches plus memory/handoff updates.

### Next steps
- Unblock two open dependencies by confirming: exact attached plan source/path, exact `relay-long-task-callback` command/interface for this repo, and whether [HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt](/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt) must be regenerated.
- Run syntax/help/smoke verification for [synthetic_maze2d_sac_her_probe.py](/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_sac_her_probe.py) and newly added analysis scripts.
- Execute one short end-to-end mini pipeline per experiment family with callback auto-analysis; fix any schema/analysis contract mismatches found.
- Launch full validation experiments sequentially via callback workflow and append evidence-backed updates after each completion to [HANDOFF_LOG.md](/root/ebm-online-rl-prototype/HANDOFF_LOG.md) and [WORKING_MEMORY.md](/root/ebm-online-rl-prototype/docs/WORKING_MEMORY.md).
- Triage untracked bundles/logs for retention policy before commit to avoid accidental artifact bloat.

## 2026-02-27T17:53:44+08:00
### Objective
- Reduce statistical uncertainty in scaffold checkpoint evaluation by increasing rollout sample count from `16` to `50` trajectories/checkpoint and reassess `h256` success trend.

### Changes
- Completed GPU re-evaluation for scaffold checkpoints:
  - `step4000`, `step8000`, `step12000`, `step16000`, `step20000`.
- Wrote consolidated summary artifacts:
  - `runs/analysis/maze2d_scaffold_reeval50_summary_20260227-175317.json`
  - `runs/analysis/maze2d_scaffold_reeval50_summary_20260227-175317.csv`

### Evidence
- Re-eval command family: `scripts/maze2d_scaffold_probe.py --algo eqm --device cuda:0 --num_eval_queries 25 --query_batch_size 2 --scaffold insert_mid ...`
- Re-eval `rollout_goal_success_rate_h256` (`new50`):
  - `4000: 0.68`, `8000: 0.96`, `12000: 0.92`, `16000: 0.88`, `20000: 0.88`
- Prior logged values at same steps from `progress_metrics.csv` (`old16`):
  - `0.4375, 0.5625, 0.8750, 0.2500, 0.7500`
- High-variance indicator: `step16000` moved from `0.25 (4/16)` to `0.88 (44/50)` under higher-cardinality eval.

### Current interpretation
- The 16-trajectory checkpoint metric is too noisy for gate decisions in this run.
- With 50 trajectories/checkpoint, scaffold performance appears consistently strong after `8k` (`h256` success ~`0.88-0.96`).
- The earlier stop+diagnose recommendation based on the `16k` dip is likely a sampling artifact.

### Next steps
- Complete matched no-scaffold re-evaluation with the same `50` trajectories/checkpoint protocol and compare scaffold vs no-scaffold at aligned steps (`4k/8k/12k/16k/20k`) before final ablation decision.

## 2026-02-27T18:25:45+08:00
### Objective
- Complete `t-0013`: compare latest EqM `scaffold=none` run against scaffold insert-mid run at `6k/12k/20k`, apply the `>=20K` online-transition gate, and record the result.

### Changes
- Added pairwise comparison artifact from source CSVs:
  - `runs/analysis/eqm_scaffold_none_vs_insertmid_t0013_gate_20260227-182527.json`
- Appended full evidence entry to `HANDOFF_LOG.md` for `task_id=t-0013`.

### Evidence
- none run (latest): `runs/analysis/maze2d_scaffold_training_20260227-172159/eqm_umaze_h64_scaffold_none_fastmon_seed0/`
- scaffold run: `runs/analysis/maze2d_scaffold_training_20260227-155628/eqm_umaze_h64_scaffold_insertmid_fastmon_seed0/`
- Stepwise metrics (`train/val/success_h256`):
  - `6000`: none `0.070325/0.072026/0.9375`, scaffold `0.070335/0.071951/0.8125`
  - `12000`: none `0.068316/0.066133/0.6875`, scaffold `0.070547/0.065665/0.8750`
  - `20000`: none `0.067595/0.063093/0.8750`, scaffold `0.066887/0.065321/0.7500`
- Gate (`>=20K` online transitions):
  - none: round `7`, `step_before_retrain=16000`, cumulative `20744`, decision `continue`
  - scaffold: round `7`, `step_before_retrain=16000`, cumulative `20848`, decision `stop+diagnose`

### Current interpretation
- In this EqM pair, `scaffold=none` is the stronger reference at `6k` and `20k`; scaffold is better only at `12k`.
- Under the current gate rule, non-scaffold remains viable while scaffold insert-mid does not.

### Next steps
- Launch matched diffuser `scaffold=none` vs `scaffold=insert_mid` training/eval under the same alignment profile and evaluate with higher-cardinality checkpoint rollouts before any scaffold adoption decision.


## 2026-02-27T10:26:29.968Z
### Objective
- Continue the Maze2D EQM validation/analysis stream on `analysis/results-2026-02-24`, with callback-ready experiment scripts, checkpoint-compatible eval tooling, and up-to-date research memory/handoff docs.

### Changes
- Updated handoff/research memory docs: `HANDOFF_LOG.md`, `docs/WORKING_MEMORY.md`, `docs/EQM_RESEARCH_FINDINGS.md`.
- Added scaffold/alignment docs and implementation notes: `docs/EQNET_ALIGNMENT_POLICY.md`, `docs/IMPLEMENTATION_NOTES_SCAFFOLD_EQM_VS_DIFFUSER.md`.
- Extended online module surface with scaffold support: `ebm_online_rl/online/__init__.py`, `ebm_online_rl/online/scaffold.py`.
- Revised core experiment/eval/probe scripts for schema/checkpoint contract alignment and callback-oriented outputs: `scripts/synthetic_maze2d_diffuser_probe.py`, `scripts/synthetic_maze2d_gcbc_her_probe.py`, `scripts/synthetic_maze2d_sac_her_probe.py`, `scripts/eval_synth_maze2d_checkpoint_prefix.py`, `scripts/eval_synth_maze2d_checkpoint_goal_suffix.py`, `scripts/exp_replan_horizon_sweep.py`, `scripts/exp_swap_matrix_maze2d.py`, `scripts/maze2d_eqm_utils.py`.
- Added new analysis/viz/probe utilities: `scripts/analysis_blockwise_locality_compare.py`, `scripts/analysis_boundary_influence_maze2d.py`, `scripts/analysis_locality_vs_noise_maze2d.py`, `scripts/analysis_subblock_locality_compare.py`, `scripts/smoothness_maze2d_compare.py`, `scripts/viz_maze2d_diffuser_compare.py`, `scripts/viz_scaffold_segments.py`, `scripts/maze2d_scaffold_probe.py`.
- Snapshot task ledger: `done=11`, `blocked=2`, `pending=0`, `running=0`, `failed=0`, `canceled=0`.

### Evidence
- Repo/context: `/root/ebm-online-rl-prototype`, branch `analysis/results-2026-02-24`.
- Command snapshot used: `git status --porcelain=v1`.
- Command snapshot used: `git diff --stat`.
- Git diff stat snapshot: `12 files changed, 2336 insertions(+), 146 deletions(-)` on tracked files.
- Untracked artifacts present include `MUJOCO_LOG.TXT`, `memory/`, multiple GPT-Pro bundle zips/directories, EQM followup bundles, new docs, and new analysis/probe scripts.
- Plan tail indicates remaining flow through analyzer compatibility checks, mini end-to-end validation, then full sequential validation launch with memory/handoff updates per run.

### Next steps
- Unblock open dependency 1: provide exact attached plan content or file path so task mapping is explicit.
- Unblock open dependency 2: confirm the required `relay-long-task-callback` command/interface for this repo.
- Confirm whether `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt` should be regenerated in this cycle.
- After unblock, execute remaining pipeline validation tasks (including mismatch fixes if found), then launch full validation experiments one-by-one and append results to `docs/WORKING_MEMORY.md` and `HANDOFF_LOG.md` after each completion.

## 2026-02-27T21:40:18+08:00
### Scaffold vs non-scaffold re-eval (50 trajectories/checkpoint) — completed
- Consolidated artifact:
  - `runs/analysis/scaffold_vs_nonscaffold_reeval50_summary_20260227-213943.json`
  - `runs/analysis/scaffold_vs_nonscaffold_reeval50_summary_20260227-213943.csv`

- EqM `h256` success (`none` vs `insert_mid`):
  - `4000`: `0.60` vs `0.68`
  - `8000`: `0.94` vs `0.96`
  - `12000`: `0.90` vs `0.92`
  - `16000`: `0.96` vs `0.88`
  - `20000`: `0.88` vs `0.88`

- Diffuser `h256` success (`none` vs `insert_mid`):
  - `5000`: `0.78` vs `0.74`
  - `10000`: `0.98` vs `1.00`
  - `15000`: `1.00` vs `0.96`
  - `20000`: `0.98` vs `0.98`

### Current interpretation
- At matched 50-sample checkpoint re-eval, scaffold effects are mixed for both EqM and Diffuser; neither algorithm shows consistent scaffold superiority.

### Next discriminating step
- Increase eval cardinality further (`n=200` trajectories/checkpoint) at a reduced checkpoint subset (`mid + final`) to test whether observed ±0.02 to ±0.04 deltas are stable vs residual sampling variance.

## 2026-02-28T08:22:10+08:00
### Poster status update
- User-requested poster cancel completed.
- Stopped active Discord score poster monitors (`PID 25904`, `PID 62043`).
- Current poster status: **stopped** (no `discord_swap_matrix_monitor.py` / `discord_score_poster.py` process active).

## 2026-02-28T11:05:57+08:00
### Non-scaffold medium/large 100K batch — live status
- Relay job: `j-20260228-082928-e6dc`
- Base dir: `runs/analysis/non_scaffold_medium_large_100k_20260228-082812/`
- Queue progress: `1/4` active (`eqm_medium_non_scaffold_100k`), `3/4` pending.
- Active run latest train line: `phase=online_round_19`, `step=94600`, `train_loss=0.03674`, `val_loss=0.03386`.
- Latest eval (`progress_metrics.csv`): `step=94000`, rollout success `h64/h128/h192/h256 = 0.00/0.00/0.00/0.00`.
- Current interpretation: training is still running normally but online rollout success remains near-zero in this first run.
- Next checkpoint action: wait for `summary.json` from run `1/4`, then decide continue vs early gate for runs `2-4`.
## 2026-02-28T13:50:41+08:00
### EqM medium near-end imagined trajectory diagnostics
- Added visual diagnostics for existing near-end checkpoints from `runs/analysis/non_scaffold_medium_large_100k_20260228-082812/eqm_medium_non_scaffold_100k/`.
- New artifacts:
  - `runs/analysis/late_ckpt_traj_viz_20260228-133624/quick_inference_plots/eqm_medium_step90000_imagined_trajs.png`
  - `runs/analysis/late_ckpt_traj_viz_20260228-133624/quick_inference_plots/eqm_medium_step95000_imagined_trajs.png`
- Visual finding: samples mostly collapse into local loops/attractor regions and often fail to reach the goal cleanly; this pattern is similar at 90k and 95k.
- Immediate implication: low rollout success in medium appears tied to inference trajectory quality (goal-reaching failure/looping), not train/val loss behavior alone.
- Next step: run the same imagined-trajectory diagnostic for other algorithm/env runs as soon as they produce near-end checkpoints.
## 2026-02-28T15:21:25+08:00
### Diffuser medium status + imagined trajectory diagnostics
- Active run: `runs/analysis/non_scaffold_medium_large_100k_20260228-082812/diffusion_medium_non_scaffold_100k/` (job `j-20260228-082928-e6dc`, still running).
- Latest logged eval (`step=20000`):
  - `imagined_goal_success_rate=1.00`, `imagined_pregoal_success_rate=0.00`
  - `rollout_goal_success_rate_h64/h128/h192/h256 = 0.00/0.00/0.00/0.00`
  - `rollout_min_goal_distance_mean_h256=8.248`, `rollout_final_goal_error_mean_h256=10.288`
- New visual artifacts:
  - `runs/analysis/late_ckpt_traj_viz_20260228-151513/diffuser_quick_inference_plots/diffuser_medium_step10000_imagined_trajs.png`
  - `runs/analysis/late_ckpt_traj_viz_20260228-151513/diffuser_quick_inference_plots/diffuser_medium_step15000_imagined_trajs.png`
  - `runs/analysis/late_ckpt_traj_viz_20260228-151513/diffuser_quick_inference_plots/diffuser_medium_step20000_imagined_trajs.png`
- Interpretation: diffuser-medium currently has endpoint-satisfying imagined samples but zero executed rollout success, consistent with execution-time incompatibility rather than pure optimization-loss failure.
- Next step: extract replay-buffer trajectory samples from medium run (round-wise) and compare to U-Maze replay trajectories to isolate where execution degrades.
## 2026-02-28T17:12:00+08:00
### Diffuser medium executed rollout trajectories (step20000) added
- New executed trajectory diagnostics generated from:
  - `runs/analysis/non_scaffold_medium_large_100k_20260228-082812/diffusion_medium_non_scaffold_100k/checkpoint_step20000.pt`
- Artifacts:
  - `runs/analysis/late_ckpt_traj_viz_20260228-173200/diffuser_rollout_plots/diffuser_medium_step20000_executed_openloop_rollouts.png`
  - `runs/analysis/late_ckpt_traj_viz_20260228-173200/diffuser_rollout_plots/diffuser_medium_step20000_imagined_vs_executed_openloop.png`
  - `runs/analysis/late_ckpt_traj_viz_20260228-173200/diffuser_rollout_plots/README.json`
- Notes:
  - To bypass known loader assertion from checkpoint config, a local patched copy was used with `max_path_length=1000`:
    - `runs/analysis/late_ckpt_traj_viz_20260228-173200/diffuser_rollout_plots/checkpoint_step20000_patched_maxpath1000.pt`
  - Plot overlap in earlier imagined figures came from `samples_per_query=6` (multiple stochastic samples overlaid per panel).
  - New executed plots use `samples_per_query=1` for visual clarity.

### Observed result
- Executed rollout success remains `0.0` (`0/8`) at step20000 under open-loop execution (`horizon=256`, threshold=0.5).
- Min-goal-distance remains large across queries (`~2.16` to `~5.59`), with nonzero wall-hit counts on some queries.

### Current implication
- Diffuser-medium failure persists in true execution even when imagined-vs-executed plotting ambiguity is removed.
- Next discriminating step remains replay-buffer trajectory sampling (medium vs umaze; round-wise) to isolate execution-degradation source.
