# GPT Pro Handoff: EBM Online RL Maze2D — Experiment Results & Next Steps

**Date:** 2026-02-21
**Branch:** `analysis/results-2026-02-21` (see below for remote URL)
**Repo:** https://github.com/MachengShen/EBM_OnlineRL
**Working dir:** `/root/ebm-online-rl-prototype`

---

## 1. Project Overview

We are building an **online RL system for Maze2D** where a **Diffuser** (trajectory diffusion model) is trained online on rollout data and plans by denoising trajectories conditioned on start+goal. The goal is to understand whether the performance advantage comes from:
- **H1**: The *collector* (data collection policy)
- **H2**: The *learner/planner* (training algorithm)
- **H3**: SAC's advantage is control-aware (better goal-directed coverage)

The primary comparison is **Diffuser** vs **SAC+HER (sparse reward)** in Maze2D.

---

## 2. Key Scripts

| Purpose | Script |
|---|---|
| Diffuser collector/learner probe | `scripts/synthetic_maze2d_diffuser_probe.py` |
| SAC+HER probe | `scripts/synthetic_maze2d_sac_her_probe.py` |
| 2×2 collector-learner swap matrix | `scripts/exp_swap_matrix_maze2d.py` |
| Execution-time ablation grid | `scripts/exp_diffuser_ablation_grid.py` |
| Collector stochasticity analyzer | `scripts/analyze_collector_stochasticity.py` |
| Posterior diversity diagnostic | `scripts/analyze_posterior_diversity.py` |
| Replan/horizon sweep | `scripts/exp_replan_horizon_sweep.py` |

### Required environment
```bash
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
PYTHON=third_party/diffuser/.venv38/bin/python3.8
# IMPORTANT: env vars must be set before process start — mujoco_py loads dynamic libs at import time
```

---

## 3. Experiment 1: Collector-Learner Swap Matrix (3-seed)

### Design
2×2 factorial: **collector** ∈ {diffuser, sac_her_sparse} × **learner** ∈ {diffuser, sac_her_sparse}
× 3 seeds × 2 modes (warmstart, frozen replay)
= 24 learner cells total

### How to run
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_swap_matrix_maze2d.py \
  --seeds 0 1 2 \
  --collectors diffuser sac_her_sparse \
  --learners diffuser sac_her_sparse \
  --modes warmstart frozen \
  --device cuda:0 \
  --base-dir runs/analysis/swap_matrix/swap_matrix_$(date +%Y%m%d-%H%M%S)
```

**Key flags:**
- `--replay_import_path PATH` / `--replay_export_path PATH` — cross-collector replay exchange
- `--disable_online_collection` — frozen-replay mode
- `wall_aware_planning=False`, `wall_aware_plan_samples=1` — non-privileged defaults (applied in script)

### Artifacts
- **Run dir:** `runs/analysis/swap_matrix/swap_matrix_20260219-231605/`
- **Results CSV:** `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`
- **Collector replays:** `runs/analysis/swap_matrix/swap_matrix_20260219-231605/collector_replays/{diffuser,sac_her_sparse}_seed{0,1,2}.npz`
- **Per-cell logs:** `runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase{1,2}_*/`

### Results (success@h256, 3-seed mean ± std)

| Mode | Collector → Learner | success@h256 |
|---|---|---|
| warmstart | **SAC → Diffuser** | **0.9722 ± 0.0393** |
| frozen | SAC → Diffuser | 0.9444 ± 0.0393 |
| frozen | SAC → SAC | 0.9167 ± 0.0000 |
| frozen | Diffuser → Diffuser | 0.8611 ± 0.0393 |
| warmstart | Diffuser → SAC | 0.8611 ± 0.0393 |
| warmstart | SAC → SAC | 0.8056 ± 0.0393 |
| frozen | Diffuser → SAC | 0.8056 ± 0.1571 |
| warmstart | Diffuser → Diffuser | 0.7500 ± 0.0680 |

### Main effects (marginals, n=12 each)

| Effect | Diffuser | SAC |
|---|---|---|
| Collector (which replay) | 0.8194 | **0.9097** |
| Learner (which algorithm) | **0.8819** | 0.8472 |

### Hypothesis verdicts

| Hypothesis | Verdict |
|---|---|
| **H1**: Diffuser collector dominates | ❌ **FALSIFIED** — SAC replay is better in 11/12 cells |
| **H2**: Diffuser learner advantage given same data | ✅ **PARTIALLY SUPPORTED** — learner main effect +3.5pp; best cell SAC→Diffuser |
| **H3**: SAC collector advantage is control-aware | ✅ **SUPPORTED** — SAC endpoint diversity > Diffuser in 11/12 consolidation cells; SAC ~61 steps faster to near-goal |

### Commit
`b9cdd15` — fix: exp_swap_matrix episode_len 64→256; add causal ablation scripts

---

## 4. Experiment 2: Collector Stochasticity Analysis

### Purpose
Characterize *why* SAC collector produces better replay than Diffuser. Test: does SAC produce better endpoint diversity (exploration) vs Diffuser's more conservative actions?

### Key artifacts
- **Consolidation (12 cells, q=10, s=40, r=8, h=192):**
  `runs/analysis/collector_stochasticity/consolidation_q10_s40_r8_h192/consolidated_overall_summary.json`
- **Visual check (seed_0, q=6, s=20, r=6, h=192):**
  `runs/analysis/collector_stochasticity/visual_check_phase1_seed0_q6_s20_r6_h192/`
  - Trajectory plots: `trajectory_plots/query_0{0..5}_trajectories.png`
  - First-hit diagnostics: `trajectory_plots/query_00_first_hit_diagnostics.png`
  - Steps-to-threshold: `steps_to_goal_threshold_0p1.json`, `steps_to_goal_threshold_0p1_summary.json`
  - Action audit: `query_00_action_magnitude_audit.json`
  - SAC cadence sensitivity: `query_00_sac_cadence_sensitivity.json`
  - Replay action norms: `collector_replay_action_norm_stats_seed0.json`

### Key findings
1. **Endpoint diversity:** SAC endpoint pairwise L2 > Diffuser in 11/12 consolidation cells
2. **Temporal speed:** SAC reaches near-goal (≤0.1) ~61 steps faster (mean, 4/6 queries where both succeed)
3. **Action magnitude:** SAC raw L2 = 1.2191 vs Diffuser = 0.7726 (rollout actions, q0)
4. **Action clipping is NOT the cause:** Diffuser clip fraction = 1.65%, SAC = 0.00%; conservativeness is method-level (denoising + plan objective)
5. **SAC cadence sensitivity:** decision_every=1 → 6/6 success; decision_every=16 → 5/6 — cadence matters

### Non-privileged planner patch (2026-02-20)
Diffuser's old default used privileged `wall_aware_planning=True, wall_aware_plan_samples=8` (selects best of 8 imagined plans by wall-hit count + goal error). This was unfair for comparison.

**Patch applied in:**
- `scripts/synthetic_maze2d_diffuser_probe.py` — plan selection now uses final-goal-error only; defaults `wall_aware_planning=False, wall_aware_plan_samples=1`
- `scripts/exp_swap_matrix_maze2d.py` — explicitly pins non-privileged defaults for all swap-matrix launches

**Old-vs-new smoke** (6q × 3r, h=128): new defaults → -5.6pp success (0.2778 vs 0.3333). Small sample, mixed per-query effects — not yet stable.

---

## 5. Experiment 3: Execution-Time Ablation Grid

### Purpose
Test GPT Pro's suggested execution-time interventions on the Diffuser *without retraining*:
- **Alpha (action_scale_mult):** ∈ {1.0, 1.2, 1.4} — scale action magnitude post-denoising
- **Beta (action_ema_beta):** ∈ {0.0, 0.5} — EMA smoothing between consecutive actions
- **Adaptive replanning:** ∈ {False, True} — replan only when predicted goal error exceeds threshold

### How to run
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_diffuser_ablation_grid.py \
  --diffuser_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0 \
  --sac_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_0 \
  --num_queries 6 --samples_per_query 20 --rollouts_per_query 6 --rollout_horizon 192 \
  --alpha-grid 1.0,1.2,1.4 --beta-grid 0.0,0.5 --adaptive-grid 0,1 --plan-samples-grid 1 \
  --out-dir runs/analysis/ablation_grid/grid_$(date +%Y%m%d-%H%M%S)
```

### Artifacts
- **Run dir:** `runs/analysis/ablation_grid/grid_20260221-134801/`
- **Results CSV:** `runs/analysis/ablation_grid/grid_20260221-134801/ablation_grid_results.csv`
- **Per-condition subdirs:** `grid_20260221-134801/alpha*_beta*_adapt*/`

### Results (12 conditions, seed_0 diagnostic, 6q×6r, h=192)

| Condition | success@0.2 | hit@0.1 | clip_frac |
|---|---|---|---|
| **alpha=1.0, beta=0.5, adapt=True** | **0.6389** | 0.333 | 0.036 |
| alpha=1.4, beta=0.5, adapt=True | 0.5833 | 0.361 | 0.527 ⚠️ |
| alpha=1.2, beta=0.5, adapt=True | 0.5556 | 0.361 | 0.363 |
| alpha=1.0, beta=0.0, adapt=False **(baseline)** | **0.5278** | 0.361 | 0.038 |
| alpha=1.0, beta=0.0, adapt=True | 0.5278 | 0.306 | 0.036 |
| alpha=1.2, beta=0.0, adapt=True | 0.5278 | 0.306 | 0.364 |
| alpha=1.4, beta=0.0, adapt=False | 0.5000 | 0.361 | 0.542 ⚠️ |
| alpha=1.4, beta=0.0, adapt=True | 0.5000 | 0.278 | 0.530 ⚠️ |
| alpha=1.0, beta=0.5, adapt=False | 0.5000 | 0.333 | 0.036 |
| alpha=1.4, beta=0.5, adapt=False | 0.5000 | 0.389 | 0.538 ⚠️ |
| alpha=1.2, beta=0.0, adapt=False | 0.4722 | 0.361 | 0.368 |
| alpha=1.2, beta=0.5, adapt=False | 0.4722 | 0.389 | 0.374 |

**SAC baseline (same checkpoint/queries):** success=0.7222

### Key findings
1. **Interaction effect:** EMA smoothing (beta=0.5) alone → NO improvement (0.5278→0.50). Adaptive replanning alone → NO improvement (0.5278→0.5278). **BOTH together → +11pp** (0.6389). Pure interaction, not additive.
2. **Alpha scaling is counterproductive:** alpha=1.4 causes 53% clip fraction — action saturation. No benefit over alpha=1.0 when combined with EMA+adaptive.
3. **Remaining gap:** Best Diffuser = 0.6389 vs SAC = 0.7222 → **-8pp gap remains**.
4. **This is a diagnostic on seed_0 only** — not a full training run comparison. EMA+adaptive are applied at *inference/collection time* on a pretrained checkpoint.

### Commits
- `9832f30` — feat: action scaling+EMA smoothing and adaptive replanning in rollout_to_goal
- `52506d6` — feat: ablation grid runner for Diffuser execution-time improvements
- `33d2484` — fix: exp_diffuser_ablation_grid arg names (underscore→hyphen)
- `85c2838` — docs: record ablation grid results

---

## 6. Open Questions (ranked by priority)

| Priority | Question | Blocking? | Resolves with |
|---|---|---|---|
| 1 | Is 3-seed matrix sufficient for paper CI? | **YES** | 5-seed swap matrix |
| 2 | Does replanning cadence/horizon explain Diffuser controllability gap? | no | `scripts/exp_replan_horizon_sweep.py` |
| 3 | Is eval_samples_per_query>1 needed for robust coverage metrics? | no | Re-run stochasticity diag with `--eval_samples_per_query 4+` |
| 4 | Does EMA+adaptive improve full training (not just inference-time diagnostic)? | no | New 3-seed swap matrix with best ablation params baked in |
| 5 | How much did privileged wall-aware planning inflate old Diffuser numbers? | no | 1-seed reruns under old vs new defaults at full scale |

---

## 7. Recommended Next Experiments

### PRIORITY 1: 5-seed swap matrix (paper CI)
```bash
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
Expected runtime: ~3.5-4h. Outputs: same CSV schema as 3-seed run.

### PRIORITY 2: Bake EMA+adaptive into full swap matrix
Re-run 3-seed swap matrix with `action_ema_beta=0.5, adaptive_replan=True` in the Diffuser collector/learner to test whether the +11pp diagnostic improvement transfers to end-to-end training.

### PRIORITY 3: Replanning cadence sweep
```bash
third_party/diffuser/.venv38/bin/python3.8 scripts/exp_replan_horizon_sweep.py \
  --diffuser_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0 \
  --decision_every_grid 1,4,8,16 --horizon_grid 64,128,192,256 \
  --num_queries 6 --rollouts_per_query 6
```

---

## 8. Implementation Notes for GPT Pro

### What NOT to change
- `GoalDataset` requires `episode_len > horizon` — do not set them equal (was a critical bug: `episode_len=64, horizon=64` produced zero samples)
- `wall_aware_planning` and plan-selection objective: the current default is non-privileged (`False`, single candidate). Reverting to wall-aware would make Diffuser use privileged maze layout info — unfair comparison
- The relay callback system (`[[relay-actions]]` blocks) fires in DM scope only; guild threads fall back to nohup

### File layout
```
scripts/
  synthetic_maze2d_diffuser_probe.py     # main Diffuser collector+learner
  synthetic_maze2d_sac_her_probe.py      # SAC+HER collector+learner
  exp_swap_matrix_maze2d.py              # 2×2 swap matrix orchestrator
  exp_diffuser_ablation_grid.py          # execution-time ablation runner
  analyze_collector_stochasticity.py    # action/trajectory diagnostics
  analyze_posterior_diversity.py        # diversity diagnostic
  exp_replan_horizon_sweep.py           # replan frequency × horizon sweep
third_party/
  diffuser/.venv38/bin/python3.8        # ONLY Python env with gym+torch+d4rl
  diffuser-maze2d/diffuser/             # Diffuser model code
runs/analysis/
  swap_matrix/swap_matrix_20260219-231605/    # 3-seed swap matrix results
  ablation_grid/grid_20260221-134801/         # 12-condition ablation grid
  collector_stochasticity/                    # visual + consolidation diagnostics
docs/
  WORKING_MEMORY.md                     # living snapshot (always current)
HANDOFF_LOG.md                          # append-only history
research_finding.txt                    # paper-level findings log
```

### Diffuser architecture (brief)
- Trajectory diffusion model conditioned on start+goal (inpainting: clamp first/last frames each denoising step)
- Plans over horizon H steps; replans every `replan_every` steps
- At each replan: sample K imagined plans from model, select best by `final_goal_error` (wall-hit scoring removed)
- Actions: denoised trajectory[0], optionally scaled by alpha and EMA-smoothed with prior action
- Training: online on rollout buffer; GoalDataset samples (s_t, g) pairs where g = s_{t+k}, k ∈ [1, episode_len-horizon]
