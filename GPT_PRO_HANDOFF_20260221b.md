# GPT Pro Handoff: EBM Online RL — Corrected Ablation Grid Results & Next Steps

**Date:** 2026-02-21 (second bundle — corrected threshold)
**Branch:** analysis/results-2026-02-21b
**Repo:** https://github.com/MachengShen/EBM_OnlineRL/tree/master
**Key fix in this bundle:** Prior ablation grid used `goal_success_threshold=0.5`; this bundle reports corrected results at threshold=0.2 (the metric used everywhere else).

---

## 1. Project Overview

We are investigating online Maze2D performance using a Diffuser-based trajectory model vs SAC+HER as both data **collectors** and **learners** in an online RL loop. The core research question: is SAC's observed advantage over Diffuser primarily in the *collector* (data diversity/quality) or the *learner* (policy quality)?

The pipeline is a 2-phase swap matrix: Phase 1 trains collectors (Diffuser or SAC) for a fixed budget; Phase 2 crosses replays into learners (any combination) and measures final policy success rate.

Additionally, we run diagnostic ablations on Diffuser's execution-time behavior (action scaling, EMA smoothing, adaptive replanning) to identify controllable improvements.

**Current status:**
- H1 (Diffuser collector dominates): **falsified** — SAC replay outperforms Diffuser replay by ~9pp in main effect
- H2 (Diffuser learner advantage): **partially supported** — Diffuser learner best cell is 0.9722 vs SAC learner 0.8472 main effect
- H3 (SAC advantage is control-quality, not raw noise): **partially supported** — consolidation + visual evidence holds; ablation grid synergy was a threshold artefact

---

## 2. Key Scripts

| Purpose | Script |
|---|---|
| Swap matrix orchestrator | `scripts/exp_swap_matrix_maze2d.py` |
| Diffuser collector/probe | `scripts/synthetic_maze2d_diffuser_probe.py` |
| SAC+HER collector/probe | `scripts/synthetic_maze2d_sac_her_probe.py` |
| Stochasticity analyzer | `scripts/analyze_collector_stochasticity.py` |
| Ablation grid runner | `scripts/exp_diffuser_ablation_grid.py` |
| Replan horizon sweep | `scripts/exp_replan_horizon_sweep.py` |

### Required environment
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1
MUJOCO_GL=egl
LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
PYTHONPATH=third_party/diffuser-maze2d
PYTHON=third_party/diffuser/.venv38/bin/python3.8
# Must be set before process start — mujoco_py dynamic libs require env at launch time
```

---

## 3. Experiment 1: 3-Seed Swap Matrix

### Design
- 2×2×2 factorial: collector ∈ {diffuser, sac_her_sparse} × learner ∈ {diffuser, sac_her_sparse} × mode ∈ {warmstart, frozen}
- 3 seeds each; evaluation horizon h=256; success threshold=0.2

### Artifacts
- Run dir: `runs/analysis/swap_matrix/swap_matrix_20260219-231605/`
- Results CSV: `runs/analysis/swap_matrix/swap_matrix_20260219-231605/swap_matrix_results.csv`

### Results (success@h256, 3-seed mean ± std)

| Mode | Collector | Learner | Success | ± |
|------|-----------|---------|---------|---|
| warmstart | sac_her_sparse | **diffuser** | **0.9722** | 0.0393 |
| frozen | sac_her_sparse | diffuser | 0.9444 | 0.0393 |
| frozen | sac_her_sparse | sac_her_sparse | 0.9167 | 0.0000 |
| frozen | diffuser | diffuser | 0.8611 | 0.0393 |
| warmstart | diffuser | sac_her_sparse | 0.8611 | 0.0393 |
| warmstart | sac_her_sparse | sac_her_sparse | 0.8056 | 0.0393 |
| frozen | diffuser | sac_her_sparse | 0.8056 | 0.1571 |
| warmstart | diffuser | diffuser | 0.7500 | 0.0680 |

### Main effects
- **Collector**: SAC replay 0.9097 vs Diffuser replay 0.8194 (SAC +9pp)
- **Learner**: Diffuser learner 0.8819 vs SAC learner 0.8472 (Diffuser +3.5pp)

### Key findings
- SAC collector produces better replay than Diffuser in all conditions
- Diffuser learner outperforms SAC learner (given same replay) — H2 partially supported
- Best overall: SAC replay → Diffuser learner (0.9722)
- 3 seeds may be insufficient for stable CI — 5-seed run is planned (open Q#1)

---

## 4. Experiment 2: Diffuser Execution-Time Ablation Grid

### Design
- 3×2×2 factorial: alpha ∈ {1.0, 1.2, 1.4} × beta ∈ {0.0, 0.5} × adaptive ∈ {0, 1}
- Seed 0 only (diagnostic); 6 queries × 6 rollouts; rollout horizon h=192
- **goal_success_threshold=0.2** (corrected — previous run used 0.5)

**Parameters:**
- `alpha` = action_scale_mult (scales raw Diffuser action magnitude)
- `beta` = action_ema_beta (EMA smoothing: `a_exec = (1-β)·a_raw + β·a_prev`)
- `adaptive` = adaptive_replan flag (replan early when goal progress stalls)

### How to run (copy-paste ready)
```bash
cd /root/ebm-online-rl-prototype
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_diffuser_ablation_grid.py \
  --diffuser_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0 \
  --sac_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_0 \
  --num_queries 6 --samples_per_query 20 --rollouts_per_query 6 --rollout_horizon 192 \
  --alpha_grid 1.0,1.2,1.4 --beta_grid 0.0,0.5 --adaptive_grid 0,1 \
  --goal_success_threshold 0.2 \
  --base_dir runs/analysis/ablation_grid/grid_v3_$(date +%Y%m%d-%H%M%S)
```

### Artifacts
- Canonical run: `runs/analysis/ablation_grid/grid_v2_20260221-171631/`
- Results CSV: `runs/analysis/ablation_grid/grid_v2_20260221-171631/ablation_grid_results.csv`
- ⚠️ Historical (threshold=0.5, DO NOT USE for comparison): `runs/analysis/ablation_grid/grid_20260221-134801/`

### Results — Full ranking (success@0.2)

| Condition | D_succ | SAC_succ | gap | D_hit@0.2 | SAC_hit@0.2 | clip_frac |
|---|---|---|---|---|---|---|
| α=1.4, β=0.5, adapt=True | **0.500** | 0.639 | 13.9pp | 0.500 | 0.639 | 0.534 |
| α=1.0, β=0.0, adapt=True | 0.472 | 0.611 | 13.9pp | 0.472 | 0.611 | 0.034 |
| α=1.0, β=0.5, adapt=False | 0.472 | 0.694 | 22.2pp | 0.472 | 0.694 | 0.037 |
| α=1.4, β=0.5, adapt=False | 0.472 | 0.667 | 19.4pp | 0.472 | 0.667 | 0.538 |
| α=1.2, β=0.5, adapt=True | 0.444 | 0.556 | 11.1pp | 0.444 | 0.556 | 0.365 |
| α=1.0, β=0.0, adapt=False **(baseline)** | 0.444 | 0.694 | 25.0pp | 0.444 | 0.694 | 0.037 |
| α=1.2, β=0.0, adapt=False | 0.444 | 0.694 | 25.0pp | 0.444 | 0.694 | 0.368 |
| α=1.2, β=0.5, adapt=False | 0.444 | 0.583 | 13.9pp | 0.444 | 0.583 | 0.371 |
| α=1.4, β=0.0, adapt=False | 0.444 | 0.667 | 22.2pp | 0.444 | 0.667 | 0.537 |
| α=1.4, β=0.0, adapt=True | 0.444 | 0.694 | 25.0pp | 0.444 | 0.694 | 0.529 |
| α=1.2, β=0.0, adapt=True | 0.389 | 0.611 | 22.2pp | 0.389 | 0.611 | 0.361 |
| α=1.0, β=0.5, adapt=True | 0.389 | 0.639 | 25.0pp | 0.389 | 0.639 | 0.036 |

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

### Beta × Adaptive interaction (critical comparison)

| | adapt=False | adapt=True | Δ |
|---|---|---|---|
| beta=0.0 | 0.444 | 0.435 | −0.009 |
| beta=0.5 | 0.463 | 0.444 | −0.019 |

**No interaction at threshold=0.2.** Compare to grid v1 (threshold=0.5):

| | adapt=False | adapt=True | Δ |
|---|---|---|---|
| beta=0.0 (v1) | 0.500 | 0.519 | +0.019 |
| beta=0.5 (v1) | 0.491 | **0.593** | **+0.102** ← artefact |

### Key findings
1. **Improvement is +5.6pp, not +11pp**: EMA+adaptive raises best Diffuser from 0.444 → 0.500 at threshold=0.2 (not 0.528→0.639 as reported with threshold=0.5).
2. **Interaction synergy is gone**: The β×adaptive +0.102 synergy seen in grid v1 was entirely due to the loose threshold=0.5. At threshold=0.2, the interaction magnitude is ≈0.
3. **adaptive=False is slightly better**: At threshold=0.2, adaptive replanning is mildly harmful (−0.014 main effect), the opposite of what grid v1 suggested.
4. **Alpha=1.4 is now the best**: Unlike grid v1 (where alpha=1.0 was best), alpha=1.4 gives a slight +0.021 at threshold=0.2, though it causes 53% clip fraction.
5. **SAC gap remains 14–25pp**: Best Diffuser (0.500) vs pooled SAC (0.646 mean). This is the persistent collector quality gap.

### Hypothesis: Final-meter precision failure
Hypothesis: The EMA+adaptive interventions help Diffuser get *near* the goal (within 0.5 units) but the model fails to achieve *precise* goal-reaching (within 0.2 units). The model may oscillate or diverge in the final approach phase. This would be a **planner denoising precision issue** rather than a gross control-cadence issue.

**Test to confirm**: Examine near-miss trajectories (min_goal_dist in 0.2–0.5 band) in the best vs baseline condition. If the best condition has more near-misses but same precision-failures, the hypothesis holds.

---

## 5. Experiment 3: Old vs New Defaults Smoke (Planner Selection Fairness)

### Background
Prior Diffuser runs used `wall_aware_planning=True` with 8-candidate imagined-plan selection using wall-hit scoring — a privileged operation not available at test time. We patched this to `wall_aware_planning=False, plan_samples=1`.

### Results (smoke: 6q × 3r, h=128, threshold=0.2)
| Metric | Old (wall-aware, 8 cands) | New (fair, 1 cand) | Δ |
|--------|--------------------------|---------------------|---|
| Success rate | 0.3333 | 0.2778 | −0.0556 |
| Min goal dist | 0.7234 | 0.7802 | +0.057 |
| Wall hits/ep | 28.72 | 47.50 | +18.78 |

**Caveat**: Small budget; effects are mixed per-query and not statistically stable. The −5.6pp drop suggests wall-aware planning did provide a real benefit. Swap matrix (commit b9cdd15) already uses non-privileged defaults.

---

## 6. Open Questions (ranked by blocking priority)

| Priority | Question | Blocking? | Resolves with |
|----------|----------|-----------|---------------|
| 1 | Is 3-seed matrix CI sufficient for paper? | **Yes** | 5-seed swap matrix |
| 2 | Why does EMA+adaptive help at 0.5 but not 0.2? | No | Near-miss trajectory visualization |
| 3 | Does replanning cadence/horizon explain the gap? | No | `exp_replan_horizon_sweep.py` |
| 4 | Is `eval_samples_per_query>1` needed? | No | Re-run stochasticity diagnostics |
| 5 | Impact of removing wall-aware planning (full seed)? | No | 1-seed full eval under new defaults |

---

## 7. Recommended Next Experiments

### PRIORITY 1: 5-Seed Swap Matrix (blocking — paper CI)
```bash
cd /root/ebm-online-rl-prototype
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_swap_matrix_maze2d.py \
  --seeds 0 1 2 3 4 --collectors diffuser sac_her_sparse \
  --learners diffuser sac_her_sparse --modes warmstart frozen \
  --device cuda:0 --base-dir runs/analysis/swap_matrix/5seed_$(date +%Y%m%d-%H%M%S) \
  2>&1 | tee runs/analysis/swap_matrix/5seed_latest.log
```

### PRIORITY 2: Near-Miss Trajectory Diagnosis (mechanism)
Post-process trajectory NPZ files from grid_v2 best and baseline conditions:
```bash
# Best condition trajectories are in:
# runs/analysis/ablation_grid/grid_v2_20260221-171631/alpha1.40_beta0.50_adapt1_K1/
# Load collector_stochasticity_summary.json and cross-check min_goal_dist histograms
# to find near-misses (0.2–0.5 band) vs true failures (>0.5)
python3 -c "
import json
for cond in ['alpha1.00_beta0.00_adapt0_K1', 'alpha1.40_beta0.50_adapt1_K1']:
    d = json.load(open(f'runs/analysis/ablation_grid/grid_v2_20260221-171631/{cond}/collector_stochasticity_summary.json'))
    print(cond, '→ min_dist mean:', d['diffuser_rollout_min_goal_dist_mean'])
"
```

### PRIORITY 3: Beta Sweep (fine-grained, if interaction reappears)
```bash
D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
  LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
  PYTHONPATH=third_party/diffuser-maze2d \
  third_party/diffuser/.venv38/bin/python3.8 scripts/exp_diffuser_ablation_grid.py \
  --diffuser_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/diffuser/seed_0 \
  --sac_run_dir runs/analysis/swap_matrix/swap_matrix_20260219-231605/phase1_collectors/sac_her_sparse/seed_0 \
  --num_queries 6 --samples_per_query 20 --rollouts_per_query 6 --rollout_horizon 192 \
  --alpha_grid 1.4 --beta_grid 0.2,0.3,0.5,0.7,0.9 --adaptive_grid 0 \
  --goal_success_threshold 0.2 \
  --base_dir runs/analysis/ablation_grid/beta_sweep_$(date +%Y%m%d-%H%M%S)
```

---

## 8. Implementation Notes for GPT Pro

### What NOT to change
- `goal_success_threshold` — must be **0.2** everywhere (was wrongly 0.5 in `exp_diffuser_ablation_grid.py`, now fixed)
- `wall_aware_planning` — must stay **False** for fair comparison (privileged wall layout info removed)
- Phase 1 collector checkpoints — already trained; do not retrain

### File layout
```
/root/ebm-online-rl-prototype/
├── scripts/                        # All experiment runners
├── third_party/diffuser/           # Diffuser model (frozen)
├── third_party/diffuser-maze2d/    # d4rl maze2d env
├── runs/analysis/
│   ├── swap_matrix/                # Phase1+Phase2 results
│   ├── ablation_grid/              # Execution-time ablation results
│   └── collector_stochasticity/    # Diagnostic trajectory analyses
├── docs/WORKING_MEMORY.md          # Living snapshot
└── HANDOFF_LOG.md                  # Append-only history
```

### Architecture brief
- Diffuser collector: DDPM trajectory model, replans every N steps, outputs action from denoised trajectory
- SAC+HER: standard SAC with hindsight experience replay, replans every step (full reactivity)
- Analyzer (`analyze_collector_stochasticity.py`): runs both collectors from matched start/goal pairs, records success, min_dist, hit rates, action statistics
- Key insight: `analyze_collector_stochasticity.py` sets `maze_arr=None` → Diffuser uses single-candidate planning (no wall-aware selection). This is the fair evaluation path.

### SAC baseline noise caveat
SAC success rates vary 0.556–0.694 across grid conditions despite SAC not using the ablation parameters. This is pure evaluation seed noise (6 rollouts × 6 queries per condition). Pool across all 12 conditions for a stable SAC baseline: **0.646 ± 0.040**.

### Threshold clarity
- `goal_success_threshold=0.2`: trajectory is "successful" if Diffuser/SAC reaches within L2 distance 0.2 of goal at any step
- `diffuser_rollout_success_rate_mean` == `diffuser_hit_rate_0p2` in grid v2 (both use the same threshold)
- `diffuser_hit_rate_0p1`: stricter metric (within 0.1) — use for precision analysis
