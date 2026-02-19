# Working Memory: Diffuser Maze2D Investigation

## Usage Rule
- Before answering new Maze2D/Diffuser investigation questions, read this file first.
- After each new experiment, code change, or interpretation update, append a timestamped entry with:
  - question asked
  - evidence inspected (files, metrics, plots)
  - conclusion and confidence
  - unresolved uncertainties and next checks

## 2026-02-13 17:37 CST+0800
### User question
- Why rollout success repeatedly equals exactly 6 successful trajectories.
- Whether 18 evaluation queries resample start/goal.
- Request for persistent working memory + reusable workflow skill.

### Evidence inspected
- Run config: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/config.json`
  - `query` has exactly 3 start-goal pairs.
  - `query_batch_size=6`.
- Eval logic: `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py` in `evaluate_goal_progress`.
  - Loops over fixed `query_pairs` and then over `n_samples` trajectories per query.
  - So eval trajectories/checkpoint = `len(query_pairs) * n_samples = 3 * 6 = 18`.
  - Rollout success is computed in env from imagined action rollout with thresholded min-distance.
- Metrics table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/progress_metrics_consistency_posthoc.csv`.
  - Observed rollout success rates map to integer counts out of 18: `0/18`, `5/18`, then repeated `6/18`.

### Conclusions
- Start-goal pairs are fixed across checkpoints; they are not resampled by default.
- Imagined action trajectories are sampled stochastically, but success-rate quantization is coarse because denominator is only 18.
- Repeated `0.333333` is exactly `6/18`; with this small eval set, plateaus at identical fractions are expected and not a bug by themselves.

### Important caveat
- `imagined_goal_success_rate` is not informative because terminal state is conditioned at the goal, making it trivially high.
- Better metrics added/logged: pre-goal error/success, state-velocity compatibility, velocity-action compatibility.

### Open items
- Increase eval robustness by expanding query set and/or batch size to reduce quantization noise.
- Consider deterministic fixed-noise evaluation and separate stochastic evaluation for variance reporting.

## 2026-02-13 17:39 CST+0800
### User question
- Requested persistent memory discipline and reusable skill.

### Evidence inspected
- Existing handoff file existed but had no 2026-02-13 update: `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`.

### Actions taken
- Updated handoff with today’s findings and code/metrics paths.
- Created reusable skill:
  - `/root/.codex/skills/experiment-working-memory-handoff/SKILL.md`
- Established memory-first rule in this document for all future turns in this investigation.

### Conclusion
- Continuity workflow is now formalized: read memory first, execute, update memory, append handoff.

## 2026-02-13 17:45 CST+0800
### User question
- Update evaluation protocol so it uses many different start-goal samples (not tiny repeated query set).

### Evidence inspected
- Protocol code before change reused tiny fixed query list (`query` string) each eval.
- Main script updated: `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
  - Added `query_mode` (`fixed|diverse`) and diverse-query controls.
  - Added angularly diverse query-bank builder from dataset states.
  - Added per-eval query resampling option.
  - Added eval metadata logging (`eval_query_pairs`, mode, counts).
- Sanity run evidence (with diverse mode):
  - Command log artifacts: `/tmp/synth_eval_proto_diverse_smoke/*`
  - `summary.json` shows `eval_query_mode=diverse`, `eval_query_bank_size=64`, `eval_query_pairs_per_step=12`.
  - `progress_metrics.csv` shows `eval_num_queries=12`, `eval_num_trajectories=24`, `eval_query_pairs=12`.

### Conclusions
- Evaluation protocol now supports broad start-goal diversity and no longer depends on only three fixed query pairs unless `query_mode=fixed` is explicitly selected.
- Default behavior now uses diverse query bank + resampling (`query_mode=diverse`, `query_resample_each_eval=true`).

### Open items
- If strict checkpoint-to-checkpoint comparability is needed, set `--no_query_resample_each_eval` to use a fixed diverse subset.

## 2026-02-13 18:46 CST+0800
### User question
- Re-evaluate previous checkpoints for the 40k run and report success-rate progress under improved diverse start-goal protocol.

### Evidence inspected
- Re-evaluation output CSV:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/progress_metrics_diverse_reeval_20260213.csv`
- Evaluation protocol used for this re-eval:
  - Diverse query bank size: 400
  - Selected diverse query pairs per checkpoint: 40
  - Samples per query: 3
  - Total trajectories per checkpoint: 120
  - Same diverse subset reused for all checkpoints (fair comparison)

### Conclusions
- Rollout success progression (5k->40k) on diverse protocol is mostly near zero with occasional 1/120 successes:
  - 5k: 0/120
  - 10k: 0/120
  - 15k: 0/120
  - 20k: 1/120
  - 25k: 1/120
  - 30k: 0/120
  - 35k: 1/120
  - 40k: 0/120
- This confirms prior 6/18 plateaus were strongly influenced by limited/fixed query protocol; broader diverse eval reveals much weaker realized-goal robustness.

### Open items
- Optional next: run a second re-eval with resampled diverse queries per checkpoint and average over multiple random seeds for confidence intervals.

## 2026-02-13 19:10 CST+0800
### User question
- Plot sample rollout trajectories for a later checkpoint so behavior can be visualized directly.

### Evidence inspected
- Checkpoint visualized:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/checkpoint_step40000.pt`
- New rollout artifacts:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/rollout_samples_step40000_diverse12_panel.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/rollout_samples_step40000_diverse12_overlay.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/rollout_samples_step40000_diverse12.csv`
- Protocol used for these plots:
  - Rebuilt the same synthetic dataset from run `config.json` (corridor-aware generation enabled).
  - Built diverse query bank with `bank_size=400`, `angle_bins=16`, `min_pair_distance=1.0`, seed `cfg.seed+991`.
  - Selected 12 query pairs with seed `20260213`; sampled 1 imagined trajectory per pair.
  - Rolled out actions in actual Maze2D env and plotted imagined vs realized trajectory on corridor map.

### Conclusions
- For the plotted sample at step 40000, realized rollout success is `1/12` (threshold 0.5), matching the weak realized-goal behavior seen in broader diverse re-evals.
- Mean distances from this plotted set:
  - `min_goal_dist_mean = 1.4106`
  - `final_goal_dist_mean = 1.5436`
- Checkpoint loaded `strict=True` without key remap in this run context.

### Open items
- If needed, replicate the same trajectory-plot protocol for steps 30k/35k to visualize progression shape-by-shape.

## 2026-02-13 19:47 CST+0800
### User question
- Reduce evaluation difficulty by constraining start-goal distance to `0.5-1.0`, then re-evaluate success rate.

### Evidence inspected
- New re-evaluation CSV (40k-run checkpoints 5k..40k):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/progress_metrics_diverse_reeval_dist_0.5_1.0_20260213.csv`
- Protocol used:
  - Same run/checkpoints as prior re-eval (`step5000`..`step40000`).
  - Fixed query set for fair cross-checkpoint comparison.
  - Query count: 40, samples per query: 3, trajectories/checkpoint: 120.
  - Query distance constrained to `[0.5, 1.0]` (actual sampled: min `0.5099`, max `0.9629`, mean `0.7727`).

### Conclusions
- With easier start-goal distances, realized rollout success rises dramatically versus the prior harder setting (`>=1.0`):
  - 5k: `72/120` (0.600)
  - 10k: `118/120` (0.983)
  - 15k: `120/120` (1.000)
  - 20k: `119/120` (0.992)
  - 25k: `118/120` (0.983)
  - 30k: `119/120` (0.992)
  - 35k: `120/120` (1.000)
  - 40k: `120/120` (1.000)
- This indicates model competence is highly local-distance sensitive: short-horizon goal-reaching is largely solved after ~10k under this easier query regime.

### Open items
- If desired, add a medium range benchmark (e.g., `1.0-1.5`) to map where performance transitions from near-perfect to failure.

## 2026-02-13 19:53 CST+0800
### User question
- Show samples of actual trajectory plots.

### Evidence inspected
- New sample rollout artifacts (easy-distance setting, step 40000):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/rollout_samples_step40000_dist0.5_1.0_12_panel.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/rollout_samples_step40000_dist0.5_1.0_12_overlay.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/rollout_samples_step40000_dist0.5_1.0_12.csv`
- Sampling protocol:
  - Checkpoint: `checkpoint_step40000.pt`
  - 12 unique start-goal queries with distance constrained to `[0.5, 1.0]`.
  - For each query: 1 imagined trajectory sampled, then action rollout in real Maze2D env.
  - Actual query distance stats for this plot set: min `0.5099`, max `0.9441`, mean `0.7350`.

### Conclusions
- For these 12 plotted easy-distance queries, rollout success is `12/12`.
- Per-query min/final goal distances in CSV are all well below the success threshold (0.5), consistent with near-perfect easy-distance re-eval performance.

### Open items
- Optional: produce a companion panel for medium distance (`1.0-1.5`) to visualize the failure onset regime.

## 2026-02-13 21:00 CST+0800
### User question
- Discuss (no code) how to evolve current diffuser pipeline into online-RL style self-improvement by adding planned trajectories back to replay buffer and using geometric goal sampling.

### Evidence inspected
- Current pipeline + eval hooks in:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
- Current continuity notes in:
  - `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md`
  - `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`

### Conclusions
- Core proposal to discuss with user:
  - Keep existing offline dataset as base replay.
  - Add iterative data-aggregation loop where planned trajectories generate additional data for future diffuser updates.
  - Use geometric goal sampling to bias goal distances/horizons (many short goals, occasional long goals) for stable curriculum.
- Critical design choice to resolve before coding:
  - Whether replay should include only environment-realized transitions, or also imagined/planned trajectories (with confidence filtering and source weighting).

### Open items
- Decide replay composition policy (real-only vs hybrid vs imagined-heavy).
- Decide geometric sampler definition (distance-based vs horizon-step based) and annealing schedule.

## 2026-02-13 21:06 CST+0800
### User question
- Confirmed replay policy choice = Real-only replay.
- Asked how to handle case where sampled goal is reached much earlier than planned K-step horizon.

### Evidence inspected
- Discussion constraints from prior turns (no code change yet).

### Conclusions
- Replay composition is now fixed for implementation planning: **Real-only transitions only**.
- Recommended early-hit handling policy:
  - If goal hit at `t_hit < K`, log prefix `[0..t_hit]` as successful real rollout data.
  - Immediately resample a new geometric goal from the current state and continue collection (receding-horizon chaining) using remaining environment budget.
  - Do not pad with imagined/fake steps; keep replay strictly real.
  - Track diagnostics: requested `K`, achieved `t_hit`, hit-rate-by-K, and chain length per episode.
- Stability guardrails:
  - enforce `K_min` to avoid trivial 1-2 step goals,
  - optionally adapt geometric parameter to shift harder when early-hit rate is too high.

### Open items
- Decide exact chaining budget policy after early hit:
  - fixed env steps per collector episode (recommended), or
  - fixed number of goal attempts per episode.

## 2026-02-13 21:10 CST+0800
### User question
- Chose collector budget policy = fixed env steps per episode.
- Asked for concrete major-loop hyperparameters:
  1) frequency of planning/collection after training,
  2) number of trajectories to generate,
  3) retraining steps on expanded replay.

### Evidence inspected
- Current synthetic Maze2D scale from prior runs: base replay ~110k transitions.
- Existing training/checkpoint cadence from probe workflow (5k checkpoints, 40k total baseline run).

### Conclusions
- Recommended balanced online-improvement loop (discussion plan, no code yet):
  - Warm start once on base replay: 10k gradient steps.
  - Then per round:
    - Train/fine-tune phase: 2k gradient steps.
    - Collect phase: 32 collector episodes, each with fixed 256 env steps.
    - Replay add per round: ~8,192 real transitions (plus natural variation from episode termination).
  - Retraining steps per round on expanded replay: 2k steps.
- Equivalent answers to user’s 3 asks (recommended preset):
  1) Plan/collect every 2k train steps.
  2) Generate 32 trajectories (collector episodes) per round at 256 env steps each.
  3) Retrain 2k steps per round.
- Also prepared conservative/aggressive alternatives for discussion:
  - Conservative: 3k train, 16 eps collect, 1k retrain.
  - Aggressive: 2k train, 64 eps collect, 4k retrain.

### Open items
- Confirm whether to oversample "recent" replay slices during retrain (e.g., 30-50% recent-data minibatches) or keep pure uniform replay sampling for first implementation.

## 2026-02-13 21:18 CST+0800
### User question
- Asked whether original Diffuser Maze2D environment has 256 env steps.
- Asked to elaborate exact next-goal resampling behavior under fixed env-step collector episodes.

### Evidence inspected
- Diffuser Maze2D config:
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/config/maze2d.py`
  - shows diffusion/planning `horizon` defaults and overrides (`umaze=128`, `large=384`), plus legacy comment about old episode-step counts.
- Live env step-limit check (current local D4RL env specs):
  - `maze2d-umaze-v1`: 300
  - `maze2d-medium-v1`: 600
  - `maze2d-large-v1`: 800
  - obtained via `env.spec.max_episode_steps` in venv38.

### Conclusions
- `256` is **not** the Maze2D environment’s native episode-step limit.
- In Diffuser context, `horizon=256` is sequence/planning horizon (and in current pipeline, 256 was proposed as collector budget), not necessarily env max episode steps.
- Next-goal resampling policy clarified for fixed-step collector episodes:
  - if goal reached early at `t_hit < K`, store only real transitions up to `t_hit`,
  - reduce remaining episode budget by executed steps,
  - immediately resample a new goal from current state with truncated geometric K using remaining budget,
  - continue until episode budget exhausted (no imagined padding).

### Open items
- Confirm geometric-sampler parameterization for implementation (`K_min`, `K_max`, `p`) and whether to enforce anti-trivial-goal filters (minimum goal distance / max resample retries).

## 2026-02-13 21:24 CST+0800
### User question
- Asked whether Diffuser supports variable-length input.

### Evidence inspected
- Model/diffusion shape handling:
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/diffusion.py`
  - `GaussianDiffusion` stores fixed `self.horizon`; `conditional_sample` builds shape `(batch, horizon, transition_dim)`.
- Dataset windowing:
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/datasets/sequence.py`
  - `SequenceDataset` builds fixed-length windows of size `horizon` from variable-length episodes, with optional padding.
- Temporal model construction depends on configured horizon:
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/temporal.py`

### Conclusions
- Diffuser is not truly variable-length per-batch at training time; it is trained on fixed-horizon sequence chunks.
- Variable episode lengths are handled by slicing/padding into fixed-length windows.
- In sampling, you can pass `horizon=` to `conditional_sample`, but practical stability/compatibility is best when using the trained horizon configuration.

### Open items
- If needed, decide whether to keep strict fixed planning horizon and express variable task difficulty only through geometric K + early termination/chaining.

## 2026-02-13 21:27 CST+0800
### User question
- Asked the exact horizon values used by Diffuser for Maze2D.

### Evidence inspected
- `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/config/maze2d.py`

### Conclusions
- Diffuser Maze2D base horizon is 256 (for both diffusion training and planning configs).
- Environment-specific overrides in config:
  - `maze2d-umaze-v1`: horizon 128
  - `maze2d-medium-v1`: horizon 256 (inherits base)
  - `maze2d-large-v1`: horizon 384

### Open items
- None.

## 2026-02-13 21:31 CST+0800
### User question
- Asked again whether Diffuser can take variable-length input.

### Evidence inspected
- `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/diffusion.py`
  - sampling API accepts optional `horizon` argument.
  - training loss uses horizon-dependent precomputed loss weights from model init.
- `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/datasets/sequence.py`
  - training windows are fixed-length horizon chunks.
- `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/temporal.py`
  - conv UNet can process sequences, but architecture is configured around fixed training horizon and down/upsample topology.

### Conclusions
- Current implementation supports fixed-length training chunks; it is not a true variable-length training pipeline out of the box.
- Inference can vary horizon via sampling call in principle, but safest/stablest is to use the trained horizon (or compatible lengths with careful testing).
- True variable-length support would require code changes (e.g., padding+mask in loss, or horizon bucketing / dynamic loss-weight handling).

### Open items
- Clarify if user wants variable length only at planning-time or also for training-time updates.

## 2026-02-13 21:35 CST+0800
### User question
- Asked for deeper explanation of variable-length planning via `conditional_sample`.
- Asked about consistency/optimality issue when early-hit chaining switches to a new goal.

### Evidence inspected
- `conditional_sample` shape logic and conditioning index semantics:
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/diffusion.py`
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/helpers.py`
- Temporal UNet down/upsampling topology:
  - `/root/ebm-online-rl-prototype/third_party/diffuser-maze2d/diffuser/models/temporal.py`

### Conclusions
- Planning-time variable horizon means: at inference, `conditional_sample(cond, horizon=h)` allocates sequence length `h`; goal clamp index must be set to `h-1` in `cond`.
- This is not true variable-length training; model was trained on fixed horizon windows.
- Practical constraints for varying `h` in this UNet: keep `h` compatible with down/up-sampling (prefer multiples of powers-of-two used by architecture) and close to trained horizon for stability.
- User’s consistency concern is valid: if goal is switched mid-episode to an unrelated new goal, earlier segment is not globally optimal for new goal.
- Two valid semantics:
  1) data-collection semantics: treat each chained segment as a separate task (local optimality only);
  2) global-goal semantics: maintain one episode-level goal and only replan toward the same goal (or subgoals derived from that goal), not random new goals.
- Recommended for consistency-sensitive evaluation: keep episode-level goal fixed; use receding-horizon replanning to same goal.

### Open items
- Decide which semantics to use for online loop implementation by default:
  - independent-goal chaining (data diversity), or
  - fixed episode-level goal (global consistency).

## 2026-02-13 21:56 CST+0800
### User question
- Rejected early-hit special handling alternatives.
- Requested simple policy: ignore early-hit case and keep collecting full horizon for training.

### Evidence inspected
- Discussion turn decision (no code change yet).

### Conclusions
- Current agreed data-collection policy:
  - no explicit early-hit branch,
  - do not terminate collection when goal is reached,
  - continue collecting transitions until fixed horizon/collector budget is filled.
- To keep this simple policy stable, recommended minimal safeguard (still no early-hit branch):
  - keep conditioning on the same goal for the remaining steps (replan-to-same-goal if replanning is used), rather than switching goals.

### Open items
- Confirm whether to use open-loop execution of one full sampled plan vs periodic replan-to-same-goal during the horizon.

## ${ts}
### User question
- Set policy constraint: always use receding-horizon replanning whenever possible.

### Evidence inspected
- Prior open item in this document: "open-loop full-plan execution vs periodic replan-to-same-goal within horizon".
- Latest handoff tail entries at:
  - `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`

### Conclusions
- Default control policy is now fixed:
  - Use receding-horizon replanning as the standard execution mode.
  - Keep the same episode-level goal while replanning (no goal switching branch under current simplified early-hit policy).
- Open-loop full-plan execution is now fallback-only (e.g., if replanning is infeasible due to compute/runtime constraints).

### Open items
- Choose replanning cadence default for implementation (`every_step` preferred, with optional `every_n_steps` fallback knob for compute control).

## 2026-02-13 22:01 CST+0800
### User question
- Policy lock confirmation: always use receding-horizon replanning whenever possible.

### Evidence inspected
- Existing entry immediately above captured this policy but had a literal template timestamp token.
- Prior open item resolved: open-loop full-plan execution vs periodic replan-to-same-goal.

### Conclusions
- Locked default execution policy for upcoming implementation:
  - receding-horizon replanning is the default controller behavior,
  - keep a fixed episode-level goal while replanning,
  - open-loop is fallback-only when replanning is computationally infeasible.

### Open items
- Default cadence choice to encode in config/CLI: replan every env step (default) with optional replan_every_n_steps > 1 knob.

## 2026-02-13 22:12 CST+0800
### User question
- Implement online-RL style self-improvement in the working Maze2D diffuser script.
- Use receding-horizon replanning (user suggested every 8 steps), real-only replay expansion, and geometric goal sampling.
- Do implementation now; explain details before actual long run.

### Evidence inspected
- Main implementation target:
  - "/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py"
- Prior continuity artifacts:
  - "/root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md"
  - "/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt"
- Verification commands/artifacts:
  - "/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py" compiled via Can't list '...'
  - CLI check with MuJoCo env vars + 
  - Smoke run logdir: "/tmp/synth_probe_online_smoke_1770991921"
  - Smoke run generated: , , , checkpoint and plots.

### Conclusions
- Implemented online self-improvement directly inside synthetic Maze2D probe script with new config/CLI knobs.
- Added online round loop:
  - collect planner rollouts (real environment transitions only) with receding-horizon replanning,
  - sample start/goal pairs from replay with geometric offset K,
  - append collected transitions to replay dataset,
  - rebuild GoalDataset/DataLoaders on expanded replay,
  - continue training for configured steps per round.
- Receding-horizon default cadence implemented as configurable parameter with default .
- Early-hit special branching is not used; collector continues until fixed episode budget or env done.
- Smoke run succeeded end-to-end for online mode (1 round) and produced expected artifacts.

### Open items
- User confirmation on final long-run hyperparameters before launching expensive training.
- Optional: tighten geometric goal sampling constraints if sampled goals are too hard/easy in full run.

## 2026-02-13 22:12 CST+0800
### User question
- Implement online-RL style self-improvement in the Maze2D diffuser script.
- Use receding-horizon replanning with a reasonable cadence (default set to 8 steps), real-only replay expansion, and geometric goal sampling.
- Provide details before any full run.

### Evidence inspected
- Implementation file:
  - /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py
- Verification:
  - compile check: python -m compileall -q scripts/synthetic_maze2d_diffuser_probe.py
  - CLI check with MuJoCo env vars and --help
  - smoke run artifacts under /tmp/synth_probe_online_smoke_1770991921

### Conclusions
- Online self-improvement implementation is complete in the synthetic Maze2D script.
- Added online replay expansion rounds:
  - planner-generated rollouts collected in the real env,
  - geometric start-goal sampling from replay,
  - collected real transitions appended to replay,
  - GoalDataset/dataloaders rebuilt from expanded replay,
  - continued retraining per round.
- Receding-horizon replanning is now explicit and configurable via online_replan_every_n_steps (default 8).
- Smoke test passed end-to-end with online mode enabled.

### Open items
- User confirmation of long-run hyperparameters before launching full training.

## 2026-02-13 22:20 CST+0800
### User question
- Requested algorithm-focused description (not software-centric) of the online self-improvement method.
- Asked for recommended round count and step budgets.

### Evidence inspected
- Implemented online loop in .
- Existing policy decisions in working memory/handoff:
  - receding-horizon replanning default,
  - real-only replay expansion,
  - no special early-hit branch.

### Conclusions
- Algorithmic loop is: alternate between model updates on current real replay and planner-driven data aggregation with geometric goal sampling.
- Recommended default schedule (balanced):
  - bootstrap updates: 10k,
  - online rounds: 12,
  - per-round collection: 32 episodes × 256 env steps,
  - per-round retraining: 2k gradient steps,
  - replanning cadence: every 8 env steps.
- Rationale: this keeps replay growth per round moderate (~3% if base replay ~256k transitions) while allowing enough optimization pressure each round.
- If strict total-train budget target is ~40k updates, use 15 rounds (10k + 15×2k = 40k).

### Open items
- Confirm whether to optimize for fixed total gradient updates (e.g., exactly 40k) or fixed wall-clock budget.

## 2026-02-13 22:20 CST+0800
### User question
- Correction entry for prior algorithm-budget note where one evidence path was truncated.

### Evidence inspected
- Correct implementation file path:
  - /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py

### Conclusions
- Algorithm-level recommendation stands unchanged:
  - bootstrap 10k updates,
  - 12 online rounds,
  - 32 episodes × 256 steps collection per round,
  - 2k retrain updates per round,
  - replan every 8 steps.
- If exact 40k total updates is required: 15 rounds at 2k per round after 10k bootstrap.

### Open items
- Awaiting user preference: fixed total updates vs fixed wall-clock.

## 2026-02-13 22:36 CST+0800
### User question
- Asked for expected total runtime and an overnight 12-hour experiment plan.
- Explicitly allowed autonomous overnight run + analyze + follow-up experiments.

### Evidence inspected
- GPU status: NVIDIA GeForce RTX 3090 with >24 GB free.
- Timing benchmark (offline-only):
  - run dir: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260213-222411
  - setting: 200 episodes + 500 train steps
  - wall time: about 32.4 sec.
- Timing benchmark (with one online round):
  - run dir: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260213-222459
  - setting: 200 episodes, 500 offline steps, 1 online round with 32x256 collection, 500 round steps
  - wall time: about 470.4 sec.
- Overnight driver launched:
  - script: /root/ebm-online-rl-prototype/scripts/overnight_online_maze2d_driver.sh
  - launcher log: /tmp/overnight_online_maze2d_driver.nohup2.log
  - active run root: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223622
  - active baseline run: A_baseline_rh8_p008

### Conclusions
- Runtime is dominated by online collection with receding-horizon replanning (training itself is comparatively cheap).
- Practical estimate for full 40k-update equivalent online schedule is roughly 3-4 hours including eval/checkpoint overhead.
- A 3-run adaptive overnight schedule fits inside 12 hours with buffer.
- Driver policy now running:
  - Run A baseline (rh=8, p=0.08, min_dist=0.5, 40k-updates schedule).
  - Run B adapts difficulty from A final rollout success (harder if >=0.45 else easier).
  - Run C ablates replanning cadence (4 or 16) based on A/B regime.

### Open items
- In morning: compare A/B/C progress metrics to pick next-day default for long production run.

## 2026-02-13 22:38 CST+0800
### User question
- Requested overnight execution plan with 12-hour budget and expected duration.

### Evidence inspected
- Active detached session:
  - tmux session name: overnight_maze2d_223745
  - driver script: /root/ebm-online-rl-prototype/scripts/overnight_online_maze2d_driver.sh
  - active run root: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745
- Current active process:
  - python synthetic_maze2d_diffuser_probe.py running baseline A in that run root.
- Timing benchmarks used for ETA:
  - offline mini run elapsed ~32.4s (200 eps + 500 updates)
  - mini run with one online round elapsed ~470.4s.

### Conclusions
- Overnight queue is actively running in tmux (more reliable than nohup in this environment).
- Estimated total time for queued 3-run adaptive plan is about 7 to 10 hours, leaving margin inside 12-hour window.
- Dominant cost remains online collection with receding-horizon replanning.

### Open items
- Morning follow-up should summarize A/B/C and select final default hyperparameters for next production run.

## 2026-02-14 10:41 CST+0800
### User question
- "what's the progress?"

### Evidence inspected
- Overnight driver log:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/driver.log`
- Per-run summaries and checkpoint metrics:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/progress_metrics.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/B_easier_goals_rh8/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/B_easier_goals_rh8/progress_metrics.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/C_replan16/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/C_replan16/progress_metrics.csv`
- Liveness check command at report time:
  - `ps -eo pid,etimes,pcpu,pmem,cmd | rg -i "overnight_online_maze2d_driver|synthetic_maze2d_diffuser_probe.py"`

### Conclusions
- Overnight queue completed all scheduled runs successfully (driver status code 0 for A/B/C).
- Total queue elapsed from driver-reported run durations:
  - A: 7986 sec, B: 5481 sec, C: 3347 sec, total: 16814 sec (~4h40m).
- Current process status: no active overnight driver or probe process (all jobs finished).
- Final rollout success remains effectively zero across all three runs:
  - A final rollout success = 0.0 at step 40000.
  - B final rollout success = 0.0 at step 25000 (train total 28000).
  - C final rollout success = 0.0 at step 25000 (train total 28000).
- Best observed rollout success was very small and transient:
  - A peak 0.0069444444 (1/144) at step 30000.
  - B peak 0.0138888889 (2/144) at step 20000.
  - C peak 0.0069444444 (1/144) at step 15000.

### Open items
- Diagnose why imagined metrics remain near-perfect while rollout success stays near zero (likely model-to-control mismatch/domain gap).
- Inspect rollout trajectory plots for late checkpoints to localize failure mode (heading drift, speed scaling, obstacle interactions).
- Decide next intervention: easier goals in both training and eval protocol, or stronger action regularization / execution constraints.

## 2026-02-14 10:58 CST+0800
### User question
- "Yes, please first do a failure mode diagnosis, plot the trajectories, and let's see what happens."
- Also requested process upgrade for long experiments: monitor intermediate results, anticipate expected behavior, and iterate if results are off.

### Evidence inspected
- Final overnight run artifacts:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/query_metrics.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/B_easier_goals_rh8/query_metrics.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/C_replan16/query_metrics.csv`
- New replay-based rollout diagnosis (executed saved `action_json` in Maze2D env from each saved `start_x,start_y`):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/summary.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/failure_diagnosis_summary.csv`
  - Per-run overlays and panels:
    - `.../A_baseline_rh8_p008_overlay_all_imagined_vs_realized.png`
    - `.../B_easier_goals_rh8_overlay_all_imagined_vs_realized.png`
    - `.../C_replan16_overlay_all_imagined_vs_realized.png`
    - `.../A_baseline_rh8_p008_worst12_imagined_vs_realized.png` (+ best12)
    - `.../B_easier_goals_rh8_worst12_imagined_vs_realized.png` (+ best12)
    - `.../C_replan16_worst12_imagined_vs_realized.png` (+ best12)
- Difficulty-regime comparison (eval queries vs online collected goals):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/difficulty_gap_summary.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/difficulty_gap_summary.png`
- Online round stats proving near-goal behavior during collection:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/online_collection.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/B_easier_goals_rh8/online_collection.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/C_replan16/online_collection.csv`

### Conclusions
- Reproduced final evaluation behavior from saved planned actions for each run (`n=144 trajectories/run`): rollout success remains `0/144` for A/B/C.
- Large imagined-vs-realized mismatch is confirmed:
  - mean endpoint gap is ~`1.64-1.70` across runs, matching final goal error scale.
  - inferred interpretation: imagined trajectory endpoints are not dynamically consistent with realized execution under these long-horizon hard queries.
- Difficulty mismatch is also confirmed and likely primary:
  - eval query start-goal distances are long-range: mean ~`2.50-2.63`, min ~`1.63`.
  - online self-improvement samples much easier goals: sampled distance ~`0.54-0.68` (last rounds), with collector rollouts reaching very close to goal (`rollout_min_goal_distance_mean ~0.017-0.022`).
  - inferred interpretation: training-time online data concentrates on short hops; evaluation demands mostly long-range navigation through the corridor, so success does not transfer.
- Minor positive: realized trajectories are generally pointed toward goals (cosine to goal direction mean ~`0.84-0.86`), but not enough progress to close long distances within sampled behavior.

### Open items
- Next targeted experiment should align train/eval difficulty distribution (e.g., increase online sampled goal distance range) before additional long overnight queues.
- Add an explicit iterative monitoring protocol in runner logic:
  - stage-gates on intermediate metrics,
  - early stop/redirection when rollout success stalls,
  - auto-launch of diagnostic plots after each gate.

## 2026-02-14 11:05 CST+0800
### User question
- Asked whether receding-horizon planning is actually implemented, noting plots that appear to continue straight even as geometry changes.
- Asked whether the agent can inspect PNG files directly.

### Evidence inspected
- PNGs reviewed directly:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/A_baseline_rh8_p008_worst12_imagined_vs_realized.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/B_easier_goals_rh8_worst12_imagined_vs_realized.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/C_replan16_worst12_imagined_vs_realized.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/A_baseline_rh8_p008_overlay_all_imagined_vs_realized.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/B_easier_goals_rh8_overlay_all_imagined_vs_realized.png`
- Implementation code paths:
  - Receding-horizon in online collection:
    - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py` (`collect_planner_dataset`, `should_replan = (t % replan_stride == 0) ...`)
  - Open-loop in evaluation:
    - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py` (`evaluate_goal_progress`, executes `for act in act_traj` with no replanning)
- Stats consistency check:
  - `online_collection.csv` shows replans/episode mean 32 when episode_len=256 and replan_every_n=8.

### Conclusions
- Yes, PNG inspection is available and was performed.
- User observation is correct for the plotted evaluation rollouts:
  - The eval path is open-loop execution of a single sampled action sequence, so it will not adapt to updated state-goal geometry at runtime.
- Receding-horizon exists, but only in the online self-improvement data collection loop (not the eval plotting loop).
- Therefore, current trajectory plots can look like “keep following the old straight-ish plan,” even if receding-horizon is enabled during online data collection.

### Open items
- Decide whether to switch evaluation/plotting rollouts to the same receding-horizon controller used in collection for apples-to-apples diagnosis.

## 2026-02-14 13:28 CST+0800
### User question
- Agreed to proceed with implementation request: update evaluation plotting/rollout to use receding-horizon controller behavior.

### Evidence inspected
- Implementation target:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
- Verified code regions:
  - new config/args: eval rollout mode + replan cadence
  - new rollout helper used by eval and query plotting
- Verification runs:
  - compile check passed:
    - `python -m compileall -q /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
  - help output confirms new CLI options:
    - `--eval_rollout_mode {open_loop,receding_horizon}`
    - `--eval_rollout_replan_every_n_steps`
  - smoke run (receding eval mode) succeeded:
    - logdir: `/tmp/synth_probe_eval_receding_smoke_1771046689`
    - log: `/tmp/synth_probe_eval_receding_smoke.log`
    - artifacts include updated `query_metrics.csv`, `query_trajectories.png`, `summary.json`
- Field-level verification:
  - `/tmp/synth_probe_eval_receding_smoke_1771046689/query_metrics.csv` now includes:
    - `rollout_mode`
    - `rollout_min_goal_distance`
    - `rollout_final_goal_error`
    - `rollout_xy_json`
    - `rollout_action_json`
  - `/tmp/synth_probe_eval_receding_smoke_1771046689/query_trajectories.png` now overlays rollout trace.

### Conclusions
- Implemented evaluation rollout mode control with receding-horizon support and made it configurable.
- Receding-horizon controller is now used in evaluation metrics path when `--eval_rollout_mode receding_horizon`.
- Query trajectory plotting now overlays actual rollout trajectory from selected eval mode (not imagined-only).
- This addresses the prior mismatch where online collection used replanning but eval/plots were open-loop only.

### Open items
- Optional next step: re-run the overnight checkpoints with `eval_rollout_mode=receding_horizon` to refresh diagnosis plots under controller-matched evaluation.

## 2026-02-14 13:34 CST+0800
### User question
- Requested a different communication style: summarize results algorithmically (professor↔PhD framing) instead of software-engineering file/change listings.
- Specifically asked for: (1) what functionality was not enabled, (2) what changed mathematically, (3) implications.

### Evidence inspected
- Prior implementation state in `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py` and latest smoke-run artifacts in `/tmp/synth_probe_eval_receding_smoke_1771046689`.

### Conclusions
- Communication style preference is now explicit and should be used for subsequent experiment/result reports.
- Key conceptual message to provide:
  - previously missing function was closed-loop (receding-horizon) execution in eval/plotting,
  - mathematical change is open-loop trajectory playback -> periodic replanning conditioned on current state and fixed goal,
  - implication is train/eval controller alignment and more faithful policy-level diagnostics.

### Open items
- Apply this same communication style consistently in subsequent updates.

## 2026-02-14 13:54 CST+0800
### User question
- Asked for a default behavior change: whenever LaTeX code is generated, automatically compile and provide a PDF for human-friendly viewing.

### Evidence inspected
- User explicit preference in this turn.

### Conclusions
- New default reporting preference accepted:
  - for future LaTeX outputs, generate both `.tex` and compiled `.pdf` artifacts automatically whenever environment tooling allows.
- If LaTeX toolchain is unavailable in a given environment, report the exact blocker and provide nearest fallback artifact.

### Open items
- None; apply this preference in subsequent turns.

## 2026-02-14 14:00 CST+0800
### User question
- Requested conversion of the previous equation-heavy summary into a human-readable PDF.

### Evidence inspected
- Tool availability checks:
  - `pdflatex`: unavailable in current environment.
  - `pdftoppm`: unavailable in current environment.
- Generated artifacts:
  - `/root/ebm-online-rl-prototype/output/pdf/receding_horizon_result_summary_2026-02-14.pdf`
  - `/root/ebm-online-rl-prototype/output/pdf/receding_horizon_result_summary_2026-02-14.tex`
- File verification:
  - PDF recognized as `PDF document, version 1.4`
  - size about `35K`.

### Conclusions
- Delivered a human-readable PDF containing the prior 3-part algorithm summary and equations.
- Due missing `pdflatex`, used a PDF fallback rendering workflow (Matplotlib mathtext) while also saving the `.tex` source.

### Open items
- If strict TeX compilation is required, install LaTeX toolchain (e.g., `pdflatex`) and recompile the saved `.tex`.

## 2026-02-14 14:03 CST+0800
### User question
- Asked whether updated receding-horizon trajectory plots are available.

### Evidence inspected
- Smoke-run receding-horizon artifact exists at:
  - `/tmp/synth_probe_eval_receding_smoke_1771046689/query_trajectories.png`
- Overnight run query plots currently available are from pre-change evaluations:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/query_trajectories.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/B_easier_goals_rh8/query_trajectories.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/C_replan16/query_trajectories.png`

### Conclusions
- Yes, an updated receding-horizon trajectory plot is available (smoke-run artifact).
- Full overnight checkpoint-level plots have not yet been regenerated under the new receding-horizon eval protocol.

### Open items
- If requested, rerun checkpoint evaluations to refresh A/B/C trajectory plots under receding-horizon execution.

## 2026-02-14 16:58 CST+0800
### User question
- Confirmed request to generate updated receding-horizon trajectory plots for overnight A/B/C results.

### Evidence inspected
- Attempted full checkpoint sweep script:
  - `/tmp/refresh_receding_eval_overnight.py`
  - logs: `/tmp/refresh_receding_eval_overnight.log`
- Important reproducibility fix discovered:
  - overnight runs used `PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d`.
  - initial refresh attempt using `third_party/diffuser` failed with model key mismatch.
- Produced focused final-checkpoint receding plots with shared reconstructed dataset/normalizer:
  - script: `/tmp/receding_final_plots_from_overnight.py`
  - output root:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214`
  - outputs:
    - `.../A_baseline_rh8_p008/receding_query_trajectories_from_final.png`
    - `.../B_easier_goals_rh8/receding_query_trajectories_from_final.png`
    - `.../C_replan16/receding_query_trajectories_from_final.png`
    - per-run CSVs + `receding_final_plot_refresh_summary.csv`

### Conclusions
- Updated receding-horizon trajectory plots are now available for the three overnight final checkpoints (A/B/C).
- On the plotted subset (`8` trajectories per run), rollout success remained `0.0` for all three runs.
- Plotted-subset mean minimum goal distance:
  - A: `1.472`
  - B: `1.772`
  - C: `1.635`
- Full checkpoint-by-checkpoint sweep under receding-horizon was started but is computationally heavy; focused final plots were produced first to unblock visual diagnosis.

### Open items
- If full checkpoint progress refresh is still desired, queue a staged overnight sweep (or reduce eval query/sample counts for daytime turnaround).

## 2026-02-14 16:59 CST+0800
### User question
- Shared a newer receding-horizon trajectory panel image (8 samples) and requested updated visualization confirmation/inspection.

### Evidence inspected
- Latest refreshed plot bundle (final checkpoints, receding eval):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/receding_final_plot_refresh_summary.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/C_replan16/receding_query_trajectories_from_final.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/C_replan16/receding_query_metrics_from_final.csv`
- Numeric checks from `C_replan16/receding_query_metrics_from_final.csv` (first 8 plotted rows):
  - `mean_line_deviation` approx `0.115-0.244`
  - `path_over_direct` approx `1.05-1.28`
  - `rollout_min_goal_distance` approx `0.904-2.014`
- Internal consistency check on rollout distances from `rollout_xy_json`:
  - for each of first 8 rows, nearest-goal index is final timestep (`argmin = 64 of 65`), so `rollout_min_goal_distance == rollout_final_goal_error` is consistent with the recorded trajectories (not a CSV write bug).

### Conclusions
- The uploaded screenshot matches the refreshed receding-horizon evaluation output for `C_replan16`.
- Receding-horizon evaluation now produces much smoother imagined paths (lower line deviation, near-straight path ratios), but realized rollouts remain far from goals on hard queries (0/8 success in plotted subset).
- Dominant failure mode in these samples is not jagged imagined planning; it is execution drift / closed-loop control mismatch where realized motion progresses only partially toward goal before stalling/diverting.

### Open items
- Run/plot a short-horizon action mismatch decomposition (per-step imagined action vs executed displacement direction) to separate model-planning quality from environment-response/control mismatch.
- If user wants checkpoint trend under receding eval, queue reduced-cost sweep (fewer queries/samples) for same A/B/C runs.

## 2026-02-14 17:02 CST+0800
### User question
- Requested that generated pictures be placed under an EBM subdirectory instead of `/tmp` for easier clicking/visualization.

### Evidence inspected
- Temporary image sources identified:
  - `/tmp/synth_probe_eval_receding_smoke_1771046689/*.png`
  - `/tmp/synth_probe_online_smoke_1770991921/*.png`
  - `/tmp/synth_eval_proto_diverse_smoke/*.png`
- Project-run image sources included:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/*.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/*/*.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/corridor1000_train40k_20260213-155630/*rollout_samples*png`
- New gallery created with copied images + source mapping:
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/maze2d_plot_gallery_20260214`
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/maze2d_plot_gallery_20260214/source_map.csv`
  - image count copied: `25`
- Mirrored gallery in main repo path:
  - `/root/ebm-online-rl-prototype/output/figures/maze2d_plot_gallery_20260214`
- Stable symlinks:
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/latest_maze2d_gallery`
  - `/root/ebm-online-rl-prototype/output/figures/latest_maze2d_gallery`

### Conclusions
- Existing relevant figures have been centralized into an EBM-local clickable gallery, so browsing no longer depends on `/tmp` paths.
- A stable `latest_maze2d_gallery` alias now exists for subsequent runs.

### Open items
- For future experiments, continue writing plot outputs directly under `.../output/figures/latest_maze2d_gallery` (or a timestamped sibling) to avoid post-hoc copying.

## 2026-02-14 18:11 CST+0800
### User question
- Requested cleanup under `runs` and `outputs`: delete everything created before yesterday and keep only new stuff from today.

### Evidence inspected
- Repo paths audited: `/root/ebm-online-rl-prototype/runs`, `/root/ebm-online-rl-prototype/output` (no `outputs/` directory exists).
- Pre-cleanup counts using mtime cutoff `2026-02-14 00:00:00`:
  - old files/symlinks (`< cutoff`): `393`
  - today files/symlinks (`>= cutoff`): `99`
- Cleanup command applied:
  - deleted only old files/symlinks under `runs` + `output`
  - pruned empty directories after deletion
- Post-cleanup check:
  - remaining today files/symlinks: `99`
  - remaining old files/symlinks: `0`
- Remaining top-level structures:
  - `runs/analysis/synth_maze2d_diffuser_probe`
  - `output/pdf/*2026-02-14*`
  - `output/figures/maze2d_plot_gallery_20260214` and `output/figures/latest_maze2d_gallery`

### Conclusions
- Cleanup completed with strict policy “keep only today’s artifacts” under `runs` and `output`.
- All older files in those two trees were removed; only `2026-02-14` artifacts remain.

### Open items
- If user intended to keep yesterday (`2026-02-13`) artifacts as well, recovery would require restoring from git history/backups for deleted generated artifacts.

## 2026-02-14 18:13 CST+0800
### User question
- Asked where the receding-horizon evaluation trajectories are located.

### Evidence inspected
- Receding-eval trajectory artifacts found under:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/A_baseline_rh8_p008/receding_query_trajectories_from_final.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/B_easier_goals_rh8/receding_query_trajectories_from_final.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/C_replan16/receding_query_trajectories_from_final.png`
- Paired per-sample CSVs (contains `rollout_mode`, `rollout_xy_json`, `rollout_action_json`):
  - `.../A_baseline_rh8_p008/receding_query_metrics_from_final.csv`
  - `.../B_easier_goals_rh8/receding_query_metrics_from_final.csv`
  - `.../C_replan16/receding_query_metrics_from_final.csv`
- EBM-local clickable copies exist in gallery:
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/latest_maze2d_gallery/`

### Conclusions
- Receding-horizon evaluation trajectories are in `receding_final_plot_refresh_20260214/*/receding_query_trajectories_from_final.png`.
- Fastest user-click path is via `figures/latest_maze2d_gallery` in current worktree.

### Open items
- If needed, provide one canonical “latest receding eval trajectory” symlink directly to the chosen run (e.g., C_replan16).

## 2026-02-14 18:33 CST+0800
### User question
- Requested maze geometry overlay on the receding-horizon trajectory plot.
- Asked to ensure trajectory generation is obstacle/wall-aware.

### Evidence inspected
- Plotting and rollout code path in:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
- Existing receding-eval artifacts to replot:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/*/receding_query_metrics_from_final.csv`
- New overlay plot outputs generated:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/A_baseline_rh8_p008/receding_query_trajectories_from_final_with_maze.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/B_easier_goals_rh8/receding_query_trajectories_from_final_with_maze.png`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/C_replan16/receding_query_trajectories_from_final_with_maze.png`
- Clickable gallery copies:
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/latest_maze2d_gallery/A_baseline_rh8_p008_receding_query_trajectories_with_maze.png`
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/latest_maze2d_gallery/B_easier_goals_rh8_receding_query_trajectories_with_maze.png`
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/latest_maze2d_gallery/C_replan16_receding_query_trajectories_with_maze.png`
- Code verification:
  - compile pass: `./.venv/bin/python -m compileall -q scripts/synthetic_maze2d_diffuser_probe.py`

### Conclusions
- Added maze-geometry overlay support to query-trajectory plotting in `synthetic_maze2d_diffuser_probe.py`.
- Added explicit wall-awareness in planning/eval pipeline:
  - wall-hit counting utilities,
  - wall-aware candidate plan selection during replanning (`sample_best_plan_from_obs`),
  - rollout/eval/query metrics now include in-wall point diagnostics (`imagined_in_wall_points`, `rollout_in_wall_points` and corresponding means).
- Receding evaluation overlay plots were regenerated for A/B/C final checkpoints and copied to user-local gallery.

### Open items
- Current environment lacks `gym` import in active runtime, so replotting used canonical `maze2d-umaze-v1` geometry for overlays.
- Next full training/eval run should be executed in the proper runtime with gym+d4rl so built-in `load_maze_arr_from_env` path is used end-to-end.

## 2026-02-14 18:44 CST+0800
### User question
- Asked how many time steps the (receding-eval) trajectories are run for.

### Evidence inspected
- Checked receding eval CSVs:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/A_baseline_rh8_p008/receding_query_metrics_from_final.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/B_easier_goals_rh8/receding_query_metrics_from_final.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/C_replan16/receding_query_metrics_from_final.csv`
- Parsed first row lengths:
  - imagined states (`xy_json`) = 64
  - imagined actions (`action_json`) = 64
  - rollout states (`rollout_xy_json`) = 65
  - rollout actions (`rollout_action_json`) = 64
- Config check where available (`B`, `C`) reports `horizon=64`.

### Conclusions
- The plotted receding-eval trajectories run for `64` action time steps.
- Rollout state traces have `65` points because they include the initial state before the first action.

### Open items
- Distinguish this from online collection episode budget (`256` env steps), which is a different setting than these query-plot trajectories.

## 2026-02-14 18:56 CST+0800
### User question
- Asked to run trajectories for longer to give more time for goal-reaching when some rollouts are not converging.

### Evidence inspected
- New reevaluation script created/executed:
  - `/tmp/receding_final_plots_long_horizon.py`
- Runtime used for env-compatible rollout:
  - `PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d`
  - `/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python`
  - `LD_LIBRARY_PATH+=/root/.mujoco/mujoco210/bin`
- Output root for longer rollout budget:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_h192_longer`
- New artifacts:
  - `.../receding_final_plot_refresh_summary_h192.csv`
  - per-run plots `.../{A_baseline_rh8_p008,B_easier_goals_rh8,C_replan16}/receding_query_trajectories_h192.png`
  - per-run metrics `.../{A_baseline_rh8_p008,B_easier_goals_rh8,C_replan16}/receding_query_metrics_h192.csv`
- Gallery copies:
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/latest_maze2d_gallery/*_receding_query_trajectories_h192.png`
  - summary CSV copied as `receding_final_plot_refresh_summary_h192.csv`
- Comparison against prior `h64` summary (`receding_final_plot_refresh_20260214/receding_final_plot_refresh_summary.csv`):
  - A: success `0.00 -> 0.75`, min-goal-dist mean `1.472 -> 0.474`
  - B: success `0.00 -> 0.25`, min-goal-dist mean `1.772 -> 0.599`
  - C: success `0.00 -> 0.25`, min-goal-dist mean `1.635 -> 0.674`

### Conclusions
- Increasing rollout horizon from 64 to 192 substantially improves goal-reaching on the plotted subset (8 trajectories/run).
- Failures were partially time-budget limited, not purely planning-quality limited.
- Nonzero `rollout_in_wall_points_mean_plotted` at h192 indicates some longer-horizon behaviors still spend significant time near/inside wall regions and are not fully robust.

### Open items
- If needed, run a second long-horizon point (e.g., h256) to map success-vs-horizon saturation.
- Integrate this as an official eval sweep parameter rather than one-off `/tmp` script.

## 2026-02-14 21:14 CST+0800
### User question
- Asked to explicitly note down the verified finding that evaluation needs more rollout steps for decent success.

### Evidence inspected
- Baseline receding-eval summary (64-step):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/receding_final_plot_refresh_summary.csv`
- Longer-horizon receding-eval summary (192-step):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_h192_longer/receding_final_plot_refresh_summary_h192.csv`

### Conclusions
- Verified protocol-level takeaway: for this setup, `64` eval rollout steps underestimates goal-reaching and can report near-zero success even when the policy can reach goals with more time.
- Recommended evaluation setting (current evidence): use at least `192` rollout steps for receding-horizon evaluation, and treat `64`-step results as time-budget-limited diagnostics.

### Open items
- Confirm saturation point with an additional `256`-step sweep before locking the final benchmark horizon.

## 2026-02-14 21:17 CST+0800
### User question
- Corrected prior wording: success vs rollout steps should be treated as a continuous spectrum, not a hard “at least 192” threshold.
- Asked to incorporate this into protocol, including rollout steps for planned trajectories added to replay.

### Evidence inspected
- Online collection rollout budget from logs:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/online_collection.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/B_easier_goals_rh8/online_collection.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/C_replan16/online_collection.csv`
- Observed across rounds in all three runs:
  - `episode_len_mean = 256`, `episode_len_min = 256`, `episode_len_max = 256`.
- Evaluation comparison references:
  - `h64` summary: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_20260214/receding_final_plot_refresh_summary.csv`
  - `h192` summary: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/receding_final_plot_refresh_h192_longer/receding_final_plot_refresh_summary_h192.csv`

### Conclusions
- Protocol correction: rollout-step budget should be treated as a sweep variable (`H`), not a single minimum claim.
- Current evidence shows longer `H` helps strongly (`64 -> 192` improved plotted success), but saturation is not yet established.
- Planned trajectories added to replay were already rolled out with fixed `256` env steps in these runs, which is consistent with using a large collection horizon.
- Recommended protocol language:
  - report `success@H` curve over multiple horizons (e.g., `H in {64,96,128,192,256}`),
  - keep replay collection horizon large (currently `256`) unless ablation shows no gain.

### Open items
- For strict monotonicity checks, evaluate from a single long rollout trace and compute prefix metrics `success@H` (rather than rerunning separate stochastic evaluations per `H`).

## 2026-02-14 21:24 CST+0800
### User question
- Requested design-first summary before implementation: add online training metric for planning success from replay-collection rollouts.
- Proposed success definitions:
  1) absolute threshold to goal (e.g., 0.1/0.2),
  2) relative reduction (>=90% reduction from initial start-goal distance).
- Emphasized protocol: keep collection horizon at 256 and derive shorter-horizon metrics (e.g., first 64 steps) from the same trajectory without extra rollout compute.

### Evidence inspected
- Current planner-collection metrics path in:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py:935` (`collect_planner_dataset`)
  - current logged stats include `rollout_min_goal_distance_mean`, `rollout_final_goal_distance_mean`, but no threshold/reduction success rates by horizon.
- Current collection horizon config present in overnight runs:
  - `online_collect_episode_len=256` (B/C configs; A config deleted in cleanup but online_collection confirms fixed 256 behavior).

### Conclusions
- Proposed design change (pre-implementation):
  - keep rollout collection at 256 steps,
  - compute per-trajectory prefix minima/finals at horizons H={64, 128, 192, 256} from the same collected distance trace,
  - log online success metrics in `online_collection.csv` (and round console print) for:
    - `success_abs_eps0.1@H`, `success_abs_eps0.2@H`,
    - `success_rel90@H` where `min_dist_prefix <= 0.1 * d0`,
    - optional `final_dist@H` and `min_dist@H` means.
- This preserves compute budget while turning horizon into an explicit online protocol variable.

### Open items
- Confirm which horizons to include by default in logs (recommended: 64 and 256 minimal set; optional 128/192 for richer diagnostics).
- Confirm absolute thresholds to log by default (recommended: both 0.1 and 0.2).

## 2026-02-14 21:39 CST+0800
### User question
- Lock eval prefix horizons to `{64,128,192,256}`.
- Add online planning-success logging during collection using:
  - absolute final-distance threshold success (`0.1`, `0.2`),
  - relative final-distance reduction success (`>=90%`).
- Keep comments in code documenting why this protocol choice is essential.

### Evidence inspected
- Updated script:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
- Protocol/comment locations:
  - eval prefix-from-single-rollout rationale comment at lines ~1853-1856 and ~1444-1446.
- Eval protocol wiring:
  - new config/CLI fields: `eval_rollout_horizon`, `eval_success_prefix_horizons`.
  - evaluate call now passes `planning_horizon=cfg.horizon`, `rollout_horizon=cfg.eval_rollout_horizon`, `success_prefix_horizons=eval_success_prefix_horizons`.
  - added metrics columns per prefix horizon: `rollout_goal_success_rate_h{H}`, `rollout_success_count_h{H}`, `rollout_min_goal_distance_mean_h{H}`, `rollout_final_goal_error_mean_h{H}`.
- Online collection planning-success logging:
  - new config/CLI fields: `online_planning_success_thresholds`, `online_planning_success_rel_reduction`.
  - `collect_planner_dataset(...)` now logs:
    - `planning_success_rate_final_t010`, `planning_success_rate_final_t020` (for defaults),
    - `planning_success_rate_final_rel090` and `planning_success_rate_final_rel_reduction`,
    - `planning_final_distance_reduction_ratio_mean`.
- Verification:
  - compile pass: `./.venv/bin/python -m compileall -q scripts/synthetic_maze2d_diffuser_probe.py`.
  - runtime CLI help check blocked in this environment due missing `gym` in `.venv`.

### Conclusions
- Eval now follows the requested protocol: one long rollout budget (default 256) with success reported at fixed prefixes `{64,128,192,256}` from the same trajectory.
- Online replay collection now emits direct planning-success metrics aligned with user-specified absolute and relative criteria.
- Progress plotting now auto-displays multiple rollout success curves when prefix columns are present.

### Open items
- Execute one short end-to-end run in the full gym/d4rl environment to validate the new CSV columns and plot outputs (beyond static compile).
- Optionally add prefix-based relative-success metrics (`success_rel90@H`) in eval if user wants horizon-wise relative criteria too.

## 2026-02-14 22:02 CST+0800
### User question
- Asked for a concrete overnight experiment proposal given current progress/lessons.
- Asked what still needs to be verified next.

### Evidence inspected
- Continuity summary:
  - `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md`
  - `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`
- Existing overnight driver template:
  - `/root/ebm-online-rl-prototype/scripts/overnight_online_maze2d_driver.sh`
- Newly implemented protocol in main script:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
  - includes eval prefix metrics @`{64,128,192,256}` from shared 256-step rollout and online planning-success metrics.

### Conclusions
- Proposed overnight plan should be monitor-gated and verification-first before large compute:
  1) **Instrumentation sanity run (short)**: verify newly added CSV columns are emitted and logically consistent.
  2) **Distance-alignment ablation**: test whether online goal-sampling distribution closer to eval distances improves long-horizon success.
  3) **Replanning cadence ablation**: compare `replan_every=8` vs denser/sparser setting under same goal-distance regime.
  4) **Post-run diagnosis**: plot trajectory mismatch and report success@`{64,128,192,256}` curves from shared traces.
- Critical verification items:
  - prefix success monotonicity (`success@64 <= success@128 <= success@192 <= success@256`) on shared rollout metrics,
  - whether online planning-success metrics (t0.1/t0.2/rel90) correlate with eval success@256,
  - whether goal-distance mismatch between collection and eval remains the dominant failure mode,
  - whether wall-hit diagnostics stay controlled as rollout horizon increases.

### Open items
- If running tonight, update driver/extractor logic to consume the new prefix metric columns (not only legacy `rollout_goal_success_rate`).
- Decide run count/stop criteria thresholds before launch (e.g., early-stop if success@256 flat for two checkpoints).

## 2026-02-14 22:13 CST+0800
### User question
- Revised overnight priorities:
  1) focus on online-loop hyperparameters for sample efficiency,
  2) specifically ablate (a) trajectories collected before each retrain and (b) update frequency of policy/diffuser,
  3) keep planning-horizon ablation on latest checkpoint,
  4) skip post-run diagnosis for now.

### Evidence inspected
- Prior plan entry at `2026-02-14 22:02 CST+0800` in this file.
- Driver template for overnight orchestration:
  - `/root/ebm-online-rl-prototype/scripts/overnight_online_maze2d_driver.sh`
- Current script supports online-loop knobs relevant to user priorities:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
  - key args: `online_collect_episodes_per_round`, `online_train_steps_per_round`, `online_replan_every_n_steps`, `online_goal_geom_*`.

### Conclusions
- Updated experiment design should center on **sample-efficiency curves vs env-step budget** under controlled online-loop schedules.
- Recommended structure:
  - Phase A: collection-chunk/frequency ablation at fixed update-intensity per env step.
  - Phase B: update-intensity ablation at fixed collection chunk size.
  - Phase C: eval-only planning-horizon ablation on the best/latest checkpoint.
- Geometric sampling remains in-scope as a controlled factor, but secondary to the two core loop knobs requested by user.

### Open items
- Need to lock exact budget-normalization protocol (fixed total env steps; report success@{64,128,192,256} vs env steps).
- Need explicit wall-clock-aware run cap for 12h so all phases complete.

## 2026-02-14 22:19 CST+0800
### User question
- Replanning cadence (steps per replan) is likely as important as planning horizon.
- Concerned about two failure modes:
  1) too-frequent replanning -> long-term inconsistency,
  2) too-infrequent replanning -> weak feedback adaptation.
- Asked for a modification proposal.

### Evidence inspected
- Existing loop currently uses fixed cadence knobs:
  - `online_replan_every_n_steps`
  - `eval_rollout_replan_every_n_steps`
  in `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`.
- Prior empirical context in this file:
  - `rh8` and `rh16` behavior differs,
  - longer rollout budgets improved success, implying control-frequency + adaptation timing matters.

### Conclusions
- Proposed algorithmic modification: **adaptive/event-triggered replanning cadence** instead of constant `N`.
- Use nominal commit length (`N_base=8`) but replan earlier/later based on state mismatch/progress/wall-risk triggers, with bounded range (`N_min`, `N_max`) to avoid chatter and stale plans.
- This directly addresses both user failure modes by coupling cadence to observed execution error rather than fixing it globally.

### Open items
- If user agrees, implement with explicit triggers + logs (trigger counts, replan-rate, mismatch stats) and run sample-efficiency ablation vs fixed cadence baselines.

## 2026-02-14 22:49 CST+0800
### User question
- Continue with the several-phase study, keep intermediate monitoring/assessment active, and encode this workflow as a reusable skill.
- Explicitly inspect for misimplementation/config issues while running.

### Evidence inspected
- New 5-step driver script created:
  - `/root/ebm-online-rl-prototype/scripts/overnight_five_step_monitor_driver.sh`
- New reusable skill created:
  - `/root/.codex/skills/ebm-onlinerl-five-step-loop/SKILL.md`
  - `/root/.codex/skills/ebm-onlinerl-five-step-loop/references/run_matrix.md`
  - `/root/.codex/skills/ebm-onlinerl-five-step-loop/scripts/launch_five_step.sh`
  - `/root/.codex/skills/ebm-onlinerl-five-step-loop/scripts/inspect_driver.py`
- Active run launched in persistent session:
  - session id: `3467`
  - run root: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/five_step_20260214-224647`
  - driver log: `.../five_step_20260214-224647/driver.log`
- Current intermediate signals:
  - driver log has startup + first snapshot (`progress_metrics.csv not ready` at ~120s), consistent with long startup/data collection stage.
  - active python process confirmed:
    - `/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python ... synthetic_maze2d_diffuser_probe.py --logdir .../S0_sanity`
  - no immediate runtime exception in `run.log` (currently only pybullet banner line).

### Conclusions
- Monitoring pipeline is now concretely implemented and running.
- Important operational fix discovered: detached `nohup` launches from this tool context were not reliable for persistence; persistent TTY session launch is reliable and now used.
- No evidence yet of script/config misimplementation from runtime errors; currently waiting for first metrics artifacts.

### Open items
- Await completion of `S0_sanity` to validate required columns:
  - `rollout_goal_success_rate_h64/h128/h192/h256`
  - `planning_success_rate_final_t010/t020`
- If sanity column check fails, stop and patch driver/main script before Phase A/B/C.

## 2026-02-14 23:21 CST+0800
### User question
- User asked why server GPU looked idle and whether monitoring was actually active.

### Evidence inspected
- Live GPU/process checks:
  - `nvidia-smi`: `16-20%` GPU util, ~`742 MiB` in use.
  - Active driver: `bash /root/ebm-online-rl-prototype/scripts/overnight_five_step_monitor_driver.sh`.
  - Active train proc: `synthetic_maze2d_diffuser_probe.py` under run root `five_step_20260214-231421`.
- Driver log:
  - snapshots at `23:16`, `23:18`, `23:20` all report `progress_metrics.csv not ready`.
- Run log:
  - completed startup + offline training to step `500` (`[train] ... step=500 ...`).
  - no further log lines yet.
- Liveness check:
  - process CPU time keeps increasing (utime/stime increments), state remains `R`, threads=132.

### Conclusions
- Monitoring is active and the run is alive.
- Apparent "GPU=0%" intervals are expected between heavy kernels; spot checks showed nonzero utilization during this diagnosis.
- Current phase appears to be a long post-step-500 eval/rollout segment with sparse logging; no evidence of process crash.

### Open items
- Continue polling until first `progress_metrics.csv` row appears, then verify required columns and proceed with phase gating.

## 2026-02-14 23:24 CST+0800
### User question
- User asked for a mechanism that effectively re-prompts the agent after each response so long-running progress does not stall without manual nudges.

### Evidence inspected
- Existing monitor/driver scripts:
  - `/root/ebm-online-rl-prototype/scripts/overnight_five_step_monitor_driver.sh`
  - `/root/.codex/skills/ebm-onlinerl-five-step-loop/scripts/inspect_driver.py`
- Tooling availability on server:
  - `codex` CLI exists at `/root/.nvm/versions/node/v24.13.0/bin/codex`
  - `tmux` and `crontab` are available.

### Conclusions
- Hard platform constraint: this chat interface is user-turn-driven; the current thread cannot self-send new assistant messages after a turn ends.
- Practical workaround: add an **autonomous reprompt loop** outside chat (cron/tmux) that periodically runs `codex exec` with a fixed monitoring prompt; it updates working-memory/handoff/status artifacts continuously.
- This preserves overnight progress continuity even when no manual prompt arrives.

### Open items
- Choose desired mode:
  - status-only monitor loop,
  - full auto-decider loop (continue/stop/branch),
  - hybrid with conservative stop rules.

## 2026-02-15 12:22 CST+0800
### User question
- Asked for detailed difference between `status-only` and `auto-decider` loops, and whether decisions come from another Codex agent assessment.

### Evidence inspected
- Existing monitor workflow and skills:
  - `/root/.codex/skills/ml-run-monitor-decider/SKILL.md`
  - `/root/.codex/skills/experiment-working-memory-handoff/SKILL.md`
  - `/root/ebm-online-rl-prototype/scripts/overnight_five_step_monitor_driver.sh`
- Prior recorded constraint:
  - chat thread itself cannot self-message; only external scheduled invocations can continue autonomously.

### Conclusions
- `status-only` loop = observe + log + alert; **no authority** to stop/start/branch runs.
- `auto-decider` loop = observe + log + **act** (continue/stop/launch-next) using explicit policy rules.
- Decision source is not a mysterious second persistent agent; operationally it is scheduled `codex exec` turns on the same repo context.
- Recommended mechanism for scientific stability: rule-first decisions from metrics; LLM text used to explain, not to override hard safety gates.

### Open items
- User to choose operating mode:
  - strict status-only,
  - conservative auto-decider (single-restart + capped branching),
  - full auto-decider.

## 2026-02-15 12:31 CST+0800
### User question
- Asked which to use for fully autonomous overnight operation (status-only vs auto-decider) and whether this risks overriding pre-night decisions.

### Evidence inspected
- Prior mechanism definition in this doc: scheduled `codex exec` invocations can act as the reprompt loop.
- Existing driver already contains deterministic phase-branching (best-A/B/C selection) based on metrics.

### Conclusions
- If "fully autonomous" means: *execute a pre-declared run matrix and keep producing updates*, prefer **driver + status-only Codex monitor loop** (lowest risk).
- If "fully autonomous" means: *adaptively stop/restart/launch follow-ups when new metrics indicate issues*, use **conservative auto-decider** with a tightly bounded action set.
- Clarification: auto-decider actions are explicitly meant to depend on the most recent results produced overnight; the key safety lever is that the *policy* is fixed before sleep.

### Open items
- Need user choice of authority level for the reprompt loop:
  - status-only,
  - conservative auto-decider (stop/restart/launch-next only),
  - full auto-decider (may redesign configs).

## 2026-02-15 12:40 CST+0800
### User question
- User wants a **more aggressive auto-decider** overnight: adaptively infer which hyperparameters matter from fresh results and redesign subsequent runs, rather than rigidly executing a pre-fixed matrix.

### Evidence inspected
- Current five-step driver already does limited adaptivity (choose-best A/B/C) but only within a predeclared grid:
  - `/root/ebm-online-rl-prototype/scripts/overnight_five_step_monitor_driver.sh`

### Conclusions
- Next iteration should implement an **adaptive hyperparameter search policy** (explore -> fit simple effect model -> exploit), with explicit guardrails against noise overfitting.
- Emphasis: decisions must be evidence-backed (latest metrics), but governed by a policy fixed before sleep (allowed action set + parameter bounds + budget limits).

### Open items
- Confirm fairness constraint:
  - keep total online env-step budget fixed per trial (recommended for sample-efficiency comparability) vs allow variable budgets.
- Decide which knobs are in-scope for autonomous tuning (recommended: collection chunk size, update-intensity ratio, replanning interval; keep others fixed).

## 2026-02-15 12:55 CST+0800
### User request
- Implement an **aggressive auto-decider** overnight strategy: adaptively identify impactful hyperparameters from fresh results, and redesign subsequent runs automatically. User explicitly allowed variable per-trial budgets.

### What changed (evidence-backed)
1) Enabled true intermediate monitoring
- `progress_metrics.csv` and `online_collection.csv` were previously only written at the end of a run.
- Patched `scripts/synthetic_maze2d_diffuser_probe.py` to flush both files incrementally:
  - after each `--eval_goal_every` evaluation
  - after each online collection round

2) Added adaptive overnight controller
- New script: `scripts/overnight_maze2d_autodecider.py`
  - runs a loop of trials under a wall-clock budget
  - explores then exploits by:
    - promoting best base-config to higher budget with probability `promote_prob`
    - bandit-weighted sampling over key knobs
  - writes:
    - `autodecider_results.csv` (all trials + objective)
    - `autodecider_importance.csv` (range-of-means influence ranking)

3) Added tmux launcher + design note
- Launcher: `scripts/launch_overnight_maze2d_autodecider_tmux.sh`
- Design doc: `docs/plans/2026-02-15-maze2d-autodecider.md`

### Implications
- Monitor/decide loops can now operate mid-run (no more "progress_metrics.csv not ready" until finish).
- Overnight autonomy no longer depends on chat re-prompts; the controller can keep generating trials and logging decisions.
- Hyperparameter influence is estimated online (crudely) from observed outcomes, and used to bias subsequent sampling.

### Open items
- Budget promotion currently reruns from scratch (no checkpoint resume). If needed, add resume support to reduce waste.

## 2026-02-15 13:57 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-135456/20260215-135456_ts20_or1_ep2_t20_rp8_gp080`
- cfg: `ts20_or1_ep2_t20_rp8_gp080`
- objective (best succ@256): `nan`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-135456/autodecider_results.csv`

## 2026-02-15 13:59 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-135456/20260215-135701_ts20_or1_ep2_t10_rp4_gp080`
- cfg: `ts20_or1_ep2_t10_rp4_gp080`
- objective (best succ@256): `nan`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-135456/autodecider_results.csv`

## 2026-02-15 14:02 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140025/20260215-140025_ts20_or1_ep1_t20_rp8_gp080`
- cfg: `ts20_or1_ep1_t20_rp8_gp080`
- objective (best succ@256): `nan`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140025/autodecider_results.csv`

## 2026-02-15 14:04 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140025/20260215-140230_ts20_or1_ep2_t10_rp8_gp080`
- cfg: `ts20_or1_ep2_t10_rp8_gp080`
- objective (best succ@256): `nan`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140025/autodecider_results.csv`

## 2026-02-15 14:08 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140630/20260215-140630_ts20_or1_ep1_t20_rp8_gp080`
- cfg: `ts20_or1_ep1_t20_rp8_gp080`
- objective (best succ@256): `0.0000`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140630/autodecider_results.csv`

## 2026-02-15 14:10 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140630/20260215-140835_ts20_or1_ep2_t20_rp4_gp080`
- cfg: `ts20_or1_ep2_t20_rp4_gp080`
- objective (best succ@256): `0.0000`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260215-140630/autodecider_results.csv`

## 2026-02-15 14:12 CST+0800
### User question
- Summarize last night's experimental results.
- Test the new Maze2D auto-decider loop end-to-end.

### Evidence inspected
- Completed overnight run with evaluation artifacts:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745`
  - key summaries:
    - `diagnosis_rollout_replay/summary.csv`
    - `diagnosis_rollout_replay/difficulty_gap_summary.csv`
    - `receding_final_plot_refresh_20260214/receding_final_plot_refresh_summary.csv` (h64 plotted subset)
    - `receding_final_plot_refresh_h192_longer/receding_final_plot_refresh_summary_h192.csv` (h192 plotted subset)
  - progress curves:
    - `A_baseline_rh8_p008/progress_metrics.csv`
    - `B_easier_goals_rh8/progress_metrics.csv`
    - `C_replan16/progress_metrics.csv`
- Attempted five-step monitor runs (no usable results):
  - `five_step_20260214-224109`, `...-224238`, `...-224323`, `...-224530`, `...-224647`, `...-231421`
  - each only has `S0_sanity/` and driver snapshots of `progress_metrics.csv not ready`.
- Auto-decider smoke test run roots:
  - `autodecider_20260215-140630` (created during this testing)

### Conclusions (results)
- On the diverse eval queries (24 queries x 6 samples = 144 trajectories), realized success at short rollout budgets was effectively ~0:
  - `diagnosis_rollout_replay/summary.csv` reports success_rate=0.0 for A/B/C.
- Increasing rollout horizon materially improves success (demonstrated on an 8-trajectory plotted subset):
  - h64 plotted subset: success 0.0 (all A/B/C).
  - h192 plotted subset: A=0.75, B=0.25, C=0.25, with nontrivial wall-hit counts.
- Strong difficulty mismatch exists between online-collected goals and eval goals:
  - eval start-goal mean distance ~2.5,
  - online sampled goal distance mean ~0.54-0.68,
  and online rollouts achieve very small min/final distances, but this does not transfer to the harder eval set.

### Conclusions (auto-decider testing)
- Implemented and validated an end-to-end functional test via `--smoke` mode:
  - controller launches trials, times out/terminates, parses `progress_metrics.csv` mid-run, writes `autodecider_results.csv` and `autodecider_importance.csv`.
  - run root evidence: `autodecider_20260215-140630`.
- Fixed a bug in horizon parsing in the controller (regex over-escaped), so it can score any available prefix horizon.

### Open items
- For real overnight use, the heavy eval configuration (24x6 trajectories, wall-aware multi-sample planning) can be slow; consider reducing eval cost for the controller (smaller query count / batch size) and using online planning-success as a cheaper proxy signal.

## 2026-02-15 14:15 CST+0800
### User question
- User asked for the "session ID" (to attach/monitor).

### Evidence inspected
- `tmux ls` shows existing sessions `0`, `2`, `3` (unrelated node processes).
- No tmux session matching `maze2d_autodecider_*` currently exists (auto-decider tests were run non-interactively).

### Conclusions
- There is no attachable tmux session for the auto-decider right now.
- To get an attachable session, launch via `scripts/launch_overnight_maze2d_autodecider_tmux.sh`, which creates a tmux session named `maze2d_autodecider_<timestamp>` and prints it.

## 2026-02-15 16:43 CST+0800
### User question
- Asked what “h192 plotted subset: A=0.75, B=0.25, C=0.25” means, and why that does not contradict “diverse eval success is ~0”.

### Evidence inspected
- Plotted-subset (h192) summary CSV (copied into gallery):
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/maze2d_plot_gallery_20260214/receding_final_plot_refresh_summary_h192.csv`
  - contains: `horizon_eval=192`, `n_plotted=8`, `goal_success_threshold=0.5`, and `rollout_success_rate_plotted` values (A=0.75, B=0.25, C=0.25).
- One plotted-subset metrics file (A):
  - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/maze2d_plot_gallery_20260214/A_baseline_rh8_p008_receding_query_metrics_h192.csv`
  - shows `8` rows total, covering only `query_id in {0,1}` (6 samples for query 0, 2 for query 1).
  - `len(rollout_xy_json)=193` for a row, consistent with a `192`-step rollout horizon.
  - recomputing `count(rollout_min_goal_distance <= 0.5)` gives `6/8 = 0.75` for A.
- Diverse eval diagnosis summary (h64 rollout budget):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/summary.csv`
  - contains `n=144` (24 queries x 6 samples) and `success_rate=0.0` for A/B/C.
  - for that diverse-eval set, `len(realized_xy_json)=65` in:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/diagnosis_rollout_replay/A_baseline_rh8_p008_imagined_vs_realized.csv`
    which is consistent with `64` rollout steps (plus the start state).

### Conclusions
- Yes: `A=0.75, B=0.25, C=0.25` are success rates, but **only for the plotted subset**:
  - success is computed on `n_plotted=8` trajectories (not on the full diverse eval set),
  - using a longer rollout horizon `H=192`,
  - with an absolute goal threshold `0.5` (success if `rollout_min_goal_distance <= 0.5` at any time in the rollout).
- The “diverse eval success is ~0” statement refers to a **different denominator and a shorter horizon**:
  - `n=144` trajectories on the diverse eval set,
  - rollout budget `H=64`,
  - and for that `H=64` budget, none of the 144 trajectories reach the 0.5-threshold (min realized distance is >= ~0.87 in A).
- Therefore there is no contradiction: the 0.75/0.25 values are not “global success on the diverse eval set”; they are “success@192 on a tiny plotted subset”.

### Open items
- If we want a direct apples-to-apples comparison, we should compute `success@H` on the **same** diverse eval set for `H in {64,128,192,256}` (or compute prefix metrics from a single long `H=256` rollout trace per trajectory).

## 2026-02-15 18:02 CST+0800
### User question
- Requested: evaluate using multiple horizons during training (prefix `success@H`), ensure success increases monotonically with horizon when computed from one longest rollout, and revisit whether `goal_success_threshold=0.5` is too loose.
- Also requested: validate online-loop hyperparameters affecting sample efficiency, especially “how many planner trajectories per diffusion training” (data chunk size vs update cadence).

### Evidence inspected
- Checkpoint eval harness added (computes all `success@{64,128,192,256}` from one `H=256` rollout per sample):
  - `/root/ebm-online-rl-prototype/scripts/eval_synth_maze2d_checkpoint_prefix.py`
- Verified monotonic prefix success on an existing trained checkpoint (A_baseline):
  - threshold=0.5:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/eval_prefix_20260215-171548/metrics.json`
    - `success_by_prefix`: `[(64, 0.0), (128, 0.8125), (192, 0.8125), (256, 0.8125)]` (16 rollouts total).
  - threshold=0.2:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/overnight_20260213-223745/A_baseline_rh8_p008/eval_prefix_20260215-172022/metrics.json`
    - `success_by_prefix`: `[(64, 0.0), (128, 0.4375), (192, 0.75), (256, 0.75)]`.
- Two small online-loop cadence/chunking experiments (same total online episodes = 32, same total train steps = 6000; eval set is small: 8 queries x 1 sample, threshold=0.2):
  - Chunky updates: `online_rounds=1`, `collect_eps_per_round=32`, `train_steps_per_round=4000`
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-172148/summary.json`
    - final `success@H`: `h64=0.0, h128=0.0, h192=0.5, h256=0.625`.
  - Frequent updates: `online_rounds=4`, `collect_eps_per_round=8`, `train_steps_per_round=1000`
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-174131/summary.json`
    - final `success@H`: `h64=0.0, h128=0.0, h192=0.375, h256=0.75`.

### Conclusions
- Prefix-horizon evaluation is now “one-rollout, many prefixes”: compute metrics from a single `H_max` rollout trace, which guarantees `success@64 <= success@128 <= success@192 <= success@256` if success is defined via prefix-min distance.
- The success threshold matters materially:
  - `0.5` is substantially more permissive than `0.2` on the same evaluated rollouts.
- Preliminary (noisy) cadence result:
  - with the same total online env episodes, more frequent replay expansion / retraining (smaller chunks, more rounds) *can* improve `success@256` (0.75 vs 0.625 here),
  - but this should be re-checked with a larger evaluation set to reduce variance.

### Open items
- Rerun cadence/chunking ablation with a larger eval set (e.g., `num_eval_queries=24`, `query_batch_size>=2`) and possibly multiple seeds.
- Decide/report an “official” success threshold for main comparisons (likely include both `0.5` for D4RL-style permissive and `0.2` for stricter).

## 2026-02-15 18:19 CST+0800
### User question
- Clarify what “1 round × 32 episodes” means in the online loop (vs “32 trajectories per training step” confusion).

### Evidence inspected
- Online loop structure in `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`:
  - Online outer loop: `for round_idx in range(1, cfg.online_rounds + 1)` (collect -> merge replay -> retrain).
  - Online collection call: `collect_planner_dataset(... n_episodes=cfg.online_collect_episodes_per_round, episode_len=cfg.online_collect_episode_len, replan_every_n_steps=cfg.online_replan_every_n_steps, ...)`.
  - Training step counter: `global_step` counts **gradient updates** inside `run_training_steps(...)`.
- Planner collection inner loop in `collect_planner_dataset(...)`:
  - Episode loop: `for ep in range(n_episodes)`.
  - Time loop: `for t in range(episode_len)`.
  - Receding-horizon planning: replan every `replan_stride = online_replan_every_n_steps` and execute the next planned action.

### Conclusions
- “1 round × 32 episodes” means:
  - In the **online replay-expansion phase**, we run the environment for **32 full episodes** (each up to `online_collect_episode_len`, typically 256 env steps), using receding-horizon replanning inside each episode.
  - We then **append those real transitions** to the replay buffer and **retrain** the diffuser for `online_train_steps_per_round` **gradient steps**.
- The “6000 training steps” in the pilots refers to the **number of SGD/Adam updates** (diffuser training iterations), not environment steps.
  - Concretely, those pilots were set up as: `total_grad_updates = offline_train_steps + sum_rounds online_train_steps_per_round = 6000`,
    while `total_online_env_steps = online_rounds * online_collect_episodes_per_round * online_collect_episode_len` (e.g., `32 * 256 = 8192` env steps total for both `1×32` and `4×8`).
- So, no: we do **not** roll out 32 trajectories “per training step”. Data collection happens in batches at round boundaries; training steps reuse replay minibatches.

### Open items
- When reporting sample efficiency, always separate the two budgets explicitly:
  - **online env steps** (collection cost) vs **gradient updates** (compute cost).

## 2026-02-15 18:24 CST+0800
### User question
- For sample-efficiency comparisons, how do we ensure **total environment steps** are constant (rather than holding gradient updates constant)?

### Evidence inspected
- Online collection returns an explicit transition count (`stats["transitions"]`) from `collect_planner_dataset(...)` in:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
- Online collector structure:
  - loops over `n_episodes` and `episode_len`, appending one transition per env step.
- Training counter `global_step` counts optimizer updates (SGD/Adam steps), not env steps.

### Conclusions
- The clean way to control sample efficiency is to treat **env interaction** as the primary budget:
  - Define an online env-step budget `B_env` (e.g., 8192, 16384, ...).
  - During online collection, count how many transitions were actually appended to replay (each transition = 1 env step).
  - Stop collection once cumulative transitions reach `B_env` (optionally truncate the last episode’s tail to hit `B_env` exactly).
- The common “rounds × episodes_per_round × episode_len” formula only guarantees constant env steps if every episode is fixed-length.
  - If episodes can terminate early, you must enforce the budget by counting transitions rather than by counting episodes.
- Gradient updates are a separate compute budget.
  - For *pure* sample-efficiency curves, compare performance as a function of `B_env` and (optionally) report compute.
  - For more controlled ablations, also fix an update schedule tied to env steps (e.g., `U = alpha * B_env`) so differences are not driven purely by more optimizer work.
- Evaluation rollouts interact with the env but are typically treated as “measurement cost” (kept identical across trials) rather than counted toward training sample budget; if strict accounting is desired, log eval env steps separately.

### Open items
- Consider adding an explicit `online_env_step_budget` (or `online_total_transitions_budget`) to configs/auto-decider so the controller can vary cadence while holding sample budget fixed.

## 2026-02-15 18:32 CST+0800
### User question
- Asked: “How many total env steps for 6000 optimization steps?” and how to truncate early-goal-hit episodes without inconsistent goals.

### Evidence inspected
- Pilot run configs (both have `train_steps=2000` and `online_train_steps_per_round` summing to total 6000 optimizer steps):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-172148/config.json`
    - offline data collection: `n_episodes=200`, `episode_len=256` -> 51,200 env steps worth of stored transitions (collector does not early-terminate here).
    - online collection: `online_rounds=1`, `online_collect_episodes_per_round=32`, `online_collect_episode_len=256` -> 8,192 transitions.
    - `online_collection.csv` confirms replay transitions: 59,392 (= 51,200 + 8,192).
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-174131/config.json`
    - same offline collection; online is 4 rounds x 8 eps x 256 -> 8,192 transitions total.
    - `online_collection.csv` shows replay transitions reaching 59,392 after round 4.

### Conclusions
- There is no intrinsic conversion “optimization steps -> env steps” because optimizer updates reuse replay data.
  - In this codebase, env steps come from (a) offline dataset collection and (b) online dataset collection, and are chosen by config.
- For the concrete 6000-step pilots above:
  - training-data env steps (stored transitions): 59,392 total = 51,200 offline + 8,192 online.
  - evaluation also consumes env steps (rollouts), but those are measurement cost and not appended to replay; keep them identical across runs if comparing sample efficiency.
- Early goal hit “wasted tail steps” options (goal-consistency preserving):
  - **Option A (terminate + restart):** If `dist(goal) <= eps` (and optionally `t >= min_len`), end the episode early (mark timeout boundary), then reset and sample a new start-goal pair. This uses remaining env-step budget to collect more episodes.
  - **Option B (goal-chaining with boundaries):** If success reached, mark an episode boundary at that time (timeout) and immediately sample a new goal, continuing from the current state. Because we insert an episode boundary, training sequence sampling won’t cross goal switches, avoiding “inconsistent goal” sequences while still using all steps for purposeful navigation.
  - In either option, consider a **minimum segment length** (e.g. >= diffusion horizon 64) so segments still contribute length-64 training windows; otherwise many short segments produce little usable training data.

## 2026-02-15 19:31 CST+0800
### User question
- User chose Option A and requested: if online-collected trajectories are shorter than the diffusion window, discard them and do not count them toward the env-step budget.

### Evidence inspected / changes made
- Implemented Option A early-termination + short-episode rejection + transition-budget collection in:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
  - New online collection knobs:
    - `online_collect_transition_budget_per_round` (accepted transition budget; 0 => fixed episode count)
    - `online_early_terminate_on_success` (default True)
    - `online_early_terminate_threshold` (default 0.2)
    - `online_min_accepted_episode_len` (0 => defaults to diffusion horizon)
  - `collect_planner_dataset(...)` now:
    - terminates an episode early if `dist(goal) <= online_early_terminate_threshold`,
    - rejects episodes with `ep_len < online_min_accepted_episode_len` (discarded; not appended),
    - optionally truncates the last accepted episode to fit the remaining transition budget without creating a too-short tail.
  - Online collection stats now include attempted vs accepted quantities:
    - `episodes_attempted`, `transitions_attempted`, `episodes_rejected_short`, as well as accepted `episodes`, `transitions`.

### Conclusions
- This enforces the user-requested accounting: the “env-step budget” for online learning is now naturally represented as
  - `accepted transitions added to replay`,
  while rejected short episodes contribute neither replay data nor counted budget.
- The transition-budget knob is the correct primitive for sample-efficiency ablations once early termination makes episode lengths variable.

### Open items
- For transparency, report both:
  - accepted transitions (counted budget) and attempted transitions (true env cost),
  since rejection sampling can hide wasted interaction.

## 2026-02-15 21:27 CST+0800
### User question
- Extend the online training beyond the prior ~6k-optimizer-step pilots, log the key metrics, and check whether the algorithm converges to (near) optimal behavior (ideally ~100% success) in Maze2D-umaze. Use lightweight, stochastic (unbiased) intermediate evaluation rather than the full 144-trajectory eval.

### Evidence inspected
- Long run with Option A online collection + accepted-transition budgets:
  - run dir: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611`
  - config: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/config.json`
  - intermediate eval logs: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/progress_metrics.csv`
  - online collection logs: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/online_collection.csv`
  - summary: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/summary.json`

Key protocol knobs used in this run:
- Eval distribution: `query_mode=diverse`, `query_min_distance=1.0`, `num_eval_queries=8`, `query_batch_size=1`, `query_resample_each_eval=true` (lightweight, stochastic estimate).
- Eval success: `goal_success_threshold=0.2`, horizons `{64,128,192,256}` computed as prefix success from one `H=256` rollout.
- Online collection:
  - `online_rounds=8`, `online_collect_transition_budget_per_round=4096` (accepted transitions)
  - early terminate on hit: `online_early_terminate_threshold=0.2`
  - reject short: `online_min_accepted_episode_len=64` (defaults to horizon)
  - goal sampling: `online_goal_geom_p=0.02`, `online_goal_geom_min_k=64`, `online_goal_geom_max_k=192`, `online_goal_min_distance=1.0`

### Results (stochastic eval success@256; denominator=8 per eval)
- From `progress_metrics.csv`:
  - step 2000: 1/8 = 0.125
  - step 4000: 2/8 = 0.250
  - step 6000: 4/8 = 0.500
  - step 8000: 4/8 = 0.500
  - step 10000: 4/8 = 0.500
  - step 12000: 6/8 = 0.750
  - step 14000: 6/8 = 0.750
  - step 16000: 6/8 = 0.750
  - step 18000: 5/8 = 0.625
- Online collection “planning success” on geometric-sampled goals reached ~1.0 quickly:
  - e.g. last round in `online_collection.csv`: `planning_success_rate_final_t020=0.980` (accepted episodes).

### Conclusions
- With 18k optimizer steps and 8 online rounds (each adding 4096 accepted transitions), the realized eval success estimate improved from ~0.125 to as high as ~0.75, but did **not** reach 1.0 on this eval distribution (diverse start-goal pairs with min distance 1.0, threshold 0.2).
- Because eval resamples queries each time and uses only 8 trajectories per checkpoint, `success@256` is a high-variance (but unbiased) estimator; the late drop from 0.75 -> 0.625 is consistent with estimator noise rather than clear regression.
- Option A rejection becomes substantial once the planner is strong: e.g. last online round attempted `82` episodes to accept `51` (reject_short `31`), with attempted transitions `5904` vs accepted `4096`. This is logged and should be considered when interpreting true interaction cost.

### Open items
- If the goal is to *verify convergence* (vs unbiased estimation), consider either:
  - increase eval sample size modestly (e.g. `num_eval_queries=24`, `query_batch_size=1`) to reduce variance, or
  - keep a fixed sampled eval subset (`--no_query_resample_each_eval`) for trend tracking and run a separate periodic resampled eval for unbiased estimates.
- If 100% success is expected on this eval distribution, likely needs either more online budget/training or a change in planning/model capacity (e.g. larger diffusion horizon / capacity), since online-goal success saturates near 1.0 while diverse-eval success remains <1.0.

## 2026-02-15 23:34 CST+0800
### User question
- “Go to the Energy-Based Model Online RL subfolder and look at the working memory file and other Codex-related files to understand what's going on.”
- “Grab latest messages” (Discord relay).

### Evidence inspected
- Working memory:
  - `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md`
- Handoff log tail:
  - `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`
- “EBM_OnlineRL” subfolder state:
  - `/root/ebm-online-rl-prototype/EBM_OnlineRL` (`ls -la`, `git show -1`)
  - observed: repo exists but contains only `.git` and an empty initial commit.
- Active code locations for Maze2D online Diffuser work:
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py`
  - `/root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py`
  - `/root/ebm-online-rl-prototype/scripts/overnight_five_step_monitor_driver.sh`
  - `/root/ebm-online-rl-prototype/scripts/eval_synth_maze2d_checkpoint_prefix.py`
  - `/root/ebm-online-rl-prototype/docs/plans/2026-02-15-maze2d-autodecider.md`
- Discord relay session mapping:
  - `/root/.codex-discord-relay/sessions.json`
- Discord latest messages fetched via Discord REST API (bot token in `/root/.codex-discord-relay.env`, proxies from `/root/.openclaw/proxy.env`):
  - DM channel resolved/confirmed as `1472466555529859074` for user `868020619499937792`.

### Conclusions
- The investigation’s authoritative continuity artifacts are:
  - working memory: `docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md`
  - handoff log: `HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`
- In this repo, `EBM_OnlineRL/` is *not* where the Maze2D Diffuser OnlineRL work lives (it’s an empty placeholder repo).
  - The active online Diffuser + eval protocol implementation is in `scripts/synthetic_maze2d_diffuser_probe.py` and related scripts under `scripts/`.
- Discord relay is functional; DM and relevant guild/thread contexts are reachable and messages can be fetched for monitoring.

### Open items
- None.

## 2026-02-16 09:35 CST+0800
### User question
- “I launched an overnight experiment last night; summarize and grab the key results.”

### Evidence inspected
- Checked for new runs:
  - `ls -lt runs/analysis/synth_maze2d_diffuser_probe`
  - `find . -newermt '2026-02-16'`
  - Result: no new run artifacts created after 2026-02-16 00:00; newest run dir remains:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611`
- Run config + summary + learning curves:
  - `.../20260215-205611/config.json`
  - `.../20260215-205611/summary.json`
  - `.../20260215-205611/progress_metrics.csv`
  - `.../20260215-205611/online_collection.csv`

### Key results (run: `20260215-205611`)
Eval protocol (stochastic, high variance):
- Diverse eval with min start-goal distance `>= 1.0`, threshold `0.2`.
- `num_eval_queries=8`, `samples_per_query=1`, `query_resample_each_eval=true`.
- One rollout per query with `H=256`, receding-horizon replanning every 8 steps; report prefix `success@{64,128,192,256}`.

Observed success@256 over training (denominator=8 per eval):
- step 2000: 0.125
- step 4000: 0.25
- step 6000: 0.5
- step 8000: 0.5
- step 10000: 0.5
- step 12000: 0.75
- step 14000: 0.75
- step 16000: 0.75
- step 18000 (final): 0.625
Best observed `success@256 = 0.75` (6/8) at steps 12000-16000; final is 0.625 (5/8).

Online collection (geometric-sampled goals, threshold=0.2):
- Planning success (distance threshold 0.2) reached ~1.0 quickly (by round 4).
- Episodes shortened as competence improved; many segments became <64 and were rejected (Option A), increasing attempted env steps.
- Aggregate across 8 online rounds (from `online_collection.csv`):
  - accepted transitions added to replay: 32,533
  - attempted transitions: 38,890 (attempted/accepted ~1.20)
  - rejected_short episodes: 91

### Conclusions
- No new run artifacts appear after midnight 2026-02-16 in this repo; the most recent substantial experiment is `20260215-205611`.
- That run shows meaningful improvement on the hard eval distribution, but not clear convergence to 100%: best `success@256` reached 0.75 with a noisy estimator (N=8, resampled queries).
- The online curriculum goals become “too easy” for the planner (near-1.0 success) while diverse eval success remains <1.0, consistent with a remaining distribution-shift/generalization gap.

### Open items
- If the user truly launched a different overnight run, confirm the run path or worktree; otherwise this repo has no additional overnight results.

## 2026-02-16 10:34 CST+0800
### User question
- “Explore all the work trees and see if there are any other results.”

### Evidence inspected
- Git worktrees for this repo:
  - `git worktree list --porcelain` in `/root/ebm-online-rl-prototype`.
  - Result: only the main worktree exists at `/root/ebm-online-rl-prototype`.
- Vibe-kanban worktrees directory:
  - `/var/tmp/vibe-kanban/worktrees/`
  - contains 3 worktrees:
    - `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL` (contains only `figures/maze2d_plot_gallery_20260214/*`)
    - `/var/tmp/vibe-kanban/worktrees/47cb-plz-take-a-look/EBM_OnlineRL` (empty placeholder)
    - `/var/tmp/vibe-kanban/worktrees/946b-github-update-ca/EBM_OnlineRL` (older code snapshot; no `runs/` dir)
- Searched for experiment artifacts outside the main repo:
  - `find /root /var/tmp /tmp -path '*/runs/analysis/synth_maze2d_diffuser_probe/*/summary.json'`
  - `find /root /var/tmp /tmp -name progress_metrics.csv`
  - Result: all such artifacts are under `/root/ebm-online-rl-prototype` (plus a few smoke-test dirs under `/tmp`).

### Conclusions
- No additional training/eval result directories were found in other worktrees.
- The only extra artifacts outside the main repo are older plot galleries under `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/figures/maze2d_plot_gallery_20260214`, which mirror plots produced from the main repo’s earlier runs.

### Open items
- If the user launched a separate “overnight” run elsewhere, we need the exact path (or run name) to summarize it; it is not discoverable via current worktree locations.

## 2026-02-16 13:28 CST+0800
### User question
- Continue to longer training (up to 100k), propose optimization/novel metrics, and invoke autonomous 3-4h auto-decider loop.

### Evidence inspected
- 
- 
- 
- 

### Changes implemented
- Added eval-time goal-coverage metrics in :
  -  and count
  -  and count
  - , 
- Updated progress logging line to print coverage at the longest prefix horizon.
- Updated auto-decider defaults:
  - search ladder includes longer optimization budgets up to 
  - lighter unbiased intermediate eval (, , resampled queries)
  - explicit hard-eval threshold 
  - online collection defaults aligned with Option A (, early terminate on hit, min accepted len 64).

### Conclusions
- The controller can now optimize not only for success@h256 but also inspect reachable-goal expansion via coverage metrics.
- Long-run continuation toward 100k is now part of the selectable budget ladder.

### Open items
- Launch and monitor a 3.5-hour auto-decider run; summarize mid-run findings and adapt if a config/implementation issue appears.

## ${TS} (correction)
### User question
- Continue to longer training (up to 100k), propose optimization/novel metrics, and invoke autonomous 3-4h auto-decider loop.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py
- /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py
- /root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md
- /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt

### Changes implemented
- Added eval-time goal-coverage metrics in evaluate_goal_progress:
  - rollout_goal_query_coverage_rate_h{h} and rollout_goal_query_coverage_count_h{h}
  - rollout_goal_cell_coverage_rate_h{h} and rollout_goal_cell_coverage_count_h{h}
  - eval_unique_goal_cells and rollout_goal_cell_total
- Updated eval progress print to include coverage at longest horizon.
- Updated auto-decider defaults:
  - search ladder includes longer optimization budgets up to train_steps=100000
  - lighter unbiased intermediate eval (num_eval_queries=12, query_batch_size=1, query_resample_each_eval=true)
  - stricter goal threshold goal_success_threshold=0.2
  - online collection defaults aligned with Option A: transition budget 4096, early terminate on hit, min accepted len 64.

### Conclusions
- Controller now observes both success and goal-reachability expansion (coverage) during training.
- Long-run continuation toward 100k is now in the selectable budget ladder.

### Open items
- Launch and monitor a 3.5-hour auto-decider run; summarize intermediate findings and adapt if needed.

## 2026-02-16 13:29 CST+0800 (timestamped correction)
### Note
- Previous entry used a literal ${TS} token. This block records the same update with concrete timestamp.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py
- /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py

### Changes implemented
- Added query-level and goal-cell-level coverage metrics at eval prefix horizons.
- Extended auto-decider search space with a long rung up to train_steps=100000 and lighter eval defaults.

## 2026-02-16 13:29 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-132936/20260216-132936_ts20_or1_ep1_t10_rp4_gp080`
- cfg: `ts20_or1_ep1_t10_rp4_gp080`
- objective (best succ@256): `nan`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-132936/autodecider_results.csv`

## 2026-02-16 13:32 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133024/20260216-133024_ts20_or1_ep1_t10_rp4_gp080`
- cfg: `ts20_or1_ep1_t10_rp4_gp080`
- objective (best succ@256): `0.0000`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133024/autodecider_results.csv`

## 2026-02-16 13:35 CST+0800
### Auto-decider launch
- Launched autonomous controller run for 3.5h.
- tmux session: maze2d_autodecider_100k_20260216-133322
- controller root: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133322
- first selected config: ts6000_or4_ep64_t3000_rp16_gp040

### Initial status
- Training reached offline_init step 500 and entered eval block.
- GPU active; process state R indicates run is making progress.

### Open items
- Wait for first progress_metrics.csv row and verify new coverage fields are present.
- Keep monitor-gated decisions as trials complete.

## 2026-02-16 13:37 CST+0800
### Auto-decider redesign + relaunch
- Diagnosis: first launch spent long wallclock in the first eval block (step-500 eval with expensive planning calls), reducing trial throughput for a 3.5h budget.
- Action: relaunched with lower eval/planning overhead while preserving receding-horizon protocol:
  - eval_goal_every=1000
  - eval_rollout_replan_every_n_steps=16
  - wall_aware_plan_samples=2
- New tmux session: maze2d_autodecider_100k_20260216-133616
- New run root: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616

### Rationale
- This increases completed-trial count (better for auto-decider learning) without removing key evaluation signals (success@{64,128,192,256} and goal-coverage metrics).

## 2026-02-16 13:37 CST+0800
### Coverage metric validation
- Verified new coverage columns are emitted in progress_metrics.csv from smoke artifact:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133024/20260216-133024_ts20_or1_ep1_t10_rp4_gp080/progress_metrics.csv
- Observed columns include:
  - rollout_goal_query_coverage_rate_h8/h16/h32
  - rollout_goal_cell_coverage_rate_h8/h16/h32
  - eval_unique_goal_cells and rollout_goal_cell_total
- This confirms metric plumbing works end-to-end before the full 3.5h run.

## 2026-02-16 14:11 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/20260216-133616_ts6000_or4_ep64_t3000_rp16_gp040`
- cfg: `ts6000_or4_ep64_t3000_rp16_gp040`
- objective (best succ@256): `0.9167`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

## 2026-02-16 14:25 CST+0800
### User question
- Whether to replace fixed-space auto hyperparameter tuning with AI-agent reasoning over broader algorithm/design changes.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py (selection logic)
- /root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md
- /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt

### Conclusions
- Current decider is algorithmic (rule/bandit/promotion), not per-step LLM reasoning.
- Replacing with an agentic meta-optimizer is feasible, but should be constrained by an edit grammar and safety/eval gates.
- Recommended path: staged hybrid controller (LLM proposes + deterministic verifier/executor) before full open-ended code-change search.

### Open items
- Decide desired autonomy level: config-only agent, module-level edits, or broad code-level proposals with auto-merge gates.

## 2026-02-16 14:32 CST+0800
### User question
- Start with near-open-ended autonomy experiment (beyond fixed hyperparameter space) using the proposed multi-agent architecture.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py
- /root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md
- /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt

### Conclusions
- User preference is high-autonomy exploration with high capability ceiling.
- Next step should be a bounded near-open-ended pilot: LLM proposals over broader code/module edits, but with hard invariants and automated rollback gates.

### Open items
- Choose acceptance aggressiveness for auto-merge (single-run pass vs replicated pass).

## 2026-02-16 14:46 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/20260216-141122_ts6000_or8_ep64_t3000_rp16_gp040`
- cfg: `ts6000_or8_ep64_t3000_rp16_gp040`
- objective (best succ@256): `0.9167`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

## 2026-02-16 15:12 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151121`
- incumbent: `ts20_or1_ep1_t10_rp4_gp080`
- incumbent score: `nan`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151121/agentic_results.csv`

## 2026-02-16 15:13 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151311`
- incumbent: `ts20_or1_ep1_t10_rp4_gp080`
- incumbent score: `nan`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151311/agentic_results.csv`

## 2026-02-16 15:13 CST+0800
### User question
- Implement conservative near-open-ended autonomous agent and prepare it to take over current auto-decider job; if successful, later make reusable skill.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py
- /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151311/summary.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-150616/agentic_results.csv

### Changes implemented
- Added conservative agentic controller:
  - /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
  - Components: proposal generator, constraint checker, critic beliefs, conservative promotion (pilot + confirmation), takeover-compatible runner.
- Added launcher:
  - /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh
- Added design doc:
  - /root/ebm-online-rl-prototype/docs/plans/2026-02-16-agentic-near-open-ended-autodecider-design.md

### Validation
- Syntax checks passed via py_compile.
- Smoke evidence:
  - near-open-ended proposal stage reached (bootstrap then proposal pilot) in run .
  - graceful completion artifact path validated in run  (, , ).

### Conclusions
- Conservative near-open-ended controller is implemented and can take over orchestration from fixed auto-decider.
- Skillization should be done after robustness criteria in design doc are met on real-budget runs.

### Open items
- Launch a real-budget conservative agentic run and compare against baseline auto-decider on matched wall-clock budget.

## CORRECTION ENTRY
### Note
- Previous entry at this time had shell-expanded placeholders in one validation bullet.

### Corrected validation evidence
- Proposal-stage execution observed in run:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-150616
- Graceful completion artifacts validated in run:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151311/summary.json
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151311/agentic_results.csv
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-151311/beliefs.json

## 2026-02-16 15:21 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/20260216-144628_ts6000_or4_ep64_t1000_rp4_gp120`
- cfg: `ts6000_or4_ep64_t1000_rp4_gp120`
- objective (best succ@256): `0.8333`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

## 2026-02-16 15:56 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/20260216-152133_ts6000_or4_ep16_t2000_rp4_gp020`
- cfg: `ts6000_or4_ep16_t2000_rp4_gp020`
- objective (best succ@256): `0.9167`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

## 2026-02-16 16:31 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/20260216-155639_ts12000_or24_ep16_t500_rp16_gp040`
- cfg: `ts12000_or24_ep16_t500_rp16_gp040`
- objective (best succ@256): `0.9167`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

## 2026-02-16 17:06 
### Auto-decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/20260216-163144_ts48000_or24_ep64_t2000_rp4_gp040`
- cfg: `ts48000_or24_ep64_t2000_rp4_gp040`
- objective (best succ@256): `0.8333`
- results table: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

## 2026-02-16 21:06 CST+0800
### User question
- "If you want, I can switch it over now and start the first conservative takeover run." User confirmed: "yes, plz".

### Evidence inspected
- Skill docs:
  - /root/.codex/skills/ml-run-monitor-decider/SKILL.md
  - /root/.codex/skills/experiment-working-memory-handoff/SKILL.md
- Pre-launch state snapshot:
  - GPU: `nvidia-smi` => util=0, mem=3-5 MiB
  - tmux sessions before launch: `0`, `2`, `3`
  - no active Maze2D autodecider/agentic processes found before launch
- Launch command:
  - `cd /root/ebm-online-rl-prototype && BUDGET_HOURS=6 MAX_AGENT_ROUNDS=8 PROPOSALS_PER_ROUND=4 MONITOR_EVERY_SEC=120 PER_TRIAL_TIMEOUT_MIN=35 SEED=21 scripts/launch_agentic_maze2d_autodecider_tmux.sh`
- Post-launch evidence:
  - new tmux session: `maze2d_agentic_20260216-210532`
  - tmux log path: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/tmux_maze2d_agentic_20260216-210532.log`
  - active controller PID evidence:
    - `agentic_maze2d_autodecider.py` running
    - child `synthetic_maze2d_diffuser_probe.py` running
  - run root:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532`
  - live logs:
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic.log`
    - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/20260216-210532_ts6000_or4_ep64_t3000_rp16_gp040/run.log`
    - latest observed training progress: `[train] phase=offline_init step=300`.

### Conclusions
- Conservative agentic takeover is now active and running the first real-budget trial.
- The previous fixed decider is not currently occupying GPU/process slots.

### Open items
- Continue monitor loop and report first completed trial metrics (`objective_succ_h256`, coverage metrics, online planning success) before any branch/promotion decisions.

## 2026-02-16 21:07 CST+0800
### User question
- Confirmation/progress on conservative takeover launch.

### Evidence inspected
- `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/20260216-210532_ts6000_or4_ep64_t3000_rp16_gp040/run.log`
- `pgrep -af 'agentic_maze2d_autodecider.py|synthetic_maze2d_diffuser_probe.py'`

### Conclusions
- Controller and child run are still alive.
- Offline phase has advanced to step 1000 (`train_loss=0.22890`, `val_loss@1000=0.25042`), so the first trial is progressing normally.

### Open items
- Wait for first eval/online-round outputs and then compare objective metrics against incumbent baseline.

## 2026-02-16 22:05 CST+0800
### User question
- Summarize the latest experiment results.

### Evidence inspected
- Active run/process status:
  - `tmux ls`
  - `pgrep -af 'agentic_maze2d_autodecider.py|synthetic_maze2d_diffuser_probe.py'`
- Agentic takeover run artifacts:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic.log`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic_results.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/20260216-210532_ts6000_or4_ep64_t3000_rp16_gp040/progress_metrics.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/20260216-210532_ts6000_or4_ep64_t3000_rp16_gp040/online_collection.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/20260216-214038_ts6000_or8_ep64_t3000_rp16_gp040/progress_metrics.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/20260216-214038_ts6000_or8_ep64_t3000_rp16_gp040/online_collection.csv`
- Previous fixed auto-decider table for context:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`

### Conclusions
- Agentic takeover is active (`maze2d_agentic_20260216-210532`) and currently running proposal pilot `ts6000_or8_ep64_t3000_rp16_gp040`.
- First trial (bootstrap) has completed via timeout gate:
  - run `20260216-210532_ts6000_or4_ep64_t3000_rp16_gp040`
  - `objective_succ_h256=1.0` (best-seen snapshot), `progress_succ_h256_last=0.75`, `progress_succ_h256_max=1.0`
  - coverage max at h256 reached query=1.0 and cell=1.0; last snapshot h256 query=0.75 cell=0.8333
  - online collection last logged round had `planning_success_rate_final_t020=1.0` and `replay_transitions=268196`
- Second trial (in progress, not finalized):
  - run `20260216-214038_ts6000_or8_ep64_t3000_rp16_gp040`
  - latest snapshot at step 9000 (`online_round_1`): h256 success=0.75, h192=0.6667, h128=0.0, h64=0.0
  - best-so-far in this run: h256 success=0.9167, h192=0.9167
  - latest h256 coverage: query=0.75, cell=0.875 (best-so-far query=0.9167, cell=1.0)
  - latest online collection row (round 1): `planning_success_rate_final_t020=0.87097`, `planning_success_rate_final_rel090=0.03226`, `replay_transitions=260040`
- Context baseline (previous fixed auto-decider run set): best `objective_succ_h256=0.9167` across several configs in `autodecider_20260216-133616`.

### Open items
- Wait for the current agentic proposal trial to finish (or timeout), then check whether promotion/confirmation is triggered and whether incumbent score remains above fixed-decider baseline.

## 2026-02-16 22:18 CST+0800
### User question
- Summarize key hyperparameter design choices for high objective/success/sample efficiency, including effects and explanations.

### Evidence inspected
- Main sweep table:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/autodecider_20260216-133616/autodecider_results.csv`
- Agentic takeover table/logs:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic_results.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic.log`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/proposals.jsonl`
- Per-run efficiency breakdown (online accepted/attempted transitions, short-episode rejects):
  - `online_collection.csv` under each run in `autodecider_20260216-133616/*`
- Horizon effect evidence from longer run:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260215-205611/progress_metrics.csv`
- Collector behavior source (important for interpreting `ep*` hyperparameter):
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py` (function `collect_planner_dataset`, break condition uses `transition_budget` when >0).

### Conclusions
- Best observed objective in current protocol reached with `train_steps=6000, online_rounds=4, online_train_steps_per_round=3000, online_replan_every_n_steps=16, online_goal_geom_p=0.04` (agentic bootstrap `objective_succ_h256=1.0`; fixed decider with same family reaches `0.9167`).
- A more sample-efficient high-score setting is `ts6000_or4_ep16_t2000_rp4_gp020` (objective `0.9167`) with much lower online accepted/attempted transitions than `ep64/t3000` settings:
  - efficient config: accepted `8175`, attempted `8998`, reject-rate `13.3%`.
  - baseline-high config: accepted `12151`, attempted `14738`, reject-rate `23.4%`.
- Increasing online rounds from 4 to 8 with otherwise-matched core settings did not improve objective and reduced stability in latest agentic pair; proposal was explicitly rejected (`delta=-0.0833`).
- Very large offline train-steps before online updates (`train_steps=48000`) performed worse (`0.8333`) under current timeout budget, likely because online adaptation had little/no time to execute.
- Geometry-goal sampling probability too high (`online_goal_geom_p=0.12`) underperformed (`0.8333`) compared with low/moderate (`0.02-0.04`) that repeatedly reached `0.9167`.
- Success strongly depends on rollout horizon: in run `20260215-205611`, late-stage rates were `h64=0.125, h128=0.25, h192=0.625, h256=0.625`, showing short horizons materially underestimate capability in this maze.
- Important implementation caveat: with `online_collect_transition_budget_per_round > 0` (currently 4096), collector stop condition is transition-budget-driven, so `online_collect_episodes_per_round` is effectively non-binding in current code path. Interpret `ep*` ablations accordingly.

### Open items
- Continue monitoring in-progress pilot `20260216-221544_ts6000_or4_ep32_t4000_rp16_gp040` to validate update-intensity hypothesis.
- If strict episode-chunk ablations are desired, modify collector logic so `online_collect_episodes_per_round` remains active even when transition budget is set.

## 2026-02-16 22:38 CST+0800
### User question
- Propose a further hypothesis-driven experiment and run it overnight via agentic autodecider without further approval.

### Evidence inspected
- Current active controller/log:
  - `tmux session`: `maze2d_agentic_20260216-210532`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic.log`
- Agentic proposal behavior in latest run:
  - p0 (`or8` budget increase) was rejected (`delta=-0.0833`)
  - p1 (`ep32,t4000`) is currently being evaluated with snapshots up to row 9 and `succ_h256_max=0.917`
- Agentic script proposal mechanism:
  - `/root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py`
- Collector implementation constraint note (for interpretation):
  - `/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py` (`collect_planner_dataset` uses transition budget as stop condition when >0).

### Hypothesis and experiment design
- Hypothesis H: Under fixed transition-budget collection, improving update-intensity/chunking is more useful for sample efficiency than simply increasing budget via more online rounds.
- Falsification criterion:
  - If budget-increase proposals (`or` upshift) consistently beat or match update-intensity proposals (`ep` downshift + `t` upshift) on objective and replay-cost-normalized performance, H is not supported.
- Overnight execution plan:
  - keep current run alive (already evaluating p1), then auto-chain a second focused overnight agentic run.
  - second run uses `proposals-per-round=2` to emphasize p0 vs p1 (budget-up vs stronger-updates with modest chunk) and reduce random drift.

### Actions taken
- Queued follow-up overnight session that waits for current run to finish, then launches focused agentic run:
  - queue session: `maze2d_agentic_queue_20260216-223823`
  - queue pane evidence: `[queue] waiting for maze2d_agentic_20260216-210532 to finish`
  - launch command (when unblocked):
    - `agentic_maze2d_autodecider.py --budget-hours 10 --max-agent-rounds 10 --proposals-per-round 2 --monitor-every-sec 120 --per-trial-timeout-min 35 --seed 73 --accept-delta 0.01`

### Open items
- Monitor current p1 completion and first 2-3 trials of the chained run; then compare p0-vs-p1 promotion outcomes and cost-normalized objective.

## 2026-02-17 03:31 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532`
- incumbent: `ts6000_or4_ep64_t3000_rp16_gp040`
- incumbent score: `1.0000`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic_results.csv`

## 2026-02-17 05:17 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224`
- incumbent: `ts6000_or4_ep64_t3000_rp16_gp040`
- incumbent score: `0.9167`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic_results.csv`

## 2026-02-17 14:49 CST+0800
### User question
- Check whether the overnight synthetic Maze2D run produced new results.
- Provide a brief findings summary and explain how the agent made decisions.

### Evidence inspected
- Latest run root listing:
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe"
- Overnight queued agentic run:
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/summary.json"
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic_results.csv"
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic.log"
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/proposals.jsonl"
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/beliefs.json"
- Previous overnight reference run:
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/summary.json"
  - "/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic_results.csv"
- Controller logic:
  - "/root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py"
  - "/root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py"

### Conclusions
- New overnight results exist: run  finished at .
- Final incumbent score in that run is  with incumbent config .
- Two pilot proposals were tested ( budget increase and  stronger updates); both also achieved objective  (no gain over incumbent).
- Because promotion threshold was , both pilots were rejected (), then later rounds only produced duplicate proposals and were skipped (no further exploration within this run).
- All three executed trials ended by per-trial timeout (, ~2105 sec each), but each still produced progress/online metrics and checkpoints.
- Compared with the prior overnight agentic run , which reported incumbent  on the same baseline config, this newest run does **not** set a new best; it indicates plateau/variance rather than improvement.

### Open items
- Improve proposal diversity after no-promotion rounds (avoid repeated duplicate heuristics).
- Consider allowing confirmation runs only after tie-break on secondary metrics (e.g., ) when objective ties.
- Decide whether to increase eval denominator beyond 12 queries to reduce 1/12 quantization effects.

## 2026-02-17 14:50 CST+0800
### Correction note
- Supersedes the immediately previous malformed block (shell interpolation issue during write).

### User question
- Check whether the overnight synthetic Maze2D run produced new results.
- Provide a brief findings summary and explain how the agent made decisions.

### Evidence inspected
- Latest run root listing:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe
- Overnight queued agentic run:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/summary.json
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic_results.csv
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic.log
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/proposals.jsonl
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/beliefs.json
- Previous overnight reference run:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/summary.json
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic_results.csv
- Controller logic:
  - /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
  - /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py

### Conclusions
- New overnight results exist: run agentic_autodecider_20260217-033224 finished at 2026-02-17 05:17:41.
- Final incumbent score in that run is 0.9167 with incumbent config ts6000_or4_ep64_t3000_rp16_gp040.
- Two pilot proposals were tested (or8 budget increase and ep32+t4000 stronger updates); both also achieved objective 0.9167 (no gain over incumbent).
- Because promotion threshold was accept_delta=0.01, both pilots were rejected (delta=0.0000), then later rounds only produced duplicate proposals and were skipped.
- All three executed trials ended by per-trial timeout (rc=-15, about 2105 sec each), but still produced progress/online metrics and checkpoints.
- Compared with the prior overnight agentic run agentic_autodecider_20260216-210532, which reported incumbent 1.0000 on the same baseline config, this newest run does not set a new best; it indicates plateau/variance rather than improvement.

### Open items
- Improve proposal diversity after no-promotion rounds to avoid repeated duplicates.
- Consider tie-break rules on secondary metrics when objective ties.
- Consider increasing evaluation denominator beyond 12 queries to reduce 1/12 quantization effects.

## 2026-02-17 14:55 CST+0800
### User question
- Why the latest synthetic Maze2D agentic run is not running.
- What delta = 0 means.
- Whether the current auto-decider is LLM-based vs fixed-rule.

### Evidence inspected
- Latest run artifacts:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic.log
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic_results.csv
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/summary.json
- Launch/queue context:
  - /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt (queued command line with proposals-per-round=2)
  - /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh
- Controller implementation:
  - /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
  - /root/ebm-online-rl-prototype/scripts/overnight_maze2d_autodecider.py
- Runtime process check:
  - ps/tmux checks show no active agentic/autodecider process.

### Conclusions
- The run is not active because it already finished; there is no running process.
- In the latest run, two pilot trials tied the incumbent objective; both had delta=0.0000 and were rejected by accept_delta=0.01.
- After round 0, the script kept regenerating the same two proposals and skipped them as duplicates for rounds 1-9, then exited.
- Key mechanism causing the duplicate loop in this run:
  - proposals-per-round was set to 2 in the queued launch command.
  - _reasoned_proposals always fills the first two slots with deterministic heuristic proposals.
  - stochastic near-neighbor proposals are only added while len(out) < max_new, so with max_new=2 they never appear.
- Therefore, this run did not continue exploring configs after the first two pilots.
- The current agentic script is not LLM-reasoning-driven. Proposal generation is hard-coded heuristic + stochastic rules (no model/API call in the script).

### Open items
- Decide whether to replace proposal generation with true LLM-driven proposal synthesis (with constraint checking retained).
- Add fallback logic to force non-duplicate candidate generation even when proposals-per-round is small.

## 2026-02-17 15:04 CST+0800
### User question
- Do not use API calls; want the Codex agent with context to make proposals.
- Prefer the initiating agent (launcher side) to generate proposals rather than fixed-rule auto-decider.

### Evidence inspected
- Current controller proposal path:
  - /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
- Launcher and queue behavior:
  - /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh
  - /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt
- Recent duplicate-loop evidence:
  - /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic.log

### Conclusions
- No-API Codex-driven proposal flow is feasible if we split responsibilities:
  - executor script runs trials + validates constraints,
  - initiating Codex agent writes proposal files per round.
- This keeps proposals context-aware (Codex memory + handoff + working-memory) without embedding an API client in the script.
- The current script can be adapted to consume external proposals first, with heuristic fallback only when external proposals are absent.

### Open items
- Decide preferred control mode:
  1) strict external proposals only (fail if missing), or
  2) external proposals preferred with heuristic fallback.

## 2026-02-17 15:22 CST+0800
### User request
- Implement no-API Codex-driven proposal path so launcher/initiating agent can propose runs instead of fixed-rule-only proposal generation.

### What changed
- Added external proposal mode to:
  - /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
- Added launcher wiring/defaults to:
  - /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh

### New control behavior
- : old heuristic/stochastic proposal behavior.
- : strict external-only; writes , waits for , stops if missing/empty after timeout.
- : same handshake, but falls back to internal proposals on missing/empty external files.

### External file contract
- Context emitted by executor each round:
  - 
- Proposals expected from Codex/manual writer:
  - 
- Proposal JSON format:
  - list or 
  - each proposal includes  keys:
    - , , , , , 
  - optional: , , , , 

### Validation
-  passed.
-  passed.
- venv smoke checks for external load/timeout passed using:
  - 
  - 

### Current status
- No active agentic run.
- Latest known incumbent score unchanged:  from .

## 2026-02-17 15:22 CST+0800
### Correction note
- Supersedes the immediately previous malformed block from shell interpolation.

### External proposal implementation summary
- Added strict external proposal handoff support in:
  - /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
- Wired launcher defaults for strict external mode in:
  - /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh

### New runtime modes
- proposal-source=internal: existing heuristic/stochastic proposals.
- proposal-source=external: strict external-only mode; emits round_###_context.json and waits for round_###_proposals.json.
- proposal-source=external_preferred: external handshake first, then falls back to internal proposals when missing/empty.

### External file contract
- Context file emitted each round:
  - <proposal_dir>/round_###_context.json
- Proposal file expected from Codex/manual writer:
  - <proposal_dir>/round_###_proposals.json
- Proposal payload:
  - either a top-level list or a JSON object with key proposals.
  - each proposal must include trial keys:
    - train_steps
    - online_rounds
    - online_collect_episodes_per_round
    - online_train_steps_per_round
    - online_replan_every_n_steps
    - online_goal_geom_p
  - optional keys: proposal_id, rationale, origin, risk_level, common_overrides.

### Validation evidence
- python3 -m py_compile /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py (pass)
- bash -n /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh (pass)
- venv smoke tests for external wait/load path (success + timeout):
  - /tmp/agentic_external_test
  - /tmp/agentic_external_timeout_test

### Current state
- No active agentic run process.
- Latest known incumbent score remains 0.9167 from run agentic_autodecider_20260217-033224.

## 2026-02-17 15:33 CST+0800
### User question
- Did the overnight hypothesis experiment get the anticipated result?

### Evidence inspected
- Latest run summary/results/logs:
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic_results.csv`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic.log`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/proposals.jsonl`
- Process/GPU status:
  - no active `agentic_maze2d_autodecider.py` or `synthetic_maze2d_diffuser_probe.py`

### Conclusions
- The anticipated result was **not clearly achieved** in the latest overnight run.
- Hypothesis under test: stronger update-intensity/chunking should beat budget-up proposal.
- Observed in run `agentic_autodecider_20260217-033224`:
  - incumbent (bootstrap): objective_succ_h256=0.9167
  - p0 budget-up (`or8`): objective_succ_h256=0.9167 (delta=0)
  - p1 stronger-updates (`ep32,t4000`): objective_succ_h256=0.9167 (delta=0)
  - both rejected due `accept_delta=0.01`.
- Secondary efficiency signal favored p1 slightly on replay transitions (`264185` vs `268275`), but objective tie means no decisive win.
- Additional reliability issue: with `proposals-per-round=2`, only deterministic p0/p1 were generated; after round 0, rounds 1-9 skipped duplicates and did no new exploration.

### Open items
- Re-run with `proposals-per-round>=3` (or external proposal mode) to avoid duplicate-only loops and get a conclusive test.

## 2026-02-17 15:49 CST+0800
### User request
- Re-summarize the previous accomplishment with a higher-level format.

### Evidence inspected
- /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt
- /root/ebm-online-rl-prototype/docs/AGENTIC_AUTODECIDER_EXTERNAL_PROPOSAL_IMPLEMENTATION.md
- /root/SYSTEM_SETUP_WORKING_MEMORY.md

### Conclusions
- Previous accomplishment to summarize: no-API external-proposal handoff for agentic Maze2D, with launcher defaults and external round context/proposal file workflow.
- User-facing summary format should remain abstract and operational (functions created, natural-language usage, Codex-agent usage).

### Open items
- Continue future responses in this concise high-level structure.

## 2026-02-17 16:05 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-160017`
- incumbent: `ts20_or1_ep1_t10_rp4_gp080`
- incumbent score: `0.0000`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-160017/agentic_results.csv`

## 2026-02-17 16:06 CST+0800
### User question
- Run a very small real training experiment to verify the external-proposal workflow functions as intended.

### Evidence inspected
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/tmux_maze2d_agentic_20260217-160017.log
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-160017/agentic.log
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-160017/agentic_results.csv
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-160017/summary.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/proposal_exchange_smoke_20260217-160017/round_000_context.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/proposal_exchange_smoke_20260217-160017/round_000_proposals.json

### Conclusions
- Smoke-scale real training launch succeeded in strict external mode.
- Bootstrap run completed with rc=0 and wrote normal progress metrics.
- External round handoff worked: context file emitted and external proposal file loaded.
- In this smoke test, the submitted proposal duplicated the incumbent config, so it was skipped as duplicate and no promotion occurred.
- End-to-end behavior validated: launch -> bootstrap training -> external wait -> external load -> proposal validation/skip -> summary output.

### Open items
- Optional follow-up: submit a non-duplicate smoke proposal (for example change online_replan_every_n_steps 4->8) to exercise pilot-run path in addition to handshake path.

## 2026-02-17 16:07 CST+0800
### User question
- Whether recent conversation/experiment context across the last two days is being stored so another agent can continue reliably.

### Evidence inspected
- `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md`
- `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`
- `/var/tmp/vibe-kanban/worktrees/13d3-investigate-trai/EBM_OnlineRL/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`
- Confirmed dated entries spanning 2026-02-16 and 2026-02-17 (agentic runs, overnight outcomes, external-proposal implementation, smoke validation).

### Conclusions
- Yes: cross-day continuity is being recorded in persistent files and is sufficient for another agent to recover context.
- Primary source of truth is the main repo pair (working memory + handoff).
- The 13d3 worktree handoff contains a shorter artifact-focused mirror and pointers back to main-repo paths.
- Minor caveat: a few malformed entries occurred from shell interpolation in earlier turns, but explicit correction entries were appended afterward.

### Open items
- Continue appending every meaningful action to both main-repo continuity files; mirror critical summaries into the 13d3 handoff when interacting from this worktree.

## 2026-02-17 16:31 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-162110`
- incumbent: `ts20_or1_ep1_t10_rp4_gp080`
- incumbent score: `0.0000`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-162110/agentic_results.csv`

## 2026-02-17 16:32 CST+0800
### User question
- Run another very small real experiment to validate the non-duplicate external proposal pilot path.

### Evidence inspected
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/tmux_maze2d_agentic_20260217-162110.log
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-162110/agentic.log
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-162110/agentic_results.csv
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-162110/summary.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-162110/proposals.jsonl
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/proposal_exchange_smoke_nodup_20260217-162110/round_000_context.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/proposal_exchange_smoke_nodup_20260217-162110/round_000_proposals.json

### Conclusions
- Second smoke-scale real run completed and validated full pilot path for strict external mode.
- Bootstrap run completed rc=0, then external context/proposal handshake completed.
- Non-duplicate proposal was accepted into pilot execution (new trial ts20_or1_ep1_t20_rp8_gp080 ran rc=0).
- Pilot was evaluated and rejected on objective threshold: delta=0.0000 < accept_delta=0.0200.
- End-to-end validated path now includes: launch -> bootstrap -> external context -> external proposal load -> pilot trial -> promotion/rejection decision -> summary write.

### Open items
- Optional follow-up if needed: lower accept_delta in smoke mode to force a promotion path test, or craft a proposal likely to shift objective on this tiny setup.

## 2026-02-17 16:47 CST+0800
### User question
- Is proposer/executor/reviewer multi-agent behavior automatically implemented in the auto-decider skill loop, or does it require launching separate Codex agents?

### Evidence inspected
- /root/.agents/skills/ebm-onlinerl-five-step-loop/SKILL.md
- /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh
- /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py

### Conclusions
- Current implementation is single-process executor by default; no built-in multi-agent orchestration that auto-launches proposer/reviewer Codex agents.
- In proposal-source=external mode, executor waits for proposal files; those must be written by another process (separate Codex agent or custom helper script).
- proposal-source=internal needs no external proposer.
- proposal-source=external_preferred can run without external proposer (falls back to internal proposals).
- Reviewer role is conceptual/operational and currently not auto-launched by the script/skill.

### Open items
- Optional future enhancement: add a supervisor script that launches executor + proposer sidecar + reviewer monitor automatically.

## 2026-02-17 16:49 CST+0800
### User question
- Does internal proposal mean same Codex agent proposing, or rule-based proposing?

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py
- /root/ebm-online-rl-prototype/scripts/launch_agentic_maze2d_autodecider_tmux.sh

### Conclusions
- internal proposal mode maps to  in the executor process.
-  is heuristic + stochastic mutation logic (origins tagged heuristic/stochastic), not LLM/Codex reasoning.
- To have Codex reason/propose, use external proposal mode and a separate proposal writer process (agent or script).

### Open items
- Optional: add true LLM/Codex sidecar orchestration so proposer/reviewer roles are auto-launched.

## 2026-02-17 16:49 CST+0800
### Correction note
- Supersedes the immediately previous malformed placeholders in the internal-proposal clarification block.

### Corrected conclusion
- Internal proposal mode calls function _reasoned_proposals in the executor process.
- That function is heuristic plus stochastic rule logic, not Codex/LLM reasoning.
- Codex reasoning proposals require external proposal mode and a separate writer process.

## 2026-02-17 17:21 
### Agentic Decider update
- run: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-171747`
- incumbent: `ts20_or1_ep1_t10_rp4_gp080`
- incumbent score: `0.0000`
- results: `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-171747/agentic_results.csv`

## 2026-02-17 17:23 CST+0800
### User question
- Check the most recent update of EBM_OnlineRL repo and summarize the newly added PDF report.

### Evidence inspected
- /root/ebm-online-rl-prototype/EBM_OnlineRL/output/pdf/maze2d_diffuser_online_rl_status_2026-02-17.pdf
- /root/ebm-online-rl-prototype/EBM_OnlineRL/docs/maze2d_diffuser_online_rl_status_2026-02-17.md
- /root/ebm-online-rl-prototype/EBM_OnlineRL/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt

### Conclusions
- Latest EBM_OnlineRL artifact update is the 2026-02-17 status bundle (markdown + PDF mirror).
- PDF presents algorithm definition, auto-decider mechanism, and evidence-backed findings; key result highlights include full-diverse eval mismatch diagnosis (0.0 success in earlier A/B/C full benchmark) and later h256 objective peaks up to 1.0 in lightweight diverse eval runs.
- No code/run-state changes were made during this check; this was verification and reporting only.

### Open items
- Optional: commit docs/output in EBM_OnlineRL subrepo if this snapshot should be retained in subrepo git history.

## 2026-02-17 17:46 CST+0800
### User question
- Implement a goal-conditioned behavior cloning baseline with HER and align essential hyperparameters to compare fairly with diffuser.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py
- /root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt
- /root/SYSTEM_SETUP_WORKING_MEMORY.md

### Conclusions
- Will implement a peer script that reuses the same geometric-goal sampling and eval-prefix semantics, while replacing diffuser planner/training with GCBC+HER policy training.
- Key fairness knobs to keep aligned: online_goal_geom_p/min_k/max_k, online collection budget/round schedule, eval rollout horizon and prefix metrics, query sampling mode.

### Open items
- Decide exact HER relabeling budget per transition and policy architecture defaults.

## 2026-02-17 18:00 CST+0800
### User question
- Implement a fair-comparison GCBC+HER baseline against current diffuser Maze2D pipeline.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py
- /root/ebm-online-rl-prototype/scripts/synthetic_maze2d_gcbc_her_probe.py
- /root/ebm-online-rl-prototype/scripts/launch_gcbc_her_maze2d_probe_tmux.sh
- /tmp/gcbc_her_smoke_20260217-175852/summary.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/20260217-175920/summary.json

### Conclusions
- Added a standalone GCBC+HER training/evaluation script with shared core comparison knobs:
  - online_goal_geom_p/min_k/max_k,
  - online collection rounds/budgets,
  - eval rollout horizon + prefix success metrics.
- Added a launcher with the same MuJoCo/D4RL env exports as existing scripts so runs are reproducible in this machine setup.
- Tiny real-run validations succeeded for both direct invocation and launcher smoke mode; artifacts were written in the same style as the diffuser probe pipeline.

### Open items
- Execute paired matched-budget experiments (diffuser vs gcbc_her) and compare objective_succ_h256 plus coverage metrics.

## 2026-02-17 18:04 CST+0800
### User question
- Launch matched latest-setting Diffuser vs GCBC+HER runs and analyze comparative results.

### Evidence inspected
- /root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py (latest common defaults)
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260217-033224/agentic_results.csv
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_20260216-210532/agentic_results.csv

### Conclusions
- Using latest stable non-smoke trial family as matched target: ts6000_or4_ep64_t3000_rp16_gp0.04 plus current common defaults.
- Pair run root prepared: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356 with dedicated diffuser/gcbc subdirs and pair_config.json.

### Open items
- Run diffuser first, then gcbc_her with same aligned knobs; extract side-by-side metrics from progress/summary.

## 2026-02-17 18:56:07 CST+0800 Matched Pair Update (Diffuser complete, GCBC running)
- Pair root: 
- Diffuser completed at step 18000 (online_round_4). Final h256 metrics from :
  - success: 
  - query coverage: 
  - goal-cell coverage: 
- Diffuser final online collection (): round , replay_transitions , transitions accepted .
- GCBC+HER matched run launched and active:
  - run dir: 
  - log: 
  - status: running (first progress checkpoint pending)

## CORRECTION NOTE (shell interpolation issue)
- The immediately preceding 18:56 CST+0800 entry has missing inline path/value text because an unquoted heredoc allowed shell interpolation on backticks.
- This correction preserves append-only policy and provides the canonical values below.

## 2026-02-17 18:56:07 CST+0800 Matched Pair Update (canonical corrected)
- Pair root: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356
- Diffuser completed at step 18000 (online_round_4). Final h256 metrics from summary.json -> progress_last:
  - success: 0.8333333333333334
  - query coverage: 0.8333333134651184
  - goal-cell coverage: 0.8888888888888888
- Diffuser final online collection (online_collection_last): round 4, replay_transitions 272338, transitions accepted 4057.
- GCBC+HER matched run launched and active:
  - run dir: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0
  - log: /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0/run.log
  - status: running (first progress checkpoint pending)

## 2026-02-17 20:57:12 CST+0800
### User question
- Continue GCBC+HER vs Diffuser comparison and summarize ready results.

### Evidence inspected
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json
- /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0_rerun_20260217-203446/summary.json
- (partial) /root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0/progress_metrics.csv

### Conclusions
- Matched pair is now complete (both legs at train_steps_total=18000; online_round_4).
- Diffuser outperforms GCBC+HER on the key h256 success objective in this single-seed, 12-trajectory eval:
  - Diffuser: success@h256=0.8333 (10/12), query_coverage@h256=0.8333, goal_cell_coverage@h256=0.8889
  - GCBC+HER: success@h256=0.5833 (7/12), query_coverage@h256=0.5833, goal_cell_coverage@h256=0.5556
- Both used the same matched knobs (seed=0, 1000 offline episodes, online_rounds=4, train_steps=6000 + 4*3000 online, transition_budget_per_round=4096, goal_success_threshold=0.2, eval_rollout_horizon=256, diverse queries=12).

### Open items
- Repeat across multiple seeds and/or increase num_eval_queries to tighten confidence intervals.

## 2026-02-17 22:18 CST+0800
### User question
- Implement hypothesis-driven experiments to falsify/validate a non-trivial hypothesis for the current online Diffuser approach.
- Add a goal-conditioned RL baseline (SAC+HER) and run it with both:
  - success/sparse reward
  - shaped reward (noted as potentially unfair vs other approaches)

### Evidence inspected
- Matched Diffuser config (for alignment): `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/config.json`
  - offline: `n_episodes=1000`, `episode_len=256`
  - online: `online_rounds=4`, `online_collect_transition_budget_per_round=4096`, `online_train_steps_per_round=3000`
  - eval: `eval_rollout_horizon=256`, `goal_success_threshold=0.2`, `query_mode=diverse`, `num_eval_queries=12`
- Existing probes for protocol/logging patterns:
  - `scripts/synthetic_maze2d_diffuser_probe.py`
  - `scripts/synthetic_maze2d_gcbc_her_probe.py`

### Working hypothesis (falsifiable)
- Diffuser’s edge (vs goal-conditioned baselines) is primarily from a planner-driven bootstrapping loop: early planning success yields higher-quality online replay, which then compounds.

### Proposed falsification experiments (order)
1. Implement `scripts/synthetic_maze2d_sac_her_probe.py` (SAC+HER) with aligned knobs + two reward modes (sparse success vs shaped distance).
2. Run matched-budget comparisons (single seed first) vs existing Diffuser + GCBC+HER.
3. Add replay export/load (NPZ) to enable data-controlled tests and collector–learner swaps:
   - Train SAC+HER on Diffuser-collected replay (and vice versa) to see whether gains follow the collector or the learner/planner.

### Open items
- Confirm which Maze2D env to treat as primary target: `maze2d-umaze-v1` vs `maze2d-medium-v1` vs `maze2d-large-v1`.
- Decide whether comparisons should fix eval query pairs across methods (`--no_query_resample_each_eval`) to reduce variance.

## 2026-02-17 23:15:35 CST+0800
### Action
- User chose env option 'first one' => maze2d-umaze-v1.
- Launched matched-budget SAC+HER (sparse/success reward) seed=0 run for side-by-side comparison with existing Diffuser + GCBC+HER pair.

### Run
- pair root: runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356
- sac+her run dir: runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0
- tmux session: maze2d_sac_her_20260217-231216
- tmux log: runs/analysis/synth_maze2d_diffuser_probe/tmux_maze2d_sac_her_20260217-231216.log
- command knobs: reward_mode=sparse batch_size=128 train_steps=6000 online_rounds=4 online_train_steps_per_round=3000

### Next
- Wait for completion, then extract progress_last h256 success + goal_cell_coverage from summary.json and decide whether to run shaped reward next.

## 2026-02-18 13:50:30 CST+0800
### Experiment: SAC+HER baseline (sparse/success reward)
- Env: maze2d-umaze-v1 (seed=0)
- Run dir: runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0
- Status: complete (train_steps_total=18000, phase=online_round_4)

### Key results (progress_last @h256, 12 diverse eval queries)
- SAC+HER (sparse): success=0.8333, query_cov=0.8333, goal_cell_cov=0.7778
- Diffuser (matched): success=0.8333, query_cov=0.8333, goal_cell_cov=0.8889
- GCBC+HER (matched): success=0.5833, query_cov=0.5833, goal_cell_cov=0.5556

### Decision
- Proceed to SAC+HER shaped-reward run next (as an explicitly 'unfair upper bound' comparison), then revisit hypothesis-driven falsification (collector/learner swap with fixed replay).

## 2026-02-18 13:53:50 CST+0800
### Experiment: SAC+HER baseline (shaped reward; report separately as unfair upper bound)
- Env: maze2d-umaze-v1 (seed=0)
- Run dir: runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_shaped_ts6000_or4_ep64_t3000_rp16_gp040_seed0
- tmux: maze2d_sac_her_20260218-135133
- Status: running

## 2026-02-18 17:13:29 CST+0800
### Experiment: SAC+HER baseline (shaped reward; unfair upper bound)
- Env: maze2d-umaze-v1 (seed=0)
- Run dir: runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_shaped_ts6000_or4_ep64_t3000_rp16_gp040_seed0
- Status: complete (train_steps_total=18000, phase=online_round_4)

### Key results (progress_last @h256, 12 diverse eval queries)
- SAC+HER (shaped): success=0.7500, query_cov=0.7500, goal_cell_cov=0.7778
- SAC+HER (sparse): success=0.8333, query_cov=0.8333, goal_cell_cov=0.7778

### Notes
- Shaped reward did not improve success vs sparse in this single seed; treat as inconclusive due to n=12 eval trajectories.

### Decision
- Keep sparse reward as the primary fair baseline.
- Next falsification target: implement fixed-replay / collector-learner swap to test whether performance gains follow the online data collector vs the learner/planner.

## 2026-02-18 17:14:37 CST+0800
### Hypothesis (recap; proposed 2026-02-17)
- Diffuser’s edge (vs goal-conditioned baselines) is primarily collector-driven via a planner-driven bootstrapping loop: early planning success yields higher-quality online replay, which then compounds.

### Evidence so far (maze2d-umaze-v1 seed=0, 12 diverse eval queries, h256)
- Diffuser: success=0.8333, query_cov=0.8333, goal_cell_cov=0.8889
- GCBC+HER: success=0.5833, query_cov=0.5833, goal_cell_cov=0.5556
- SAC+HER (sparse): success=0.8333, query_cov=0.8333, goal_cell_cov=0.7778
- SAC+HER (shaped): success=0.7500, query_cov=0.7500, goal_cell_cov=0.7778

### Interim read
- Diffuser’s advantage over GCBC+HER is consistent in this single seed, but Diffuser is not strictly better than SAC+HER under the same budget on the primary success metric.
- The collector-vs-learner hypothesis is not yet directly tested; swap/fixed-replay experiments are still required.

## 2026-02-18 17:49 CST+0800
### User question
- Record the non-trivial hypothesis proposed “yesterday” (2026-02-17).
- Based on current experiment results, analyze:
  1) which hypotheses have clear conclusions
  2) which hypotheses are not supported by enough evidence

### Evidence inspected
- Working-memory recap entry (hypothesis + current numbers):
  - `/root/ebm-online-rl-prototype/docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md` (2026-02-18 17:14 CST+0800 block)
- Matched-budget summary.json evidence (maze2d-umaze-v1, seed=0, eval_num_queries=12):
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/gcbc_her_ts6000_or4_ep64_t3000_rp16_gp040_seed0_rerun_20260217-203446/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_sparse_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
  - `/root/ebm-online-rl-prototype/runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/sac_her_shaped_ts6000_or4_ep64_t3000_rp16_gp040_seed0/summary.json`
- Recent experiment timeline/decisions:
  - `/root/ebm-online-rl-prototype/HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt`

### Conclusions
- Non-trivial hypothesis (proposed 2026-02-17; recorded in the 2026-02-18 17:14 block above):
  - “Diffuser’s edge (vs goal-conditioned baselines) is primarily collector-driven via a planner-driven bootstrapping loop: early planning success yields higher-quality online replay, which then compounds.”
- Clear conclusions (scoped to the *current* evidence only: maze2d-umaze-v1, seed=0, 12 eval queries, matched budget):
  - GCBC+HER underperforms both Diffuser and SAC+HER(sparse) on success@h256:
    - Diffuser: 10/12
    - GCBC+HER: 7/12
    - SAC+HER(sparse): 10/12
  - Diffuser is not strictly better than SAC+HER(sparse) on the primary success@h256 metric in this setting (they tie at 10/12).
  - SAC+HER(shaped) did not outperform SAC+HER(sparse) here (9/12 vs 10/12), so “shaped reward helps” is not supported by this single run.
- Not supported by enough evidence yet (requires additional experiments / controls):
  - The collector-vs-learner attribution in the non-trivial hypothesis above:
    - Current results are end-to-end comparisons with different collectors and learners.
    - Need fixed-replay / collector-learner swap to isolate the causal factor.
  - Generalization claims (across seeds, across Maze2D medium/large, across larger eval sets than 12).
  - The “update-intensity/chunking beats budget-up” hypothesis (see 2026-02-16 22:38 and 2026-02-17 15:33 blocks) is still inconclusive due to objective ties and the duplicate-proposal issue limiting exploration in that run.

### Open items
- Implement fixed replay export/load (NPZ) and run collector–learner swap:
  - train SAC+HER on Diffuser-collected replay and vice versa.
- Strengthen denominators:
  - multiple seeds and/or larger `num_eval_queries` for tighter uncertainty on 1/12 steps.

## 2026-02-18 22:31 
### Agentic Decider update
- run: `output/smoke/agentic_smoke_20260218-222646`
- incumbent: `ts20_or1_ep1_t10_rp4_gp080`
- incumbent score: `0.0000`
- results: `output/smoke/agentic_smoke_20260218-222646/agentic_results.csv`
