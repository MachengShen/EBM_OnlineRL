# Implementation Notes: Scaffold EqM vs Diffuser

Last updated: 2026-02-27

## Scope implemented
- Added shared scaffold helper utilities in `ebm_online_rl/online/scaffold.py`:
  - `build_anchor_times`
  - `extract_anchor_xy`
  - `apply_pos_only_anchors_`
- Added sampler-side scaffold insertion support in `scripts/synthetic_maze2d_diffuser_probe.py`:
  - EqM path: insertion during iterative refinement with pos-only anchor clamping.
  - Diffuser path: insertion during reverse denoising loop with pos-only anchor clamping.
  - Optional EqM eta anneal support (`eqm_eta_start`, `eqm_eta_end`) in conditional sampling.
- Added unified probe entrypoint:
  - `scripts/maze2d_scaffold_probe.py`
  - supports `--algo {eqm,diffuser}`, `--scaffold {none,insert_mid}`, stride/insert controls, smoothness metrics, and rollout success metrics.

## Easy-to-mess-up sections and safeguards
- Packing convention:
  - Maze2D path in this pipeline uses `[act | obs]`.
  - Position anchors always use `x[:, :, act_dim:act_dim+2]`.
  - Guardrail: helper functions validate tensor rank and index bounds.
- Anchor insertion order:
  - Correct order per step: update sample -> endpoint conditioning -> insert anchors once -> re-apply anchors every remaining step.
  - Guardrail: implemented explicitly in both EqM and Diffuser loops.
- Cross-module EqM detection:
  - `isinstance` can fail when modules are dynamically loaded under different names.
  - Guardrail: switched to duck-typed EqM detection (`n_eqm_steps` + `step_size`).
- Import path fragility in diffuser venv:
  - `ebm_online_rl` package is not always importable under default venv path.
  - Guardrail: explicitly inject repo root into `sys.path` in scripts before importing local modules.

## Sanity checks run
- Syntax:
  - `python -m py_compile` on all modified files.
- Runtime smoke (`--help`):
  - `scripts/maze2d_scaffold_probe.py --help` under diffuser venv.
- Runtime smoke (EqM scaffold path):
  - `scripts/maze2d_scaffold_probe.py --algo eqm ... --scaffold insert_mid ...` with `num_eval_queries=1`, `query_batch_size=1`.
- Runtime smoke (Diffuser scaffold path):
  - `scripts/maze2d_scaffold_probe.py --algo diffuser ... --scaffold insert_mid --diff_steps 16 ...` with `num_eval_queries=1`, `query_batch_size=1`.

## Reused implementation from parallel worktree
- Inspected parallel worktree:
  - `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d`
- Reusable parts for this scaffold task were limited:
  - That branch mainly introduced EqNet denoiser integration and ablation tooling.
  - No direct scaffold insertion implementation existed to lift as-is.

## Outstanding uncertainty to discuss before long runs
- `build_anchor_times` currently excludes endpoints by default at application-time (via scaffold settings), not in the helper itself.
- Diffuser `--diff_steps` override currently truncates denoising depth from checkpoint `n_timesteps`; this is intentional for compute matching but should be called out when comparing absolute performance.
- Eta anneal behavior is sampling-time only; training objective is unchanged.
