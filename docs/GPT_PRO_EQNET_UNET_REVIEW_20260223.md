# GPT-Pro Review Packet: EqNet vs UNet (Maze2D)

- Generated: 2026-02-23 11:38:20 
- Branch: `feature/eqnet-maze2d`
- Commit: `e99252d` (base implementation commit for this branch)
- Run locator: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/LAST_EQNET_3SEED_RUN.txt`
- Canonical run root: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504`
- Core caveat: results are **n=3 seeds per architecture**; treat as directional evidence.

## 1) Scripts Used

| Purpose | Script |
|---|---|
| Main ablation launcher | `scripts/ablation_maze2d_eqnet_vs_unet.sh` |
| Per-run train/eval (both UNet and EqNet via flag) | `scripts/synthetic_maze2d_diffuser_probe.py` |
| Aggregation/report generation | `scripts/analyze_ablation_eqnet_vs_unet.py` |
| (Pipeline continuation after EqNet completion) | `scripts/resume_after_eqnet.sh` |

## 2) Exact Commands Executed

### 2.1 Relay launch command

```bash
cd /root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d && mkdir -p runs/analysis/eqnet_vs_unet && RUN_ROOT=runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_$(date +%Y%m%d-%H%M%S) && mkdir -p $RUN_ROOT && printf '%s\n' $RUN_ROOT > runs/analysis/eqnet_vs_unet/LAST_EQNET_3SEED_RUN.txt && echo [start] $RUN_ROOT && bash scripts/ablation_maze2d_eqnet_vs_unet.sh --env maze2d-umaze-v1 --seeds 0,1,2 --device cuda:0 --base-dir $RUN_ROOT 2>&1 | tee $RUN_ROOT/launcher.log
```

### 2.2 First UNet probe command (from launcher log)

```bash
/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3 /root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/scripts/synthetic_maze2d_diffuser_probe.py --env maze2d-umaze-v1 --seed 0 --logdir runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/unet/seed_0 --query_mode diverse --goal_success_threshold 0.2 --denoiser_arch unet --n_episodes 400 --episode_len 256 --horizon 64 --train_steps 6000 --batch_size 128 --online_self_improve --online_rounds 4 --online_collect_episodes_per_round 64 --online_collect_episode_len 256 --online_train_steps_per_round 3000 --online_replan_every_n_steps 16 --num_eval_queries 12 --query_bank_size 256 --query_batch_size 1 --query_min_distance 1.0 --eval_goal_every 3000 --eval_rollout_horizon 256 --eval_rollout_replan_every_n_steps 16 --eval_success_prefix_horizons 64,128,192,256 --save_checkpoint_every 5000 --model_dim 64 --model_dim_mults 1,2,4 --n_diffusion_steps 64 --device cuda:0
```

### 2.3 First EqNet probe command (from launcher log)

```bash
/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3 /root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/scripts/synthetic_maze2d_diffuser_probe.py --env maze2d-umaze-v1 --seed 0 --logdir runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet/seed_0 --query_mode diverse --goal_success_threshold 0.2 --denoiser_arch eqnet --n_episodes 400 --episode_len 256 --horizon 64 --train_steps 6000 --batch_size 128 --online_self_improve --online_rounds 4 --online_collect_episodes_per_round 64 --online_collect_episode_len 256 --online_train_steps_per_round 3000 --online_replan_every_n_steps 16 --num_eval_queries 12 --query_bank_size 256 --query_batch_size 1 --query_min_distance 1.0 --eval_goal_every 3000 --eval_rollout_horizon 256 --eval_rollout_replan_every_n_steps 16 --eval_success_prefix_horizons 64,128,192,256 --save_checkpoint_every 5000 --model_dim 64 --model_dim_mults 1,2,4 --n_diffusion_steps 64 --device cuda:0
```

## 3) Result Summary (n=3 per architecture)

| Arch | Params | Success mean±std | Min-goal-dist mean±std | Final-goal-dist mean±std | Wall-hits mean±std |
|---|---:|---:|---:|---:|---:|
| UNet | 3973510 | 0.7778 ± 0.2097 | 0.3838 ± 0.3258 | 0.4397 ± 0.3242 | 84.5556 ± 20.2581 |
| EqNet | 1607880 | 0.2778 ± 0.1273 | 0.8307 ± 0.0837 | 0.9634 ± 0.0978 | 89.9167 ± 27.7768 |

### EqNet minus UNet deltas

| Metric | Delta (EqNet - UNet) |
|---|---:|
| Success mean | -0.5000 |
| Min-goal-dist mean | +0.4469 |
| Final-goal-dist mean | +0.5237 |
| Wall-hits mean | +5.3611 |

## 4) Per-Seed Table

| Seed | UNet success@h256 | EqNet success@h256 | Δ success | UNet min-dist | EqNet min-dist | Δ min-dist | UNet final-dist | EqNet final-dist | Δ final-dist |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5833 | 0.1667 | -0.4167 | 0.6806 | 0.7449 | +0.0643 | 0.7362 | 0.8514 | +0.1152 |
| 1 | 0.7500 | 0.4167 | -0.3333 | 0.4356 | 0.9121 | +0.4766 | 0.4895 | 1.0073 | +0.5178 |
| 2 | 1.0000 | 0.2500 | -0.7500 | 0.0352 | 0.8351 | +0.7999 | 0.0935 | 1.0315 | +0.9381 |

## 5) Artifact Paths

- Summary JSON: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet_vs_unet_summary.json`
- Per-seed rows CSV: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet_vs_unet_rows.csv`
- Markdown summary: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet_vs_unet_summary.md`
- Success curve plot: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet_vs_unet_success_curve.png`
- Full launcher log: `/root/ebm-online-rl-prototype/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/launcher.log`

## 6) Implementation-Level Audit Checks for GPT-Pro

1. Verify EqNet adapter call mapping (`model(x, cond, t)` -> EqNet expected inputs) in the implementation path.
2. Confirm EqNet receives the same conditioning and normalization pipeline as UNet.
3. Check whether EqNet needs architecture-specific optimization settings (e.g., LR, warmup, batch) instead of UNet defaults.
4. Validate that horizon constraints and padding assumptions match EqNet internals for all train/eval phases.
5. Compare training-loss dynamics by architecture from per-seed logs to detect optimization instability.
6. Re-run with larger seed count before hard conclusions; current signal is from n=3 only.
