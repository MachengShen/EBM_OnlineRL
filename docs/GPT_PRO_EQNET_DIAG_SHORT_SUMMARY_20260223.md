# EqNet Diagnostic Short Summary (2026-02-23)

## Scope
This note summarizes the corrected EqNet-focused diagnostic requested after the UNet/EqNet transcription mismatch was resolved.

## What was run
- Replay source: `runs/analysis/expert_dataset_unet_diag/maze2d_umaze_d4rl_replay_full.npz`
  - transitions: 1,000,000
  - episodes: 12,459
- Matched offline replay-fit setup (same seed and train budget as prior UNet diagnostic):
  - EqNet run: `runs/analysis/expert_dataset_unet_diag/eqnet_offline_expert_noeval_seed0_20260223-142032/`
  - UNet baseline run: `runs/analysis/expert_dataset_unet_diag/unet_offline_expert_noeval_seed0_20260223-140459/`
  - train steps: 600
  - device: CPU
  - val logging: enabled (`val_every=25`, `val_batches=20`)

## Key results (train/val)
From `runs/analysis/expert_dataset_unet_diag/eqnet_vs_unet_expert_diag_compare.csv`:

| Metric | EqNet | UNet | EqNet - UNet |
|---|---:|---:|---:|
| step1 train loss | 3.0410 | 0.8589 | +2.1821 |
| step1 val loss | 3.0356 | 0.8625 | +2.1732 |
| final train loss (step600) | 0.3565 | 0.2333 | +0.1232 |
| final val loss (step600) | 0.4533 | 0.2401 | +0.2132 |
| final val-train gap | 0.0968 | 0.0068 | +0.0900 |

## Interpretation
- EqNet does fit the expert replay (loss decreases over training), but converges to a clearly worse loss regime than UNet in this matched setup.
- The much larger final validation gap for EqNet (`+0.0968` vs `+0.0068`) suggests poorer generalization/stability under current EqNet hyperparameters and integration.
- This supports the hypothesis that EqNet underperformance is not only an online-collector artifact; a gap already appears in offline replay-fit efficiency.

## Caveat
- Final fixed-query rollout evaluation for this EqNet pass was terminated after step-600 training due long CPU runtime tail.
- Therefore, this diagnostic is train/val-focused; planning rollout metrics for this exact EqNet run are not included.

## Recommended next diagnostic wave (EqNet only)
1. Budget-matched EqNet width sweep: `eqnet_model_dim = eqnet_emb_dim in {64,96,128}`.
2. LR sensitivity sweep: `{2e-4, 1e-4, 5e-5}`.
3. Depth/kernel sweep: `eqnet_n_layers in {8,16,25}`, `eqnet_kernel_expansion_rate in {2,5}`.
4. Keep all non-swept settings fixed to isolate factors.

## Primary artifacts
- `runs/analysis/expert_dataset_unet_diag/eqnet_offline_expert_noeval_seed0_20260223-142032/overfit_summary_eqnet_train_only.json`
- `runs/analysis/expert_dataset_unet_diag/unet_offline_expert_noeval_seed0_20260223-140459/overfit_summary.json`
- `runs/analysis/expert_dataset_unet_diag/eqnet_vs_unet_expert_diag_compare.json`
- `runs/analysis/expert_dataset_unet_diag/eqnet_vs_unet_expert_diag_compare.csv`
