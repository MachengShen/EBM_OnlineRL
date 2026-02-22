# EqNet Maze2D Ablation Summary

Status: implementation complete; experiment execution pending.

## Objective
Evaluate whether replacing the Diffuser Temporal U-Net denoiser with EqNet improves Maze2D online self-improvement performance under matched training/collection/eval budgets.

## Commands
```bash
# Smoke
bash scripts/ablation_maze2d_eqnet_vs_unet.sh --smoke

# Full (3 seeds by default: 0,1,2)
bash scripts/ablation_maze2d_eqnet_vs_unet.sh --env maze2d-umaze-v1 --device cuda:0
```

## Implementation References
- EqNet adapter: `scripts/eqnet_adapter.py`
- Denoiser switch + wiring: `scripts/synthetic_maze2d_diffuser_probe.py`
- Swap matrix passthrough flags: `scripts/exp_swap_matrix_maze2d.py`
- Ablation launcher: `scripts/ablation_maze2d_eqnet_vs_unet.sh`
- Ablation analyzer: `scripts/analyze_ablation_eqnet_vs_unet.py`

## Reproducibility
- Upstream reference repo: `https://github.com/rvl-lab-utoronto/diffusion-stitching`
- Inspected commit for integration planning: `d27cf2ab7bf760dc62742b34e7bacf4e83ea9562`
- If local checkout exists at `third_party_external/diffusion-stitching`, the launcher writes commit hash to `runs/.../diffusion_stitching_commit.txt`.

## Expected Outputs
- `runs/analysis/eqnet_vs_unet/<run>/eqnet_vs_unet_rows.csv`
- `runs/analysis/eqnet_vs_unet/<run>/eqnet_vs_unet_summary.json`
- `runs/analysis/eqnet_vs_unet/<run>/eqnet_vs_unet_summary.md`
- `runs/analysis/eqnet_vs_unet/<run>/eqnet_vs_unet_success_curve.png`

## Notes
- EqNet path currently requires power-of-two planning horizon.
- For fairness, keep all non-denoiser settings matched between `unet` and `eqnet`.
