#!/usr/bin/env python3
"""Boundary-vs-interior influence analysis for EqM and Diffuser.

Tests two predictions from the "position=scaffold, action=local-adjustment" mental model:

1. **Boundary position influence**: When we pick an interior output position
   (pivot at t=H/2), does the gradient w.r.t. input positions show elevated
   influence at the boundary timesteps (t=0, t=H-1) compared to interior
   timesteps at the same offset?

2. **Full influence heatmap**: For each pivot position across the full horizon,
   what does the complete gradient profile look like? This reveals whether
   boundary-adjacent pivots have systematically different influence patterns.

Transition packing: [ax, ay | x, y, vx, vy]
  act: dims 0:2, pos: dims 2:4, vel: dims 4:6

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \\
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \\
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \\
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \\
    scripts/analysis_boundary_influence_maze2d.py \\
    --eqm_checkpoint <path> --diffuser_checkpoint <path> [--n_probes 128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    load_diffuser_model_and_dataset,
    load_eqm_model_and_dataset,
)

SUBBLOCK_SLICES = {
    "act": slice(0, 2),
    "pos": slice(2, 4),
    "vel": slice(4, 6),
}


def full_influence_profile(denoiser, x, pivot_t, out_block, in_block, t_embed, device):
    """Compute raw grad norm at EVERY input timestep for a fixed pivot output timestep.

    Returns: np.ndarray [H] of raw grad norms.
    """
    x = x.clone().detach().requires_grad_(True)
    t_tensor = torch.full((1,), t_embed, dtype=torch.long, device=device)
    y = denoiser(x, {}, t_tensor)  # [1, H, D]

    out_sl = SUBBLOCK_SLICES[out_block]
    in_sl = SUBBLOCK_SLICES[in_block]

    # Random unit projection on output sub-block at pivot
    u = torch.randn(2, device=device)
    u = u / (u.norm() + 1e-12)
    scalar = (y[0, pivot_t, out_sl] * u).sum()

    grad = torch.autograd.grad(scalar, x, create_graph=False)[0]  # [1, H, D]
    raw_norms = grad[0, :, in_sl].norm(dim=1)  # [H]
    return raw_norms.detach().cpu().numpy()


def collect_heatmap(denoiser, dataset, device, t_embed, horizon,
                    n_probes, out_block, in_block, pivot_positions):
    """Collect influence heatmaps: for each pivot position, average the full influence profile.

    Returns: dict[pivot_t -> np.ndarray[H]] of mean raw grad norms.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    # Accumulate per pivot
    accum = {p: np.zeros(horizon, dtype=np.float64) for p in pivot_positions}
    counts = {p: 0 for p in pivot_positions}

    for i in range(n_probes):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        traj = batch.trajectories.to(device)  # [1, H, D]

        for p in pivot_positions:
            profile = full_influence_profile(denoiser, traj, p, out_block, in_block,
                                             t_embed, device)
            accum[p] += profile
            counts[p] += 1

        if (i + 1) % 32 == 0:
            print(f"    {i+1}/{n_probes}")

    result = {}
    for p in pivot_positions:
        result[p] = accum[p] / max(counts[p], 1)
    return result


def print_boundary_analysis(heatmaps, H, label, out_block, in_block):
    """Analyze whether boundary timesteps show elevated influence."""
    print(f"\n{'='*90}")
    print(f"BOUNDARY ANALYSIS: {out_block}_out <- {in_block}_in ({label})")
    print(f"{'='*90}")

    # Use interior pivot (H/2)
    mid = H // 2
    if mid not in heatmaps:
        print("  [no interior pivot available]")
        return

    profile = heatmaps[mid]

    # Boundary timesteps
    boundary_ts = [0, 1, H - 2, H - 1]
    # Interior timesteps at similar offsets from pivot
    # For pivot=H/2, offset to boundary ~= H/2
    # Compare boundary influence to interior influence at same offset
    print(f"\n  Pivot at t={mid} (interior). Full profile (selected timesteps):\n")
    print(f"  {'t_in':>5} {'raw_norm':>12} {'offset':>8} {'location':>12}")
    print(f"  {'-'*40}")

    for t in range(H):
        off = abs(t - mid)
        loc = "BOUNDARY" if t <= 1 or t >= H - 2 else ("near-bnd" if t <= 3 or t >= H - 4 else "interior")
        if t <= 4 or t >= H - 5 or t == mid or off <= 2:
            print(f"  {t:>5} {profile[t]:>12.6f} {off:>8} {loc:>12}")

    # Aggregate comparison
    bnd_vals = [profile[t] for t in [0, 1, H - 2, H - 1] if t != mid]
    near_bnd = [profile[t] for t in [2, 3, H - 4, H - 3] if t != mid]

    # Interior: same range of offsets as boundary but measured from mid
    bnd_offsets = [abs(t - mid) for t in [0, 1, H - 2, H - 1]]
    int_ring = []
    for off in bnd_offsets:
        for t in [mid - off, mid + off]:
            if 4 <= t <= H - 5:
                int_ring.append(profile[t])

    print(f"\n  Boundary (t=0,1,{H-2},{H-1}): mean={np.mean(bnd_vals):.6f}")
    print(f"  Near-boundary (t=2,3,{H-4},{H-3}): mean={np.mean(near_bnd):.6f}")
    if int_ring:
        print(f"  Interior at same offset: mean={np.mean(int_ring):.6f}")
        ratio = np.mean(bnd_vals) / (np.mean(int_ring) + 1e-12)
        print(f"  Boundary/Interior ratio: {ratio:.3f}x")


def print_pivot_comparison(heatmaps, H, label, out_block, in_block):
    """Compare influence profiles when pivot is at boundary vs interior."""
    print(f"\n{'='*90}")
    print(f"PIVOT COMPARISON: {out_block}_out <- {in_block}_in ({label})")
    print(f"{'='*90}")

    # Get boundary and interior pivots
    bnd_pivots = [p for p in sorted(heatmaps.keys()) if p <= 2 or p >= H - 3]
    int_pivots = [p for p in sorted(heatmaps.keys()) if H // 4 <= p <= 3 * H // 4]

    if not bnd_pivots or not int_pivots:
        print("  [insufficient pivots]")
        return

    # For each pivot, report: raw norm at offset 0, offset 1, total influence
    print(f"\n  {'pivot':>5} {'location':>10} {'off0_norm':>12} {'off1_norm':>12} "
          f"{'off2_norm':>12} {'total':>12} {'off0/total':>12}")
    print(f"  {'-'*75}")

    for p in sorted(heatmaps.keys()):
        prof = heatmaps[p]
        loc = "BOUNDARY" if p <= 1 or p >= H - 2 else ("near-bnd" if p <= 3 or p >= H - 4 else "interior")
        off0 = prof[p]
        off1 = prof[p + 1] if p + 1 < H else prof[p - 1]
        off2_candidates = [prof[t] for t in [p - 2, p + 2] if 0 <= t < H]
        off2 = np.mean(off2_candidates) if off2_candidates else 0
        total = np.sum(prof)
        print(f"  {p:>5} {loc:>10} {off0:>12.6f} {off1:>12.6f} {off2:>12.6f} "
              f"{total:>12.4f} {off0/total:>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqm_checkpoint", required=True)
    parser.add_argument("--diffuser_checkpoint", required=True)
    parser.add_argument("--n_probes", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or f"runs/analysis/boundary_influence_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] EqM...")
    eqm_model, eqm_dataset, eqm_cfg = load_eqm_model_and_dataset(args.eqm_checkpoint, device)
    eqm_horizon = int(eqm_cfg["horizon"])
    H = eqm_horizon

    print("[load] Diffuser...")
    diff_model, diff_dataset, diff_cfg = load_diffuser_model_and_dataset(args.diffuser_checkpoint, device)

    eqm_den = eqm_model.model; eqm_den.eval()
    diff_den = diff_model.model; diff_den.eval()

    # Pivot positions: boundary + near-boundary + interior sampling
    pivot_positions = sorted(set([
        0, 1, 2, 3, 4,              # start boundary zone
        H // 4, H // 3,             # interior left
        H // 2,                      # center
        2 * H // 3, 3 * H // 4,     # interior right
        H - 5, H - 4, H - 3, H - 2, H - 1  # goal boundary zone
    ]))
    pivot_positions = [p for p in pivot_positions if 0 <= p < H]

    print(f"\n[config] H={H}, pivots={pivot_positions}, n_probes={args.n_probes}")

    all_results = {}

    # Run for both models, for pos→pos and act→act (and vel→act for action local-adjustment)
    block_pairs = [("pos", "pos"), ("act", "act"), ("act", "vel"), ("act", "pos")]

    for out_b, in_b in block_pairs:
        pair_key = f"{out_b}_out__{in_b}_in"
        print(f"\n{'='*70}")
        print(f"Block pair: {pair_key}")
        print(f"{'='*70}")

        print(f"\n  [Diffuser t_embed=0]")
        diff_hm = collect_heatmap(diff_den, diff_dataset, device, t_embed=0,
                                  horizon=H, n_probes=args.n_probes,
                                  out_block=out_b, in_block=in_b,
                                  pivot_positions=pivot_positions)
        all_results[f"diffuser_{pair_key}"] = {str(k): v.tolist() for k, v in diff_hm.items()}

        print(f"\n  [EqM t_embed=0]")
        eqm_hm = collect_heatmap(eqm_den, eqm_dataset, device, t_embed=0,
                                 horizon=H, n_probes=args.n_probes,
                                 out_block=out_b, in_block=in_b,
                                 pivot_positions=pivot_positions)
        all_results[f"eqm_{pair_key}"] = {str(k): v.tolist() for k, v in eqm_hm.items()}

        # Print analyses
        print_boundary_analysis(diff_hm, H, "Diffuser", out_b, in_b)
        print_boundary_analysis(eqm_hm, H, "EqM", out_b, in_b)
        print_pivot_comparison(diff_hm, H, "Diffuser", out_b, in_b)
        print_pivot_comparison(eqm_hm, H, "EqM", out_b, in_b)

    # Save JSON
    json_path = outdir / "boundary_influence.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] {json_path}")

    # Compact summary
    print(f"\n{'='*90}")
    print(f"COMPACT SUMMARY: BOUNDARY vs INTERIOR INFLUENCE (clean data, t=0)")
    print(f"{'='*90}\n")

    for out_b, in_b in block_pairs:
        pair_key = f"{out_b}_out__{in_b}_in"
        mid = H // 2

        for model_label, prefix in [("Diffuser", "diffuser"), ("EqM", "eqm")]:
            key = f"{prefix}_{pair_key}"
            hm = {int(k): np.array(v) for k, v in all_results[key].items()}

            if mid not in hm:
                continue

            prof = hm[mid]
            # Offset-0 norm at boundary pivots vs interior pivots
            bnd_off0 = np.mean([hm[p][p] for p in [0, 1, H - 2, H - 1] if p in hm])
            int_off0 = np.mean([hm[p][p] for p in hm if H // 4 <= p <= 3 * H // 4])

            print(f"  {pair_key:22s} {model_label:>8}: "
                  f"bnd_pivot off0={bnd_off0:.4f}  int_pivot off0={int_off0:.4f}  "
                  f"ratio={bnd_off0/(int_off0+1e-12):.3f}x")

    print(f"\nResults saved to: {outdir}")


if __name__ == "__main__":
    main()
