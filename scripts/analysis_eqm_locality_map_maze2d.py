#!/usr/bin/env python3
"""H2 locality analysis: temporal influence bandedness of EqM denoiser on Maze2D.

Measures how the denoiser output at timestep t depends on inputs at timestep t',
using a VJP (vector-Jacobian product) trick to avoid computing the full Jacobian.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/analysis_eqm_locality_map_maze2d.py --checkpoint <path> [--n_samples 256]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    load_eqm_model_and_dataset,
)


def locality_profile(denoiser, x: torch.Tensor) -> tuple:
    """Compute influence profile for one random pivot timestep.

    Args:
        denoiser: raw TemporalUnet model(x, cond, t) -> output
        x: trajectory tensor [1, T, D], requires_grad will be set

    Returns:
        (influence: np.ndarray [T], pivot_t: int)
    """
    x = x.clone().detach().requires_grad_(True)
    T = x.shape[1]
    D = x.shape[2]
    device = x.device

    t0 = torch.zeros((1,), dtype=torch.long, device=device)
    # Forward pass through denoiser (3-arg signature: x, cond, time)
    # Pass empty cond dict — cond is unused by TemporalUnet internally
    y = denoiser(x, {}, t0)  # [1, T, D]

    # Random pivot timestep and projection direction
    t = int(torch.randint(0, T, (1,)).item())
    u = torch.randn((D,), device=device)

    # VJP: d(y[0,t,:] . u) / dx
    scalar = (y[0, t, :] * u).sum()
    grad = torch.autograd.grad(scalar, x, create_graph=False)[0]  # [1, T, D]

    # Influence = L2 norm of gradient per timestep
    infl = grad[0].norm(dim=1)  # [T]
    infl = infl / (infl.max() + 1e-12)
    return infl.detach().cpu().numpy(), t


def main():
    parser = argparse.ArgumentParser(description="H2: EqM locality influence map on Maze2D")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EqM checkpoint")
    parser.add_argument("--n_samples", type=int, default=256, help="Number of VJP probes")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[H2 locality] Loading checkpoint: {args.checkpoint}")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    horizon = int(cfg["horizon"])
    print(f"[H2 locality] Model loaded. horizon={horizon}, n_samples={args.n_samples}")

    # Setup output dir
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path(f"runs/analysis/eqm_locality_h2_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Get raw denoiser (TemporalUnet)
    denoiser = model.model
    denoiser.eval()

    # Collect influence profiles over many samples
    # offset_influences[offset] = list of influence values
    offset_influences = {d: [] for d in range(horizon)}

    rng = np.random.default_rng(42)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    for i in range(args.n_samples):
        # Get a trajectory from dataset (normalized, [1, H, D])
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        traj = batch.trajectories.to(device)  # [1, H, D]

        infl, pivot = locality_profile(denoiser, traj)

        # Record influence by offset from pivot
        for t_prime in range(horizon):
            offset = abs(t_prime - pivot)
            offset_influences[offset].append(float(infl[t_prime]))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{args.n_samples}] done")

    # Compute summary statistics per offset
    results = []
    for offset in range(horizon):
        vals = offset_influences[offset]
        if vals:
            results.append({
                "offset": offset,
                "mean_influence": float(np.mean(vals)),
                "std_influence": float(np.std(vals)),
                "median_influence": float(np.median(vals)),
                "n_probes": len(vals),
            })

    # Save CSV
    csv_path = outdir / "locality_profile.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["offset", "mean_influence", "std_influence", "median_influence", "n_probes"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[H2 locality] Saved CSV: {csv_path}")

    # Save JSON summary
    summary = {
        "hypothesis": "H2",
        "description": "Temporal locality of EqM denoiser vector field",
        "checkpoint": str(args.checkpoint),
        "n_samples": args.n_samples,
        "horizon": horizon,
        "peak_offset_0_mean": results[0]["mean_influence"] if results else None,
        "half_decay_offset": None,
        "results": results,
    }
    # Find half-decay offset
    if results and results[0]["mean_influence"] > 0:
        peak = results[0]["mean_influence"]
        for r in results:
            if r["mean_influence"] < peak * 0.5:
                summary["half_decay_offset"] = r["offset"]
                break

    json_path = outdir / "locality_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H2 locality] Saved JSON: {json_path}")

    # Plot
    offsets = [r["offset"] for r in results]
    means = [r["mean_influence"] for r in results]
    stds = [r["std_influence"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(offsets, means, yerr=stds, fmt="o-", markersize=3, capsize=2, alpha=0.8)
    ax.set_xlabel("Temporal offset |t - t'|")
    ax.set_ylabel("Normalized influence (mean +/- std)")
    ax.set_title(f"H2: EqM denoiser locality profile (n={args.n_samples}, H={horizon})")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% decay")
    if summary["half_decay_offset"] is not None:
        ax.axvline(x=summary["half_decay_offset"], color="g", linestyle="--", alpha=0.5,
                   label=f"half-decay @ offset={summary['half_decay_offset']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = outdir / "locality_profile.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H2 locality] Saved plot: {plot_path}")

    # Print key results
    print(f"\n=== H2 Locality Results ===")
    print(f"  Peak (offset 0): {summary['peak_offset_0_mean']:.4f}")
    print(f"  Half-decay offset: {summary['half_decay_offset']}")
    if len(results) >= 5:
        print(f"  Influence at offset 5: {results[5]['mean_influence']:.4f}")
    if len(results) >= 10:
        print(f"  Influence at offset 10: {results[10]['mean_influence']:.4f}")


if __name__ == "__main__":
    main()
