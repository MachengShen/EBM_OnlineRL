#!/usr/bin/env python3
"""H2 follow-up: blockwise locality probes across regimes.

Decomposes the VJP influence profile into 4 blocks:
  act_out__act_in, act_out__obs_in, obs_out__act_in, obs_out__obs_in

Regimes: dataset, corrupted (gamma sweep), eqm_iter (k sweep).

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/analysis_eqm_locality_map_maze2d_blockwise.py \
    --checkpoint <path> --probe_regime dataset --n_probes 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    load_eqm_model_and_dataset,
    normalize_condition,
)
from diffuser.models.helpers import apply_conditioning

PROBE_TYPES = [
    "act_out__act_in",
    "act_out__obs_in",
    "obs_out__act_in",
    "obs_out__obs_in",
]


def blockwise_locality_probe(denoiser, x, probe_type, device):
    """Single VJP probe returning per-timestep influence for one block pair.

    Args:
        denoiser: model callable (x, cond, t) -> y
        x: trajectory [1, H, D], detached
        probe_type: one of PROBE_TYPES
        device: torch device

    Returns:
        (influence_vector [H], pivot_t int)
    """
    x = x.clone().detach().requires_grad_(True)
    t0 = torch.zeros((1,), dtype=torch.long, device=device)
    y = denoiser(x, {}, t0)  # [1, H, D]

    H = x.shape[1]
    t = int(torch.randint(0, H, (1,)).item())

    out_block, in_block = probe_type.split("__")

    # Compute scalar projection on output block
    if out_block == "act_out":
        u = torch.randn((ACT_DIM,), device=device)
        scalar = (y[0, t, :ACT_DIM] * u).sum()
    else:  # obs_out
        u = torch.randn((OBS_DIM,), device=device)
        scalar = (y[0, t, ACT_DIM:] * u).sum()

    grad = torch.autograd.grad(scalar, x, create_graph=False)[0]  # [1, H, D]

    # Extract influence on input block
    if in_block == "act_in":
        infl = grad[0, :, :ACT_DIM].norm(dim=1)  # [H]
    else:  # obs_in
        infl = grad[0, :, ACT_DIM:].norm(dim=1)  # [H]

    infl = infl / (infl.max() + 1e-12)
    return infl.detach().cpu().numpy(), t


def generate_samples(model, dataset, regime, device, n_batch=64, **kwargs):
    """Generate trajectory samples for a given regime.

    Args:
        regime: 'dataset', 'corrupted', 'eqm_iter'
        kwargs: gamma (for corrupted), k (for eqm_iter)

    Returns:
        list of trajectory tensors [1, H, D]
    """
    horizon = model.horizon
    samples = []

    if regime == "dataset":
        indices = np.random.choice(len(dataset), size=min(n_batch, len(dataset)), replace=False)
        for idx in indices:
            batch = dataset[idx]
            traj = torch.tensor(batch.trajectories, dtype=torch.float32, device=device)
            if traj.ndim == 2:
                traj = traj.unsqueeze(0)  # [1, H, D]
            samples.append(traj)

    elif regime == "corrupted":
        gamma = kwargs.get("gamma", 0.5)
        indices = np.random.choice(len(dataset), size=min(n_batch, len(dataset)), replace=False)
        for idx in indices:
            batch = dataset[idx]
            x_clean = torch.tensor(batch.trajectories, dtype=torch.float32, device=device)
            if x_clean.ndim == 2:
                x_clean = x_clean.unsqueeze(0)
            eps = torch.randn_like(x_clean)
            x_gamma = gamma * x_clean + (1 - gamma) * eps
            samples.append(x_gamma)

    elif regime == "eqm_iter":
        k_target = kwargs.get("k", 0)
        # Generate EqM iterates from noise with start/goal conditioning
        indices = np.random.choice(len(dataset), size=min(n_batch, len(dataset)), replace=False)
        for idx in indices:
            batch = dataset[idx]
            cond = {}
            for t_key, val in batch.conditions.items():
                val_t = torch.tensor(val, dtype=torch.float32, device=device)
                if val_t.ndim == 1:
                    val_t = val_t.unsqueeze(0)
                cond[t_key] = val_t

            x = torch.randn((1, horizon, TRANSITION_DIM), device=device)
            x = apply_conditioning(x, cond, ACT_DIM)
            t0 = torch.zeros((1,), dtype=torch.long, device=device)

            with torch.no_grad():
                for step in range(k_target):
                    grad = model.model(x, cond, t0)
                    x = x - float(model.step_size) * grad
                    if model.clip_denoised:
                        x = torch.clamp(x, -1.0, 1.0)
                    x = apply_conditioning(x, cond, ACT_DIM)

            samples.append(x.detach())

    return samples


def run_blockwise_probes(model, dataset, device, regime, n_probes, probe_types, **kwargs):
    """Run blockwise locality probes and return aggregated results."""
    horizon = model.horizon
    denoiser = model.model

    samples = generate_samples(model, dataset, regime, device, n_batch=n_probes, **kwargs)

    results = {}
    for pt in probe_types:
        offsets = defaultdict(list)
        done = 0
        for sample in samples:
            if done >= n_probes:
                break
            infl, pivot_t = blockwise_locality_probe(denoiser, sample, pt, device)
            for s in range(horizon):
                off = abs(s - pivot_t)
                offsets[off].append(infl[s])
            done += 1

        # Aggregate by offset
        rows = []
        for off in sorted(offsets.keys()):
            vals = np.array(offsets[off])
            rows.append({
                "offset": off,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "n": len(vals),
            })
        results[pt] = rows
        if done > 0:
            peak = rows[0]["mean"] if rows else 0
            off1 = rows[1]["mean"] if len(rows) > 1 else 0
            print(f"  {pt}: peak={peak:.4f}, off1={off1:.4f}, ratio={peak/(off1+1e-12):.1f}x")

    return results


def save_results(results, outdir, regime, regime_param=""):
    """Save CSV + JSON for each probe type."""
    os.makedirs(outdir, exist_ok=True)
    suffix = f"_{regime_param}" if regime_param else ""

    summary = {"regime": regime, "param": regime_param, "probe_types": {}}

    for pt, rows in results.items():
        csv_path = os.path.join(outdir, f"locality_{regime}{suffix}_{pt}.csv")
        with open(csv_path, "w") as f:
            f.write("offset,mean,std,median,n\n")
            for r in rows:
                f.write(f"{r['offset']},{r['mean']:.6f},{r['std']:.6f},{r['median']:.6f},{r['n']}\n")

        peak = rows[0]["mean"] if rows else 0
        off1 = rows[1]["mean"] if len(rows) > 1 else 0
        summary["probe_types"][pt] = {
            "peak_offset0": peak,
            "offset1_mean": off1,
            "ratio_0_to_1": peak / (off1 + 1e-12),
        }

    json_path = os.path.join(outdir, f"locality_{regime}{suffix}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {outdir}")

    return summary


def plot_blockwise(all_results, outdir, regime_label):
    """Plot all 4 probe types in a 2x2 grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot: matplotlib not available]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"H2 Blockwise Locality — {regime_label}", fontsize=14)

    for idx, pt in enumerate(PROBE_TYPES):
        ax = axes[idx // 2][idx % 2]
        if pt not in all_results:
            ax.set_title(pt)
            continue
        rows = all_results[pt]
        offsets = [r["offset"] for r in rows]
        means = [r["mean"] for r in rows]
        stds = [r["std"] for r in rows]
        ax.errorbar(offsets, means, yerr=stds, fmt="-o", markersize=2, capsize=2)
        ax.set_title(pt.replace("__", " → "))
        ax.set_xlabel("Temporal offset |t - t'|")
        ax.set_ylabel("Normalized influence")
        ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    path = os.path.join(outdir, f"locality_blockwise_{regime_label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description="H2 blockwise locality probes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_probes", type=int, default=512)
    parser.add_argument("--probe_regime", default="dataset",
                        choices=["dataset", "corrupted", "eqm_iter", "all"])
    parser.add_argument("--gamma_list", default="0.0,0.25,0.5,0.75,0.9")
    parser.add_argument("--k_list", default="0,1,2,5,10,25")
    parser.add_argument("--probe_types", default=",".join(PROBE_TYPES))
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    probe_types = [p.strip() for p in args.probe_types.split(",")]
    gammas = [float(g) for g in args.gamma_list.split(",")]
    k_list = [int(k) for k in args.k_list.split(",")]

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or f"runs/analysis/eqm_h2_blockwise_{ts}"
    os.makedirs(outdir, exist_ok=True)

    print(f"[H2 blockwise] Loading checkpoint: {args.checkpoint}")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    print(f"[H2 blockwise] horizon={model.horizon}, n_probes={args.n_probes}")

    regimes = [args.probe_regime] if args.probe_regime != "all" else ["dataset", "corrupted", "eqm_iter"]
    all_summaries = {}

    for regime in regimes:
        if regime == "dataset":
            print(f"\n--- Regime: dataset ---")
            results = run_blockwise_probes(model, dataset, device, "dataset", args.n_probes, probe_types)
            summary = save_results(results, outdir, "dataset")
            plot_blockwise(results, outdir, "dataset")
            all_summaries["dataset"] = summary

        elif regime == "corrupted":
            for gamma in gammas:
                print(f"\n--- Regime: corrupted, gamma={gamma} ---")
                results = run_blockwise_probes(
                    model, dataset, device, "corrupted", args.n_probes, probe_types, gamma=gamma
                )
                param = f"g{gamma:.2f}".replace(".", "")
                summary = save_results(results, outdir, "corrupted", param)
                plot_blockwise(results, outdir, f"corrupted_gamma={gamma:.2f}")
                all_summaries[f"corrupted_g{gamma}"] = summary

        elif regime == "eqm_iter":
            for k in k_list:
                print(f"\n--- Regime: eqm_iter, k={k} ---")
                results = run_blockwise_probes(
                    model, dataset, device, "eqm_iter", args.n_probes, probe_types, k=k
                )
                param = f"k{k:02d}"
                summary = save_results(results, outdir, "eqm_iter", param)
                plot_blockwise(results, outdir, f"eqm_iter_k={k}")
                all_summaries[f"eqm_iter_k{k}"] = summary

    # Write combined summary
    combined_path = os.path.join(outdir, "blockwise_combined_summary.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[H2 blockwise] All results saved to {outdir}")


if __name__ == "__main__":
    main()
