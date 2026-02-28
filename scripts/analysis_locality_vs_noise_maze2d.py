#!/usr/bin/env python3
"""Locality-vs-noise analysis: EqM and Diffuser at multiple noise levels on Maze2D.

For the Diffuser, measures the temporal locality (VJP influence bandwidth)
of the U-Net denoiser at different denoising timesteps:
  t≈0       (near-clean data)
  t≈T/4     (early denoising)
  t≈T/2     (mid denoising)
  t≈3T/4    (late denoising)
  t≈T-1     (pure noise)

For EqM, measures locality at different noise levels added to clean data:
  Same q_sample noise schedule as Diffuser for direct comparison,
  but always using EqM's t=0 timestep embedding (since EqM is time-invariant).

Both models share the same TemporalUnet backbone so the difference is
purely in how training shaped the weights.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \\
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \\
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \\
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \\
    scripts/analysis_locality_vs_noise_maze2d.py \\
    --eqm_checkpoint <path> \\
    --diffuser_checkpoint <path> \\
    [--n_samples 256] [--noise_timesteps 0,16,32,48,63]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    load_diffuser_model_and_dataset,
    load_eqm_model_and_dataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# VJP locality measurement
# ─────────────────────────────────────────────────────────────────────────────

def locality_profile(denoiser, x, t_embed):
    """Compute influence profile via VJP for one random pivot.

    Args:
        denoiser: TemporalUnet — model(x, cond, t) -> output
        x: trajectory [1, H, D], will clone+require_grad
        t_embed: integer timestep to pass to the model

    Returns:
        (influence: np.ndarray [H], pivot_t: int)
    """
    x = x.clone().detach().requires_grad_(True)
    H = x.shape[1]
    D = x.shape[2]
    device = x.device

    t_tensor = torch.full((1,), t_embed, dtype=torch.long, device=device)
    y = denoiser(x, {}, t_tensor)  # [1, H, D]

    pivot = int(torch.randint(0, H, (1,)).item())
    u = torch.randn((D,), device=device)

    scalar = (y[0, pivot, :] * u).sum()
    grad = torch.autograd.grad(scalar, x, create_graph=False)[0]  # [1, H, D]

    infl = grad[0].norm(dim=1)  # [H]
    infl = infl / (infl.max() + 1e-12)
    return infl.detach().cpu().numpy(), pivot


def collect_locality_stats(denoiser, dataset, device, t_embed, horizon,
                           n_samples, noise_fn=None, label=""):
    """Collect aggregated locality profile.

    Args:
        denoiser: TemporalUnet
        dataset: GoalDataset to iterate
        device: torch device
        t_embed: timestep to pass as model embedding
        horizon: trajectory length
        n_samples: number of VJP probes
        noise_fn: optional fn(x_clean) -> x_noisy. If None, use x_clean directly.
        label: string label for progress printing

    Returns:
        dict with offset -> mean/std/median influence
    """
    offset_influences = {d: [] for d in range(horizon)}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    for i in range(n_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        traj = batch.trajectories.to(device)  # [1, H, D]

        if noise_fn is not None:
            traj = noise_fn(traj)

        infl, pivot = locality_profile(denoiser, traj, t_embed)

        for t_prime in range(horizon):
            offset = abs(t_prime - pivot)
            offset_influences[offset].append(float(infl[t_prime]))

        if (i + 1) % 64 == 0:
            print(f"    [{label}] {i+1}/{n_samples} probes done")

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
    return results


def find_half_decay(results):
    """Find the first offset where mean influence drops below 50% of peak."""
    if not results or results[0]["mean_influence"] <= 0:
        return None
    peak = results[0]["mean_influence"]
    for r in results:
        if r["mean_influence"] < peak * 0.5:
            return r["offset"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Locality-vs-noise: EqM and Diffuser at multiple noise levels")
    parser.add_argument("--eqm_checkpoint", required=True)
    parser.add_argument("--diffuser_checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=256,
                        help="VJP probes per condition")
    parser.add_argument("--noise_timesteps", type=str, default="0,16,32,48,63",
                        help="Comma-separated diffusion timesteps to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    noise_ts = [int(x) for x in args.noise_timesteps.split(",")]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or f"runs/analysis/locality_vs_noise_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────
    print("[load] Loading EqM checkpoint...")
    eqm_model, eqm_dataset, eqm_cfg = load_eqm_model_and_dataset(
        args.eqm_checkpoint, device)
    eqm_horizon = int(eqm_cfg["horizon"])

    print("[load] Loading Diffuser checkpoint...")
    diff_model, diff_dataset, diff_cfg = load_diffuser_model_and_dataset(
        args.diffuser_checkpoint, device)
    diff_horizon = int(diff_cfg["horizon"])
    n_diff_steps = int(diff_cfg.get("n_diffusion_steps", 64))

    print(f"[info] EqM h={eqm_horizon}, Diffuser h={diff_horizon}, "
          f"n_diff_steps={n_diff_steps}")
    print(f"[info] Test timesteps: {noise_ts}")
    print(f"[info] Probes per condition: {args.n_samples}")

    # Get raw denoiser (TemporalUnet) from each model
    eqm_denoiser = eqm_model.model
    eqm_denoiser.eval()
    diff_denoiser = diff_model.model
    diff_denoiser.eval()

    # Print noise schedule at test timesteps
    print(f"\n[schedule] Diffuser noise schedule (cosine, T={n_diff_steps}):")
    for t in noise_ts:
        if t < n_diff_steps:
            alpha_bar = float(diff_model.alphas_cumprod[t])
            snr_db = 10 * np.log10(alpha_bar / (1 - alpha_bar + 1e-12))
            print(f"  t={t:3d}: alpha_bar={alpha_bar:.4f}, "
                  f"sqrt_alpha={float(diff_model.sqrt_alphas_cumprod[t]):.4f}, "
                  f"sqrt_1-alpha={float(diff_model.sqrt_one_minus_alphas_cumprod[t]):.4f}, "
                  f"SNR={snr_db:.1f} dB")

    # ── Run experiments ──────────────────────────────────────────────────
    all_results = {}

    for t in noise_ts:
        t_clamped = min(t, n_diff_steps - 1)
        print(f"\n{'='*70}")
        print(f"Noise level: t={t_clamped} / {n_diff_steps-1}")
        print(f"{'='*70}")

        # Build q_sample noise function using Diffuser's schedule
        sqrt_alpha = float(diff_model.sqrt_alphas_cumprod[t_clamped])
        sqrt_one_minus_alpha = float(diff_model.sqrt_one_minus_alphas_cumprod[t_clamped])

        def make_noise_fn(sa, soma):
            """Closure to capture schedule values."""
            def noise_fn(x_clean):
                noise = torch.randn_like(x_clean)
                return sa * x_clean + soma * noise
            return noise_fn

        noise_fn = make_noise_fn(sqrt_alpha, sqrt_one_minus_alpha)

        # ── Diffuser locality at this noise level ────────────────────
        print(f"\n  [diffuser] Measuring locality at t={t_clamped}...")
        diff_results = collect_locality_stats(
            diff_denoiser, diff_dataset, device,
            t_embed=t_clamped,
            horizon=diff_horizon,
            n_samples=args.n_samples,
            noise_fn=noise_fn,
            label=f"diff_t{t_clamped}",
        )
        diff_hd = find_half_decay(diff_results)
        print(f"  [diffuser] peak={diff_results[0]['mean_influence']:.4f}, "
              f"half_decay={diff_hd}")
        all_results[f"diffuser_t{t_clamped}"] = {
            "model": "diffuser",
            "noise_timestep": t_clamped,
            "t_embed": t_clamped,
            "sqrt_alpha_bar": sqrt_alpha,
            "sqrt_1m_alpha_bar": sqrt_one_minus_alpha,
            "half_decay_offset": diff_hd,
            "peak_influence": diff_results[0]["mean_influence"],
            "profile": diff_results,
        }

        # ── EqM locality at same noise level ─────────────────────────
        print(f"\n  [eqm] Measuring locality at noise level t={t_clamped} (t_embed=0)...")
        eqm_results = collect_locality_stats(
            eqm_denoiser, eqm_dataset, device,
            t_embed=0,  # EqM always uses t=0
            horizon=eqm_horizon,
            n_samples=args.n_samples,
            noise_fn=noise_fn,
            label=f"eqm_noise{t_clamped}",
        )
        eqm_hd = find_half_decay(eqm_results)
        print(f"  [eqm] peak={eqm_results[0]['mean_influence']:.4f}, "
              f"half_decay={eqm_hd}")
        all_results[f"eqm_noise{t_clamped}"] = {
            "model": "eqm",
            "noise_timestep": t_clamped,
            "t_embed": 0,
            "sqrt_alpha_bar": sqrt_alpha,
            "sqrt_1m_alpha_bar": sqrt_one_minus_alpha,
            "half_decay_offset": eqm_hd,
            "peak_influence": eqm_results[0]["mean_influence"],
            "profile": eqm_results,
        }

    # ── Save results ─────────────────────────────────────────────────────
    # JSON
    json_path = outdir / "locality_vs_noise_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] JSON: {json_path}")

    # CSV summary
    csv_path = outdir / "locality_vs_noise_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "noise_t", "t_embed", "sqrt_alpha_bar",
                         "peak_influence", "half_decay_offset",
                         "influence_offset1", "influence_offset5",
                         "influence_offset10", "influence_offset_max"])
        for key, res in all_results.items():
            prof = res["profile"]
            i1 = prof[1]["mean_influence"] if len(prof) > 1 else None
            i5 = prof[5]["mean_influence"] if len(prof) > 5 else None
            i10 = prof[10]["mean_influence"] if len(prof) > 10 else None
            imax = prof[-1]["mean_influence"] if prof else None
            writer.writerow([
                res["model"], res["noise_timestep"], res["t_embed"],
                f"{res['sqrt_alpha_bar']:.4f}",
                f"{res['peak_influence']:.4f}",
                res["half_decay_offset"],
                f"{i1:.4f}" if i1 is not None else "",
                f"{i5:.4f}" if i5 is not None else "",
                f"{i10:.4f}" if i10 is not None else "",
                f"{imax:.4f}" if imax is not None else "",
            ])
    print(f"[save] CSV: {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    # Plot 1: Diffuser locality profiles at different noise levels
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.set_title("Diffuser: locality profile at different noise levels", fontsize=11)
    for t in noise_ts:
        key = f"diffuser_t{min(t, n_diff_steps-1)}"
        if key in all_results:
            prof = all_results[key]["profile"]
            offsets = [r["offset"] for r in prof]
            means = [r["mean_influence"] for r in prof]
            sa = all_results[key]["sqrt_alpha_bar"]
            ax.plot(offsets, means, "o-", markersize=2, alpha=0.8,
                    label=f"t={min(t,n_diff_steps-1)} (sqrt_α={sa:.3f})")
    ax.set_xlabel("Temporal offset |t - t'|")
    ax.set_ylabel("Normalized influence")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.3, label="50% decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: EqM locality profiles at different noise levels
    ax = axes[1]
    ax.set_title("EqM: locality profile at different noise levels (t_embed=0)", fontsize=11)
    for t in noise_ts:
        key = f"eqm_noise{min(t, n_diff_steps-1)}"
        if key in all_results:
            prof = all_results[key]["profile"]
            offsets = [r["offset"] for r in prof]
            means = [r["mean_influence"] for r in prof]
            sa = all_results[key]["sqrt_alpha_bar"]
            ax.plot(offsets, means, "o-", markersize=2, alpha=0.8,
                    label=f"noise_t={min(t,n_diff_steps-1)} (sqrt_α={sa:.3f})")
    ax.set_xlabel("Temporal offset |t - t'|")
    ax.set_ylabel("Normalized influence")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.3, label="50% decay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Temporal Locality vs Noise Level: Diffuser vs EqM (Maze2D)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plot_path = outdir / "locality_vs_noise_profiles.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Profile plot: {plot_path}")

    # Plot 3: Half-decay offset vs noise level (comparison)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    diff_ts = []
    diff_hds = []
    eqm_ts = []
    eqm_hds = []
    for t in noise_ts:
        tc = min(t, n_diff_steps - 1)
        dk = f"diffuser_t{tc}"
        ek = f"eqm_noise{tc}"
        if dk in all_results and all_results[dk]["half_decay_offset"] is not None:
            diff_ts.append(tc)
            diff_hds.append(all_results[dk]["half_decay_offset"])
        if ek in all_results and all_results[ek]["half_decay_offset"] is not None:
            eqm_ts.append(tc)
            eqm_hds.append(all_results[ek]["half_decay_offset"])

    ax2.plot(diff_ts, diff_hds, "s-", color="#9933cc", markersize=8,
             linewidth=2, label="Diffuser", alpha=0.8)
    ax2.plot(eqm_ts, eqm_hds, "o-", color="#00bb55", markersize=8,
             linewidth=2, label="EqM", alpha=0.8)
    ax2.set_xlabel("Noise level (diffusion timestep t)", fontsize=11)
    ax2.set_ylabel("Half-decay offset (timesteps)", fontsize=11)
    ax2.set_title("Temporal Influence Bandwidth vs Noise Level\n"
                   "(higher = less local = more temporal coherence)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    hd_path = outdir / "half_decay_vs_noise.png"
    fig2.savefig(hd_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[save] Half-decay plot: {hd_path}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("LOCALITY VS NOISE LEVEL — SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Model':<12} {'Noise t':>8} {'t_embed':>8} {'sqrt_α':>8} "
          f"{'Peak':>8} {'HD offset':>10} {'Inf@1':>8} {'Inf@5':>8} {'Inf@10':>8}")
    print("-" * 90)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        prof = r["profile"]
        i1 = prof[1]["mean_influence"] if len(prof) > 1 else 0
        i5 = prof[5]["mean_influence"] if len(prof) > 5 else 0
        i10 = prof[10]["mean_influence"] if len(prof) > 10 else 0
        print(f"{r['model']:<12} {r['noise_timestep']:>8} {r['t_embed']:>8} "
              f"{r['sqrt_alpha_bar']:>8.4f} {r['peak_influence']:>8.4f} "
              f"{str(r['half_decay_offset']):>10} {i1:>8.4f} {i5:>8.4f} {i10:>8.4f}")

    print(f"\nResults saved to: {outdir}")


if __name__ == "__main__":
    main()
