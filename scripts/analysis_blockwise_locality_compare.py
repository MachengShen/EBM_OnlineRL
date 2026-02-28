#!/usr/bin/env python3
"""Blockwise locality comparison: EqM vs Diffuser, raw grad norms at offsets 0-4.

Reports both raw (unnormalized) and normalized grad norms for each block pair
(obs→obs, act→act, obs→act, act→obs) at multiple noise levels.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/analysis_blockwise_locality_compare.py \
    --eqm_checkpoint <path> \
    --diffuser_checkpoint <path> \
    [--n_probes 256] [--noise_timesteps 0,32,63]
"""
from __future__ import annotations

import argparse
import json
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
    load_diffuser_model_and_dataset,
    load_eqm_model_and_dataset,
)

BLOCK_PAIRS = [
    "obs_out__obs_in",
    "act_out__act_in",
    "obs_out__act_in",
    "act_out__obs_in",
]


def blockwise_probe_raw(denoiser, x, block_pair, t_embed, device):
    """Single VJP probe returning RAW (unnormalized) per-timestep grad norms.

    Returns:
        (raw_grad_norms: np.ndarray [H], pivot_t: int)
    """
    x = x.clone().detach().requires_grad_(True)
    t_tensor = torch.full((1,), t_embed, dtype=torch.long, device=device)
    y = denoiser(x, {}, t_tensor)  # [1, H, D]

    H = x.shape[1]
    pivot = int(torch.randint(0, H, (1,)).item())

    out_block, in_block = block_pair.split("__")

    # Random projection on output block
    if out_block == "act_out":
        u = torch.randn((ACT_DIM,), device=device)
        u = u / (u.norm() + 1e-12)  # unit vector for scale consistency
        scalar = (y[0, pivot, :ACT_DIM] * u).sum()
    else:  # obs_out
        u = torch.randn((OBS_DIM,), device=device)
        u = u / (u.norm() + 1e-12)
        scalar = (y[0, pivot, ACT_DIM:] * u).sum()

    grad = torch.autograd.grad(scalar, x, create_graph=False)[0]  # [1, H, D]

    # Extract grad norm on input block
    if in_block == "act_in":
        raw_norms = grad[0, :, :ACT_DIM].norm(dim=1)  # [H]
    else:  # obs_in
        raw_norms = grad[0, :, ACT_DIM:].norm(dim=1)  # [H]

    return raw_norms.detach().cpu().numpy(), pivot


def collect_blockwise_stats(denoiser, dataset, device, t_embed, horizon,
                            n_probes, noise_fn, block_pairs, max_offset=10):
    """Collect raw grad norm stats per offset per block pair."""
    # offset_data[block_pair][offset] = list of raw grad norms
    offset_data = {bp: defaultdict(list) for bp in block_pairs}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    for i in range(n_probes):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        traj = batch.trajectories.to(device)  # [1, H, D]
        if noise_fn is not None:
            traj = noise_fn(traj)

        for bp in block_pairs:
            raw_norms, pivot = blockwise_probe_raw(denoiser, traj, bp, t_embed, device)
            for s in range(horizon):
                off = abs(s - pivot)
                if off <= max_offset:
                    offset_data[bp][off].append(float(raw_norms[s]))

        if (i + 1) % 64 == 0:
            print(f"    {i+1}/{n_probes} probes done")

    # Aggregate
    results = {}
    for bp in block_pairs:
        rows = []
        for off in sorted(offset_data[bp].keys()):
            vals = np.array(offset_data[bp][off])
            rows.append({
                "offset": off,
                "raw_mean": float(np.mean(vals)),
                "raw_std": float(np.std(vals)),
                "raw_median": float(np.median(vals)),
                "n": len(vals),
            })
        # Add normalized versions (normalized by offset-0 mean)
        peak = rows[0]["raw_mean"] if rows else 1.0
        for r in rows:
            r["norm_mean"] = r["raw_mean"] / (peak + 1e-12)
        results[bp] = rows
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eqm_checkpoint", required=True)
    parser.add_argument("--diffuser_checkpoint", required=True)
    parser.add_argument("--n_probes", type=int, default=256)
    parser.add_argument("--noise_timesteps", type=str, default="0,32,63")
    parser.add_argument("--max_offset", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    noise_ts = [int(x) for x in args.noise_timesteps.split(",")]

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or f"runs/analysis/blockwise_locality_compare_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    print("[load] EqM...")
    eqm_model, eqm_dataset, eqm_cfg = load_eqm_model_and_dataset(
        args.eqm_checkpoint, device)
    eqm_horizon = int(eqm_cfg["horizon"])

    print("[load] Diffuser...")
    diff_model, diff_dataset, diff_cfg = load_diffuser_model_and_dataset(
        args.diffuser_checkpoint, device)
    diff_horizon = int(diff_cfg["horizon"])
    n_diff_steps = int(diff_cfg.get("n_diffusion_steps", 64))

    eqm_denoiser = eqm_model.model; eqm_denoiser.eval()
    diff_denoiser = diff_model.model; diff_denoiser.eval()

    # Print noise schedule
    print(f"\n[schedule] Cosine schedule (T={n_diff_steps}):")
    for t in noise_ts:
        tc = min(t, n_diff_steps - 1)
        sa = float(diff_model.sqrt_alphas_cumprod[tc])
        soma = float(diff_model.sqrt_one_minus_alphas_cumprod[tc])
        print(f"  t={tc}: sqrt_alpha={sa:.4f}, sqrt_1-alpha={soma:.4f}")

    all_results = {}

    for t in noise_ts:
        tc = min(t, n_diff_steps - 1)
        sa = float(diff_model.sqrt_alphas_cumprod[tc])
        soma = float(diff_model.sqrt_one_minus_alphas_cumprod[tc])

        def make_noise_fn(sa_, soma_):
            def fn(x):
                return sa_ * x + soma_ * torch.randn_like(x)
            return fn
        noise_fn = make_noise_fn(sa, soma)

        print(f"\n{'='*70}")
        print(f"NOISE t={tc} (sqrt_alpha={sa:.4f})")
        print(f"{'='*70}")

        # Diffuser
        print(f"\n  [diffuser t_embed={tc}]")
        diff_res = collect_blockwise_stats(
            diff_denoiser, diff_dataset, device, t_embed=tc,
            horizon=diff_horizon, n_probes=args.n_probes,
            noise_fn=noise_fn, block_pairs=BLOCK_PAIRS,
            max_offset=args.max_offset)
        all_results[f"diffuser_t{tc}"] = diff_res

        # EqM
        print(f"\n  [eqm t_embed=0, noise_level=t{tc}]")
        eqm_res = collect_blockwise_stats(
            eqm_denoiser, eqm_dataset, device, t_embed=0,
            horizon=eqm_horizon, n_probes=args.n_probes,
            noise_fn=noise_fn, block_pairs=BLOCK_PAIRS,
            max_offset=args.max_offset)
        all_results[f"eqm_t{tc}"] = eqm_res

    # ── Save JSON ────────────────────────────────────────────────────────
    json_path = outdir / "blockwise_locality_compare.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] {json_path}")

    # ── Print summary tables ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("BLOCKWISE LOCALITY: RAW GRAD NORMS AT OFFSETS 0-4")
    print(f"{'='*100}")

    for t in noise_ts:
        tc = min(t, n_diff_steps - 1)
        sa = float(diff_model.sqrt_alphas_cumprod[tc])
        print(f"\n--- Noise level t={tc} (sqrt_alpha={sa:.4f}) ---\n")

        for bp in BLOCK_PAIRS:
            print(f"  {bp.replace('__', ' → ')}:")
            diff_rows = all_results[f"diffuser_t{tc}"][bp]
            eqm_rows = all_results[f"eqm_t{tc}"][bp]

            print(f"    {'Offset':>6}  {'Diff raw':>10} {'Diff norm':>10}  "
                  f"{'EqM raw':>10} {'EqM norm':>10}  {'Ratio D/E':>10}")
            for off in range(min(5, len(diff_rows), len(eqm_rows))):
                dr = diff_rows[off]
                er = eqm_rows[off]
                ratio = dr["raw_mean"] / (er["raw_mean"] + 1e-12)
                print(f"    {off:>6}  {dr['raw_mean']:>10.6f} {dr['norm_mean']:>10.4f}  "
                      f"{er['raw_mean']:>10.6f} {er['norm_mean']:>10.4f}  {ratio:>10.3f}")
            print()

    # ── Compact table: just obs→obs offsets 0-4 at clean (t=0) ───────
    print(f"\n{'='*100}")
    print("COMPACT: obs→obs raw grad norms, offsets 0-4")
    print(f"{'='*100}\n")
    print(f"{'Model':<12} {'Noise':>6} {'Off0':>10} {'Off1':>10} {'Off2':>10} "
          f"{'Off3':>10} {'Off4':>10}  {'1/0 ratio':>10}")
    print("-" * 90)
    for t in noise_ts:
        tc = min(t, n_diff_steps - 1)
        for model_label, key in [("Diffuser", f"diffuser_t{tc}"),
                                   ("EqM", f"eqm_t{tc}")]:
            rows = all_results[key]["obs_out__obs_in"]
            vals = [rows[i]["raw_mean"] if i < len(rows) else 0 for i in range(5)]
            ratio_1_0 = vals[1] / (vals[0] + 1e-12)
            print(f"{model_label:<12} {tc:>6} {vals[0]:>10.6f} {vals[1]:>10.6f} "
                  f"{vals[2]:>10.6f} {vals[3]:>10.6f} {vals[4]:>10.6f}  {ratio_1_0:>10.4f}")
    print()

    # Same for act→act
    print(f"{'Model':<12} {'Noise':>6} {'Off0':>10} {'Off1':>10} {'Off2':>10} "
          f"{'Off3':>10} {'Off4':>10}  {'1/0 ratio':>10}")
    print("-" * 90)
    for t in noise_ts:
        tc = min(t, n_diff_steps - 1)
        for model_label, key in [("Diffuser", f"diffuser_t{tc}"),
                                   ("EqM", f"eqm_t{tc}")]:
            rows = all_results[key]["act_out__act_in"]
            vals = [rows[i]["raw_mean"] if i < len(rows) else 0 for i in range(5)]
            ratio_1_0 = vals[1] / (vals[0] + 1e-12)
            print(f"{model_label:<12} {tc:>6} {vals[0]:>10.6f} {vals[1]:>10.6f} "
                  f"{vals[2]:>10.6f} {vals[3]:>10.6f} {vals[4]:>10.6f}  {ratio_1_0:>10.4f}")

    print(f"\nResults saved to: {outdir}")


if __name__ == "__main__":
    main()
