#!/usr/bin/env python3
"""H1 follow-up: dynamics alignment sweeps across corruption and EqM iterate regimes.

Three descent metrics: dot-sign fraction, cosine similarity, small-step decrease.
Gamma sweep (corrupted samples) and k sweep (EqM iterates from noise).

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/analysis_eqm_maze2d_dyn_alignment_sweeps.py \
    --checkpoint <eqm_ckpt> --dyn_ckpt <forward_dyn.pt>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    load_eqm_model_and_dataset,
)
from diffuser.models.helpers import apply_conditioning

# Add project root for models
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from ebm_online_rl.models.forward_dynamics import ForwardDynamics


def unpack_sa(x, obs_dim, act_dim):
    """Unpack [act|obs] trajectory into states and actions."""
    a = x[:, :-1, :act_dim]             # [B, H-1, act]
    s = x[:, :, act_dim:act_dim + obs_dim]  # [B, H, obs]
    return s, a


def j_dyn(x, f_dyn, lam_u=0.0):
    """Compute dynamics residual objective (differentiable)."""
    s, a = unpack_sa(x, OBS_DIM, ACT_DIM)
    s_next_pred = f_dyn(s[:, :-1, :], a)  # [B, H-1, obs]
    s_next_true = s[:, 1:, :]              # [B, H-1, obs]
    dyn_loss = ((s_next_pred - s_next_true) ** 2).mean()
    if lam_u > 0:
        dyn_loss = dyn_loss + lam_u * (a ** 2).mean()
    return dyn_loss


def generate_samples_for_regime(model, dataset, device, regime, n, **kwargs):
    """Generate trajectory samples for a given regime.

    Returns list of (x_tensor [1,H,D], cond_dict) tuples.
    """
    horizon = model.horizon
    samples = []
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    for idx in indices:
        batch = dataset[idx]
        cond = {}
        for t_key, val in batch.conditions.items():
            val_t = torch.tensor(val, dtype=torch.float32, device=device)
            if val_t.ndim == 1:
                val_t = val_t.unsqueeze(0)
            cond[t_key] = val_t

        if regime == "dataset":
            x = torch.tensor(batch.trajectories, dtype=torch.float32, device=device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            samples.append((x, cond))

        elif regime == "corrupted":
            gamma = kwargs.get("gamma", 0.5)
            x_clean = torch.tensor(batch.trajectories, dtype=torch.float32, device=device)
            if x_clean.ndim == 2:
                x_clean = x_clean.unsqueeze(0)
            eps = torch.randn_like(x_clean)
            x_gamma = gamma * x_clean + (1 - gamma) * eps
            x_gamma = apply_conditioning(x_gamma, cond, ACT_DIM)
            samples.append((x_gamma, cond))

        elif regime == "eqm_iter":
            k_target = kwargs.get("k", 0)
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
            samples.append((x.detach(), cond))

    return samples


def compute_alignment_metrics(model, f_dyn, samples, device, lam_u=0.0, eps_step=0.01):
    """Compute 3 alignment metrics + blockwise cosine on a batch of samples.

    Returns dict of aggregated metrics.
    """
    denoiser = model.model
    metrics = {
        "dot_positive_frac": [],
        "cosine_overall": [],
        "cosine_action": [],
        "cosine_state": [],
        "small_step_descent_frac": [],
        "j_dyn_value": [],
    }

    for x_detach, cond in samples:
        # 1. Compute grad_J
        x = x_detach.clone().detach().requires_grad_(True)
        loss = j_dyn(x, f_dyn, lam_u=lam_u)
        grad_J = torch.autograd.grad(loss, x, create_graph=False)[0]  # [1, H, D]

        # 2. Compute EqM field f(x)
        with torch.no_grad():
            t0 = torch.zeros((1,), dtype=torch.long, device=device)
            f_field = denoiser(x_detach, {}, t0)  # [1, H, D]

        # Flatten for metrics (use reshape for non-contiguous tensors)
        f_flat = f_field.reshape(-1)
        g_flat = grad_J.reshape(-1)

        # Metric 1: dot product sign (is -f a descent direction?)
        dot = (g_flat * f_flat).sum().item()
        metrics["dot_positive_frac"].append(1.0 if dot > 0 else 0.0)

        # Metric 2: cosine similarity (overall, action-only, state-only)
        cos = F.cosine_similarity(f_flat.unsqueeze(0), g_flat.unsqueeze(0)).item()
        metrics["cosine_overall"].append(cos)

        # Action-only
        f_act = f_field[:, :, :ACT_DIM].reshape(-1)
        g_act = grad_J[:, :, :ACT_DIM].reshape(-1)
        cos_act = F.cosine_similarity(f_act.unsqueeze(0), g_act.unsqueeze(0)).item()
        metrics["cosine_action"].append(cos_act)

        # State-only
        f_obs = f_field[:, :, ACT_DIM:].reshape(-1)
        g_obs = grad_J[:, :, ACT_DIM:].reshape(-1)
        cos_obs = F.cosine_similarity(f_obs.unsqueeze(0), g_obs.unsqueeze(0)).item()
        metrics["cosine_state"].append(cos_obs)

        # Metric 3: small normalized step decrease
        with torch.no_grad():
            f_norm = f_field / (f_field.norm() + 1e-12)
            x1 = x_detach - eps_step * f_norm
            j_before = j_dyn(x_detach, f_dyn, lam_u=lam_u).item()
            j_after = j_dyn(x1, f_dyn, lam_u=lam_u).item()
            metrics["small_step_descent_frac"].append(1.0 if j_after < j_before else 0.0)
            metrics["j_dyn_value"].append(j_before)

    # Aggregate
    agg = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        agg[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "n": len(arr),
        }
    return agg


def main():
    parser = argparse.ArgumentParser(description="H1 alignment sweeps")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dyn_ckpt", required=True, help="Path to trained ForwardDynamics .pt")
    parser.add_argument("--lam_u", type=float, default=0.0)
    parser.add_argument("--lam_u_secondary", type=float, default=1e-3)
    parser.add_argument("--gamma_list", default="0.0,0.25,0.5,0.75,0.9,0.99")
    parser.add_argument("--k_list", default="0,1,2,5,10,25")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--eps_step", type=float, default=0.01)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gammas = [float(g) for g in args.gamma_list.split(",")]
    k_list = [int(k) for k in args.k_list.split(",")]

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or f"runs/analysis/eqm_h1_alignment_sweeps_{ts}"
    os.makedirs(outdir, exist_ok=True)

    # Load EqM model
    print(f"[H1 sweeps] Loading EqM checkpoint: {args.checkpoint}")
    model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    print(f"[H1 sweeps] horizon={model.horizon}")

    # Load forward dynamics
    print(f"[H1 sweeps] Loading ForwardDynamics: {args.dyn_ckpt}")
    f_dyn = ForwardDynamics(OBS_DIM, ACT_DIM).to(device)
    dyn_state = torch.load(args.dyn_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in dyn_state:
        f_dyn.load_state_dict(dyn_state["model_state_dict"])
    else:
        f_dyn.load_state_dict(dyn_state)
    f_dyn.eval()

    all_results = {}

    # --- Gamma sweep (corrupted) ---
    print(f"\n=== Gamma sweep (corrupted samples) ===")
    for gamma in gammas:
        print(f"\n  gamma={gamma:.2f}")
        samples = generate_samples_for_regime(
            model, dataset, device, "corrupted", args.n_samples, gamma=gamma
        )
        for lam_u_label, lam_u_val in [("lam0", args.lam_u), ("lam1e3", args.lam_u_secondary)]:
            agg = compute_alignment_metrics(model, f_dyn, samples, device, lam_u=lam_u_val, eps_step=args.eps_step)
            key = f"gamma_{gamma:.2f}_{lam_u_label}"
            all_results[key] = {
                "regime": "corrupted",
                "gamma": gamma,
                "lam_u": lam_u_val,
                **agg,
            }
            dot_f = agg["dot_positive_frac"]["mean"]
            cos_o = agg["cosine_overall"]["mean"]
            cos_a = agg["cosine_action"]["mean"]
            desc = agg["small_step_descent_frac"]["mean"]
            j_val = agg["j_dyn_value"]["mean"]
            print(f"    [{lam_u_label}] dot+={dot_f:.3f} cos={cos_o:.3f} cos_act={cos_a:.3f} desc={desc:.3f} J={j_val:.6f}")

    # --- K sweep (EqM iterates) ---
    print(f"\n=== K sweep (EqM iterates from noise) ===")
    for k in k_list:
        print(f"\n  k={k}")
        samples = generate_samples_for_regime(
            model, dataset, device, "eqm_iter", args.n_samples, k=k
        )
        for lam_u_label, lam_u_val in [("lam0", args.lam_u), ("lam1e3", args.lam_u_secondary)]:
            agg = compute_alignment_metrics(model, f_dyn, samples, device, lam_u=lam_u_val, eps_step=args.eps_step)
            key = f"k_{k:02d}_{lam_u_label}"
            all_results[key] = {
                "regime": "eqm_iter",
                "k": k,
                "lam_u": lam_u_val,
                **agg,
            }
            dot_f = agg["dot_positive_frac"]["mean"]
            cos_o = agg["cosine_overall"]["mean"]
            cos_a = agg["cosine_action"]["mean"]
            desc = agg["small_step_descent_frac"]["mean"]
            j_val = agg["j_dyn_value"]["mean"]
            print(f"    [{lam_u_label}] dot+={dot_f:.3f} cos={cos_o:.3f} cos_act={cos_a:.3f} desc={desc:.3f} J={j_val:.6f}")

    # --- Dataset baseline ---
    print(f"\n=== Dataset baseline ===")
    samples = generate_samples_for_regime(model, dataset, device, "dataset", args.n_samples)
    for lam_u_label, lam_u_val in [("lam0", args.lam_u), ("lam1e3", args.lam_u_secondary)]:
        agg = compute_alignment_metrics(model, f_dyn, samples, device, lam_u=lam_u_val, eps_step=args.eps_step)
        key = f"dataset_{lam_u_label}"
        all_results[key] = {"regime": "dataset", "lam_u": lam_u_val, **agg}
        dot_f = agg["dot_positive_frac"]["mean"]
        cos_o = agg["cosine_overall"]["mean"]
        cos_a = agg["cosine_action"]["mean"]
        desc = agg["small_step_descent_frac"]["mean"]
        j_val = agg["j_dyn_value"]["mean"]
        print(f"  [{lam_u_label}] dot+={dot_f:.3f} cos={cos_o:.3f} cos_act={cos_a:.3f} desc={desc:.3f} J={j_val:.6f}")

    # Save all results
    out_path = os.path.join(outdir, "alignment_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[H1 sweeps] All results saved to {outdir}")

    # --- Generate summary plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: gamma sweep (lam0 only)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("H1 Alignment vs Corruption (gamma sweep, lam_u=0)")
        gamma_vals = sorted(gammas)
        for metric_idx, (metric_name, label) in enumerate([
            ("dot_positive_frac", "Dot+ fraction"),
            ("cosine_overall", "Cosine (overall)"),
            ("small_step_descent_frac", "Small-step descent"),
        ]):
            ax = axes[metric_idx]
            means = [all_results[f"gamma_{g:.2f}_lam0"][metric_name]["mean"] for g in gamma_vals]
            ax.plot(gamma_vals, means, "o-", linewidth=2)
            ax.set_xlabel("gamma (1=clean, 0=noise)")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "alignment_gamma_sweep.png"), dpi=150)
        plt.close()

        # Plot 2: k sweep (lam0 only)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("H1 Alignment vs EqM Iterate (k sweep, lam_u=0)")
        for metric_idx, (metric_name, label) in enumerate([
            ("dot_positive_frac", "Dot+ fraction"),
            ("cosine_overall", "Cosine (overall)"),
            ("small_step_descent_frac", "Small-step descent"),
        ]):
            ax = axes[metric_idx]
            means = [all_results[f"k_{k:02d}_lam0"][metric_name]["mean"] for k in k_list]
            ax.plot(k_list, means, "s-", linewidth=2, color="tab:orange")
            ax.set_xlabel("EqM step k")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "alignment_k_sweep.png"), dpi=150)
        plt.close()

        # Plot 3: blockwise cosine (action vs state) across gamma
        fig, ax = plt.subplots(figsize=(8, 5))
        cos_act = [all_results[f"gamma_{g:.2f}_lam0"]["cosine_action"]["mean"] for g in gamma_vals]
        cos_obs = [all_results[f"gamma_{g:.2f}_lam0"]["cosine_state"]["mean"] for g in gamma_vals]
        ax.plot(gamma_vals, cos_act, "o-", label="Action-only cosine", linewidth=2)
        ax.plot(gamma_vals, cos_obs, "s-", label="State-only cosine", linewidth=2)
        ax.set_xlabel("gamma (1=clean, 0=noise)")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Blockwise Cosine: Action vs State (gamma sweep)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "alignment_blockwise_cosine_gamma.png"), dpi=150)
        plt.close()

        print(f"  Plots saved to {outdir}")
    except ImportError:
        print("  [skip plots: matplotlib not available]")


if __name__ == "__main__":
    main()
