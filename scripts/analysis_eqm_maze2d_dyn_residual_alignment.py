#!/usr/bin/env python3
"""H1 MPC-like surrogate descent analysis for EqM on Maze2D.

1. Trains a 1-step forward dynamics model on GoalDataset (normalized space).
2. Defines J_dyn(x) = mean||s_{t+1} - f_dyn(s_t, a_t)||^2 + lam_u * mean||a_t||^2
3. Tests whether EqM vector field aligns with -grad_J_dyn and reduces J_dyn.

Usage:
  D4RL_SUPPRESS_IMPORT_ERROR=1 MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH \
    PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d \
    /root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8 \
    scripts/analysis_eqm_maze2d_dyn_residual_alignment.py \
      --checkpoint <path> [--dyn_train_steps 20000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add project root for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from ebm_online_rl.models.forward_dynamics import ForwardDynamics

from maze2d_eqm_utils import (
    ACT_DIM,
    OBS_DIM,
    TRANSITION_DIM,
    load_eqm_model_and_dataset,
)


def unpack_sa(x: torch.Tensor, obs_dim: int, act_dim: int):
    """Unpack [act | obs] trajectory into states and actions.

    Maze2D packing: dims [:act_dim] = actions, dims [act_dim:] = observations.

    Args:
        x: [B, H, D] packed trajectory

    Returns:
        s: [B, H, obs_dim] — states at all timesteps
        a: [B, H-1, act_dim] — actions at all but last timestep
    """
    a = x[:, :-1, :act_dim]                          # [B, H-1, act]
    s = x[:, :, act_dim:act_dim + obs_dim]            # [B, H, obs]
    return s, a


def J_dyn(
    x: torch.Tensor,
    f_dyn: ForwardDynamics,
    obs_dim: int,
    act_dim: int,
    lam_u: float = 1e-3,
) -> torch.Tensor:
    """Differentiable dynamics-compatibility surrogate objective.

    Args:
        x: [B, H, D] packed trajectory (requires_grad for gradient computation)
        f_dyn: trained forward dynamics model
        obs_dim, act_dim: dimension sizes
        lam_u: action regularization weight

    Returns:
        J: [B] scalar per batch element
    """
    s, a = unpack_sa(x, obs_dim, act_dim)
    s_pred = f_dyn(s[:, :-1, :], a)              # [B, H-1, obs]
    resid = s[:, 1:, :] - s_pred
    dyn = (resid ** 2).mean(dim=(-1, -2))         # [B]
    u = (a ** 2).mean(dim=(-1, -2))               # [B]
    return dyn + lam_u * u


def train_forward_dynamics(
    dataset,
    device: torch.device,
    obs_dim: int,
    act_dim: int,
    train_steps: int = 20000,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 256,
    depth: int = 2,
) -> ForwardDynamics:
    """Train a ForwardDynamics model from GoalDataset (normalized space)."""
    f_dyn = ForwardDynamics(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, depth=depth).to(device)
    optimizer = torch.optim.Adam(f_dyn.parameters(), lr=lr)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    print(f"[H1 dynamics] Training ForwardDynamics for {train_steps} steps...")
    losses = []
    for step in range(train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        traj = batch.trajectories.to(device)  # [B, H, D]
        s, a = unpack_sa(traj, obs_dim, act_dim)

        # Training target: predict s_{t+1} from (s_t, a_t)
        s_t = s[:, :-1, :]    # [B, H-1, obs]
        s_tp1 = s[:, 1:, :]   # [B, H-1, obs]

        s_pred = f_dyn(s_t, a)
        loss = ((s_pred - s_tp1) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if (step + 1) % 2000 == 0:
            recent_loss = np.mean(losses[-500:])
            print(f"  step {step+1}/{train_steps}, loss={recent_loss:.6f}")

    print(f"[H1 dynamics] Training done. Final loss={np.mean(losses[-500:]):.6f}")
    return f_dyn


def compute_alignment(
    eqm_model,
    f_dyn: ForwardDynamics,
    dataset,
    device: torch.device,
    obs_dim: int,
    act_dim: int,
    n_samples: int = 256,
    step_size: float = 0.1,
    lam_u: float = 1e-3,
) -> dict:
    """Compute gradient alignment between EqM vector field and -grad(J_dyn)."""
    denoiser = eqm_model.model  # raw TemporalUnet
    denoiser.eval()
    f_dyn.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    cosine_sims = []
    cosine_sims_state = []
    cosine_sims_action = []
    j_before_list = []
    j_after_list = []
    j_decreases = 0

    for i in range(n_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        traj = batch.trajectories.to(device)  # [1, H, D]

        # Compute grad_J = gradient of J_dyn w.r.t. x
        x = traj.clone().detach().requires_grad_(True)
        j_val = J_dyn(x, f_dyn, obs_dim, act_dim, lam_u)
        grad_J = torch.autograd.grad(j_val.sum(), x, create_graph=False)[0]  # [1, H, D]

        # Get EqM vector field: f = denoiser(x, {}, t0)
        x_detach = traj.clone().detach()
        t0 = torch.zeros((1,), dtype=torch.long, device=device)
        with torch.no_grad():
            f = denoiser(x_detach, {}, t0)  # [1, H, D]

        # EqM update: x_new = x - step_size * f
        # If f aligns with +grad_J, then x - step_size*f moves in -grad_J direction (descent)
        # So we want cosine_sim(f, grad_J) > 0 for descent

        # Flatten for cosine similarity
        f_flat = f.reshape(-1)
        g_flat = grad_J.reshape(-1).detach()

        cos_sim = torch.nn.functional.cosine_similarity(
            f_flat.unsqueeze(0), g_flat.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)

        # State-only and action-only cosine similarity
        # State dims: [:, :, act_dim:]  Action dims: [:, :, :act_dim]
        f_state = f[:, :, act_dim:].reshape(-1)
        g_state = grad_J[:, :, act_dim:].reshape(-1).detach()
        f_action = f[:, :-1, :act_dim].reshape(-1)
        g_action = grad_J[:, :-1, :act_dim].reshape(-1).detach()

        if f_state.norm() > 1e-10 and g_state.norm() > 1e-10:
            cos_state = torch.nn.functional.cosine_similarity(
                f_state.unsqueeze(0), g_state.unsqueeze(0)
            ).item()
            cosine_sims_state.append(cos_state)

        if f_action.norm() > 1e-10 and g_action.norm() > 1e-10:
            cos_action = torch.nn.functional.cosine_similarity(
                f_action.unsqueeze(0), g_action.unsqueeze(0)
            ).item()
            cosine_sims_action.append(cos_action)

        # Test descent: does J_dyn decrease after one EqM step?
        with torch.no_grad():
            j_before = J_dyn(x_detach, f_dyn, obs_dim, act_dim, lam_u).item()
            x_new = x_detach - step_size * f
            j_after = J_dyn(x_new, f_dyn, obs_dim, act_dim, lam_u).item()
        j_before_list.append(j_before)
        j_after_list.append(j_after)
        if j_after < j_before:
            j_decreases += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_samples}] mean_cos={np.mean(cosine_sims):.4f}, "
                  f"descent_frac={j_decreases/(i+1):.3f}")

    return {
        "n_samples": n_samples,
        "cosine_sim_mean": float(np.mean(cosine_sims)),
        "cosine_sim_median": float(np.median(cosine_sims)),
        "cosine_sim_std": float(np.std(cosine_sims)),
        "cosine_sim_state_mean": float(np.mean(cosine_sims_state)) if cosine_sims_state else None,
        "cosine_sim_action_mean": float(np.mean(cosine_sims_action)) if cosine_sims_action else None,
        "descent_fraction": j_decreases / max(1, n_samples),
        "j_before_mean": float(np.mean(j_before_list)),
        "j_after_mean": float(np.mean(j_after_list)),
        "j_decrease_mean": float(np.mean([b - a for b, a in zip(j_before_list, j_after_list)])),
        "step_size": step_size,
        "lam_u": lam_u,
    }


def plot_j_dyn_across_eqm_steps(
    eqm_model,
    f_dyn: ForwardDynamics,
    dataset,
    device: torch.device,
    obs_dim: int,
    act_dim: int,
    n_trajectories: int = 5,
    n_eqm_steps: int = 25,
    step_size: float = 0.1,
    lam_u: float = 1e-3,
    outdir: Path = Path("."),
):
    """Plot J_dyn value across EqM refinement steps for a few trajectories."""
    from diffuser.models.helpers import apply_conditioning

    denoiser = eqm_model.model
    denoiser.eval()
    f_dyn.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    fig, ax = plt.subplots(figsize=(10, 5))

    for traj_i in range(n_trajectories):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Get conditions from the batch
        cond = {k: v.to(device) for k, v in batch.conditions.items()}

        # Start from noise, apply conditioning
        H = batch.trajectories.shape[1]
        x = torch.randn((1, H, TRANSITION_DIM), device=device)
        x = apply_conditioning(x, cond, ACT_DIM)

        j_values = []
        t0 = torch.zeros((1,), dtype=torch.long, device=device)

        with torch.no_grad():
            for step_i in range(n_eqm_steps + 1):
                j_val = J_dyn(x, f_dyn, obs_dim, act_dim, lam_u).item()
                j_values.append(j_val)

                if step_i < n_eqm_steps:
                    grad = denoiser(x, cond, t0)
                    x = x - step_size * grad
                    if eqm_model.clip_denoised:
                        x = torch.clamp(x, -1.0, 1.0)
                    x = apply_conditioning(x, cond, ACT_DIM)

        ax.plot(range(n_eqm_steps + 1), j_values, marker=".", alpha=0.7, label=f"traj {traj_i}")

    ax.set_xlabel("EqM step")
    ax.set_ylabel("J_dyn")
    ax.set_title(f"H1: J_dyn across EqM refinement steps (n_traj={n_trajectories})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = outdir / "j_dyn_across_eqm_steps.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H1] Saved J_dyn plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="H1: MPC-like surrogate alignment analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dyn_ckpt", type=str, default=None, help="Pre-trained dynamics model (skip training)")
    parser.add_argument("--dyn_train_steps", type=int, default=20000)
    parser.add_argument("--dyn_hidden", type=int, default=256)
    parser.add_argument("--dyn_depth", type=int, default=2)
    parser.add_argument("--n_alignment_samples", type=int, default=256)
    parser.add_argument("--n_plot_trajectories", type=int, default=8)
    parser.add_argument("--lam_u", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[H1 alignment] Loading EqM checkpoint: {args.checkpoint}")
    eqm_model, dataset, cfg = load_eqm_model_and_dataset(args.checkpoint, device)
    horizon = int(cfg["horizon"])
    step_size = float(cfg.get("eqm_step_size", 0.1))
    print(f"[H1 alignment] Model loaded. horizon={horizon}")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path(f"runs/analysis/eqm_dyn_alignment_h1_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Train or load forward dynamics model
    if args.dyn_ckpt:
        print(f"[H1 alignment] Loading dynamics model: {args.dyn_ckpt}")
        f_dyn = ForwardDynamics(
            obs_dim=OBS_DIM, act_dim=ACT_DIM,
            hidden=args.dyn_hidden, depth=args.dyn_depth,
        ).to(device)
        f_dyn.load_state_dict(torch.load(args.dyn_ckpt, map_location=device, weights_only=True))
    else:
        f_dyn = train_forward_dynamics(
            dataset, device, OBS_DIM, ACT_DIM,
            train_steps=args.dyn_train_steps,
            hidden=args.dyn_hidden,
            depth=args.dyn_depth,
        )
        dyn_path = outdir / "forward_dynamics.pt"
        torch.save(f_dyn.state_dict(), dyn_path)
        print(f"[H1 alignment] Saved dynamics model: {dyn_path}")

    # Compute alignment
    print(f"\n[H1 alignment] Computing gradient alignment (n={args.n_alignment_samples})...")
    alignment = compute_alignment(
        eqm_model, f_dyn, dataset, device,
        OBS_DIM, ACT_DIM,
        n_samples=args.n_alignment_samples,
        step_size=step_size,
        lam_u=args.lam_u,
    )

    # Plot J_dyn across EqM steps
    print(f"\n[H1 alignment] Plotting J_dyn across EqM steps...")
    plot_j_dyn_across_eqm_steps(
        eqm_model, f_dyn, dataset, device,
        OBS_DIM, ACT_DIM,
        n_trajectories=args.n_plot_trajectories,
        n_eqm_steps=int(cfg.get("eqm_steps", 25)),
        step_size=step_size,
        lam_u=args.lam_u,
        outdir=outdir,
    )

    # Save results
    summary = {
        "hypothesis": "H1",
        "description": "MPC-like surrogate descent alignment",
        "checkpoint": str(args.checkpoint),
        "horizon": horizon,
        "eqm_step_size": step_size,
        "dyn_train_steps": args.dyn_train_steps,
        "alignment": alignment,
    }
    with open(outdir / "dyn_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== H1 Alignment Results ===")
    print(f"  Cosine sim (overall):     mean={alignment['cosine_sim_mean']:.4f} "
          f"median={alignment['cosine_sim_median']:.4f}")
    print(f"  Cosine sim (state-only):  {alignment['cosine_sim_state_mean']:.4f}" if alignment['cosine_sim_state_mean'] else "  State cosine: N/A")
    print(f"  Cosine sim (action-only): {alignment['cosine_sim_action_mean']:.4f}" if alignment['cosine_sim_action_mean'] else "  Action cosine: N/A")
    print(f"  Descent fraction:         {alignment['descent_fraction']:.3f}")
    print(f"  J_dyn decrease (mean):    {alignment['j_decrease_mean']:.6f}")
    print(f"\nSaved to: {outdir}")


if __name__ == "__main__":
    main()
