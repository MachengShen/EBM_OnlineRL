#!/usr/bin/env python3
"""Diagnostics for original Diffuser Maze2D checkpoints.

Produces:
1) Boundary consistency metrics across checkpoints.
2) Denoised trajectory plots (state XY + state/action time series).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--diffuser_root",
        type=Path,
        default=Path("/root/ebm-online-rl-prototype/third_party/diffuser-maze2d"),
    )
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, default="maze2d-umaze-v1")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--steps", type=str, default="0,1000,2000,3000")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument(
        "--condition_mode",
        type=str,
        choices=["in_distribution", "plan_goal_zero_vel"],
        default="plan_goal_zero_vel",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, required=True)
    return p.parse_args()


def load_components(diffuser_root: Path):
    sys.path.insert(0, str(diffuser_root))
    import diffuser.datasets as datasets  # pylint: disable=import-error
    import diffuser.utils as utils  # pylint: disable=import-error
    from diffuser.guides.policies import Policy  # pylint: disable=import-error

    return datasets, utils, Policy


def parse_steps(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def set_env_state(env, obs: np.ndarray) -> None:
    qpos = obs[:2]
    qvel = obs[2:]
    if hasattr(env, "set_state"):
        env.set_state(qpos, qvel)
    else:
        env.unwrapped.set_state(qpos, qvel)


def compute_metrics_for_sample(
    env,
    obs_seq: np.ndarray,
    act_seq: np.ndarray,
    cond_start: np.ndarray,
    cond_goal: np.ndarray,
) -> Dict[str, float]:
    diffs_xy = np.linalg.norm(obs_seq[1:, :2] - obs_seq[:-1, :2], axis=1)
    mid_slice = diffs_xy[1:-1] if len(diffs_xy) > 2 else diffs_xy
    mid_jump = float(np.mean(mid_slice))
    start_jump = float(diffs_xy[0])
    end_jump = float(diffs_xy[-1])

    env_res = []
    for t in range(len(act_seq)):
        set_env_state(env, obs_seq[t])
        nxt, _, _, _ = env.step(act_seq[t])
        env_res.append(float(np.linalg.norm(nxt - obs_seq[t + 1])))
    env_res = np.asarray(env_res, dtype=np.float64)
    env_mid = float(np.mean(env_res[1:-1])) if len(env_res) > 2 else float(np.mean(env_res))

    return {
        "start_cond_err": float(np.linalg.norm(obs_seq[0] - cond_start)),
        "goal_cond_err": float(np.linalg.norm(obs_seq[-1] - cond_goal)),
        "start_jump_xy": start_jump,
        "mid_jump_xy": mid_jump,
        "end_jump_xy": end_jump,
        "start_jump_ratio": float(start_jump / (mid_jump + 1e-8)),
        "end_jump_ratio": float(end_jump / (mid_jump + 1e-8)),
        "env_res_start": float(env_res[0]),
        "env_res_mid": env_mid,
        "env_res_end": float(env_res[-1]),
        "env_res_start_ratio": float(env_res[0] / (env_mid + 1e-8)),
        "env_res_end_ratio": float(env_res[-1] / (env_mid + 1e-8)),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets, utils, Policy = load_components(args.diffuser_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = utils.load_config(str(args.run_dir), "dataset_config.pkl")
    model_config = utils.load_config(str(args.run_dir), "model_config.pkl")
    diffusion_config = utils.load_config(str(args.run_dir), "diffusion_config.pkl")

    dataset = dataset_config()
    model = model_config()
    diffusion = diffusion_config(model).to(torch.device(args.device)).eval()
    policy = Policy(diffusion, dataset.normalizer)
    env = datasets.load_environment(args.dataset)

    sample_index = max(0, min(args.sample_index, len(dataset.indices) - 1))
    path_ind, start, end = dataset.indices[sample_index]
    start_obs = dataset.fields.observations[path_ind, start].copy()
    goal_obs = dataset.fields.observations[path_ind, end - 1].copy()
    cond_start = start_obs.copy()
    cond_goal = goal_obs.copy()
    if args.condition_mode == "plan_goal_zero_vel":
        cond_goal[2:] = 0.0

    cond = {0: cond_start, dataset.horizon - 1: cond_goal}
    steps = parse_steps(args.steps)

    all_rows: List[Dict[str, float]] = []
    sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for step in steps:
        ckpt_path = args.run_dir / f"state_{step}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=args.device)
        key = "ema" if "ema" in ckpt else "model"
        diffusion.load_state_dict(ckpt[key])
        diffusion.eval()

        _, traj = policy(cond, batch_size=args.n_samples)
        acts = np.asarray(traj.actions)  # [B, H, act_dim]
        obs = np.asarray(traj.observations)  # [B, H, obs_dim]
        sample_cache[step] = (obs, acts)

        metric_rows = []
        for i in range(obs.shape[0]):
            row = compute_metrics_for_sample(
                env=env,
                obs_seq=obs[i],
                act_seq=acts[i, :-1] if acts.shape[1] == obs.shape[1] else acts[i],
                cond_start=cond_start,
                cond_goal=cond_goal,
            )
            metric_rows.append(row)

        row_mean: Dict[str, float] = {"step": float(step)}
        for k in metric_rows[0].keys():
            row_mean[k] = float(np.mean([r[k] for r in metric_rows]))
        all_rows.append(row_mean)

    all_rows.sort(key=lambda x: x["step"])
    if not all_rows:
        raise RuntimeError("No checkpoints evaluated.")

    csv_path = args.out_dir / f"maze2d_branch_boundary_metrics_{args.condition_mode}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = args.out_dir / f"maze2d_branch_boundary_metrics_{args.condition_mode}.json"
    summary_path.write_text(json.dumps(all_rows, indent=2))

    # XY denoised trajectories grid
    steps_present = [int(r["step"]) for r in all_rows]
    n = len(steps_present)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    for i, step in enumerate(steps_present):
        ax = axes[i // ncols][i % ncols]
        obs, _ = sample_cache[step]
        for j in range(obs.shape[0]):
            ax.plot(obs[j, :, 0], obs[j, :, 1], alpha=0.35, lw=1.5)
        ax.scatter([cond_start[0]], [cond_start[1]], c="green", s=60, label="start")
        ax.scatter([cond_goal[0]], [cond_goal[1]], c="red", s=60, label="goal")
        ax.set_title(f"step={step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"Original Maze2D Diffuser Denoised XY ({args.condition_mode})")
    fig.tight_layout()
    xy_plot = args.out_dir / f"maze2d_branch_denoised_xy_{args.condition_mode}.png"
    fig.savefig(xy_plot, dpi=180)
    plt.close(fig)

    # State/action series at final checkpoint, sample 0
    final_step = steps_present[-1]
    obs_final, acts_final = sample_cache[final_step]
    obs0 = obs_final[0]
    acts0 = acts_final[0]
    t_obs = np.arange(obs0.shape[0])
    t_act = np.arange(acts0.shape[0])

    fig2, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    axs[0].plot(obs0[:, 0], obs0[:, 1], lw=2)
    axs[0].scatter([cond_start[0]], [cond_start[1]], c="green", s=60)
    axs[0].scatter([cond_goal[0]], [cond_goal[1]], c="red", s=60)
    axs[0].set_title(f"Final ckpt step={final_step}: sample-0 XY")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(alpha=0.3)

    for d, name in enumerate(["x", "y", "vx", "vy"]):
        axs[1].plot(t_obs, obs0[:, d], label=name)
    axs[1].set_title("Denoised state sequence")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("state")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    for d in range(acts0.shape[1]):
        axs[2].plot(t_act, acts0[:, d], label=f"a{d}")
    axs[2].set_title("Denoised action sequence")
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("action")
    axs[2].grid(alpha=0.3)
    axs[2].legend()
    fig2.tight_layout()
    series_plot = args.out_dir / f"maze2d_branch_denoised_state_action_{args.condition_mode}.png"
    fig2.savefig(series_plot, dpi=180)
    plt.close(fig2)

    # Boundary metrics plot
    steps_f = [r["step"] for r in all_rows]
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4.5))
    ax3[0].plot(steps_f, [r["start_jump_ratio"] for r in all_rows], marker="o", label="start/mid jump")
    ax3[0].plot(steps_f, [r["end_jump_ratio"] for r in all_rows], marker="o", label="end/mid jump")
    ax3[0].axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
    ax3[0].set_title("Boundary jump ratio")
    ax3[0].set_xlabel("checkpoint")
    ax3[0].set_ylabel("ratio")
    ax3[0].grid(alpha=0.3)
    ax3[0].legend()

    ax3[1].plot(steps_f, [r["env_res_start_ratio"] for r in all_rows], marker="o", label="start/mid env-res")
    ax3[1].plot(steps_f, [r["env_res_end_ratio"] for r in all_rows], marker="o", label="end/mid env-res")
    ax3[1].axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
    ax3[1].set_title("Boundary env residual ratio")
    ax3[1].set_xlabel("checkpoint")
    ax3[1].set_ylabel("ratio")
    ax3[1].grid(alpha=0.3)
    ax3[1].legend()
    fig3.tight_layout()
    metric_plot = args.out_dir / f"maze2d_branch_boundary_ratios_{args.condition_mode}.png"
    fig3.savefig(metric_plot, dpi=180)
    plt.close(fig3)

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {xy_plot}")
    print(f"wrote {series_plot}")
    print(f"wrote {metric_plot}")


if __name__ == "__main__":
    main()
