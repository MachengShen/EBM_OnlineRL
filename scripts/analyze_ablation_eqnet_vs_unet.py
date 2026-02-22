#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _discover_summary_files(base_dir: Path) -> List[Path]:
    files = sorted(base_dir.glob("*/seed_*/summary.json"))
    if files:
        return files
    return sorted(base_dir.glob("**/summary.json"))


def _arch_seed_from_path(path: Path) -> Tuple[str, int]:
    arch = path.parent.parent.name
    seed_tok = path.parent.name
    if seed_tok.startswith("seed_"):
        try:
            seed = int(seed_tok.split("_", 1)[1])
        except Exception:
            seed = -1
    else:
        seed = -1
    return arch, seed


def _max_prefix(summary: Dict) -> int:
    pref = summary.get("eval_success_prefix_horizons", [])
    if isinstance(pref, list) and pref:
        vals = [int(x) for x in pref]
        return max(vals)
    return 256


def _progress_metrics(summary: Dict, h: int) -> Dict[str, float]:
    p = summary.get("progress_last", {}) or {}
    success_key = f"rollout_goal_success_rate_h{h}"
    min_dist_key = f"rollout_min_goal_distance_mean_h{h}"
    final_dist_key = f"rollout_final_goal_error_mean_h{h}"
    return {
        "success": float(p.get(success_key, np.nan)),
        "min_goal_dist": float(p.get(min_dist_key, p.get("rollout_min_goal_distance_mean", np.nan))),
        "final_goal_dist": float(p.get(final_dist_key, p.get("rollout_final_goal_error_mean", np.nan))),
    }


def _load_progress_curve(run_dir: Path, arch: str, seed: int, h: int) -> pd.DataFrame:
    path = run_dir / "progress_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    success_col = f"rollout_goal_success_rate_h{h}"
    if success_col not in df.columns:
        cands = [c for c in df.columns if c.startswith("rollout_goal_success_rate_h")]
        if not cands:
            return pd.DataFrame()
        success_col = sorted(cands, key=lambda c: int(c.split("h", 1)[1]))[-1]
    step_col = "global_step" if "global_step" in df.columns else ("step" if "step" in df.columns else None)
    if step_col is None:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "arch": arch,
            "seed": seed,
            "global_step": pd.to_numeric(df[step_col], errors="coerce"),
            "success": pd.to_numeric(df[success_col], errors="coerce"),
        }
    ).dropna()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate EqNet vs UNet Maze2D ablation outputs.")
    ap.add_argument("--base-dir", type=Path, required=True)
    args = ap.parse_args()

    base_dir = args.base_dir.resolve()
    files = _discover_summary_files(base_dir)
    if not files:
        raise SystemExit(f"No summary.json files found under {base_dir}")

    rows: List[Dict] = []
    curve_rows: List[pd.DataFrame] = []
    for f in files:
        try:
            summary = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        arch, seed = _arch_seed_from_path(f)
        arch = str(summary.get("denoiser_arch", arch))
        h = _max_prefix(summary)
        pm = _progress_metrics(summary, h)
        row = {
            "arch": arch,
            "seed": int(seed),
            "run_dir": str(f.parent),
            "summary_path": str(f),
            "train_steps_total": float(summary.get("train_steps_total", np.nan)),
            "denoiser_trainable_params": float(summary.get("denoiser_trainable_params", np.nan)),
            "success_h": int(h),
            "success_final": pm["success"],
            "min_goal_dist_final": pm["min_goal_dist"],
            "final_goal_dist_final": pm["final_goal_dist"],
            "query_rollout_in_wall_points_mean": float(summary.get("query_rollout_in_wall_points_mean", np.nan)),
        }
        rows.append(row)

        curve_df = _load_progress_curve(f.parent, arch, int(seed), h)
        if not curve_df.empty:
            curve_rows.append(curve_df)

    rows_df = pd.DataFrame(rows).sort_values(["arch", "seed"]).reset_index(drop=True)
    rows_csv = base_dir / "eqnet_vs_unet_rows.csv"
    rows_df.to_csv(rows_csv, index=False)

    agg = (
        rows_df.groupby("arch", dropna=False)
        .agg(
            n=("success_final", "count"),
            success_mean=("success_final", "mean"),
            success_std=("success_final", "std"),
            min_goal_dist_mean=("min_goal_dist_final", "mean"),
            min_goal_dist_std=("min_goal_dist_final", "std"),
            final_goal_dist_mean=("final_goal_dist_final", "mean"),
            final_goal_dist_std=("final_goal_dist_final", "std"),
            wall_hits_mean=("query_rollout_in_wall_points_mean", "mean"),
            wall_hits_std=("query_rollout_in_wall_points_mean", "std"),
            denoiser_params_mean=("denoiser_trainable_params", "mean"),
        )
        .reset_index()
        .sort_values("arch")
    )

    summary_obj: Dict = {
        "base_dir": str(base_dir),
        "num_runs": int(len(rows_df)),
        "conditions": agg.to_dict(orient="records"),
    }
    summary_json = base_dir / "eqnet_vs_unet_summary.json"
    summary_json.write_text(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# EqNet vs UNet Maze2D Ablation Summary",
        "",
        f"- Base dir: `{base_dir}`",
        f"- Runs discovered: {len(rows_df)}",
        "",
        "| arch | n | success mean±std | min-goal-dist mean±std | final-goal-dist mean±std | wall-hits mean±std | params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        md_lines.append(
            "| {arch} | {n} | {sm:.4f} ± {ss:.4f} | {mm:.4f} ± {ms:.4f} | {fm:.4f} ± {fs:.4f} | {wm:.4f} ± {ws:.4f} | {pm:.0f} |".format(
                arch=r["arch"],
                n=int(r["n"]),
                sm=float(r["success_mean"]),
                ss=float(0.0 if pd.isna(r["success_std"]) else r["success_std"]),
                mm=float(r["min_goal_dist_mean"]),
                ms=float(0.0 if pd.isna(r["min_goal_dist_std"]) else r["min_goal_dist_std"]),
                fm=float(r["final_goal_dist_mean"]),
                fs=float(0.0 if pd.isna(r["final_goal_dist_std"]) else r["final_goal_dist_std"]),
                wm=float(r["wall_hits_mean"]),
                ws=float(0.0 if pd.isna(r["wall_hits_std"]) else r["wall_hits_std"]),
                pm=float(r["denoiser_params_mean"]),
            )
        )
    md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append(f"- Rows CSV: `{rows_csv}`")
    md_lines.append(f"- Summary JSON: `{summary_json}`")

    curve_png = base_dir / "eqnet_vs_unet_success_curve.png"
    if curve_rows:
        curves = pd.concat(curve_rows, ignore_index=True)
        curve_agg = (
            curves.groupby(["arch", "global_step"], dropna=False)["success"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["arch", "global_step"])
        )
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for arch in sorted(curve_agg["arch"].unique()):
            sub = curve_agg[curve_agg["arch"] == arch]
            x = sub["global_step"].to_numpy(dtype=np.float64)
            mean = sub["mean"].to_numpy(dtype=np.float64)
            std = sub["std"].fillna(0.0).to_numpy(dtype=np.float64)
            ax.plot(x, mean, linewidth=2.2, label=arch)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("global_step")
        ax.set_ylabel("rollout goal success")
        ax.set_title("EqNet vs UNet success curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(curve_png, dpi=180)
        plt.close(fig)
        md_lines.append(f"- Success curve: `{curve_png}`")

    summary_md = base_dir / "eqnet_vs_unet_summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[done] {rows_csv}")
    print(f"[done] {summary_json}")
    print(f"[done] {summary_md}")
    if curve_png.exists():
        print(f"[done] {curve_png}")


if __name__ == "__main__":
    main()
