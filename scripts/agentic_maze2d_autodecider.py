#!/usr/bin/env python3
from __future__ import annotations

"""
Conservative near-open-ended autonomous controller for Maze2D online Diffuser.

This script is intended to replace the fixed-space selector in
`overnight_maze2d_autodecider.py` when desired.

Design:
- Hypothesis/proposal agent: generates broader change proposals
  (trial config + selected common-arg overrides).
- Constraint checker: enforces "still EBM online-RL" invariants.
- Runner/monitor: reuses the existing run utility for consistency.
- Critic: tracks simple parameter-level effect beliefs from outcomes.
- Promotion policy: conservative two-stage confirmation before incumbent update.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

import overnight_maze2d_autodecider as base


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe(v: Any) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _isfinite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _log(path: Path, msg: str) -> None:
    line = f"[agentic] {_now()} {msg}\n"
    print(line, end="")
    _append(path, line)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    rationale: str
    trial: base.TrialConfig
    common_overrides: Mapping[str, Any]
    origin: str
    risk_level: str  # low, med

    def short_name(self) -> str:
        return f"{self.proposal_id}_{self.trial.short_name()}"


def _neighbor(values: Sequence[Any], cur: Any, direction: int) -> Any:
    vals = list(values)
    if len(vals) == 0:
        return cur
    try:
        idx = vals.index(cur)
    except ValueError:
        idx = 0
    j = max(0, min(len(vals) - 1, idx + int(direction)))
    return vals[j]


def _default_common_args(smoke: bool) -> Dict[str, Any]:
    # Matches current auto-decider defaults to keep apples-to-apples baselines.
    common = {
        "device": "cuda:0",
        "n_episodes": 1000,
        "episode_len": 256,
        "max_path_length": 256,
        "horizon": 64,
        "n_diffusion_steps": 64,
        "model_dim": 64,
        "model_dim_mults": "1,2,4",
        "learning_rate": 2e-4,
        "batch_size": 128,
        "val_every": 500,
        "val_batches": 20,
        "eval_goal_every": 1000,
        "save_checkpoint_every": 5000,
        "query_mode": "diverse",
        "num_eval_queries": 12,
        "query_batch_size": 1,
        "query_resample_each_eval": True,
        "query_min_distance": 1.0,
        "goal_success_threshold": 0.2,
        "online_self_improve": True,
        "online_collect_episode_len": 256,
        "online_collect_transition_budget_per_round": 4096,
        "online_goal_geom_min_k": 64,
        "online_goal_geom_max_k": 192,
        "online_goal_min_distance": 1.0,
        "online_early_terminate_on_success": True,
        "online_early_terminate_threshold": 0.2,
        "online_min_accepted_episode_len": 64,
        "eval_rollout_replan_every_n_steps": 16,
        "wall_aware_plan_samples": 2,
        "eval_rollout_horizon": 256,
        "eval_success_prefix_horizons": "64,128,192,256",
        "online_planning_success_thresholds": "0.1,0.2",
        "online_planning_success_rel_reduction": 0.9,
    }
    if smoke:
        common.update(
            {
                "device": "cpu",
                "n_episodes": 4,
                "episode_len": 32,
                "max_path_length": 32,
                "horizon": 16,
                "n_diffusion_steps": 16,
                "model_dim": 32,
                "model_dim_mults": "1,2",
                "batch_size": 16,
                "val_every": 10,
                "val_batches": 1,
                "eval_goal_every": 10,
                "save_checkpoint_every": 0,
                "num_eval_queries": 4,
                "query_bank_size": 32,
                "query_batch_size": 1,
                "eval_rollout_horizon": 32,
                "eval_success_prefix_horizons": "8,16,32",
                "online_collect_episode_len": 32,
                "online_collect_transition_budget_per_round": 32,
                "online_goal_geom_min_k": 4,
                "online_goal_geom_max_k": 16,
                "online_goal_min_distance": 0.3,
                "online_early_terminate_threshold": 0.1,
                "online_min_accepted_episode_len": 8,
            }
        )
    return common


def _default_space(smoke: bool) -> base.SearchSpace:
    if smoke:
        return base.SearchSpace(
            train_steps=[20, 40],
            online_rounds=[1, 2],
            online_collect_episodes_per_round=[1, 2],
            online_train_steps_per_round=[10, 20],
            online_replan_every_n_steps=[4, 8],
            online_goal_geom_p=[0.08],
        )
    return base.default_search_space()


def _initial_trial(space: base.SearchSpace, smoke: bool) -> base.TrialConfig:
    if smoke:
        # Keep smoke runs short so proposal/confirmation logic can be validated quickly.
        ts = min(space.train_steps)
        or_ = min(space.online_rounds)
        ep = min(space.online_collect_episodes_per_round)
        tr = min(space.online_train_steps_per_round)
        rp = min(space.online_replan_every_n_steps)
        gp = min(space.online_goal_geom_p)
    else:
        # Conservative incumbent from the currently-strong setting seen in logs.
        ts = min(space.train_steps, key=lambda v: abs(v - 6000))
        or_ = min(space.online_rounds, key=lambda v: abs(v - 4))
        ep = min(space.online_collect_episodes_per_round, key=lambda v: abs(v - 64))
        tr = min(space.online_train_steps_per_round, key=lambda v: abs(v - 3000))
        rp = min(space.online_replan_every_n_steps, key=lambda v: abs(v - 16))
        gp = min(space.online_goal_geom_p, key=lambda v: abs(v - 0.04))
    return base.TrialConfig(
        train_steps=int(ts),
        online_rounds=int(or_),
        online_collect_episodes_per_round=int(ep),
        online_train_steps_per_round=int(tr),
        online_replan_every_n_steps=int(rp),
        online_goal_geom_p=float(gp),
    )


def _objective(row: Mapping[str, Any]) -> float:
    return _safe(row.get("objective_succ_h256", float("nan")))


def _proposal_key(p: Proposal) -> Tuple[Any, ...]:
    return (
        tuple(sorted(p.trial.to_args().items())),
        tuple(sorted((str(k), str(v)) for k, v in p.common_overrides.items())),
    )


def _check_constraints(
    proposal: Proposal,
    base_common: Mapping[str, Any],
) -> Tuple[bool, str]:
    merged = dict(base_common)
    merged.update(dict(proposal.common_overrides))

    # Core identity constraints: still the same algorithm family and eval protocol.
    if not bool(merged.get("online_self_improve", False)):
        return False, "online_self_improve must stay enabled"
    if int(merged.get("horizon", 0)) <= 0:
        return False, "horizon must be > 0"
    if int(merged.get("eval_rollout_horizon", 0)) < int(merged.get("horizon", 1)):
        return False, "eval_rollout_horizon must be >= horizon"
    if int(merged.get("online_collect_episode_len", 0)) < int(merged.get("horizon", 1)):
        return False, "online_collect_episode_len must be >= horizon"
    if float(merged.get("goal_success_threshold", 1.0)) <= 0.0 or float(merged.get("goal_success_threshold", 1.0)) > 1.0:
        return False, "goal_success_threshold must be in (0,1]"
    if int(merged.get("num_eval_queries", 0)) <= 0:
        return False, "num_eval_queries must be > 0"
    if int(merged.get("query_batch_size", 0)) <= 0:
        return False, "query_batch_size must be > 0"

    # Prefix horizon consistency.
    prefix_raw = str(merged.get("eval_success_prefix_horizons", ""))
    pref = [int(x.strip()) for x in prefix_raw.split(",") if x.strip()]
    if len(pref) == 0:
        return False, "eval_success_prefix_horizons cannot be empty"
    if max(pref) > int(merged.get("eval_rollout_horizon", 0)):
        return False, "max prefix horizon cannot exceed eval_rollout_horizon"
    if sorted(pref) != pref:
        return False, "prefix horizons must be sorted"

    # Keep proposals conservative by limiting risky eval/collection changes.
    if int(merged.get("eval_rollout_replan_every_n_steps", 0)) <= 0:
        return False, "eval_rollout_replan_every_n_steps must be > 0"
    if int(merged.get("wall_aware_plan_samples", 0)) <= 0:
        return False, "wall_aware_plan_samples must be > 0"
    return True, "ok"


def _coerce_trial_config(raw: Mapping[str, Any]) -> base.TrialConfig:
    required = (
        "train_steps",
        "online_rounds",
        "online_collect_episodes_per_round",
        "online_train_steps_per_round",
        "online_replan_every_n_steps",
        "online_goal_geom_p",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"missing trial keys: {missing}")
    return base.TrialConfig(
        train_steps=int(float(raw["train_steps"])),
        online_rounds=int(float(raw["online_rounds"])),
        online_collect_episodes_per_round=int(float(raw["online_collect_episodes_per_round"])),
        online_train_steps_per_round=int(float(raw["online_train_steps_per_round"])),
        online_replan_every_n_steps=int(float(raw["online_replan_every_n_steps"])),
        online_goal_geom_p=float(raw["online_goal_geom_p"]),
    )


def _coerce_external_proposal(raw: Mapping[str, Any], idx: int) -> Proposal:
    trial_raw = raw.get("trial")
    if not isinstance(trial_raw, Mapping):
        raise ValueError("proposal.trial must be a mapping")
    trial = _coerce_trial_config(trial_raw)

    common_overrides = raw.get("common_overrides", {})
    if common_overrides is None:
        common_overrides = {}
    if not isinstance(common_overrides, Mapping):
        raise ValueError("proposal.common_overrides must be a mapping")

    proposal_id = str(raw.get("proposal_id") or f"ext{idx}")
    rationale = str(raw.get("rationale") or "external proposal")
    origin = str(raw.get("origin") or "external")
    risk_level = str(raw.get("risk_level") or raw.get("risk") or "med").lower()
    if risk_level not in {"low", "med"}:
        risk_level = "med"

    return Proposal(
        proposal_id=proposal_id,
        rationale=rationale,
        trial=trial,
        common_overrides=dict(common_overrides),
        origin=origin,
        risk_level=risk_level,
    )


def _load_external_proposals(path: Path, max_new: int) -> List[Proposal]:
    raw_obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_obj, Mapping):
        entries = raw_obj.get("proposals", [])
    else:
        entries = raw_obj
    if not isinstance(entries, list):
        raise ValueError("external proposal file must be a list or {\"proposals\": [...]}")
    out: List[Proposal] = []
    for i, entry in enumerate(entries):
        if len(out) >= max_new:
            break
        if not isinstance(entry, Mapping):
            raise ValueError(f"proposal index {i} is not a mapping")
        out.append(_coerce_external_proposal(entry, i))
    return out


def _summarize_recent_rows(rows: Sequence[Mapping[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    keep = (
        "run_name",
        "role",
        "proposal_id",
        "proposal_origin",
        "objective_succ_h256",
        "progress_succ_h256_last",
        "progress_succ_h256_max",
        "progress_goal_cov_query_h256_last",
        "progress_goal_cov_query_h256_max",
        "online_planning_success_rate_final_t020",
        "online_planning_success_rate_final_rel090",
        "online_replay_transitions_last",
        "rc",
    )
    out: List[Dict[str, Any]] = []
    for row in rows[-max(0, int(limit)) :]:
        d: Dict[str, Any] = {}
        for k in keep:
            if k in row:
                d[k] = row[k]
        out.append(d)
    return out


def _serialize_tried_keys(tried: Iterable[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for trial_items, common_items in sorted(tried, key=lambda x: str(x)):
        out.append(
            {
                "trial": {k: v for k, v in trial_items},
                "common_overrides": {k: v for k, v in common_items},
            }
        )
    return out


def _build_round_context(
    *,
    round_idx: int,
    incumbent_trial: base.TrialConfig,
    incumbent_common: Mapping[str, Any],
    incumbent_score: float,
    all_rows: Sequence[Mapping[str, Any]],
    tried: Iterable[Tuple[Any, ...]],
    max_new: int,
    accept_delta: float,
    require_confirmation: bool,
    space: base.SearchSpace,
    base_common: Mapping[str, Any],
    proposal_file: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": "agentic_external_round_context_v1",
        "created_at": _now(),
        "round": int(round_idx),
        "proposal_requirements": {
            "proposal_file": str(proposal_file),
            "max_new": int(max_new),
            "accept_delta": float(accept_delta),
            "require_confirmation": bool(require_confirmation),
            "notes": "Provide up to max_new proposals. Proposals violating constraints may be rejected.",
        },
        "incumbent": {
            "trial": incumbent_trial.to_args(),
            "common_overrides": dict(incumbent_common),
            "score": float(incumbent_score) if _isfinite(incumbent_score) else None,
        },
        "recent_results_tail": _summarize_recent_rows(all_rows),
        "tried_configs": _serialize_tried_keys(tried),
        "space": asdict(space),
        "base_common_args": dict(base_common),
        "proposal_template": {
            "proposals": [
                {
                    "proposal_id": "p0",
                    "rationale": "short reason",
                    "origin": "external_codex",
                    "risk_level": "low",
                    "trial": incumbent_trial.to_args(),
                    "common_overrides": {},
                }
            ]
        },
    }


def _wait_for_external_proposals(
    *,
    round_idx: int,
    proposal_dir: Path,
    max_new: int,
    wait_timeout_sec: int,
    poll_sec: float,
    context_payload: Mapping[str, Any],
    log_path: Path,
) -> Tuple[Optional[List[Proposal]], str, Path]:
    proposal_dir.mkdir(parents=True, exist_ok=True)
    context_path = proposal_dir / f"round_{round_idx:03d}_context.json"
    proposals_path = proposal_dir / f"round_{round_idx:03d}_proposals.json"
    _write_json(context_path, context_payload)
    _log(
        log_path,
        (
            f"await_external_proposals round={round_idx} context={context_path} "
            f"proposals={proposals_path} timeout_sec={wait_timeout_sec}"
        ),
    )

    started = time.time()
    last_error = ""
    sleep_s = max(0.5, float(poll_sec))
    while True:
        if proposals_path.exists():
            try:
                proposals = _load_external_proposals(proposals_path, max_new=max_new)
            except Exception as e:
                msg = str(e)
                if msg != last_error:
                    _log(log_path, f"external_proposals_parse_error round={round_idx} err={msg}")
                    last_error = msg
            else:
                _log(log_path, f"loaded_external_proposals round={round_idx} count={len(proposals)}")
                return proposals, "ok", proposals_path
        if int(wait_timeout_sec) > 0 and (time.time() - started) >= float(wait_timeout_sec):
            _log(log_path, f"external_proposals_timeout round={round_idx}")
            return None, "timeout", proposals_path
        time.sleep(sleep_s)


def _reasoned_proposals(
    *,
    rng: random.Random,
    space: base.SearchSpace,
    incumbent_trial: base.TrialConfig,
    incumbent_common: Mapping[str, Any],
    incumbent_score: float,
    recent_rows: Sequence[Mapping[str, Any]],
    max_new: int,
) -> List[Proposal]:
    out: List[Proposal] = []

    # Read recent signals for lightweight "mental model" heuristics.
    recent = recent_rows[-1] if len(recent_rows) > 0 else {}
    succ_last = _safe(recent.get("progress_succ_h256_last", incumbent_score))
    cov_last = _safe(recent.get("progress_goal_cov_query_h256_last", float("nan")))

    # H1: If success trails coverage, increase planning feedback frequency.
    if _isfinite(succ_last) and _isfinite(cov_last) and cov_last - succ_last > 0.10:
        trial = base.TrialConfig(
            train_steps=incumbent_trial.train_steps,
            online_rounds=incumbent_trial.online_rounds,
            online_collect_episodes_per_round=incumbent_trial.online_collect_episodes_per_round,
            online_train_steps_per_round=incumbent_trial.online_train_steps_per_round,
            online_replan_every_n_steps=int(_neighbor(space.online_replan_every_n_steps, incumbent_trial.online_replan_every_n_steps, -1)),
            online_goal_geom_p=incumbent_trial.online_goal_geom_p,
        )
        out.append(
            Proposal(
                proposal_id=f"p{len(out)}",
                rationale="coverage > success; tighten replanning cadence",
                trial=trial,
                common_overrides={"eval_rollout_replan_every_n_steps": max(1, int(incumbent_common.get("eval_rollout_replan_every_n_steps", 16) // 2))},
                origin="heuristic",
                risk_level="low",
            )
        )

    # H2: If near saturation, promote budget.
    if _isfinite(incumbent_score) and incumbent_score >= 0.70:
        levels = space.budget_levels()
        try:
            idx = levels.index((incumbent_trial.train_steps, incumbent_trial.online_rounds))
        except ValueError:
            idx = -1
        if 0 <= idx < len(levels) - 1:
            ts2, or2 = levels[idx + 1]
            out.append(
                Proposal(
                    proposal_id=f"p{len(out)}",
                    rationale="high score; increase budget for robustness",
                    trial=base.TrialConfig(
                        train_steps=int(ts2),
                        online_rounds=int(or2),
                        online_collect_episodes_per_round=incumbent_trial.online_collect_episodes_per_round,
                        online_train_steps_per_round=incumbent_trial.online_train_steps_per_round,
                        online_replan_every_n_steps=incumbent_trial.online_replan_every_n_steps,
                        online_goal_geom_p=incumbent_trial.online_goal_geom_p,
                    ),
                    common_overrides={},
                    origin="heuristic",
                    risk_level="low",
                )
            )

    # H3: Throughput/credit assignment tradeoff.
    out.append(
        Proposal(
            proposal_id=f"p{len(out)}",
            rationale="test stronger updates with modest data chunk",
            trial=base.TrialConfig(
                train_steps=incumbent_trial.train_steps,
                online_rounds=incumbent_trial.online_rounds,
                online_collect_episodes_per_round=int(_neighbor(space.online_collect_episodes_per_round, incumbent_trial.online_collect_episodes_per_round, -1)),
                online_train_steps_per_round=int(_neighbor(space.online_train_steps_per_round, incumbent_trial.online_train_steps_per_round, +1)),
                online_replan_every_n_steps=incumbent_trial.online_replan_every_n_steps,
                online_goal_geom_p=incumbent_trial.online_goal_geom_p,
            ),
            common_overrides={},
            origin="heuristic",
            risk_level="low",
        )
    )

    # Add a few randomized near-neighbor proposals for near-open-ended exploration.
    while len(out) < max_new:
        d = rng.choice([-1, +1])
        dd = rng.choice([-1, +1])
        trial = base.TrialConfig(
            train_steps=int(rng.choice(space.train_steps)),
            online_rounds=int(rng.choice(space.online_rounds)),
            online_collect_episodes_per_round=int(_neighbor(space.online_collect_episodes_per_round, incumbent_trial.online_collect_episodes_per_round, d)),
            online_train_steps_per_round=int(_neighbor(space.online_train_steps_per_round, incumbent_trial.online_train_steps_per_round, dd)),
            online_replan_every_n_steps=int(_neighbor(space.online_replan_every_n_steps, incumbent_trial.online_replan_every_n_steps, rng.choice([-1, +1]))),
            online_goal_geom_p=float(_neighbor(space.online_goal_geom_p, incumbent_trial.online_goal_geom_p, rng.choice([-1, +1]))),
        )
        common_overrides = {}
        if rng.random() < 0.5:
            common_overrides["eval_goal_every"] = int(rng.choice([500, 1000, 1500]))
        if rng.random() < 0.4:
            common_overrides["wall_aware_plan_samples"] = int(rng.choice([1, 2, 4]))
        out.append(
            Proposal(
                proposal_id=f"p{len(out)}",
                rationale="near-neighbor exploratory mutation",
                trial=trial,
                common_overrides=common_overrides,
                origin="stochastic",
                risk_level="med" if len(common_overrides) else "low",
            )
        )
    return out[:max_new]


def _update_beliefs(
    beliefs: Dict[str, Dict[str, float]],
    proposal: Proposal,
    score_delta: float,
) -> None:
    params = dict(proposal.trial.to_args())
    params.update({f"common.{k}": v for k, v in proposal.common_overrides.items()})
    for k, v in params.items():
        key = f"{k}={v}"
        ent = beliefs.setdefault(key, {"n": 0.0, "sum_delta": 0.0, "mean_delta": 0.0})
        ent["n"] += 1.0
        ent["sum_delta"] += float(score_delta)
        ent["mean_delta"] = ent["sum_delta"] / max(1.0, ent["n"])


def _env_for_runs(root: Path) -> Dict[str, str]:
    ld_mj = "/root/.mujoco/mujoco210/bin"
    env = dict(os.environ)
    env.update(
        {
            "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", "") + ":" + ld_mj,
            "MUJOCO_GL": "egl",
            "D4RL_SUPPRESS_IMPORT_ERROR": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(root / "third_party/diffuser-maze2d"),
        }
    )
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/root/ebm-online-rl-prototype"))
    ap.add_argument("--python", type=Path, default=Path("/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python"))
    ap.add_argument("--main-script", type=Path, default=Path("/root/ebm-online-rl-prototype/scripts/synthetic_maze2d_diffuser_probe.py"))
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory (default: runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_<timestamp>).",
    )
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--monitor-every-sec", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-trial-timeout-min", type=float, default=35.0)
    ap.add_argument("--max-agent-rounds", type=int, default=8)
    ap.add_argument("--proposals-per-round", type=int, default=4)
    ap.add_argument(
        "--proposal-source",
        choices=["internal", "external", "external_preferred"],
        default="internal",
        help="Where proposals come from: internal heuristics, strict external files, or external with internal fallback.",
    )
    ap.add_argument(
        "--proposal-dir",
        type=Path,
        default=None,
        help="Directory for external proposal handoff files (round_*_context/proposals.json).",
    )
    ap.add_argument("--proposal-wait-timeout-sec", type=int, default=1800)
    ap.add_argument("--proposal-poll-sec", type=float, default=10.0)
    ap.add_argument("--accept-delta", type=float, default=0.02, help="Minimum objective gain required vs incumbent.")
    ap.add_argument("--require-confirmation", action="store_true", default=True)
    ap.add_argument("--confirm-seed-offset", type=int, default=9973)
    args = ap.parse_args()

    root = args.root
    py = args.python
    main_py = args.main_script
    if not root.exists():
        raise SystemExit(f"missing root: {root}")
    if not py.exists():
        raise SystemExit(f"missing python: {py}")
    if not main_py.exists():
        raise SystemExit(f"missing main script: {main_py}")

    rng = random.Random(int(args.seed))
    space = _default_space(smoke=bool(args.smoke))
    base_common = _default_common_args(smoke=bool(args.smoke))
    env = _env_for_runs(root)

    base_dir = args.base_dir if args.base_dir else (root / "runs/analysis/synth_maze2d_diffuser_probe" / f"agentic_autodecider_{_stamp()}")
    if base_dir.exists() and any(base_dir.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty base_dir: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir = args.proposal_dir if args.proposal_dir else (base_dir / "proposal_exchange")
    proposal_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "agentic.log"
    results_path = base_dir / "agentic_results.csv"
    proposals_path = base_dir / "proposals.jsonl"
    beliefs_path = base_dir / "beliefs.json"
    _write_json(beliefs_path, {})

    _write_json(
        base_dir / "agentic_config.json",
        {
            "created_at": _now(),
            "base_dir": str(base_dir),
            "root": str(root),
            "python": str(py),
            "main_script": str(main_py),
            "budget_hours": float(args.budget_hours),
            "max_agent_rounds": int(args.max_agent_rounds),
            "proposals_per_round": int(args.proposals_per_round),
            "proposal_source": str(args.proposal_source),
            "proposal_dir": str(proposal_dir),
            "proposal_wait_timeout_sec": int(args.proposal_wait_timeout_sec),
            "proposal_poll_sec": float(args.proposal_poll_sec),
            "accept_delta": float(args.accept_delta),
            "require_confirmation": bool(args.require_confirmation),
            "space": asdict(space),
            "base_common_args": base_common,
        },
    )

    _log(log_path, f"start base_dir={base_dir}")
    deadline = datetime.now() + timedelta(hours=float(args.budget_hours))

    incumbent_trial = _initial_trial(space, smoke=bool(args.smoke))
    incumbent_common: Dict[str, Any] = {}
    incumbent_score = float("nan")
    beliefs: Dict[str, Dict[str, float]] = {}
    all_rows: List[Dict[str, Any]] = []
    tried: set[Tuple[Any, ...]] = set()

    # Bootstrap incumbent from one trial.
    _log(log_path, f"bootstrap incumbent={incumbent_trial.short_name()}")
    row0 = base._run_one_trial(
        py=py,
        main_py=main_py,
        root=root,
        base_dir=base_dir,
        driver_log=log_path,
        common_args=base_common,
        trial=incumbent_trial,
        monitor_every_sec=int(args.monitor_every_sec),
        env=env,
        max_wallclock_sec=int(float(args.per_trial_timeout_min) * 60.0) if args.per_trial_timeout_min else None,
    )
    row0["role"] = "bootstrap"
    row0["proposal_id"] = "bootstrap"
    all_rows.append(row0)
    incumbent_score = _objective(row0)
    tried.add((tuple(sorted(incumbent_trial.to_args().items())), tuple()))
    pd.DataFrame(all_rows).to_csv(results_path, index=False)

    for round_idx in range(int(args.max_agent_rounds)):
        if datetime.now() >= deadline:
            _log(log_path, f"budget_exhausted round={round_idx}")
            break

        max_new = max(1, int(args.proposals_per_round))
        proposal_source = str(args.proposal_source)
        proposals: List[Proposal] = []
        if proposal_source == "internal":
            proposals = _reasoned_proposals(
                rng=rng,
                space=space,
                incumbent_trial=incumbent_trial,
                incumbent_common=dict(base_common, **incumbent_common),
                incumbent_score=incumbent_score,
                recent_rows=all_rows,
                max_new=max_new,
            )
        else:
            proposal_file = proposal_dir / f"round_{round_idx:03d}_proposals.json"
            context = _build_round_context(
                round_idx=round_idx,
                incumbent_trial=incumbent_trial,
                incumbent_common=incumbent_common,
                incumbent_score=incumbent_score,
                all_rows=all_rows,
                tried=tried,
                max_new=max_new,
                accept_delta=float(args.accept_delta),
                require_confirmation=bool(args.require_confirmation),
                space=space,
                base_common=base_common,
                proposal_file=proposal_file,
            )
            loaded, status, used_path = _wait_for_external_proposals(
                round_idx=round_idx,
                proposal_dir=proposal_dir,
                max_new=max_new,
                wait_timeout_sec=int(args.proposal_wait_timeout_sec),
                poll_sec=float(args.proposal_poll_sec),
                context_payload=context,
                log_path=log_path,
            )
            if loaded is None:
                if proposal_source == "external":
                    _log(log_path, f"round={round_idx} strict_external_no_proposals status={status} stop")
                    break
                _log(log_path, f"round={round_idx} external_missing status={status} fallback=internal")
                proposals = _reasoned_proposals(
                    rng=rng,
                    space=space,
                    incumbent_trial=incumbent_trial,
                    incumbent_common=dict(base_common, **incumbent_common),
                    incumbent_score=incumbent_score,
                    recent_rows=all_rows,
                    max_new=max_new,
                )
            elif len(loaded) == 0:
                if proposal_source == "external":
                    _log(log_path, f"round={round_idx} strict_external_empty_proposals file={used_path} stop")
                    break
                _log(log_path, f"round={round_idx} external_empty_proposals file={used_path} fallback=internal")
                proposals = _reasoned_proposals(
                    rng=rng,
                    space=space,
                    incumbent_trial=incumbent_trial,
                    incumbent_common=dict(base_common, **incumbent_common),
                    incumbent_score=incumbent_score,
                    recent_rows=all_rows,
                    max_new=max_new,
                )
            else:
                proposals = loaded
        _log(
            log_path,
            (
                f"round={round_idx} proposals={len(proposals)} source={proposal_source} "
                f"incumbent={incumbent_trial.short_name()} score={incumbent_score:.4f}"
            ),
        )

        accepted_this_round = False
        for p in proposals:
            ok, reason = _check_constraints(p, dict(base_common, **incumbent_common))
            p_payload = {
                "ts": _now(),
                "round": int(round_idx),
                "proposal_id": p.proposal_id,
                "short_name": p.short_name(),
                "origin": p.origin,
                "risk": p.risk_level,
                "rationale": p.rationale,
                "trial": p.trial.to_args(),
                "common_overrides": dict(p.common_overrides),
                "constraint_ok": bool(ok),
                "constraint_reason": reason,
            }
            _append(proposals_path, json.dumps(p_payload, sort_keys=True) + "\n")
            if not ok:
                _log(log_path, f"reject proposal={p.short_name()} reason={reason}")
                continue
            key = _proposal_key(p)
            if key in tried:
                _log(log_path, f"skip proposal={p.short_name()} reason=duplicate")
                continue
            tried.add(key)

            merged_common = dict(base_common, **incumbent_common, **dict(p.common_overrides))
            _log(log_path, f"pilot proposal={p.short_name()} rationale={p.rationale}")
            pilot = base._run_one_trial(
                py=py,
                main_py=main_py,
                root=root,
                base_dir=base_dir,
                driver_log=log_path,
                common_args=merged_common,
                trial=p.trial,
                monitor_every_sec=int(args.monitor_every_sec),
                env=env,
                max_wallclock_sec=int(float(args.per_trial_timeout_min) * 60.0) if args.per_trial_timeout_min else None,
            )
            pilot["role"] = "pilot"
            pilot["proposal_id"] = p.proposal_id
            pilot["proposal_rationale"] = p.rationale
            pilot["proposal_origin"] = p.origin
            all_rows.append(pilot)
            pd.DataFrame(all_rows).to_csv(results_path, index=False)

            pilot_obj = _objective(pilot)
            delta = pilot_obj - incumbent_score if _isfinite(incumbent_score) and _isfinite(pilot_obj) else float("-inf")
            _update_beliefs(beliefs, p, delta if _isfinite(delta) else -1.0)
            _write_json(beliefs_path, beliefs)

            if not _isfinite(pilot_obj) or not _isfinite(incumbent_score):
                _log(log_path, f"reject pilot proposal={p.short_name()} reason=non_finite objective")
                continue
            if delta < float(args.accept_delta):
                _log(log_path, f"reject pilot proposal={p.short_name()} delta={delta:.4f} < accept_delta={args.accept_delta:.4f}")
                continue

            # Conservative confirmation: second run with different seed.
            if bool(args.require_confirmation):
                confirm_common = dict(merged_common)
                confirm_seed = int(args.seed) + int(args.confirm_seed_offset) + int(round_idx)
                confirm_common["seed"] = int(confirm_seed)
                _log(log_path, f"confirm proposal={p.short_name()} seed={confirm_seed}")
                confirm = base._run_one_trial(
                    py=py,
                    main_py=main_py,
                    root=root,
                    base_dir=base_dir,
                    driver_log=log_path,
                    common_args=confirm_common,
                    trial=p.trial,
                    monitor_every_sec=int(args.monitor_every_sec),
                    env=env,
                    max_wallclock_sec=int(float(args.per_trial_timeout_min) * 60.0) if args.per_trial_timeout_min else None,
                )
                confirm["role"] = "confirm"
                confirm["proposal_id"] = p.proposal_id
                confirm["proposal_rationale"] = p.rationale
                confirm["proposal_origin"] = p.origin
                all_rows.append(confirm)
                pd.DataFrame(all_rows).to_csv(results_path, index=False)
                confirm_obj = _objective(confirm)
                if not _isfinite(confirm_obj):
                    _log(log_path, f"reject confirm proposal={p.short_name()} reason=non_finite objective")
                    continue
                confirm_delta = confirm_obj - incumbent_score
                if confirm_delta < float(args.accept_delta) * 0.5:
                    _log(
                        log_path,
                        f"reject confirm proposal={p.short_name()} confirm_delta={confirm_delta:.4f} "
                        f"< {0.5 * float(args.accept_delta):.4f}",
                    )
                    continue

            incumbent_trial = p.trial
            incumbent_common = dict(incumbent_common, **dict(p.common_overrides))
            incumbent_score = pilot_obj
            accepted_this_round = True
            _log(
                log_path,
                f"promote proposal={p.short_name()} new_incumbent_score={incumbent_score:.4f}",
            )
            break

        if not accepted_this_round:
            _log(log_path, f"round={round_idx} no_promotion")

    # Persist final summary and handoff snippets.
    summary = {
        "finished_at": _now(),
        "base_dir": str(base_dir),
        "proposal_source": str(args.proposal_source),
        "proposal_dir": str(proposal_dir),
        "incumbent_trial": incumbent_trial.to_args(),
        "incumbent_common_overrides": incumbent_common,
        "incumbent_score": float(incumbent_score) if _isfinite(incumbent_score) else None,
        "results_csv": str(results_path),
        "beliefs_json": str(beliefs_path),
        "proposals_jsonl": str(proposals_path),
    }
    _write_json(base_dir / "summary.json", summary)

    try:
        wm = root / "docs/WORKING_MEMORY_DIFFUSER_MAZE2D.md"
        hf = root / "HANDOFF_SUMMARY_FOR_NEXT_CODEX.txt"
        _append(
            hf,
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z%z')}] [agentic] "
            f"done run={base_dir.name} incumbent_score={incumbent_score:.4f}\n",
        )
        _append(
            wm,
            "\n"
            f"## {datetime.now().strftime('%Y-%m-%d %H:%M %Z%z')}\n"
            "### Agentic Decider update\n"
            f"- run: `{base_dir}`\n"
            f"- incumbent: `{incumbent_trial.short_name()}`\n"
            f"- incumbent score: `{incumbent_score:.4f}`\n"
            f"- results: `{results_path}`\n",
        )
    except Exception as e:
        _log(log_path, f"warn failed to append wm/handoff: {e}")

    _log(log_path, f"done summary={base_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
