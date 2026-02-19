#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


ROUND_CTX_RE = re.compile(r"round_(\d+)_context\.json$")
START_BASE_DIR_RE = re.compile(r"start base_dir=(.+)$")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z%z")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _log(path: Path, msg: str) -> None:
    line = f"[orchestrator] {_now()} {msg}\n"
    print(line, end="")
    _append(path, line)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def _coerce_space_values(values: Iterable[Any], incumbent: Any) -> List[Any]:
    out: List[Any] = []
    inc = incumbent
    for v in values:
        vv = v
        if isinstance(inc, int) and not isinstance(inc, bool):
            vv = _as_int(v, inc)
        elif isinstance(inc, float):
            vv = _as_float(v, inc)
        out.append(vv)
    return out


def _ordered_alternatives(values: Sequence[Any], incumbent: Any) -> List[Any]:
    vals = list(values)
    alts = [v for v in vals if v != incumbent]
    if _is_number(incumbent):
        alts.sort(key=lambda x: abs(_as_float(x, 0.0) - _as_float(incumbent, 0.0)))
    return alts


def _trial_key(trial: Mapping[str, Any], common_overrides: Mapping[str, Any]) -> Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]:
    trial_items = tuple(sorted((str(k), str(v)) for k, v in trial.items()))
    common_items = tuple(sorted((str(k), str(v)) for k, v in common_overrides.items()))
    return trial_items, common_items


def _extract_tried(context: Mapping[str, Any]) -> Set[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]]:
    tried: Set[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]] = set()
    raw = context.get("tried_configs", [])
    if not isinstance(raw, list):
        return tried
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        trial = entry.get("trial", {})
        common = entry.get("common_overrides", {})
        if isinstance(trial, Mapping) and isinstance(common, Mapping):
            tried.add(_trial_key(trial, common))
    return tried


def _build_proposals_from_context(
    *,
    context: Mapping[str, Any],
    round_idx: int,
    max_new: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    incumbent = context.get("incumbent", {})
    incumbent_trial_raw = incumbent.get("trial", {}) if isinstance(incumbent, Mapping) else {}
    if not isinstance(incumbent_trial_raw, Mapping):
        incumbent_trial_raw = {}
    incumbent_trial = dict(incumbent_trial_raw)
    if not incumbent_trial:
        return []

    space = context.get("space", {})
    if not isinstance(space, Mapping):
        space = {}

    tried = _extract_tried(context)
    seen: Set[Tuple[Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]] = set()
    proposals: List[Dict[str, Any]] = []

    def add_candidate(trial: Mapping[str, Any], rationale: str, risk: str = "low") -> bool:
        key = _trial_key(trial, {})
        if key in tried or key in seen:
            return False
        seen.add(key)
        proposals.append(
            {
                "proposal_id": f"orch_r{round_idx}_p{len(proposals)}",
                "rationale": rationale,
                "origin": "orchestrator_proposer",
                "risk_level": risk,
                "trial": dict(trial),
                "common_overrides": {},
            }
        )
        return True

    priority_dims = [
        "online_replan_every_n_steps",
        "online_train_steps_per_round",
        "online_collect_episodes_per_round",
        "online_rounds",
        "train_steps",
        "online_goal_geom_p",
    ]

    for dim in priority_dims:
        if len(proposals) >= max_new:
            break
        cur = incumbent_trial.get(dim)
        values = space.get(dim, [])
        if not isinstance(values, list):
            continue
        vals = _coerce_space_values(values, cur)
        for alt in _ordered_alternatives(vals, cur):
            trial = dict(incumbent_trial)
            trial[dim] = alt
            if add_candidate(trial, f"single-dimension mutation on {dim}"):
                break

    if len(proposals) < max_new:
        rp = "online_replan_every_n_steps"
        tr = "online_train_steps_per_round"
        rp_cur = incumbent_trial.get(rp)
        tr_cur = incumbent_trial.get(tr)
        rp_vals = _coerce_space_values(space.get(rp, []), rp_cur) if isinstance(space.get(rp, []), list) else []
        tr_vals = _coerce_space_values(space.get(tr, []), tr_cur) if isinstance(space.get(tr, []), list) else []
        rp_alt = _ordered_alternatives(rp_vals, rp_cur)
        tr_alt = _ordered_alternatives(tr_vals, tr_cur)
        if rp_alt and tr_alt:
            trial = dict(incumbent_trial)
            trial[rp] = rp_alt[0]
            trial[tr] = tr_alt[0]
            add_candidate(trial, "paired mutation: replanning cadence + update intensity", risk="med")

    dims_for_random = [d for d in priority_dims if isinstance(space.get(d, []), list) and len(space.get(d, [])) > 0]
    attempts = 0
    while len(proposals) < max_new and attempts < 128 and dims_for_random:
        attempts += 1
        trial = dict(incumbent_trial)
        mutate_dims = rng.sample(dims_for_random, k=min(len(dims_for_random), rng.choice([1, 2])))
        for dim in mutate_dims:
            cur = trial.get(dim)
            vals = _coerce_space_values(space.get(dim, []), cur)
            alts = _ordered_alternatives(vals, cur)
            if alts:
                trial[dim] = rng.choice(alts)
        add_candidate(trial, "stochastic local mutation", risk="med")

    if not proposals:
        add_candidate(incumbent_trial, "fallback incumbent proposal (all nearby configs appear tried)")

    return proposals[: max(1, int(max_new))]


def _parse_round_idx(path: Path) -> Optional[int]:
    m = ROUND_CTX_RE.fullmatch(path.name)
    if not m:
        return None
    return int(m.group(1))


def _read_latest_row(results_csv: Path) -> Optional[Dict[str, Any]]:
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return dict(rows[-1])


@dataclass
class SharedState:
    orchestrator_log: Path
    proposal_dir: Path
    proposer_log: Path
    reviewer_log: Path
    executor_stdout_log: Path
    base_dir: Optional[Path] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    executor_done: threading.Event = field(default_factory=threading.Event)

    def set_base_dir(self, value: Path) -> None:
        with self.lock:
            if self.base_dir is None:
                self.base_dir = value

    def get_base_dir(self) -> Optional[Path]:
        with self.lock:
            return self.base_dir


def _pump_executor(proc: subprocess.Popen[str], state: SharedState) -> None:
    with state.executor_stdout_log.open("a", encoding="utf-8") as out:
        assert proc.stdout is not None
        for raw in proc.stdout:
            out.write(raw)
            out.flush()
            line = raw.strip()
            m = START_BASE_DIR_RE.search(line)
            if m:
                path = Path(m.group(1).strip())
                state.set_base_dir(path)
                _log(state.orchestrator_log, f"detected_base_dir={path}")


def _proposer_loop(state: SharedState, poll_sec: float, seed: int, fallback_max_new: int) -> None:
    rng = random.Random(seed + 101)
    handled: Set[int] = set()
    sleep_s = max(0.5, float(poll_sec))
    _log(state.proposer_log, f"start proposal_dir={state.proposal_dir}")
    while not state.stop_event.is_set():
        progress = False
        for ctx_path in sorted(state.proposal_dir.glob("round_*_context.json")):
            round_idx = _parse_round_idx(ctx_path)
            if round_idx is None or round_idx in handled:
                continue
            proposal_path = state.proposal_dir / f"round_{round_idx:03d}_proposals.json"
            if proposal_path.exists():
                handled.add(round_idx)
                continue
            try:
                context = json.loads(ctx_path.read_text(encoding="utf-8"))
            except Exception as e:
                _log(state.proposer_log, f"round={round_idx} context_parse_error err={e}")
                continue
            req = context.get("proposal_requirements", {})
            max_new = _as_int(req.get("max_new", fallback_max_new), fallback_max_new)
            proposals = _build_proposals_from_context(
                context=context,
                round_idx=round_idx,
                max_new=max_new,
                rng=rng,
            )
            if not proposals:
                _log(state.proposer_log, f"round={round_idx} no_proposals_generated")
                continue
            _write_json(proposal_path, {"proposals": proposals})
            _log(state.proposer_log, f"round={round_idx} wrote_proposals={proposal_path} count={len(proposals)}")
            handled.add(round_idx)
            progress = True

        if state.executor_done.is_set() and not progress:
            break
        time.sleep(sleep_s)

    _log(state.proposer_log, "stop")


def _reviewer_loop(state: SharedState, poll_sec: float) -> None:
    sleep_s = max(1.0, float(poll_sec))
    last_signature = ""
    _log(state.reviewer_log, "start")
    while not state.stop_event.is_set():
        base_dir = state.get_base_dir()
        if base_dir is None:
            if state.executor_done.is_set():
                break
            time.sleep(sleep_s)
            continue

        metrics: Dict[str, Any] = {"base_dir": str(base_dir)}
        results_csv = base_dir / "agentic_results.csv"
        row = _read_latest_row(results_csv)
        if row is not None:
            metrics["latest_run_name"] = row.get("run_name")
            metrics["latest_role"] = row.get("role")
            metrics["latest_proposal_id"] = row.get("proposal_id")
            metrics["latest_objective_succ_h256"] = row.get("objective_succ_h256")
            metrics["latest_online_round_last"] = row.get("online_round_last")
        log_path = base_dir / "agentic.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            metrics["agentic_log_size"] = len(text)
            metrics["event_promote_count"] = text.count("promote proposal=")
            metrics["event_reject_pilot_count"] = text.count("reject pilot proposal=")
            metrics["event_no_promotion_count"] = text.count(" no_promotion")
            metrics["event_loaded_external_count"] = text.count("loaded_external_proposals")

        signature = json.dumps(metrics, sort_keys=True, default=str)
        if signature != last_signature:
            payload = {"ts": _now(), "metrics": metrics}
            _append(state.reviewer_log, json.dumps(payload, sort_keys=True) + "\n")
            last_signature = signature

        if state.executor_done.is_set():
            break
        time.sleep(sleep_s)

    _log(state.reviewer_log, "stop")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Supervisor that auto-runs executor + proposer + reviewer sidecars for agentic Maze2D."
    )
    ap.add_argument("--root", type=Path, default=Path("/root/ebm-online-rl-prototype"))
    ap.add_argument("--python", type=Path, default=Path("/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python"))
    ap.add_argument("--executor-script", type=Path, default=Path("/root/ebm-online-rl-prototype/scripts/agentic_maze2d_autodecider.py"))
    ap.add_argument("--main-script", type=Path, default=None, help="Optional override passed through to executor.")
    ap.add_argument("--orchestrator-dir", type=Path, default=None)
    ap.add_argument("--proposal-dir", type=Path, default=None)
    ap.add_argument("--proposal-source", choices=["external", "external_preferred", "internal"], default="external")
    ap.add_argument("--budget-hours", type=float, default=6.0)
    ap.add_argument("--monitor-every-sec", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-trial-timeout-min", type=float, default=35.0)
    ap.add_argument("--max-agent-rounds", type=int, default=8)
    ap.add_argument("--proposals-per-round", type=int, default=4)
    ap.add_argument("--proposal-wait-timeout-sec", type=int, default=1800)
    ap.add_argument("--proposal-poll-sec", type=float, default=10.0)
    ap.add_argument("--accept-delta", type=float, default=0.02)
    ap.add_argument("--proposer-poll-sec", type=float, default=2.0)
    ap.add_argument("--reviewer-poll-sec", type=float, default=20.0)
    args = ap.parse_args()

    root = args.root
    py = args.python
    executor_script = args.executor_script
    if not root.exists():
        raise SystemExit(f"missing root: {root}")
    if not py.exists():
        raise SystemExit(f"missing python: {py}")
    if not executor_script.exists():
        raise SystemExit(f"missing executor script: {executor_script}")
    if args.main_script and (not args.main_script.exists()):
        raise SystemExit(f"missing main script override: {args.main_script}")

    orchestrator_dir = (
        args.orchestrator_dir
        if args.orchestrator_dir
        else root / "runs/analysis/synth_maze2d_diffuser_probe" / f"orchestrator_{_stamp()}"
    )
    orchestrator_dir.mkdir(parents=True, exist_ok=True)
    proposal_dir = args.proposal_dir if args.proposal_dir else (orchestrator_dir / "proposal_exchange")
    proposal_dir.mkdir(parents=True, exist_ok=True)

    orchestrator_log = orchestrator_dir / "orchestrator.log"
    executor_stdout_log = orchestrator_dir / "executor_stdout.log"
    proposer_log = orchestrator_dir / "proposer.log"
    reviewer_log = orchestrator_dir / "reviewer_notes.jsonl"
    summary_path = orchestrator_dir / "summary.json"
    config_path = orchestrator_dir / "config.json"

    _write_json(
        config_path,
        {
            "created_at": _now(),
            "root": str(root),
            "python": str(py),
            "executor_script": str(executor_script),
            "main_script": str(args.main_script) if args.main_script else None,
            "orchestrator_dir": str(orchestrator_dir),
            "proposal_dir": str(proposal_dir),
            "proposal_source": str(args.proposal_source),
            "budget_hours": float(args.budget_hours),
            "monitor_every_sec": int(args.monitor_every_sec),
            "seed": int(args.seed),
            "smoke": bool(args.smoke),
            "per_trial_timeout_min": float(args.per_trial_timeout_min),
            "max_agent_rounds": int(args.max_agent_rounds),
            "proposals_per_round": int(args.proposals_per_round),
            "proposal_wait_timeout_sec": int(args.proposal_wait_timeout_sec),
            "proposal_poll_sec": float(args.proposal_poll_sec),
            "accept_delta": float(args.accept_delta),
            "proposer_poll_sec": float(args.proposer_poll_sec),
            "reviewer_poll_sec": float(args.reviewer_poll_sec),
        },
    )

    _log(orchestrator_log, f"start orchestrator_dir={orchestrator_dir}")
    _log(orchestrator_log, f"proposal_dir={proposal_dir}")

    cmd: List[str] = [
        str(py),
        str(executor_script),
        "--root",
        str(root),
        "--python",
        str(py),
        "--budget-hours",
        str(args.budget_hours),
        "--monitor-every-sec",
        str(args.monitor_every_sec),
        "--seed",
        str(args.seed),
        "--per-trial-timeout-min",
        str(args.per_trial_timeout_min),
        "--max-agent-rounds",
        str(args.max_agent_rounds),
        "--proposals-per-round",
        str(args.proposals_per_round),
        "--proposal-source",
        str(args.proposal_source),
        "--proposal-dir",
        str(proposal_dir),
        "--proposal-wait-timeout-sec",
        str(args.proposal_wait_timeout_sec),
        "--proposal-poll-sec",
        str(args.proposal_poll_sec),
        "--accept-delta",
        str(args.accept_delta),
    ]
    if args.main_script:
        cmd.extend(["--main-script", str(args.main_script)])
    if args.smoke:
        cmd.append("--smoke")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _log(orchestrator_log, f"launched_executor pid={proc.pid}")

    state = SharedState(
        orchestrator_log=orchestrator_log,
        proposal_dir=proposal_dir,
        proposer_log=proposer_log,
        reviewer_log=reviewer_log,
        executor_stdout_log=executor_stdout_log,
    )

    t_pump = threading.Thread(target=_pump_executor, args=(proc, state), daemon=True)
    t_prop = threading.Thread(
        target=_proposer_loop,
        args=(state, args.proposer_poll_sec, int(args.seed), int(args.proposals_per_round)),
        daemon=True,
    )
    t_rev = threading.Thread(target=_reviewer_loop, args=(state, args.reviewer_poll_sec), daemon=True)
    t_pump.start()
    t_prop.start()
    t_rev.start()

    rc = proc.wait()
    state.executor_done.set()
    state.stop_event.set()
    _log(orchestrator_log, f"executor_finished rc={rc}")

    t_pump.join(timeout=10.0)
    t_prop.join(timeout=10.0)
    t_rev.join(timeout=10.0)

    summary = {
        "finished_at": _now(),
        "return_code": int(rc),
        "orchestrator_dir": str(orchestrator_dir),
        "proposal_dir": str(proposal_dir),
        "base_dir": str(state.get_base_dir()) if state.get_base_dir() else None,
        "executor_stdout_log": str(executor_stdout_log),
        "orchestrator_log": str(orchestrator_log),
        "proposer_log": str(proposer_log),
        "reviewer_log": str(reviewer_log),
        "config_json": str(config_path),
    }
    _write_json(summary_path, summary)
    _log(orchestrator_log, f"done summary={summary_path}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

