#!/usr/bin/env python3
"""
Discord experiment monitor for swap-matrix / locomotion experiments.

Understands two directory layouts:

1. Swap-matrix layout (maze2d):
     phase1_collectors/{collector}/seed_{N}/
     phase2_learners/{mode}/{collector}_to_{learner}/seed_{N}/

2. Flat layout (locomotion, ablation grids, etc.):
     {condition}/{env}/seed_{N}/   or any nested dir with summary.json

Posts a compact, fixed-width Discord update every --interval seconds.

── Example output (swap-matrix) ─────────────────────────────────────────────
🧪 maze2d-medium  |  6.2h  |  2026-02-22 14:35 UTC
■■■□□□□□□□□□□□□□□□□□  6/24 done, 1 ⟳, 17 pending

COLLECTORS       s0     s1     s2
  diffuser     0.611  0.750  0.583 ✓
  sac          —      —      —     ✗

WARMSTART        s0     s1     s2
  diff→diff    0.083  —      —     ⟳
  diff→sac     —      —      —
  sac→diff     —      —      —
  sac→sac      —      —      —
FROZEN           s0     s1     s2
  diff→diff    —      —      —
  ...

⟳ warmstart/diff→diff/s0  step=3k/18k  roll@256=0.083
─────────────────────────────────────────────────────────────────────────────

Usage:
    python3 scripts/discord_swap_matrix_monitor.py \\
        --base-dir runs/analysis/swap_matrix/maze2d_medium_20260222-145304 \\
        --label "maze2d-medium" \\
        --interval 1800

Env vars: DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────

PING_USER_ID    = "868020619499937792"     # Research lead
DEFAULT_CHANNEL = "1472061023778242744"    # Main research channel

# Metrics tried in order; first found wins
SUCCESS_KEYS = [
    "progress_last.rollout_goal_success_rate_h256",
    "progress_last.rollout_goal_success_rate",
    "rollout_success_rate_h256",
    "final_success_rate",
    "score",
]
PROGRESS_RE   = re.compile(r"\[progress\].*?roll(?:out_success|@256)=([0-9.]+)")
TRAIN_STEP_RE = re.compile(r"step=\s*([0-9]+)")
TOTAL_STEP_RE = re.compile(r"train_steps_total.*?([0-9]+)")

# Short display names
COLL_SHORT = {"diffuser": "diff", "sac_her_sparse": "sac",
              "diffuser_online": "diff-online", "sac_scratch": "sac-0"}
MODE_SHORT = {"warmstart": "WARMSTART", "frozen": "FROZEN"}


# ── Data extraction ───────────────────────────────────────────────────────────

def _nested_get(d: dict, dotted: str, default=None):
    parts = dotted.split(".", 1)
    val = d.get(parts[0], default)
    if len(parts) == 1 or not isinstance(val, dict):
        return val
    return _nested_get(val, parts[1], default)


def _extract_success(summary: dict) -> Optional[float]:
    for k in SUCCESS_KEYS:
        v = _nested_get(summary, k)
        if v is not None:
            return round(float(v), 3)
    return None


def _rc_from_summary(summary: dict) -> int:
    return int(summary.get("rc", 0)) if "rc" in summary else 0


def _tail_bytes(path: Path, n: int = 8192) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _last_progress(log_path: Path) -> Optional[str]:
    tail = _tail_bytes(log_path)
    for line in reversed(tail.splitlines()):
        if "[progress]" in line:
            return line.strip()
    return None


def _step_from_progress(prog: str) -> Optional[int]:
    m = TRAIN_STEP_RE.search(prog)
    return int(m.group(1)) if m else None


def _success_from_progress(prog: str) -> Optional[float]:
    m = PROGRESS_RE.search(prog)
    return float(m.group(1)) if m else None


# ── Directory scanner ─────────────────────────────────────────────────────────

class Cell:
    """One experiment run (one directory)."""
    __slots__ = ("path", "label", "status", "success", "step", "progress_line", "rc")

    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.status = "pending"   # pending | running | done | failed
        self.success: Optional[float] = None
        self.step: Optional[int] = None
        self.progress_line: Optional[str] = None
        self.rc: int = 0

    def load(self):
        summary_p = self.path / "summary.json"
        log_p     = self.path / "stdout_stderr.log"
        if summary_p.exists():
            try:
                d = json.loads(summary_p.read_text())
                self.success = _extract_success(d)
                self.rc = int(d.get("rc", 0))
                self.status = "failed" if self.rc != 0 else "done"
            except Exception:
                self.status = "failed"
        elif log_p.exists():
            self.status = "running"
            prog = _last_progress(log_p)
            if prog:
                self.progress_line = prog
                self.step = _step_from_progress(prog)
                self.success = _success_from_progress(prog)


def scan(base_dir: Path) -> Tuple[str, List[Cell]]:
    """
    Detect layout, scan cells.
    Returns (layout_type, cells) where layout_type is 'swap_matrix' or 'flat'.
    """
    phase2 = base_dir / "phase2_learners"
    phase1 = base_dir / "phase1_collectors"
    if phase2.exists() or phase1.exists():
        return "swap_matrix", _scan_swap_matrix(base_dir)
    return "flat", _scan_flat(base_dir)


def _scan_swap_matrix(base_dir: Path) -> List[Cell]:
    cells = []
    phase1 = base_dir / "phase1_collectors"
    phase2 = base_dir / "phase2_learners"

    if phase1.exists():
        for coll_dir in sorted(phase1.iterdir()):
            if not coll_dir.is_dir():
                continue
            for seed_dir in sorted(coll_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                label = f"p1/{coll_dir.name}/{seed_dir.name}"
                c = Cell(seed_dir, label)
                c.load()
                cells.append(c)

    if phase2.exists():
        for mode_dir in sorted(phase2.iterdir()):
            if not mode_dir.is_dir():
                continue
            for cond_dir in sorted(mode_dir.iterdir()):
                if not cond_dir.is_dir():
                    continue
                for seed_dir in sorted(cond_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    label = f"{mode_dir.name}/{cond_dir.name}/{seed_dir.name}"
                    c = Cell(seed_dir, label)
                    c.load()
                    cells.append(c)
    return cells


def _scan_flat(base_dir: Path) -> List[Cell]:
    cells = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if "summary.json" in filenames or "stdout_stderr.log" in filenames:
            p = Path(dirpath)
            label = str(p.relative_to(base_dir))
            c = Cell(p, label)
            c.load()
            cells.append(c)
    return sorted(cells, key=lambda x: x.label)


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_score(v: Optional[float], status: str) -> str:
    if v is not None:
        return f"{v:.3f}"
    if status == "failed":
        return " ✗  "
    if status == "running":
        return "  ⟳  "
    return "  —  "


def _status_icon(status: str) -> str:
    return {"done": "✓", "failed": "✗", "running": "⟳", "pending": "·"}[status]


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return "■" * filled + "□" * (width - filled)


def _swap_matrix_table(cells: List[Cell]) -> str:
    """Build compact fixed-width table for swap-matrix layout."""
    # Index cells by (phase, mode, collector, learner, seed)
    p1: Dict[Tuple, Cell] = {}   # (coll, seed) -> Cell
    p2: Dict[Tuple, Cell] = {}   # (mode, coll, learner, seed) -> Cell

    collectors_seen = set()
    modes_seen      = set()
    colls_p2_seen   = set()
    learners_seen   = set()
    seeds_seen      = set()

    for c in cells:
        parts = c.label.split("/")
        if parts[0] == "p1" and len(parts) >= 3:
            coll = parts[1]
            seed = int(parts[2].replace("seed_", ""))
            p1[(coll, seed)] = c
            collectors_seen.add(coll)
            seeds_seen.add(seed)
        elif len(parts) >= 3:
            # mode/coll_to_learner/seed_N
            mode = parts[0]
            cond = parts[1]  # e.g. "diffuser_to_sac_her_sparse"
            seed = int(parts[2].replace("seed_", ""))
            if "_to_" in cond:
                coll_p, lrn_p = cond.split("_to_", 1)
            else:
                coll_p, lrn_p = cond, cond
            p2[(mode, coll_p, lrn_p, seed)] = c
            modes_seen.add(mode)
            colls_p2_seen.add(coll_p)
            learners_seen.add(lrn_p)
            seeds_seen.add(seed)

    seeds    = sorted(seeds_seen)
    s_hdr    = "  ".join(f"s{s}" for s in seeds)
    col_w    = 6   # width per seed column

    def _row(label: str, cells_row: List[Optional[Cell]]) -> str:
        scores = []
        for c in cells_row:
            if c is None:
                scores.append("  —  ")
            else:
                scores.append(_fmt_score(c.success, c.status))
        return f"  {label:<14}" + "  ".join(f"{s:>5}" for s in scores)

    lines = []

    # Phase 1 collectors
    if p1:
        lines.append(f"COLLECTORS     {s_hdr}")
        for coll in sorted(collectors_seen):
            row_cells = [p1.get((coll, s)) for s in seeds]
            # Overall status icon for this collector
            statuses = [c.status if c else "pending" for c in row_cells]
            icon = " ✓" if all(s == "done" for s in statuses) else \
                   " ✗" if any(s == "failed" for s in statuses) else \
                   " ⟳" if any(s == "running" for s in statuses) else ""
            lines.append(_row(COLL_SHORT.get(coll, coll), row_cells) + icon)
        lines.append("")

    # Phase 2 learners — one block per mode
    conditions = [(c, l) for c in sorted(colls_p2_seen) for l in sorted(learners_seen)]
    for mode in sorted(modes_seen):
        lines.append(f"{MODE_SHORT.get(mode, mode.upper()):<14} {s_hdr}")
        for coll, lrn in conditions:
            row_cells = [p2.get((mode, coll, lrn, s)) for s in seeds]
            label = f"{COLL_SHORT.get(coll,'?')}→{COLL_SHORT.get(lrn,'?')}"
            lines.append(_row(label, row_cells))
        lines.append("")

    return "\n".join(lines).rstrip()


def format_message(label: str, layout: str, cells: List[Cell],
                   elapsed_h: Optional[float] = None) -> str:
    now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elapsed_s = f"  |  {elapsed_h:.1f}h" if elapsed_h is not None else ""
    header    = f"<@{PING_USER_ID}>  🧪 **{label}**{elapsed_s}  |  {now_str}"

    total   = len(cells)
    done    = sum(1 for c in cells if c.status == "done")
    failed  = sum(1 for c in cells if c.status == "failed")
    running = sum(1 for c in cells if c.status == "running")
    pending = total - done - failed - running

    bar   = _progress_bar(done + failed, total)
    parts = [f"{done}/{total} done"]
    if running:
        parts.append(f"{running} ⟳")
    if failed:
        parts.append(f"{failed} ✗")
    if pending:
        parts.append(f"{pending} pending")
    summary_line = f"{bar}  {', '.join(parts)}"

    lines = [header, summary_line, ""]

    if layout == "swap_matrix":
        lines.append("```")
        lines.append(_swap_matrix_table(cells))
        lines.append("```")
    else:
        # Flat layout: simple table
        lines.append("```")
        lines.append(f"{'Cell':<48}  {'s@h256':>6}  {'st':>2}")
        lines.append("─" * 60)
        for c in cells:
            sc   = f"{c.success:.3f}" if c.success is not None else "  —   "
            icon = _status_icon(c.status)
            cell = c.label[-47:] if len(c.label) > 47 else c.label
            lines.append(f"{cell:<48}  {sc:>6}  {icon:>2}")
        lines.append("```")

    # Running cell detail
    running_cells = [c for c in cells if c.status == "running"]
    if running_cells:
        lines.append("")
        for rc in running_cells[:3]:
            step_s = f"  step={rc.step:,}" if rc.step else ""
            succ_s = f"  roll@256={rc.success:.3f}" if rc.success is not None else ""
            # Shorten label: drop p2/phase prefix
            lbl = rc.label.replace("phase2_learners/", "").replace("p1/", "p1:")
            lines.append(f"⟳ `{lbl}`{step_s}{succ_s}")
        if len(running_cells) > 3:
            lines.append(f"  _(+ {len(running_cells) - 3} more running)_")

    text = "\n".join(lines)
    if len(text) > 1950:
        text = text[:1947] + "…"
    return text


# ── Discord ───────────────────────────────────────────────────────────────────

def post_to_discord(token: str, channel_id: str, text: str, dry_run: bool = False) -> bool:
    if dry_run:
        print("─" * 70)
        print(text)
        print("─" * 70)
        print(f"[dry-run] {len(text)} chars")
        return True

    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{channel_id}/messages",
        data=payload,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "ExperimentMonitor/2.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[monitor] Posted (HTTP {resp.status})", flush=True)
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[monitor] Discord HTTP {e.code}: {body}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[monitor] Error: {e}", file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-dir",  required=True)
    ap.add_argument("--label",     default="Experiment",
                    help="Short human-readable name for the Discord header")
    ap.add_argument("--token",     default=os.environ.get("DISCORD_BOT_TOKEN", ""))
    ap.add_argument("--channel",   default=os.environ.get("DISCORD_CHANNEL_ID", DEFAULT_CHANNEL))
    ap.add_argument("--interval",  type=int, default=1800,
                    help="Seconds between posts (default 1800 = 30 min)")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print to stdout instead of posting")
    ap.add_argument("--once",      action="store_true",
                    help="Post once and exit")
    args = ap.parse_args()

    if not args.dry_run and not args.token:
        print("ERROR: --token or DISCORD_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: {base_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    start = time.time()
    print(f"[monitor] base={base_dir}  channel={args.channel}  interval={args.interval}s",
          flush=True)

    while True:
        layout, cells = scan(base_dir)
        elapsed_h = (time.time() - start) / 3600.0
        text = format_message(args.label, layout, cells, elapsed_h=elapsed_h)
        post_to_discord(args.token, args.channel, text, dry_run=args.dry_run)

        done    = sum(1 for c in cells if c.status in ("done", "failed"))
        running = sum(1 for c in cells if c.status == "running")
        print(f"[monitor] done={done}/{len(cells)} running={running}  "
              f"next in {args.interval}s", flush=True)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
