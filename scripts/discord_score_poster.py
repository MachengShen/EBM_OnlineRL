#!/usr/bin/env python3
"""
Discord periodic score poster for locomotion collector experiments.
Reads long_latest.log every N minutes, extracts eval scores,
and posts a compact formatted table to a Discord channel/thread.

Usage:
    python3 scripts/discord_score_poster.py \
        --log runs/analysis/locomotion_collector/long_latest.log \
        --token "$DISCORD_BOT_TOKEN" \
        --channel "$DISCORD_CHANNEL_ID" \
        --interval 1800
"""
import argparse
import ast
import os
import re
import sys
import time
import urllib.request
import urllib.error
import json
from datetime import datetime, timezone

# ── Regex patterns ──────────────────────────────────────────────────────────

# e.g. "  [diffuser_warmstart_sac] ep=300/1000  grad_steps=380308  score=0.39"
RE_EVAL = re.compile(
    r"\[(?P<cond>diffuser_warmstart_sac|sac_scratch|gcbc_diffuser)\]"
    r"\s+ep=(?P<ep>\d+)/(?P<total>\d+)"
    r"\s+grad_steps=(?P<grad>\d+)"
    r"\s+score=(?P<score>[0-9.eE+\-]+)"
)

# e.g. "  Collection done in 222.0s  mean_score=0.43"
RE_COLLECTION = re.compile(r"Collection done in [\d.]+s\s+mean_score=(?P<score>[0-9.eE+\-]+)")

# e.g. "  RECORD: {'condition': 'gcbc_diffuser', 'env': 'hopper-medium-expert-v2', ...}"
RE_RECORD = re.compile(r"RECORD:\s+(\{.+\})")

# Header line like "ENV=hopper-medium-expert-v2  CONDITION=diffuser_warmstart_sac  SEED=0"
RE_HEADER = re.compile(r"ENV=(?P<env>[a-z0-9_\-]+)\s+CONDITION=(?P<cond>[a-z_]+)\s+SEED=(?P<seed>\d+)")

# Condition short names for table display
COND_SHORT = {
    "diffuser_warmstart_sac": "diff→SAC",
    "sac_scratch":            "SAC-0   ",
    "gcbc_diffuser":          "GCBC    ",
}
ENV_SHORT = {
    "hopper-medium-expert-v2":  "hopper",
    "walker2d-medium-expert-v2": "walker",
}

# ── Log parser ───────────────────────────────────────────────────────────────

def parse_log(log_path):
    """Return parsed state from the log file."""
    state = {
        # (env, cond, seed) -> {ep, total, grad, score}
        "evals": {},
        # env -> {score, done}
        "collections": {},
        # list of completed RECORD dicts
        "records": [],
        # current header context
        "current_env": None,
        "current_cond": None,
        "current_seed": None,
        "log_lines": 0,
    }

    try:
        with open(log_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return state

    state["log_lines"] = len(lines)

    for line in lines:
        m = RE_HEADER.search(line)
        if m:
            state["current_env"]  = m.group("env")
            state["current_cond"] = m.group("cond")
            state["current_seed"] = int(m.group("seed"))
            continue

        m = RE_COLLECTION.search(line)
        if m and state["current_env"]:
            env = state["current_env"]
            if env not in state["collections"]:
                state["collections"][env] = []
            state["collections"][env].append(float(m.group("score")))
            continue

        m = RE_EVAL.search(line)
        if m:
            env  = state["current_env"]  or "?"
            cond = m.group("cond")
            seed = state["current_seed"] if state["current_seed"] is not None else "?"
            key  = (env, cond, seed)
            state["evals"][key] = {
                "ep":    int(m.group("ep")),
                "total": int(m.group("total")),
                "grad":  int(m.group("grad")),
                "score": float(m.group("score")),
            }
            continue

        m = RE_RECORD.search(line)
        if m:
            try:
                rec = ast.literal_eval(m.group(1))
                state["records"].append(rec)
            except Exception:
                pass

    return state


PING_USER_ID = "868020619499937792"  # Macheng — guaranteed push notification


def format_table(state, elapsed_h=None):
    """Return a Discord-friendly message string."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"<@{PING_USER_ID}> **Locomotion Collector — Score Update** ({now_str})"]
    if elapsed_h is not None:
        lines.append(f"Elapsed: {elapsed_h:.1f}h")
    lines.append("")

    evals = state["evals"]
    if not evals:
        lines.append("_No eval data yet — training initialising…_")
        return "\n".join(lines)

    # Gather all (env, cond) pairs seen
    envs  = sorted({k[0] for k in evals})
    conds = ["diffuser_warmstart_sac", "sac_scratch", "gcbc_diffuser"]
    seeds = sorted({k[2] for k in evals})

    # One table per env
    for env in envs:
        env_label = ENV_SHORT.get(env, env)
        lines.append(f"**{env_label}**")
        lines.append("```")
        header = f"{'Condition':<12} {'Seed':>4}  {'Ep':>8}  {'Score':>6}"
        lines.append(header)
        lines.append("-" * len(header))
        for cond in conds:
            cond_label = COND_SHORT.get(cond, cond)
            for seed in seeds:
                key = (env, cond, seed)
                if key in evals:
                    d = evals[key]
                    pct = d["ep"] / d["total"] * 100
                    lines.append(
                        f"{cond_label:<12} {seed:>4}  "
                        f"{d['ep']:>4}/{d['total']:<4} ({pct:>3.0f}%)  "
                        f"{d['score']:>5.3f}"
                    )
                else:
                    lines.append(f"{cond_label:<12} {seed:>4}  {'—':>8}  {'—':>6}")
        lines.append("```")

    # Collection quality
    if state["collections"]:
        lines.append("**Diffuser collection** (mean norm score)")
        for env, scores in sorted(state["collections"].items()):
            env_label = ENV_SHORT.get(env, env)
            avg = sum(scores) / len(scores)
            lines.append(f"  {env_label}: {avg:.3f} ({len(scores)} run(s))")
        lines.append("")

    # Completed conditions (from RECORD lines)
    if state["records"]:
        done = len(state["records"])
        lines.append(f"Completed conditions: {done}")

    lines.append(f"Log lines: {state['log_lines']}")
    return "\n".join(lines)


# ── Discord sender ────────────────────────────────────────────────────────────

def post_to_discord(token, channel_id, text, dry_run=False):
    if dry_run:
        print("=== DRY RUN — would post ===")
        print(text)
        return True

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "DiscordScorePoster/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
        print(f"[poster] Posted to Discord (HTTP {status})")
        return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[poster] Discord HTTP error {e.code}: {body}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[poster] Discord error: {e}", file=sys.stderr)
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",      default="runs/analysis/locomotion_collector/long_latest.log")
    ap.add_argument("--token",    default=os.environ.get("DISCORD_BOT_TOKEN", ""))
    ap.add_argument("--channel",  default=os.environ.get("DISCORD_CHANNEL_ID", ""))
    ap.add_argument("--interval", type=int, default=1800, help="Seconds between posts (default 1800=30m)")
    ap.add_argument("--dry-run",  action="store_true", help="Print instead of posting to Discord")
    ap.add_argument("--once",     action="store_true", help="Post once and exit")
    args = ap.parse_args()

    if not args.dry_run and not args.token:
        print("ERROR: --token or DISCORD_BOT_TOKEN required", file=sys.stderr)
        sys.exit(1)
    if not args.dry_run and not args.channel:
        print("ERROR: --channel or DISCORD_CHANNEL_ID required", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    print(f"[poster] Starting. Log={args.log}  Channel={args.channel}  Interval={args.interval}s")

    while True:
        elapsed_h = (time.time() - start_time) / 3600.0
        state  = parse_log(args.log)
        text   = format_table(state, elapsed_h=elapsed_h)
        post_to_discord(args.token, args.channel, text, dry_run=args.dry_run)

        if args.once:
            break

        print(f"[poster] Sleeping {args.interval}s until next post…")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
