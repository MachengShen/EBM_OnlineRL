#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/ebm-online-rl-prototype"
PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
MAIN_SCRIPT="${ROOT}/scripts/synthetic_maze2d_diffuser_probe.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
BASE_DIR="${ROOT}/runs/analysis/synth_maze2d_diffuser_probe/five_step_${STAMP}"
mkdir -p "${BASE_DIR}"
DRIVER_LOG="${BASE_DIR}/driver.log"

LD_MJ="/root/.mujoco/mujoco210/bin"

BUDGET_SEC=$((12 * 3600))
START_SEC="$(date +%s)"
MONITOR_EVERY_SEC="${MONITOR_EVERY_SEC:-300}"

log() {
  local msg="$1"
  echo "[driver] $(date '+%Y-%m-%d %H:%M:%S %Z') ${msg}" | tee -a "${DRIVER_LOG}"
}

# Always emit a terminal status line, so unexpected exits are visible in logs.
trap 'rc=$?; log "driver_exit rc=${rc}"' EXIT

remaining_budget_sec() {
  local now
  now="$(date +%s)"
  echo $(( BUDGET_SEC - (now - START_SEC) ))
}

COMMON_ARGS=(
  --device cuda:0
  --n_episodes 1000
  --episode_len 256
  --max_path_length 256
  --horizon 64
  --n_diffusion_steps 64
  --model_dim 64
  --model_dim_mults 1,2,4
  --learning_rate 2e-4
  --batch_size 128
  --val_every 500
  --val_batches 20
  # We need intermediate checkpoints for monitor/decide loops; 5000 is too sparse
  # for short ablation runs (~2k-3k train steps).
  --eval_goal_every 500
  --save_checkpoint_every 5000
  --query_mode diverse
  --num_eval_queries 24
  --query_batch_size 6
  --online_self_improve
  --online_collect_episode_len 256
  --online_goal_geom_p 0.08
  --online_goal_geom_min_k 8
  --online_goal_geom_max_k 96
  --online_goal_min_distance 0.5
  --eval_rollout_horizon 256
  --eval_success_prefix_horizons 64,128,192,256
  --online_planning_success_thresholds 0.1,0.2
  --online_planning_success_rel_reduction 0.9
)

snapshot_metrics() {
  local run_dir="$1"
  local phase_tag="$2"
  local progress_csv="${run_dir}/progress_metrics.csv"
  local online_csv="${run_dir}/online_collection.csv"
  if [[ ! -f "${progress_csv}" ]]; then
    log "snapshot phase=${phase_tag} run_dir=${run_dir} progress_metrics.csv not ready"
    return
  fi
  "${PY}" - <<'PY' "${progress_csv}" "${online_csv}" "${phase_tag}" | tee -a "${DRIVER_LOG}"
import math
import os
import sys
import pandas as pd

progress_path, online_path, phase = sys.argv[1], sys.argv[2], sys.argv[3]

def f(v):
    try:
        return float(v)
    except Exception:
        return float('nan')

try:
    p = pd.read_csv(progress_path)
except Exception as e:
    print(f"[snapshot] phase={phase} progress_read_error={e}")
    raise SystemExit(0)

if len(p) == 0:
    print(f"[snapshot] phase={phase} progress_empty=1")
    raise SystemExit(0)

row = p.iloc[-1]
step = int(row.get("step", -1))
h64 = f(row.get("rollout_goal_success_rate_h64", row.get("rollout_goal_success_rate", float('nan'))))
h128 = f(row.get("rollout_goal_success_rate_h128", float('nan')))
h192 = f(row.get("rollout_goal_success_rate_h192", float('nan')))
h256 = f(row.get("rollout_goal_success_rate_h256", row.get("rollout_goal_success_rate", float('nan'))))
monotone_ok = True
vals = [h64, h128, h192, h256]
for a, b in zip(vals, vals[1:]):
    if not (math.isnan(a) or math.isnan(b) or a <= b + 1e-9):
        monotone_ok = False
        break

msg = (
    f"[snapshot] phase={phase} step={step} "
    f"succ_h64={h64:.4f} succ_h128={h128:.4f} succ_h192={h192:.4f} succ_h256={h256:.4f} "
    f"prefix_monotone={int(monotone_ok)}"
)
print(msg)

if os.path.exists(online_path):
    try:
        o = pd.read_csv(online_path)
        if len(o) > 0:
            r = o.iloc[-1]
            t010 = f(r.get("planning_success_rate_final_t010", float('nan')))
            t020 = f(r.get("planning_success_rate_final_t020", float('nan')))
            rel = f(r.get("planning_success_rate_final_rel090", r.get("planning_success_rate_final_rel_reduction", float('nan'))))
            print(f"[snapshot] phase={phase} planning_t010={t010:.4f} planning_t020={t020:.4f} planning_rel090={rel:.4f}")
    except Exception as e:
        print(f"[snapshot] phase={phase} online_read_error={e}")
PY
}

extract_last_h256() {
  local run_dir="$1"
  local progress_csv="${run_dir}/progress_metrics.csv"
  if [[ ! -f "${progress_csv}" ]]; then
    echo "nan"
    return
  fi
  "${PY}" - <<'PY' "${progress_csv}"
import sys
import pandas as pd
path = sys.argv[1]
try:
    df = pd.read_csv(path)
    if len(df) == 0:
        print("nan")
    elif "rollout_goal_success_rate_h256" in df.columns and df["rollout_goal_success_rate_h256"].notna().any():
        print(float(df["rollout_goal_success_rate_h256"].dropna().iloc[-1]))
    elif "rollout_goal_success_rate" in df.columns and df["rollout_goal_success_rate"].notna().any():
        print(float(df["rollout_goal_success_rate"].dropna().iloc[-1]))
    else:
        print("nan")
except Exception:
    print("nan")
PY
}

validate_new_columns() {
  local run_dir="$1"
  local progress_csv="${run_dir}/progress_metrics.csv"
  local online_csv="${run_dir}/online_collection.csv"
  if [[ ! -f "${progress_csv}" || ! -f "${online_csv}" ]]; then
    return 1
  fi
  "${PY}" - <<'PY' "${progress_csv}" "${online_csv}"
import sys
import pandas as pd
p = pd.read_csv(sys.argv[1])
o = pd.read_csv(sys.argv[2])
need_p = [
    "rollout_goal_success_rate_h64",
    "rollout_goal_success_rate_h128",
    "rollout_goal_success_rate_h192",
    "rollout_goal_success_rate_h256",
]
need_o = [
    "planning_success_rate_final_t010",
    "planning_success_rate_final_t020",
]
missing_p = [c for c in need_p if c not in p.columns]
missing_o = [c for c in need_o if c not in o.columns]
if missing_p or missing_o:
    print(f"missing_progress={missing_p} missing_online={missing_o}")
    raise SystemExit(1)
print("ok")
PY
}

run_case() {
  local name="$1"; shift
  local run_dir="${BASE_DIR}/${name}"
  mkdir -p "${run_dir}"
  local run_log="${run_dir}/run.log"
  local t0 t1 elapsed status pid

  log "launch name=${name} run_dir=${run_dir} remaining_budget_sec=$(remaining_budget_sec)"
  t0="$(date +%s)"

  set +e
  (
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${LD_MJ}" \
    MUJOCO_GL=egl \
    D4RL_SUPPRESS_IMPORT_ERROR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${ROOT}/third_party/diffuser-maze2d" \
    "${PY}" "${MAIN_SCRIPT}" \
      --logdir "${run_dir}" \
      "${COMMON_ARGS[@]}" \
      "$@"
  ) > "${run_log}" 2>&1 &
  pid=$!

  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${MONITOR_EVERY_SEC}"
    snapshot_metrics "${run_dir}" "${name}"
  done

  wait "${pid}"
  status=$?
  set -e

  t1="$(date +%s)"
  elapsed=$(( t1 - t0 ))
  log "done name=${name} status=${status} elapsed_sec=${elapsed}"
  snapshot_metrics "${run_dir}" "${name}-final"

  if [[ ${status} -ne 0 ]]; then
    log "failure name=${name} see_log=${run_log}"
    return ${status}
  fi
}

choose_best() {
  local names=("$@")
  local best_name=""
  local best_val="-1"
  local v
  for n in "${names[@]}"; do
    v="$(extract_last_h256 "${BASE_DIR}/${n}")"
    log "score name=${n} succ_h256=${v}"
    if "${PY}" - <<'PY' "${v}" "${best_val}"
import math, sys
v = float(sys.argv[1]) if sys.argv[1] not in {"nan", ""} else float('nan')
b = float(sys.argv[2])
raise SystemExit(0 if (not math.isnan(v) and v > b) else 1)
PY
    then
      best_name="${n}"
      best_val="${v}"
    fi
  done
  if [[ -z "${best_name}" ]]; then
    best_name="${names[0]}"
  fi
  echo "${best_name}"
}

log "start_ts=${STAMP}"
log "base_dir=${BASE_DIR}"

if (( $(remaining_budget_sec) < 10800 )); then
  log "remaining budget too short for full 5-step protocol; exiting"
  exit 0
fi

# Step 1: instrumentation sanity run.
run_case "S0_sanity" \
  --train_steps 1000 \
  --online_rounds 2 \
  --online_collect_episodes_per_round 8 \
  --online_train_steps_per_round 500 \
  --online_replan_every_n_steps 8

if ! validate_new_columns "${BASE_DIR}/S0_sanity" > /tmp/s0_validate.txt 2>&1; then
  log "sanity_column_check_failed detail=$(cat /tmp/s0_validate.txt)"
  log "stopping to avoid misconfigured overnight spend"
  exit 2
fi
log "sanity_column_check=ok"

if (( $(remaining_budget_sec) < 9000 )); then
  log "budget low after sanity; stopping"
  exit 0
fi

# Step 2: Phase A (collection chunk / update cadence at matched totals).
run_case "A1_chunk16_r12_t750" \
  --train_steps 3000 \
  --online_rounds 12 \
  --online_collect_episodes_per_round 16 \
  --online_train_steps_per_round 750 \
  --online_replan_every_n_steps 8

run_case "A2_chunk32_r6_t1500" \
  --train_steps 3000 \
  --online_rounds 6 \
  --online_collect_episodes_per_round 32 \
  --online_train_steps_per_round 1500 \
  --online_replan_every_n_steps 8

run_case "A3_chunk64_r3_t3000" \
  --train_steps 3000 \
  --online_rounds 3 \
  --online_collect_episodes_per_round 64 \
  --online_train_steps_per_round 3000 \
  --online_replan_every_n_steps 8

BEST_A="$(choose_best A1_chunk16_r12_t750 A2_chunk32_r6_t1500 A3_chunk64_r3_t3000)"
log "best_phase_A=${BEST_A}"

if (( $(remaining_budget_sec) < 6300 )); then
  log "budget low after phase A; stopping"
  exit 0
fi

# Step 3: Phase B (update-intensity ablation anchored on best A cadence).
case "${BEST_A}" in
  A1_chunk16_r12_t750)
    B_COLLECT=16; B_ROUNDS=12; B_BASE_T=750 ;;
  A2_chunk32_r6_t1500)
    B_COLLECT=32; B_ROUNDS=6; B_BASE_T=1500 ;;
  *)
    B_COLLECT=64; B_ROUNDS=3; B_BASE_T=3000 ;;
esac

B_LOW_T=$(( B_BASE_T / 2 ))
if (( B_LOW_T < 250 )); then B_LOW_T=250; fi
B_HIGH_T=$(( B_BASE_T * 2 ))

run_case "B1_low_update_t${B_LOW_T}" \
  --train_steps 3000 \
  --online_rounds "${B_ROUNDS}" \
  --online_collect_episodes_per_round "${B_COLLECT}" \
  --online_train_steps_per_round "${B_LOW_T}" \
  --online_replan_every_n_steps 8

run_case "B2_high_update_t${B_HIGH_T}" \
  --train_steps 3000 \
  --online_rounds "${B_ROUNDS}" \
  --online_collect_episodes_per_round "${B_COLLECT}" \
  --online_train_steps_per_round "${B_HIGH_T}" \
  --online_replan_every_n_steps 8

BEST_AB="$(choose_best "${BEST_A}" "B1_low_update_t${B_LOW_T}" "B2_high_update_t${B_HIGH_T}")"
log "best_phase_AB=${BEST_AB}"

if (( $(remaining_budget_sec) < 3600 )); then
  log "budget low after phase B; stopping"
  exit 0
fi

# Step 4: Phase C replanning-step ablation (user-priority).
run_case "C1_replan4" \
  --train_steps 3000 \
  --online_rounds "${B_ROUNDS}" \
  --online_collect_episodes_per_round "${B_COLLECT}" \
  --online_train_steps_per_round "${B_BASE_T}" \
  --online_replan_every_n_steps 4

run_case "C2_replan8" \
  --train_steps 3000 \
  --online_rounds "${B_ROUNDS}" \
  --online_collect_episodes_per_round "${B_COLLECT}" \
  --online_train_steps_per_round "${B_BASE_T}" \
  --online_replan_every_n_steps 8

run_case "C3_replan16" \
  --train_steps 3000 \
  --online_rounds "${B_ROUNDS}" \
  --online_collect_episodes_per_round "${B_COLLECT}" \
  --online_train_steps_per_round "${B_BASE_T}" \
  --online_replan_every_n_steps 16

BEST_C="$(choose_best C1_replan4 C2_replan8 C3_replan16)"
log "best_phase_C=${BEST_C}"

# Step 5: Reflection summary table.
"${PY}" - <<'PY' "${BASE_DIR}" | tee -a "${DRIVER_LOG}"
import json
import math
import os
import sys
import pandas as pd

base = sys.argv[1]
rows = []
for name in sorted(os.listdir(base)):
    run_dir = os.path.join(base, name)
    if not os.path.isdir(run_dir):
        continue
    p = os.path.join(run_dir, "progress_metrics.csv")
    o = os.path.join(run_dir, "online_collection.csv")
    if not os.path.exists(p):
        continue
    try:
        dfp = pd.read_csv(p)
    except Exception:
        continue
    if len(dfp) == 0:
        continue
    r = dfp.iloc[-1]
    entry = {
        "run": name,
        "step": int(r.get("step", -1)),
        "succ_h64": float(r.get("rollout_goal_success_rate_h64", float('nan'))),
        "succ_h128": float(r.get("rollout_goal_success_rate_h128", float('nan'))),
        "succ_h192": float(r.get("rollout_goal_success_rate_h192", float('nan'))),
        "succ_h256": float(r.get("rollout_goal_success_rate_h256", r.get("rollout_goal_success_rate", float('nan')))),
    }
    if os.path.exists(o):
        try:
            dfo = pd.read_csv(o)
            if len(dfo) > 0:
                ro = dfo.iloc[-1]
                entry["plan_t010"] = float(ro.get("planning_success_rate_final_t010", float('nan')))
                entry["plan_t020"] = float(ro.get("planning_success_rate_final_t020", float('nan')))
                entry["plan_rel090"] = float(ro.get("planning_success_rate_final_rel090", ro.get("planning_success_rate_final_rel_reduction", float('nan'))))
        except Exception:
            pass
    rows.append(entry)

out_csv = os.path.join(base, "five_step_summary.csv")
pd.DataFrame(rows).sort_values("succ_h256", ascending=False).to_csv(out_csv, index=False)
print(f"[summary] path={out_csv} n_runs={len(rows)}")
PY

log "completed all scheduled phases"
log "summary_csv=${BASE_DIR}/five_step_summary.csv"
