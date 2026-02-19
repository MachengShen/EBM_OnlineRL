#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "usage: overnight_online_maze2d_driver.sh"
  echo ""
  echo "Runs a fixed 12h sequence of online Maze2D Diffuser runs and writes artifacts under:"
  echo "  runs/analysis/synth_maze2d_diffuser_probe/overnight_<timestamp>/"
  echo ""
  echo "This script currently does not accept CLI flags; edit the file to change the schedule."
  exit 0
fi

ROOT="/root/ebm-online-rl-prototype"
PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
MAIN_SCRIPT="${ROOT}/scripts/synthetic_maze2d_diffuser_probe.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
BASE_DIR="${ROOT}/runs/analysis/synth_maze2d_diffuser_probe/overnight_${STAMP}"
mkdir -p "${BASE_DIR}"
DRIVER_LOG="${BASE_DIR}/driver.log"

LD_MJ="/root/.mujoco/mujoco210/bin"

echo "[driver] start_ts=${STAMP}" | tee -a "${DRIVER_LOG}"
echo "[driver] base_dir=${BASE_DIR}" | tee -a "${DRIVER_LOG}"

BUDGET_SEC=$((12 * 3600))
START_SEC="$(date +%s)"

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
  --eval_goal_every 5000
  --save_checkpoint_every 5000
  --query_mode diverse
  --num_eval_queries 24
  --query_batch_size 6
  --online_self_improve
  --online_collect_episodes_per_round 32
  --online_collect_episode_len 256
  --online_goal_geom_min_k 8
  --online_goal_geom_max_k 96
)

remaining_budget_sec() {
  local now
  now="$(date +%s)"
  echo $(( BUDGET_SEC - (now - START_SEC) ))
}

extract_last_rollout_success() {
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
    if len(df) == 0 or "rollout_goal_success_rate" not in df.columns:
        print("nan")
    else:
        v = float(df["rollout_goal_success_rate"].dropna().iloc[-1]) if df["rollout_goal_success_rate"].notna().any() else float("nan")
        print(v)
except Exception:
    print("nan")
PY
}

run_case() {
  local name="$1"; shift
  local run_dir="${BASE_DIR}/${name}"
  mkdir -p "${run_dir}"
  local run_log="${run_dir}/run.log"
  local t0 t1 status elapsed

  t0="$(date +%s)"
  echo "[driver] launch name=${name} run_dir=${run_dir}" | tee -a "${DRIVER_LOG}"
  echo "[driver] remaining_budget_sec=$(remaining_budget_sec)" | tee -a "${DRIVER_LOG}"

  set +e
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${LD_MJ}" \
  MUJOCO_GL=egl \
  D4RL_SUPPRESS_IMPORT_ERROR=1 \
  PYTHONPATH="${ROOT}/third_party/diffuser-maze2d" \
  "${PY}" "${MAIN_SCRIPT}" \
    --logdir "${run_dir}" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    > "${run_log}" 2>&1
  status=$?
  set -e

  t1="$(date +%s)"
  elapsed=$(( t1 - t0 ))
  echo "[driver] done name=${name} status=${status} elapsed_sec=${elapsed}" | tee -a "${DRIVER_LOG}"
  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[driver] summary_path=${run_dir}/summary.json" | tee -a "${DRIVER_LOG}"
  fi
}

if (( $(remaining_budget_sec) < 7200 )); then
  echo "[driver] remaining budget too short, exiting early." | tee -a "${DRIVER_LOG}"
  exit 0
fi

# Run A: baseline 40k-update equivalent schedule.
run_case "A_baseline_rh8_p008" \
  --train_steps 10000 \
  --online_rounds 15 \
  --online_train_steps_per_round 2000 \
  --online_replan_every_n_steps 8 \
  --online_goal_geom_p 0.08 \
  --online_goal_min_distance 0.5

A_DIR="${BASE_DIR}/A_baseline_rh8_p008"
A_SUCC="$(extract_last_rollout_success "${A_DIR}")"
echo "[driver] A_last_rollout_success=${A_SUCC}" | tee -a "${DRIVER_LOG}"

if (( $(remaining_budget_sec) < 5400 )); then
  echo "[driver] budget low after A, stopping." | tee -a "${DRIVER_LOG}"
  exit 0
fi

# Run B: adapt goal-difficulty curriculum based on run A outcome.
if "${PY}" - <<'PY' "${A_SUCC}"
import sys, math
v = float(sys.argv[1]) if sys.argv[1] not in {"nan", ""} else float("nan")
print("1" if (not math.isnan(v) and v >= 0.45) else "0")
PY
then
  :
fi
B_MODE="$("${PY}" - <<'PY' "${A_SUCC}"
import sys, math
v = float(sys.argv[1]) if sys.argv[1] not in {"nan", ""} else float("nan")
print("harder" if (not math.isnan(v) and v >= 0.45) else "easier")
PY
)"
echo "[driver] B_mode=${B_MODE}" | tee -a "${DRIVER_LOG}"

if [[ "${B_MODE}" == "harder" ]]; then
  run_case "B_harder_goals_rh8" \
    --train_steps 8000 \
    --online_rounds 10 \
    --online_train_steps_per_round 2000 \
    --online_replan_every_n_steps 8 \
    --online_goal_geom_p 0.05 \
    --online_goal_min_distance 0.8
else
  run_case "B_easier_goals_rh8" \
    --train_steps 8000 \
    --online_rounds 10 \
    --online_train_steps_per_round 2000 \
    --online_replan_every_n_steps 8 \
    --online_goal_geom_p 0.12 \
    --online_goal_min_distance 0.4
fi

if (( $(remaining_budget_sec) < 5400 )); then
  echo "[driver] budget low after B, stopping." | tee -a "${DRIVER_LOG}"
  exit 0
fi

# Run C: replanning-cadence ablation (compute vs control quality).
if [[ "${B_MODE}" == "harder" ]]; then
  run_case "C_replan4" \
    --train_steps 8000 \
    --online_rounds 10 \
    --online_train_steps_per_round 2000 \
    --online_replan_every_n_steps 4 \
    --online_goal_geom_p 0.08 \
    --online_goal_min_distance 0.5
else
  run_case "C_replan16" \
    --train_steps 8000 \
    --online_rounds 10 \
    --online_train_steps_per_round 2000 \
    --online_replan_every_n_steps 16 \
    --online_goal_geom_p 0.08 \
    --online_goal_min_distance 0.5
fi

echo "[driver] completed all scheduled runs." | tee -a "${DRIVER_LOG}"
