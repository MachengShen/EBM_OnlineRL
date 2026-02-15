#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/ebm-online-rl-prototype"
PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
SCRIPT="${ROOT}/scripts/overnight_maze2d_autodecider.py"

STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION="maze2d_autodecider_${STAMP}"

BUDGET_HOURS="${BUDGET_HOURS:-12}"
MAX_TRIALS="${MAX_TRIALS:-12}"
MONITOR_EVERY_SEC="${MONITOR_EVERY_SEC:-300}"
SEED="${SEED:-0}"

LOG_DIR="${ROOT}/runs/analysis/synth_maze2d_diffuser_probe"
mkdir -p "${LOG_DIR}"
TMUX_LOG="${LOG_DIR}/tmux_${SESSION}.log"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && '${PY}' '${SCRIPT}' --budget-hours '${BUDGET_HOURS}' --max-trials '${MAX_TRIALS}' --monitor-every-sec '${MONITOR_EVERY_SEC}' --seed '${SEED}' 2>&1 | tee -a '${TMUX_LOG}'"

echo "started tmux session: ${SESSION}"
echo "tmux log: ${TMUX_LOG}"
echo "attach: tmux attach -t ${SESSION}"

