#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/ebm-online-rl-prototype"
PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
SCRIPT="${ROOT}/scripts/agentic_maze2d_autodecider.py"

STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION="maze2d_agentic_${STAMP}"

BUDGET_HOURS="${BUDGET_HOURS:-6}"
MAX_AGENT_ROUNDS="${MAX_AGENT_ROUNDS:-8}"
PROPOSALS_PER_ROUND="${PROPOSALS_PER_ROUND:-4}"
PROPOSAL_SOURCE="${PROPOSAL_SOURCE:-external}"
PROPOSAL_WAIT_TIMEOUT_SEC="${PROPOSAL_WAIT_TIMEOUT_SEC:-1800}"
PROPOSAL_POLL_SEC="${PROPOSAL_POLL_SEC:-10}"
MONITOR_EVERY_SEC="${MONITOR_EVERY_SEC:-120}"
PER_TRIAL_TIMEOUT_MIN="${PER_TRIAL_TIMEOUT_MIN:-35}"
SEED="${SEED:-0}"
SMOKE="${SMOKE:-0}"
BASE_DIR="${BASE_DIR:-}"

LOG_DIR="${ROOT}/runs/analysis/synth_maze2d_diffuser_probe"
mkdir -p "${LOG_DIR}"
TMUX_LOG="${LOG_DIR}/tmux_${SESSION}.log"
PROPOSAL_DIR="${PROPOSAL_DIR:-${LOG_DIR}/proposal_exchange_${SESSION}}"
mkdir -p "${PROPOSAL_DIR}"

CMD=(
  "cd '${ROOT}' && '${PY}' '${SCRIPT}'"
  "--budget-hours '${BUDGET_HOURS}'"
  "--max-agent-rounds '${MAX_AGENT_ROUNDS}'"
  "--proposals-per-round '${PROPOSALS_PER_ROUND}'"
  "--proposal-source '${PROPOSAL_SOURCE}'"
  "--proposal-dir '${PROPOSAL_DIR}'"
  "--proposal-wait-timeout-sec '${PROPOSAL_WAIT_TIMEOUT_SEC}'"
  "--proposal-poll-sec '${PROPOSAL_POLL_SEC}'"
  "--monitor-every-sec '${MONITOR_EVERY_SEC}'"
  "--per-trial-timeout-min '${PER_TRIAL_TIMEOUT_MIN}'"
  "--seed '${SEED}'"
)
if [[ -n "${BASE_DIR}" ]]; then
  CMD+=("--base-dir '${BASE_DIR}'")
fi
if [[ "${SMOKE}" == "1" ]]; then
  CMD+=("--smoke")
fi

tmux new-session -d -s "${SESSION}" \
  "${CMD[*]} 2>&1 | tee -a '${TMUX_LOG}'"

echo "started tmux session: ${SESSION}"
echo "tmux log: ${TMUX_LOG}"
echo "proposal source: ${PROPOSAL_SOURCE}"
echo "proposal dir: ${PROPOSAL_DIR}"
echo "external proposal file pattern: ${PROPOSAL_DIR}/round_###_proposals.json"
echo "attach: tmux attach -t ${SESSION}"
