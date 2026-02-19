#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/ebm-online-rl-prototype"
PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
SCRIPT="${ROOT}/scripts/agentic_role_orchestrator.py"

STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION="maze2d_role_orch_${STAMP}"

BUDGET_HOURS="${BUDGET_HOURS:-6}"
MAX_AGENT_ROUNDS="${MAX_AGENT_ROUNDS:-8}"
PROPOSALS_PER_ROUND="${PROPOSALS_PER_ROUND:-4}"
PROPOSAL_SOURCE="${PROPOSAL_SOURCE:-external}"
PROPOSAL_WAIT_TIMEOUT_SEC="${PROPOSAL_WAIT_TIMEOUT_SEC:-1800}"
PROPOSAL_POLL_SEC="${PROPOSAL_POLL_SEC:-10}"
PROPOSER_POLL_SEC="${PROPOSER_POLL_SEC:-2}"
REVIEWER_POLL_SEC="${REVIEWER_POLL_SEC:-20}"
MONITOR_EVERY_SEC="${MONITOR_EVERY_SEC:-120}"
PER_TRIAL_TIMEOUT_MIN="${PER_TRIAL_TIMEOUT_MIN:-35}"
ACCEPT_DELTA="${ACCEPT_DELTA:-0.02}"
SEED="${SEED:-0}"
SMOKE="${SMOKE:-0}"

LOG_DIR="${ROOT}/runs/analysis/synth_maze2d_diffuser_probe"
mkdir -p "${LOG_DIR}"
TMUX_LOG="${LOG_DIR}/tmux_${SESSION}.log"
ORCH_DIR="${ORCH_DIR:-${LOG_DIR}/orchestrator_${SESSION}}"
PROPOSAL_DIR="${PROPOSAL_DIR:-${ORCH_DIR}/proposal_exchange}"
mkdir -p "${ORCH_DIR}" "${PROPOSAL_DIR}"

CMD=(
  "cd '${ROOT}' && '${PY}' '${SCRIPT}'"
  "--root '${ROOT}'"
  "--python '${PY}'"
  "--orchestrator-dir '${ORCH_DIR}'"
  "--proposal-dir '${PROPOSAL_DIR}'"
  "--budget-hours '${BUDGET_HOURS}'"
  "--max-agent-rounds '${MAX_AGENT_ROUNDS}'"
  "--proposals-per-round '${PROPOSALS_PER_ROUND}'"
  "--proposal-source '${PROPOSAL_SOURCE}'"
  "--proposal-wait-timeout-sec '${PROPOSAL_WAIT_TIMEOUT_SEC}'"
  "--proposal-poll-sec '${PROPOSAL_POLL_SEC}'"
  "--proposer-poll-sec '${PROPOSER_POLL_SEC}'"
  "--reviewer-poll-sec '${REVIEWER_POLL_SEC}'"
  "--monitor-every-sec '${MONITOR_EVERY_SEC}'"
  "--per-trial-timeout-min '${PER_TRIAL_TIMEOUT_MIN}'"
  "--accept-delta '${ACCEPT_DELTA}'"
  "--seed '${SEED}'"
)
if [[ "${SMOKE}" == "1" ]]; then
  CMD+=("--smoke")
fi

tmux new-session -d -s "${SESSION}" \
  "${CMD[*]} 2>&1 | tee -a '${TMUX_LOG}'"

echo "started tmux session: ${SESSION}"
echo "tmux log: ${TMUX_LOG}"
echo "orchestrator dir: ${ORCH_DIR}"
echo "proposal dir: ${PROPOSAL_DIR}"
echo "external proposal file pattern: ${PROPOSAL_DIR}/round_###_proposals.json"
echo "attach: tmux attach -t ${SESSION}"

