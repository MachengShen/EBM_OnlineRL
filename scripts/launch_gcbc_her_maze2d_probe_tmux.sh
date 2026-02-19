#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/ebm-online-rl-prototype"
PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
SCRIPT="${ROOT}/scripts/synthetic_maze2d_gcbc_her_probe.py"
LD_MJ="/root/.mujoco/mujoco210/bin"

STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION="maze2d_gcbc_her_${STAMP}"

SMOKE="${SMOKE:-0}"
SEED="${SEED:-0}"
MONITOR_TAG="${MONITOR_TAG:-gcbc_her}"

# Fair-comparison defaults (aligned to current diffuser auto-decider baseline family).
N_EPISODES="${N_EPISODES:-1000}"
EPISODE_LEN="${EPISODE_LEN:-256}"
HORIZON="${HORIZON:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAL_EVERY="${VAL_EVERY:-500}"
VAL_BATCHES="${VAL_BATCHES:-20}"
EVAL_GOAL_EVERY="${EVAL_GOAL_EVERY:-1000}"
SAVE_CHECKPOINT_EVERY="${SAVE_CHECKPOINT_EVERY:-5000}"

ONLINE_SELF_IMPROVE="${ONLINE_SELF_IMPROVE:-1}"
ONLINE_ROUNDS="${ONLINE_ROUNDS:-4}"
ONLINE_TRAIN_STEPS_PER_ROUND="${ONLINE_TRAIN_STEPS_PER_ROUND:-3000}"
ONLINE_COLLECT_TRANSITION_BUDGET_PER_ROUND="${ONLINE_COLLECT_TRANSITION_BUDGET_PER_ROUND:-4096}"
ONLINE_COLLECT_EPISODES_PER_ROUND="${ONLINE_COLLECT_EPISODES_PER_ROUND:-64}"
ONLINE_COLLECT_EPISODE_LEN="${ONLINE_COLLECT_EPISODE_LEN:-256}"
ONLINE_REPLAN_EVERY_N_STEPS="${ONLINE_REPLAN_EVERY_N_STEPS:-16}"
ONLINE_GOAL_GEOM_P="${ONLINE_GOAL_GEOM_P:-0.04}"
ONLINE_GOAL_GEOM_MIN_K="${ONLINE_GOAL_GEOM_MIN_K:-64}"
ONLINE_GOAL_GEOM_MAX_K="${ONLINE_GOAL_GEOM_MAX_K:-192}"
ONLINE_GOAL_MIN_DISTANCE="${ONLINE_GOAL_MIN_DISTANCE:-1.0}"
ONLINE_EARLY_TERMINATE_THRESHOLD="${ONLINE_EARLY_TERMINATE_THRESHOLD:-0.2}"
ONLINE_MIN_ACCEPTED_EPISODE_LEN="${ONLINE_MIN_ACCEPTED_EPISODE_LEN:-64}"

QUERY_MODE="${QUERY_MODE:-diverse}"
NUM_EVAL_QUERIES="${NUM_EVAL_QUERIES:-12}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-1}"
QUERY_MIN_DISTANCE="${QUERY_MIN_DISTANCE:-1.0}"
GOAL_SUCCESS_THRESHOLD="${GOAL_SUCCESS_THRESHOLD:-0.2}"
EVAL_ROLLOUT_MODE="${EVAL_ROLLOUT_MODE:-receding_horizon}"
EVAL_ROLLOUT_REPLAN_EVERY_N_STEPS="${EVAL_ROLLOUT_REPLAN_EVERY_N_STEPS:-16}"
EVAL_ROLLOUT_HORIZON="${EVAL_ROLLOUT_HORIZON:-256}"
EVAL_SUCCESS_PREFIX_HORIZONS="${EVAL_SUCCESS_PREFIX_HORIZONS:-64,128,192,256}"

GCBC_HIDDEN_DIMS="${GCBC_HIDDEN_DIMS:-256,256}"
GCBC_HER_K_PER_TRANSITION="${GCBC_HER_K_PER_TRANSITION:-4}"
GCBC_FUTURE_SAMPLE_ATTEMPTS="${GCBC_FUTURE_SAMPLE_ATTEMPTS:-16}"

if [[ "${SMOKE}" == "1" ]]; then
  N_EPISODES=4
  EPISODE_LEN=32
  HORIZON=16
  TRAIN_STEPS=40
  BATCH_SIZE=16
  VAL_EVERY=10
  VAL_BATCHES=1
  EVAL_GOAL_EVERY=10
  SAVE_CHECKPOINT_EVERY=0
  ONLINE_ROUNDS=1
  ONLINE_TRAIN_STEPS_PER_ROUND=20
  ONLINE_COLLECT_TRANSITION_BUDGET_PER_ROUND=32
  ONLINE_COLLECT_EPISODES_PER_ROUND=1
  ONLINE_COLLECT_EPISODE_LEN=32
  ONLINE_REPLAN_EVERY_N_STEPS=4
  ONLINE_GOAL_GEOM_P=0.08
  ONLINE_GOAL_GEOM_MIN_K=4
  ONLINE_GOAL_GEOM_MAX_K=16
  ONLINE_GOAL_MIN_DISTANCE=0.3
  ONLINE_EARLY_TERMINATE_THRESHOLD=0.1
  ONLINE_MIN_ACCEPTED_EPISODE_LEN=8
  NUM_EVAL_QUERIES=4
  QUERY_BATCH_SIZE=1
  EVAL_ROLLOUT_REPLAN_EVERY_N_STEPS=4
  EVAL_ROLLOUT_HORIZON=32
  EVAL_SUCCESS_PREFIX_HORIZONS="8,16,32"
fi

LOG_DIR="${ROOT}/runs/analysis/synth_maze2d_diffuser_probe"
mkdir -p "${LOG_DIR}"
TMUX_LOG="${LOG_DIR}/tmux_${SESSION}.log"

CMD=(
  "cd '${ROOT}' && LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:-}:${LD_MJ}\" MUJOCO_GL=egl D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONPATH=\"${ROOT}/third_party/diffuser-maze2d\" '${PY}' '${SCRIPT}'"
  "--seed '${SEED}'"
  "--n_episodes '${N_EPISODES}'"
  "--episode_len '${EPISODE_LEN}'"
  "--horizon '${HORIZON}'"
  "--train_steps '${TRAIN_STEPS}'"
  "--batch_size '${BATCH_SIZE}'"
  "--val_every '${VAL_EVERY}'"
  "--val_batches '${VAL_BATCHES}'"
  "--eval_goal_every '${EVAL_GOAL_EVERY}'"
  "--save_checkpoint_every '${SAVE_CHECKPOINT_EVERY}'"
  "--query_mode '${QUERY_MODE}'"
  "--num_eval_queries '${NUM_EVAL_QUERIES}'"
  "--query_batch_size '${QUERY_BATCH_SIZE}'"
  "--query_min_distance '${QUERY_MIN_DISTANCE}'"
  "--query_resample_each_eval"
  "--goal_success_threshold '${GOAL_SUCCESS_THRESHOLD}'"
  "--eval_rollout_mode '${EVAL_ROLLOUT_MODE}'"
  "--eval_rollout_replan_every_n_steps '${EVAL_ROLLOUT_REPLAN_EVERY_N_STEPS}'"
  "--eval_rollout_horizon '${EVAL_ROLLOUT_HORIZON}'"
  "--eval_success_prefix_horizons '${EVAL_SUCCESS_PREFIX_HORIZONS}'"
  "--online_rounds '${ONLINE_ROUNDS}'"
  "--online_train_steps_per_round '${ONLINE_TRAIN_STEPS_PER_ROUND}'"
  "--online_collect_transition_budget_per_round '${ONLINE_COLLECT_TRANSITION_BUDGET_PER_ROUND}'"
  "--online_collect_episodes_per_round '${ONLINE_COLLECT_EPISODES_PER_ROUND}'"
  "--online_collect_episode_len '${ONLINE_COLLECT_EPISODE_LEN}'"
  "--online_replan_every_n_steps '${ONLINE_REPLAN_EVERY_N_STEPS}'"
  "--online_goal_geom_p '${ONLINE_GOAL_GEOM_P}'"
  "--online_goal_geom_min_k '${ONLINE_GOAL_GEOM_MIN_K}'"
  "--online_goal_geom_max_k '${ONLINE_GOAL_GEOM_MAX_K}'"
  "--online_goal_min_distance '${ONLINE_GOAL_MIN_DISTANCE}'"
  "--online_early_terminate_threshold '${ONLINE_EARLY_TERMINATE_THRESHOLD}'"
  "--online_min_accepted_episode_len '${ONLINE_MIN_ACCEPTED_EPISODE_LEN}'"
  "--gcbc_hidden_dims '${GCBC_HIDDEN_DIMS}'"
  "--gcbc_her_k_per_transition '${GCBC_HER_K_PER_TRANSITION}'"
  "--gcbc_future_sample_attempts '${GCBC_FUTURE_SAMPLE_ATTEMPTS}'"
)

if [[ "${ONLINE_SELF_IMPROVE}" == "1" ]]; then
  CMD+=("--online_self_improve")
else
  CMD+=("--no_online_self_improve")
fi

tmux new-session -d -s "${SESSION}" \
  "${CMD[*]} 2>&1 | tee -a '${TMUX_LOG}'"

echo "started tmux session: ${SESSION}"
echo "tmux log: ${TMUX_LOG}"
echo "tag: ${MONITOR_TAG}"
echo "attach: tmux attach -t ${SESSION}"
