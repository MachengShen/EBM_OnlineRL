#!/usr/bin/env bash
set -euo pipefail

# Wait for EqNet ablation to finish, then resume paused experiments.
# Uses the relay job exit_code file as the completion signal.

EQNET_EXIT_CODE="/root/.codex-discord-relay/jobs/discord:1472061022239195304:thread:1473203408256368795/j-20260222-195504-8810/exit_code"
REPO="/root/ebm-online-rl-prototype"
PYTHON="third_party/diffuser/.venv38/bin/python3"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

echo "[resume] $(date -Is) Waiting for EqNet ablation to finish..."
echo "[resume] Polling: $EQNET_EXIT_CODE"

while [ ! -f "$EQNET_EXIT_CODE" ]; do
    sleep 60
done

EQNET_RC=$(cat "$EQNET_EXIT_CODE")
echo "[resume] $(date -Is) EqNet ablation finished with exit code: $EQNET_RC"

# Kill the EqNet main-channel poster if still alive
POSTER_PID_FILE="$REPO/.worktrees/eqnet-maze2d/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_3seed_20260222-195504/eqnet_main_channel_monitor.pid"
if [ -f "$POSTER_PID_FILE" ]; then
    POSTER_PID=$(cat "$POSTER_PID_FILE")
    kill "$POSTER_PID" 2>/dev/null && echo "[resume] Killed EqNet poster (PID $POSTER_PID)" || echo "[resume] EqNet poster already gone"
fi

# Brief pause to let GPU memory free up
sleep 10

echo ""
echo "================================================================"
echo "[resume] $(date -Is) Phase 1: maze2d-medium swap matrix (resume)"
echo "================================================================"
cd "$REPO"
$PYTHON scripts/exp_swap_matrix_maze2d.py \
    --env maze2d-medium-v1 \
    --base-dir runs/analysis/swap_matrix/maze2d_medium_20260222-145304 \
    --seeds 0,1,2 --device cuda:0

echo ""
echo "================================================================"
echo "[resume] $(date -Is) Phase 2: maze2d-large swap matrix"
echo "================================================================"
$PYTHON scripts/exp_swap_matrix_maze2d.py \
    --env maze2d-large-v1 \
    --base-dir runs/analysis/swap_matrix/maze2d_large_20260222-145304 \
    --seeds 0,1,2 --device cuda:0

echo ""
echo "================================================================"
echo "[resume] $(date -Is) Phase 3: Locomotion swap matrix"
echo "================================================================"
$PYTHON scripts/exp_locomotion_collector_study.py \
    --base-dir runs/analysis/locomotion_collector/loco_swap_20260222-145304 \
    --seeds 0,1,2 --device cuda:0

echo ""
echo "[resume] $(date -Is) All phases complete."
