#!/bin/bash
# Launch all experiments sequentially on one GPU.
# 1) Maze2d medium swap matrix
# 2) Maze2d large swap matrix
# 3) Locomotion online Diffuser swap matrix (hopper + walker2d)

set -euo pipefail

REPO=/root/ebm-online-rl-prototype
PYTHON=$REPO/third_party/diffuser/.venv38/bin/python3
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1

echo "=== Experiment batch started at $(date) ==="
echo "=== Timestamp: $TIMESTAMP ==="

# ---- 1. Maze2d Medium Swap Matrix ----
echo ""
echo "============================================================"
echo "  PHASE 1: Maze2d Medium Swap Matrix"
echo "============================================================"
MAZE_MEDIUM_DIR=$REPO/runs/analysis/swap_matrix/maze2d_medium_$TIMESTAMP

$PYTHON $REPO/scripts/exp_swap_matrix_maze2d.py \
    --env maze2d-medium-v1 \
    --base-dir "$MAZE_MEDIUM_DIR" \
    --seeds 0,1,2 \
    --device cuda:0 \
    2>&1 | tee "$REPO/runs/analysis/swap_matrix/maze2d_medium_${TIMESTAMP}.log"

echo "[DONE] Maze2d Medium at $(date)"

# ---- 2. Maze2d Large Swap Matrix ----
echo ""
echo "============================================================"
echo "  PHASE 2: Maze2d Large Swap Matrix"
echo "============================================================"
MAZE_LARGE_DIR=$REPO/runs/analysis/swap_matrix/maze2d_large_$TIMESTAMP

$PYTHON $REPO/scripts/exp_swap_matrix_maze2d.py \
    --env maze2d-large-v1 \
    --base-dir "$MAZE_LARGE_DIR" \
    --seeds 0,1,2 \
    --device cuda:0 \
    2>&1 | tee "$REPO/runs/analysis/swap_matrix/maze2d_large_${TIMESTAMP}.log"

echo "[DONE] Maze2d Large at $(date)"

# ---- 3. Locomotion Online Diffuser Swap Matrix ----
echo ""
echo "============================================================"
echo "  PHASE 3: Locomotion Online Diffuser (Hopper + Walker2d)"
echo "============================================================"
LOCO_DIR=$REPO/runs/analysis/locomotion_collector/loco_swap_$TIMESTAMP

$PYTHON $REPO/scripts/exp_locomotion_collector_study.py \
    --envs "hopper-medium-expert-v2,walker2d-medium-expert-v2" \
    --conditions "diffuser_online,sac_scratch,sac_collects_diffuser_learns,diffuser_collects_sac_learns" \
    --seeds 0,1,2 \
    --device cuda:0 \
    --out-dir "$LOCO_DIR" \
    --n-online-episodes 100 \
    --max-episode-steps 1000 \
    --eval-every-episodes 20 \
    --n-eval-episodes 5 \
    --diffuser-train-steps-per-round 1000 \
    --diffuser-train-batch-size 32 \
    --collect-per-round 10 \
    2>&1 | tee "$REPO/runs/analysis/locomotion_collector/loco_swap_${TIMESTAMP}.log"

echo "[DONE] Locomotion at $(date)"

echo ""
echo "=== ALL EXPERIMENTS COMPLETED at $(date) ==="
