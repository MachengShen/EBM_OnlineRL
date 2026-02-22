#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_ID="maze2d-umaze-v1"
SEEDS=("0" "1" "2")
DEVICE="cuda:0"
SMOKE=0
BASE_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_ID="$2"
      shift 2
      ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--env maze2d-umaze-v1] [--seeds 0,1,2] [--device cuda:0] [--base-dir PATH] [--smoke]"
      exit 2
      ;;
  esac
done

if [[ -z "$BASE_DIR" ]]; then
  BASE_DIR="$ROOT/runs/analysis/eqnet_vs_unet/eqnet_vs_unet_$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$BASE_DIR"

PYTHON="${PYTHON:-/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON"
  exit 2
fi

export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin"
else
  export LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"
fi
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="/root/ebm-online-rl-prototype/third_party/diffuser-maze2d"
else
  export PYTHONPATH="/root/ebm-online-rl-prototype/third_party/diffuser-maze2d:${PYTHONPATH}"
fi

if [[ -d "$ROOT/third_party_external/diffusion-stitching/.git" ]]; then
  git -C "$ROOT/third_party_external/diffusion-stitching" rev-parse HEAD > "$BASE_DIR/diffusion_stitching_commit.txt"
fi

if [[ "$SMOKE" -eq 1 ]]; then
  COMMON_ARGS=(
    --n_episodes 8
    --episode_len 32
    --horizon 16
    --train_steps 50
    --batch_size 16
    --online_self_improve
    --online_rounds 1
    --online_collect_episodes_per_round 2
    --online_collect_episode_len 32
    --online_train_steps_per_round 20
    --online_replan_every_n_steps 4
    --num_eval_queries 4
    --query_bank_size 32
    --query_batch_size 1
    --query_min_distance 0.3
    --eval_goal_every 10
    --eval_rollout_horizon 32
    --eval_rollout_replan_every_n_steps 4
    --eval_success_prefix_horizons 8,16,32
    --save_checkpoint_every 0
    --model_dim 32
    --model_dim_mults 1,2
    --n_diffusion_steps 16
    --device cpu
  )
else
  COMMON_ARGS=(
    --n_episodes 400
    --episode_len 256
    --horizon 64
    --train_steps 6000
    --batch_size 128
    --online_self_improve
    --online_rounds 4
    --online_collect_episodes_per_round 64
    --online_collect_episode_len 256
    --online_train_steps_per_round 3000
    --online_replan_every_n_steps 16
    --num_eval_queries 12
    --query_bank_size 256
    --query_batch_size 1
    --query_min_distance 1.0
    --eval_goal_every 3000
    --eval_rollout_horizon 256
    --eval_rollout_replan_every_n_steps 16
    --eval_success_prefix_horizons 64,128,192,256
    --save_checkpoint_every 5000
    --model_dim 64
    --model_dim_mults 1,2,4
    --n_diffusion_steps 64
    --device "$DEVICE"
  )
fi

echo "Launching EqNet vs UNet ablation"
echo "  base_dir: $BASE_DIR"
echo "  env:      $ENV_ID"
echo "  seeds:    ${SEEDS[*]}"
echo "  smoke:    $SMOKE"

for ARCH in unet eqnet; do
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$BASE_DIR/$ARCH/seed_$SEED"
    mkdir -p "$RUN_DIR"
    LOG="$RUN_DIR/stdout_stderr.log"
    CMD=(
      "$PYTHON" "$ROOT/scripts/synthetic_maze2d_diffuser_probe.py"
      --env "$ENV_ID"
      --seed "$SEED"
      --logdir "$RUN_DIR"
      --query_mode diverse
      --goal_success_threshold 0.2
      --denoiser_arch "$ARCH"
      "${COMMON_ARGS[@]}"
    )
    echo
    echo "[run] arch=$ARCH seed=$SEED"
    echo "[cmd] ${CMD[*]}"
    "${CMD[@]}" 2>&1 | tee "$LOG"
  done
done

python3 "$ROOT/scripts/analyze_ablation_eqnet_vs_unet.py" --base-dir "$BASE_DIR"
echo
echo "[done] Outputs:"
echo "  - $BASE_DIR/eqnet_vs_unet_rows.csv"
echo "  - $BASE_DIR/eqnet_vs_unet_summary.json"
echo "  - $BASE_DIR/eqnet_vs_unet_summary.md"
if [[ -f "$BASE_DIR/eqnet_vs_unet_success_curve.png" ]]; then
  echo "  - $BASE_DIR/eqnet_vs_unet_success_curve.png"
fi
