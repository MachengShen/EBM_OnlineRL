# EBM Online RL Prototype

Minimal online goal-reaching prototype with diffusion planning in a 2D point-mass environment.

## What is implemented
- `PointMass2D` continuous environment (`s in [-1,1]^2`, `a in [-0.1,0.1]^2`, episode len 50).
- Replay buffer storing full episodes and sampling packed trajectory segments `[B, H+1, obs+act]`.
- Diffusion model over trajectories with Diffuser-style inpainting:
  - clamp start state `s0` and terminal state `sH` during every reverse denoising step.
- Online loop:
  - warmup random collection,
  - train diffusion model on replay,
  - MPC-style planning from diffusion model for data collection,
  - periodic evaluation on uniformly sampled goals.
- Logging + artifacts:
  - `metrics.jsonl`,
  - checkpoints in `checkpoints/`,
  - `success_rate.png`.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
pip install -r requirements.txt
```

## Run (smoke test)
```bash
python scripts/online_pointmass_goal_diffuser.py \
  --device cuda:0 \
  --total_env_steps 2000 \
  --warmup_steps 500 \
  --train_every 500 \
  --gradient_steps 20 \
  --batch_size 32 \
  --n_diffusion_steps 8 \
  --model_base_dim 16 \
  --model_dim_mults 1,2 \
  --eval_every 1000 \
  --n_eval_episodes 20
```

CPU execution is allowed (for example, `--device cpu`) but will be slow due to diffusion MPC planning.

## Recommended run
```bash
python scripts/online_pointmass_goal_diffuser.py --device cuda:0
```

## Maze2D (Online Diffuser / EBM-OnlineRL)

This repo also contains a Maze2D online Diffuser workflow driven by:
- a per-run probe/trainer: `scripts/synthetic_maze2d_diffuser_probe.py`
- an *agentic* autonomous controller (primary): `scripts/agentic_maze2d_autodecider.py`
- a tmux launcher for the agentic controller: `scripts/launch_agentic_maze2d_autodecider_tmux.sh`
- an older rule-based selector (legacy): `scripts/overnight_maze2d_autodecider.py`

### Objective Invariants (Protocol Guardrails)
If you refactor, preserve these unless you are intentionally changing the benchmark protocol:
- Primary env ids: `maze2d-umaze-v1` (and related Maze2D variants when explicitly configured).
- Success metric definition: `goal_dist <= goal_success_threshold` (default threshold `0.2` in the online workflow).
- Diverse evaluation protocol defaults (used for selection/monitoring):
  - `query_mode=diverse`
  - `query_min_distance=1.0`
  - `eval_rollout_horizon=256`
  - prefix success horizons derived from the *same realized rollout*: `eval_success_prefix_horizons=64,128,192,256`
- Online collection protocol defaults:
  - `online_self_improve=true`
  - Option-A style early terminate on success enabled
  - collection budget expressed as `online_collect_transition_budget_per_round` (accepted transitions)

### Setup Notes
Maze2D runs use a MuJoCo + D4RL + Diffuser environment that is currently managed under `third_party/`.
`third_party/` is intentionally not tracked by git (it is large); see the plan docs for the exact env vars.

Minimum env vars for Maze2D scripts:
```bash
export ROOT=/root/ebm-online-rl-prototype
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONPATH="${ROOT}/third_party/diffuser-maze2d"
export PY="${ROOT}/third_party/diffuser/.venv38/bin/python"
```

### Run (Agentic Autodecider)
Start an agentic controller in tmux (external proposals by default):
```bash
cd /root/ebm-online-rl-prototype
PROPOSAL_SOURCE=external \
SEED=0 \
scripts/launch_agentic_maze2d_autodecider_tmux.sh
```

The run root will be created under:
- `runs/analysis/synth_maze2d_diffuser_probe/agentic_autodecider_<timestamp>/`

### Run (Single Probe)
```bash
cd /root/ebm-online-rl-prototype
${PY} scripts/synthetic_maze2d_diffuser_probe.py --help
```
