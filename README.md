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

Note: this training script is GPU-only and will error out if CUDA is unavailable.
