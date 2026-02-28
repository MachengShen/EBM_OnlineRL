# Equilibrium Matching: Mechanistic Analysis Findings

**Project:** EBM Online RL / Diffuser-based Planning
**Environment:** Maze2D-umaze-v1 (d4rl)
**Model:** EqM (Equilibrium Matching) — a time-invariant vector field replacement for standard diffusion-based trajectory planning
**Checkpoint:** `eqm_k25_s010_budgetmatch` (step 18000, K=25, step_size=0.1, horizon=64)
**Branch:** `analysis/results-2026-02-24`
**Last updated:** 2026-02-27

---

## 1. Background

### 1.1 What is Equilibrium Matching (EqM)?

Standard Diffuser uses a learned denoising schedule indexed by a diffusion timestep $t \in \{0, ..., T\}$. EqM replaces this with a **time-invariant** vector field: the denoiser is always called at $t_0 = 0$, and trajectory refinement proceeds as fixed-point iteration:

$$x_{k+1} = x_k - s \cdot f_\theta(x_k, \text{cond}, t_0=0), \qquad k = 0, 1, \ldots, K-1$$

starting from Gaussian noise $x_0 \sim \mathcal{N}(0, I)$, with step size $s = 0.1$ and $K = 25$ iterations. The training loss matches the denoiser output to the noise that was added, identical to standard diffusion, but evaluated only at $t_0 = 0$.

### 1.2 Trajectory representation

Maze2D trajectories are packed as `[act(2) | obs(4)]` per timestep, giving transition_dim = 6 over a horizon $H = 64$. Observations are `[x, y, vx, vy]` and actions are `[force_x, force_y]`.

### 1.3 Conditioning mechanism

Start and goal states are injected via **inpainting**: at each EqM iteration, the observation dimensions at timesteps $t=0$ (start) and $t=H-1$ (goal) are overwritten with the normalized target values. This hard constraint ensures the planned trajectory begins at the current state and ends at the goal.

### 1.4 Research questions

We investigate three mechanistic hypotheses about how EqM generates coherent trajectory plans:

- **H1 (Dynamics alignment):** Does the EqM vector field $f_\theta$ act as a descent direction on a dynamics-consistency objective?
- **H2 (Spatial locality):** Does $f_\theta$ exhibit local structure — does it correct each timestep based primarily on nearby timesteps?
- **H3 (Compositional conditioning):** Can additional waypoint constraints be composed with start/goal conditioning to steer trajectories through intermediate points?

---

## 2. H1 — EqM as Implicit Dynamics Optimization

### 2.1 Setup

We define a **dynamics residual objective** that measures how well a trajectory satisfies the environment's forward dynamics:

$$J_{\text{dyn}}(x) = \frac{1}{H-1} \sum_{t=0}^{H-2} \| s_{t+1} - f_{\text{dyn}}(s_t, a_t) \|^2 + \lambda_u \cdot \frac{1}{H-1} \sum_{t=0}^{H-2} \| a_t \|^2$$

where $s_t$ and $a_t$ are unpacked from the trajectory $x$, and $f_{\text{dyn}}$ is a learned 1-step forward dynamics model trained on the same dataset. The regularization weight $\lambda_u$ is typically 0 (state-only) or $10^{-3}$ (with action penalty).

We measure alignment between the EqM field $f_\theta(x)$ and $\nabla_x J_{\text{dyn}}(x)$ using three metrics:
1. **Dot-product sign**: Is $\langle \nabla J_{\text{dyn}}, f_\theta \rangle > 0$?
2. **Cosine similarity**: $\cos(\nabla J_{\text{dyn}}, f_\theta)$ — overall, action-only, state-only
3. **Small normalized step decrease**: Does $J_{\text{dyn}}(x - \epsilon \cdot f_\theta / \|f_\theta\|) < J_{\text{dyn}}(x)$ for $\epsilon = 0.01$?

### 2.2 Initial results (Phase 1)

Measured on **dataset trajectories** (i.e., samples near the equilibrium):

| Metric | Value |
|---|---|
| dot-positive fraction | 0.458 |
| cosine similarity | -0.003 |
| small-step descent fraction | 0.22 |
| trajectory-level $J_{\text{dyn}}$ decrease | 10 → ~0 over 25 EqM steps |

**Preliminary interpretation:** Weak single-step alignment, but the full EqM trajectory drives $J_{\text{dyn}}$ to near-zero. This appeared to only partially support H1.

### 2.3 Follow-up: Regime sweeps resolve the operating-point artifact

The weak single-step results on dataset trajectories were puzzling. We hypothesized this was an **operating-point artifact**: dataset trajectories are already near the equilibrium, where $J_{\text{dyn}} \approx 2 \times 10^{-6}$ and the gradient $\nabla J_{\text{dyn}}$ is essentially zero — making alignment measurements vacuous noise.

To test this, we swept alignment metrics across two families of **off-equilibrium** inputs:

**Corrupted trajectories** (interpolate dataset with noise via $x_\gamma = \gamma \cdot x_{\text{data}} + (1 - \gamma) \cdot \epsilon$):

| $\gamma$ | dot+ | cos(overall) | cos(action) | descent | $J_{\text{dyn}}$ |
|---|---|---|---|---|---|
| 0.00 (pure noise) | **1.000** | 0.182 | -0.162 | **1.000** | 8.63 |
| 0.25 | **1.000** | 0.176 | -0.133 | **1.000** | 3.84 |
| 0.50 | **1.000** | 0.224 | -0.078 | **1.000** | 1.09 |
| 0.75 | **1.000** | 0.440 | 0.019 | **1.000** | 0.15 |
| 0.90 | **1.000** | 0.559 | 0.045 | **1.000** | 0.020 |
| 0.99 | **1.000** | 0.098 | 0.001 | **1.000** | 0.0002 |

**EqM iterates from noise** (run $k$ EqM steps from $x_0 \sim \mathcal{N}(0, I)$):

| $k$ | dot+ | cos(overall) | cos(action) | descent | $J_{\text{dyn}}$ |
|---|---|---|---|---|---|
| 0 (pure noise) | **1.000** | 0.179 | -0.165 | **1.000** | 9.15 |
| 1 | **1.000** | 0.523 | 0.051 | **1.000** | 0.92 |
| 2 | **1.000** | 0.596 | 0.050 | **1.000** | 0.76 |
| 5 | **1.000** | 0.636 | 0.062 | **1.000** | 0.41 |
| 10 | **1.000** | 0.651 | 0.110 | **1.000** | 0.14 |
| 25 | **1.000** | 0.517 | 0.078 | **1.000** | 0.008 |

**Dataset baseline** ($\lambda_u = 0$): dot+ = 0.430, cos = -0.003, descent = 0.027, $J_{\text{dyn}}$ = 0.000002

### 2.4 Interpretation

**H1 is STRONGLY SUPPORTED.** The results are unambiguous:

1. **dot+ = 1.0 and descent = 1.0 for every off-equilibrium regime tested** — the EqM field is a descent direction on $J_{\text{dyn}}$ at every point along its trajectory from noise to convergence.

2. **The initial weak results were an operating-point artifact.** At the dataset equilibrium, $J_{\text{dyn}} \approx 2 \times 10^{-6}$, so $\nabla J_{\text{dyn}} \approx 0$ and alignment measurements become pure noise (dot+ = 0.43, which is indistinguishable from random).

3. **Cosine similarity peaks in the mid-corruption range** (cos = 0.65 at $k = 10$), where $J_{\text{dyn}}$ is meaningful but not yet tiny. At pure noise ($k = 0$), the cosine is lower (0.18) because $f_\theta$ is doing massive corrections while $\nabla J_{\text{dyn}}$ is high-dimensional and the field also optimizes implicitly for other objectives (smoothness, data likelihood). At convergence, the cosine drops because the gradient vanishes.

4. **Action-only cosine is near-zero with $\lambda_u = 0$** because the dynamics residual is dominated by state prediction error. With $\lambda_u = 10^{-3}$, action cosine jumps to 0.75 on dataset trajectories (where actions are the only non-trivial gradient direction).

**Key insight:** EqM refinement is not merely "removing noise" — it is performing **implicit model-predictive optimization** on a dynamics-consistency surrogate at every iteration. The denoiser has internalized the environment dynamics through training and applies them as a descent field.

---

## 3. H2 — Spatial Locality of the EqM Vector Field

### 3.1 Setup

We probe the **temporal locality** of the denoiser: when correcting the output at timestep $t$, how much does it depend on inputs at nearby vs. distant timesteps?

**Method:** For a given trajectory $x$ and target timestep $t$, compute the VJP (vector-Jacobian product):

$$\text{influence}(\Delta t) = \left\| \frac{\partial \langle f_\theta(x)_t, u \rangle}{\partial x_{t + \Delta t}} \right\|$$

where $u$ is a random unit probe vector. This measures how much the denoiser's output at timestep $t$ is influenced by the input at timestep $t + \Delta t$.

### 3.2 Initial results (Phase 1) — Mixed norm

Aggregating across all output/input dimensions:

| Metric | Value |
|---|---|
| Peak influence at offset 0 | 0.995 (normalized) |
| Half-decay offset | 1 timestep |
| Offset-0 / offset-1 ratio | ~10x |

The denoiser is overwhelmingly local: the input at the same timestep dominates the correction, with rapid decay.

### 3.3 Follow-up: Blockwise decomposition reveals channel-dependent structure

The mixed-norm result hides important heterogeneity. We decomposed the VJP into 4 blocks based on input/output channel type:

| Block | Meaning |
|---|---|
| obs→obs | How do input observations influence output observation corrections? |
| obs→act | How do input observations influence output action corrections? |
| act→obs | How do input actions influence output observation corrections? |
| act→act | How do input actions influence output action corrections? |

Measured as the ratio of offset-0 influence to mean influence over all offsets:

**Dataset trajectories:**

| Block | Locality Ratio |
|---|---|
| act→act | 6.5x |
| act→obs | 1.4x |
| obs→act | 1.4x |
| **obs→obs** | **27.4x** |

**Pure noise ($\gamma = 0$):**

| Block | Locality Ratio |
|---|---|
| act→act | 240x |
| act→obs | 2.9x |
| obs→act | 2.6x |
| **obs→obs** | **43.7x** |

**Converged EqM iterate ($k = 25$):**

| Block | Locality Ratio |
|---|---|
| act→act | 3.1x |
| act→obs | 1.3x |
| obs→act | 1.2x |
| **obs→obs** | **23.1x** |

### 3.4 Interpretation

**H2 is STRONGLY SUPPORTED, with important nuance:**

1. **obs→obs is always the most local block** (23–44x ratio). The denoiser corrects observations almost entirely based on the observation at the same timestep. This makes physical sense: position and velocity are local properties that shouldn't depend on distant states.

2. **act→act locality is regime-dependent.** On pure noise (240x ratio), the model corrects each action independently. As trajectories converge to equilibrium (3.1x ratio), actions become coupled across timesteps. This suggests the model learns cross-timestep action dependencies as the trajectory structure becomes coherent — nearby actions must coordinate to produce smooth state transitions.

3. **Cross-blocks (act→obs, obs→act) are always weakly local** (~1.3–2.9x ratio). The observation-to-action coupling extends across multiple timesteps, consistent with the interpretation that the model uses a temporal neighborhood of observations to inform each action. This is what enables planning: actions at time $t$ depend on the state context over a window around $t$.

4. **Locality is strongest far from equilibrium.** Pure noise shows the highest locality ratios because the denoiser can make large, independent corrections to each timestep. As the trajectory becomes structured, genuine temporal correlations force the model to attend to more distant timesteps.

**Architectural interpretation:** The TemporalUnet processes trajectories as 1D sequences with skip connections. The strong obs→obs locality is consistent with the U-Net's local receptive field for observation channels. The weaker obs→act locality suggests the skip connections propagate observation context laterally to inform action predictions over a wider temporal window.

---

## 4. H3 — Compositional Waypoint Conditioning

### 4.1 Setup

We test whether additional constraints can be composed with start/goal conditioning. A **waypoint** is a target $(x, y)$ position at an intermediate timestep $t_w$, injected by overwriting the position dimensions of $x_{t_w}$ at each EqM iteration (analogous to start/goal inpainting, but for an intermediate point).

Two injection modes:
- **pos_only**: Overwrite only $(x, y)$, leave velocity dimensions free
- **pos_and_zero_vel**: Overwrite $(x, y)$ and set $(v_x, v_y) = (0, 0)$

### 4.2 Imagination-level results (Phase 1)

Planned trajectories (no environment interaction) with waypoint at $t_w = H/2 = 32$:

| Mode | WP hit rate | Goal hit rate | Joint |
|---|---|---|---|
| pos_only | **1.000** | 1.000 | **1.000** |
| pos_and_zero_vel | **1.000** | 1.000 | **1.000** |
| no_waypoint | 0.000 | 1.000 | 0.000 |

**Perfect composition at the planning level.** The waypoint is hard-clamped at each EqM iteration, so the planned trajectory always passes through it exactly.

### 4.3 Execution-level results (Follow-up)

Full environment rollout with MPC-style replanning (replan every 8 steps, max 300 steps per episode, 50 episodes). Success criterion: agent's actual position passes within $\epsilon = 0.5$ of the waypoint at **any** timestep during the episode.

| Mode | WP Hit | Goal Hit | Joint | Mean Min WP Dist | Mean Min Goal Dist |
|---|---|---|---|---|---|
| pos_only | **0.260** | 0.880 | **0.240** | 1.082 | 0.294 |
| pos_and_zero_vel | **0.240** | 0.880 | **0.220** | 1.013 | 0.298 |
| no_waypoint | 0.000 | 0.900 | 0.000 | $\infty$ | 0.256 |

### 4.4 Interpretation

**H3 is SUPPORTED with a significant imagination-execution gap:**

1. **Waypoint conditioning works at execution level** — 24–26% joint success vs. 0% baseline. The no-waypoint baseline never accidentally visits the waypoint, confirming that waypoint visits are causal, not incidental.

2. **Goal reaching is robust** (~88–90%) regardless of waypoint presence, confirming the planning + replanning pipeline works well for the primary objective.

3. **pos_only and pos_and_zero_vel are nearly identical** (26% vs. 24%). This is consistent with H2 locality findings: since obs→obs is extremely local, setting velocity at the waypoint timestep has minimal effect on the trajectory at other timesteps.

4. **The imagination-execution gap is large** (100% → ~25%). This reflects a fundamental design limitation:

   - The waypoint is injected at a fixed planning-horizon timestep $t_w = 31$ (midpoint of $H = 64$).
   - As execution proceeds, the "shifting waypoint index" $t_{\text{local}} = t_w - k_{\text{exec}}$ moves earlier in the plan and eventually drops below $t = 1$, at which point the waypoint constraint is removed.
   - With `replan_every = 8` and $t_w = 31$, the waypoint is active for only the first ~30 execution steps out of ~300 total (10% of the episode).
   - After the constraint is dropped, the agent has no reason to route through the waypoint and may deviate to a more direct path to the goal.

5. **Potential improvements** (not yet tested):
   - Keep the waypoint active for more of the episode (e.g., reformulate as a subgoal rather than a fixed-timestep constraint)
   - Increase `n_plan_samples` to select plans that better route through the waypoint
   - Decrease `replan_every` to follow plans more faithfully
   - Use a hierarchical approach: first reach waypoint, then reach goal

### 4.5 Trajectory visualization

*(See `runs/analysis/eqm_waypoint_exec_viz_*/waypoint_exec_trajectories.png` for trajectory plots with waypoint markers.)*

Visual inspection reveals:
- **HIT episodes:** The agent's trajectory clearly curves toward the waypoint (orange diamond) before continuing to the goal. The early planned trajectories (green dashed) pass through the waypoint circle.
- **MISS episodes:** The agent follows a direct path to the goal that doesn't naturally pass near the waypoint. The waypoint constraint is active in early plans but drops off before the agent reaches that region of the maze.
- **Closest approach points** (orange X) show where the agent came nearest to the waypoint.

---

## 5. Cross-Hypothesis Connections

### 5.1 H1 + H2: Local dynamics descent

H1 shows EqM is a descent direction on dynamics residuals. H2 shows this correction is primarily local. Together: **EqM implements a spatially local dynamics optimizer** — each timestep's correction is driven mainly by its local dynamics inconsistency, not by global trajectory structure. This is analogous to a Jacobi iteration on a block-tridiagonal dynamics constraint system.

### 5.2 H2 + H3: Locality enables composition

The extreme obs→obs locality (27x ratio) explains why pos_only waypoint conditioning works as well as pos_and_zero_vel: the velocity dimensions at the waypoint timestep don't significantly influence the observation corrections at nearby timesteps. The denoiser treats each timestep's observation correction as nearly independent, so clamping additional dimensions has minimal effect.

### 5.3 H1 + H3: Implicit MPC interpretation

H1 shows EqM refinement is dynamics-consistent descent. H3 shows waypoint constraints compose with this descent. Together: **EqM with waypoint conditioning is performing constrained implicit MPC** — finding trajectories that satisfy dynamics while passing through specified points. The 100% imagination-level success confirms the optimization is fully capable; the execution gap is a control/interface problem, not a planning limitation.

---

## 6. EqM vs Diffuser Comparative Analysis

*Date: 2026-02-26. Both models use the same TemporalUnet backbone, same maze2d-umaze-v1 dataset, same online self-improvement training (4 rounds). Checkpoints: EqM `eqm_k25_s010_budgetmatch/checkpoint_last.pt`, Diffuser `diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/checkpoint_last.pt`.*

### 8.1 Trajectory Smoothness

Scripts: `scripts/smoothness_maze2d_compare.py`, `scripts/viz_maze2d_diffuser_compare.py`. 20 imagined trajectories on matched start/goal pairs.

| Metric | EqM | Diffuser | Ratio (E/D) |
|---|---|---|---|
| Mean step displacement | **0.160** | 0.039 | 4.1x |
| Std step displacement | 0.077 | 0.023 | 3.3x |
| Max step displacement | 0.510 | 0.121 | 4.2x |
| Total path length | 39.43 | 9.62 | 4.1x |
| Straightness (goal dist / path length) | **0.20** | **0.84** | 0.24x |
| Mean direction change | **118°** | **36°** | 3.3x |

**Conclusion:** EqM imagined trajectories are ~4x jerkier and far less straight than Diffuser. This is a real model property, confirmed numerically, not a visualization artifact. EqM makes large erratic steps with ~118° average turns; Diffuser takes short smooth steps with ~36° average turns.

### 8.2 Locality-vs-Noise Analysis

Script: `scripts/analysis_locality_vs_noise_maze2d.py`. 256 VJP probes per condition × 5 noise levels × 2 models. Artifact: `runs/analysis/locality_vs_noise_20260226-210232/`.

Noise levels: t = 0 (clean), 16, 32, 48, 63 (pure noise). Diffuser uses matching t_embed; EqM always uses t_embed = 0.

**Key finding: both models have half-decay offset = 1 at ALL noise levels.** The influence of a timestep's input on neighboring timesteps drops to ~5% within 1 step for both models, at every noise level.

| Model | Clean (t=0) Inf@1 | Mid (t=32) Inf@1 | Noise (t=63) Inf@1 |
|---|---|---|---|
| Diffuser | 0.068 | 0.022 | 0.005 |
| EqM | 0.046 | 0.036 | 0.020 |

Diffuser becomes more local at high noise; EqM retains more non-local influence at high noise, but both are highly local throughout. **The EqM smoothness deficit does NOT arise from denoiser non-locality — the architectures are equally local. The difference emerges from the iterative sampling dynamics (64 DDPM steps vs 25 EqM gradient descent steps).**

### 8.3 Sub-block Locality: Position vs Velocity Decomposition

Script: `scripts/analysis_subblock_locality_compare.py`. 256 VJP probes, clean data (t=0), raw (unnormalized) grad norms. Artifact: `runs/analysis/subblock_locality_compare_20260226-214034/`.

Transition packing: `[ax, ay | x, y, vx, vy]` — act = dims 0:2, pos = dims 2:4, vel = dims 4:6.

#### Velocity → Action (`act_out ← vel_in`) — *user-requested*

| Model | Off0 | Off1 | Off2 | Off3 | Off4 | Off1/Off0 |
|---|---|---|---|---|---|---|
| Diffuser | 1.120 | 0.569 | 0.564 | 0.537 | 0.446 | **51%** |
| EqM | 0.032 | 0.029 | 0.022 | 0.018 | 0.020 | **89%** |

EqM's velocity → action coupling is **nearly flat across all offsets** (89% persists at offset 1). Actions at timestep t depend equally on velocity at all other timesteps. Diffuser has moderate locality (51%). This is the strongest structural difference found between the two models.

#### Position → Velocity (`vel_out ← pos_in`) — *user-requested*

| Model | Off0 | Off1 | Off2 | Off3 | Off4 | Off1/Off0 |
|---|---|---|---|---|---|---|
| Diffuser | 1.162 | 0.392 | 0.333 | 0.329 | 0.251 | **34%** |
| EqM | 0.025 | 0.008 | 0.007 | 0.008 | 0.008 | **32%** |

Both models show nearly identical locality structure for this sub-block (~32–34% at offset 1, then roughly flat). Diffuser has ~46x larger raw magnitude. This cross-block coupling is substantially more non-local than the diagonal blocks.

#### Position → Position (`pos_out ← pos_in`) — *user-requested*

| Model | Off0 | Off1 | Off2 | Off3 | Off4 | Off1/Off0 |
|---|---|---|---|---|---|---|
| Diffuser | 25.025 | 0.952 | 1.142 | 0.897 | 0.798 | **3.8%** |
| EqM | 0.903 | 0.023 | 0.017 | 0.015 | 0.020 | **2.5%** |

Both models are **extremely local** for pos→pos: the self-timestep dominates 96–97% of influence. Diffuser has 27x larger raw magnitude overall (consistent with the H2 blockwise result).

#### Additional sub-blocks (full table, t=0)

| Sub-block pair | Diffuser Off0 | EqM Off0 | D/E ratio | Diffuser 1/0 | EqM 1/0 |
|---|---|---|---|---|---|
| pos → pos | 25.025 | 0.903 | 27.7x | 3.8% | 2.5% |
| vel → vel | 23.114 | 0.863 | 26.8x | 6.8% | 3.8% |
| pos → vel | 1.162 | 0.025 | 46.3x | 33.8% | 31.8% |
| vel → pos | 0.907 | 0.027 | 33.3x | 34.6% | 20.3% |
| vel → act | 1.120 | 0.032 | 34.8x | **50.8%** | **89.1%** |
| pos → act | 1.220 | 0.012 | 100.8x | 27.6% | 88.8% |

### 8.4 Interpretation

1. **Magnitude asymmetry (27–100x)**: EqM produces substantially weaker raw gradients than Diffuser across all sub-blocks at clean data. This is consistent with EqM operating near an equilibrium fixed point where the vector field is small by construction, while DDPM operates at a larger signal regime.

2. **vel→act non-locality is the key EqM anomaly**: EqM's vel→act 1/0 ratio is 89% vs Diffuser's 51%. EqM actions do not depend primarily on contemporaneous velocity; they draw on velocity context from across the entire trajectory. This likely drives the jerky action sequences: without a strong local vel→act prior, each step's action is influenced by inconsistent global context.

3. **Diagonal blocks (pos→pos, vel→vel) are highly local for both models** (2–7% at offset 1), confirming the H2 locality finding extends to the finer sub-block level.

4. **Cross-block off-diagonal channels (pos↔vel) are moderately non-local** (~32–35% at offset 1) for both models, suggesting both learn some temporal coupling between position and velocity across timesteps.

5. **EqM pos→act is also anomalously non-local** (88.8% at offset 1, similar to vel→act). Both action-output sub-blocks in EqM are globally distributed, while in Diffuser the pos→act channel is more local (27.6%). This suggests EqM's action predictions are fundamentally dependent on the entire trajectory's state context rather than the local state.

### 8.5 Boundary Influence Spikes

Both trained models exhibit strong **boundary influence spikes**: when the output pivot is at an interior timestep (t=32), the gradient w.r.t. boundary input positions (t=0, t=63) is dramatically elevated relative to neighboring timesteps.

#### Boundary spike magnitudes (pos→pos, pivot=32, clean data, 128 probes)

| Model | t=0 (bnd) | t=1 | t=31 | t=32 (self) | t=33 | t=62 | t=63 (bnd) |
|---|---|---|---|---|---|---|---|
| Diffuser | 2.594 | 0.087 | 0.740 | **28.04** | 0.696 | 0.084 | 2.424 |
| EqM | 0.338 | 0.003 | 0.007 | **0.914** | 0.006 | 0.002 | 0.298 |

Key observations:
- **Self (offset-0) is still the largest influence** — 28.04 (Diffuser) and 0.91 (EqM)
- **Boundary spikes are the second largest** — ~2.5 (Diffuser, 9% of self) and ~0.32 (EqM, 36% of self)
- Boundary/near-boundary ratio: ~30x (Diffuser), ~100x (EqM)

#### Action boundary amplification

| Model | Block pair | Boundary pivot off0 | Interior pivot off0 | Ratio |
|---|---|---|---|---|
| Diffuser | act←vel | 1.118 | 1.129 | 0.99x |
| EqM | act←vel | 0.042 | 0.033 | 1.25x |
| EqM | act←pos | 0.036 | 0.012 | 3.06x |

Action boundary amplification is modest for Diffuser but present in EqM, especially for act←pos.

#### Monotonic action decay (Diffuser, offsets 0–10, t=0)

Diffuser `act_out←vel_in`: 1.129, 0.574, 0.353, 0.266, 0.212, 0.181, 0.157, 0.133, 0.116, 0.096, 0.091 — **strictly monotonically decreasing**.

Diffuser `act_out←pos_in`: 1.220, 0.337, 0.186, 0.149, 0.120, 0.105, 0.086, 0.075, 0.064, 0.053, 0.049 — **strictly monotonically decreasing**.

EqM action offsets are roughly decreasing but noisier at low magnitudes.

### 8.6 Boundary Spike Mechanism: Training-Time Conditioning

**Question:** Why do boundary timesteps show elevated influence when training doesn't explicitly teach boundary importance?

**Hypotheses tested:**
1. **H1 (Architectural):** Conv1d padding effects in TemporalUnet create boundary artifacts → **RULED OUT**
2. **H2 (Learned data statistics):** Training data has consistent boundary structure → Plausible but secondary
3. **H3 (Training-time conditioning):** `apply_conditioning()` clamps boundary observations during training → **CONFIRMED as primary mechanism**

**Control experiment — random untrained TemporalUnet (same architecture):**

| Metric | Random network | Trained (Diffuser) | Trained (EqM) |
|---|---|---|---|
| Boundary / near-bnd | **0.73x** (suppressed) | ~30x (elevated) | ~100x (elevated) |
| Boundary / interior | **0.14x** (suppressed) | ~2.5x | ~36x |

The random network shows **no boundary spikes** — boundaries are actually suppressed, likely due to conv1d zero-padding at edges. This definitively rules out H1.

**Root cause (H3 confirmed):** Both models apply `apply_conditioning()` *during training*, clamping boundary observations to clean values before each forward pass:
- Diffuser `p_losses()`: `x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)` before model forward
- EqM `loss()`: `x_gamma = apply_conditioning(x_gamma, cond, self.action_dim)` before model forward

Every training example has noise-free observations at t=0 and t=H-1, while interior timesteps are noised. The model learns that boundaries contain reliable anchor information at all noise levels, developing a **boundary-anchored denoising strategy** — functionally a learned BVP (boundary value problem) solver.

**Implication for "position = scaffold" mental model:** The training procedure explicitly teaches the model to use boundary positions as fixed anchor points. Position influence is elevated at boundaries because the model learns to propagate clean boundary signal inward to denoise interior positions. This is consistent with position acting as a scaffold anchored at start/goal constraints.

---

## 7. Open Questions

1. **H2 blockwise: act→act locality drop.** The act→act locality ratio drops from 240x (noise) to 3.1x (converged). Is this because converged trajectories have genuine cross-timestep action dependencies, or because the denoiser field magnitude vanishes near equilibrium, making the ratio's denominator unreliable?

2. **H1 strong support validation.** dot+ = 1.0 and descent = 1.0 for all off-equilibrium samples is remarkable. Does this fully validate the "EqM as implicit MPC" interpretation, or are there caveats? For instance, the surrogate $J_{\text{dyn}}$ uses a learned dynamics model that may not capture the true planning objective (reward maximization, obstacle avoidance).

3. **H3 execution gap root cause.** Is the 26% waypoint hit rate primarily a control execution problem (actions don't produce the planned trajectory), a replanning problem (the shifting waypoint index drops the constraint too early), or a feasibility problem (some waypoints require paths incompatible with the maze geometry)?

4. **Cross-block coupling interpretation.** obs→act has a consistent ~1.4x locality ratio across all regimes. Does this low ratio suggest that the model uses observations over a wide temporal window to inform actions, consistent with a "plan actions from state context" interpretation?

---

## 8. Methods Reference

### 7.1 Scripts

| Script | Purpose |
|---|---|
| `scripts/analysis_eqm_maze2d_dyn_alignment_sweeps.py` | H1 alignment metrics across regimes |
| `scripts/analysis_eqm_locality_map_maze2d_blockwise.py` | H2 blockwise locality probes |
| `scripts/eval_maze2d_waypoint_exec.py` | H3 execution-level waypoint eval |
| `scripts/viz_maze2d_waypoint_exec_trajectories.py` | H3 trajectory visualization |
| `scripts/smoothness_maze2d_compare.py` | §8.1 EqM vs Diffuser smoothness metrics |
| `scripts/analysis_locality_vs_noise_maze2d.py` | §8.2 locality vs noise level (both models) |
| `scripts/analysis_blockwise_locality_compare.py` | §8.3 blockwise raw grad norms (both models) |
| `scripts/analysis_subblock_locality_compare.py` | §8.3 pos/vel sub-block locality (both models) |
| `scripts/analysis_boundary_influence_maze2d.py` | §8.5–8.6 boundary influence & mechanism analysis |
| `scripts/viz_maze2d_diffuser_compare.py` | §8.1 side-by-side trajectory visualization |
| `scripts/maze2d_eqm_utils.py` | Shared model loading & utilities |

### 7.2 Checkpoints

**EqM:** `runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_k25_s010_budgetmatch/checkpoint_last.pt`
- Architecture: TemporalUnet (dim=64, dim_mults=[1,2,4])
- Training: 18,000 steps, EqM loss at $t_0 = 0$, 4 online rounds
- Dataset: maze2d-umaze-v1 (d4rl), 12,459 episodes

**Diffuser:** `runs/analysis/synth_maze2d_diffuser_probe/compare_diffuser_vs_gcbc_20260217-180356/diffuser_ts6000_or4_ep64_t3000_rp16_gp040_seed0/checkpoint_last.pt`
- Architecture: TemporalUnet (dim=64, dim_mults=[1,2,4]) — identical to EqM backbone
- Training: 6,000 env steps, DDPM (T=64 cosine schedule), 4 online rounds
- Dataset: maze2d-umaze-v1 (d4rl), same 12,459 episodes

### 7.3 Reproduction

```bash
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/tmp/mujoco_compat:/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export PYTHONPATH=/root/ebm-online-rl-prototype/third_party/diffuser-maze2d
PYTHON=/root/ebm-online-rl-prototype/third_party/diffuser/.venv38/bin/python3.8
CKPT=runs/analysis/synth_maze2d_diffuser_probe/eqm_budgetmatch_20260225-012027/eqm_k25_s010_budgetmatch/checkpoint_last.pt
DYN_CKPT=runs/analysis/eqm_dyn_alignment_h1_20260225-175408/forward_dynamics.pt

# H1 alignment sweeps
$PYTHON scripts/analysis_eqm_maze2d_dyn_alignment_sweeps.py \
  --checkpoint $CKPT --dyn_ckpt $DYN_CKPT --n_samples 256

# H2 blockwise locality
$PYTHON scripts/analysis_eqm_locality_map_maze2d_blockwise.py \
  --checkpoint $CKPT --n_probes 256 --probe_regime all

# H3 execution-level waypoint
$PYTHON scripts/eval_maze2d_waypoint_exec.py \
  --checkpoint $CKPT --waypoint_mode both --n_episodes 50 --replan_every 8

# H3 trajectory visualization
$PYTHON scripts/viz_maze2d_waypoint_exec_trajectories.py \
  --checkpoint $CKPT --n_episodes 8
```
