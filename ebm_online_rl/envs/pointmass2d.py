from __future__ import annotations

import numpy as np


class PointMass2D:
    """2D goal-reaching environment with selectable first/second-order dynamics."""

    def __init__(
        self,
        episode_length: int = 50,
        state_limit: float = 1.0,
        unbounded_state_space: bool = False,
        state_sample_std: float = 1.0,
        action_limit: float = 0.1,
        success_threshold: float = 0.05,
        dynamics_model: str = "first_order",
        double_integrator_dt: float = 0.1,
        initial_velocity_std: float = 0.0,
        velocity_damping: float = 0.0,
        velocity_clip: float = 5.0,
    ) -> None:
        self.episode_length = int(episode_length)
        self.state_limit = float(state_limit)
        self.unbounded_state_space = bool(unbounded_state_space)
        self.state_sample_std = float(state_sample_std)
        self.action_limit = float(action_limit)
        self.success_threshold = float(success_threshold)
        self.dynamics_model = str(dynamics_model)
        self.double_integrator_dt = float(double_integrator_dt)
        self.initial_velocity_std = float(initial_velocity_std)
        self.velocity_damping = float(velocity_damping)
        self.velocity_clip = float(velocity_clip)

        self.position_dim = 2
        self.act_dim = 2
        if self.dynamics_model == "first_order":
            self.obs_dim = self.position_dim
        elif self.dynamics_model == "double_integrator":
            self.obs_dim = self.position_dim * 2
        else:
            raise ValueError(
                f"Unknown dynamics_model={self.dynamics_model}. "
                "Expected one of: first_order, double_integrator."
            )

        self._rng = np.random.default_rng(0)
        self._t = 0
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.goal = np.zeros(self.obs_dim, dtype=np.float32)

    def sample_position(self, rng: np.random.Generator | None = None) -> np.ndarray:
        sampler = rng if rng is not None else self._rng
        if self.unbounded_state_space:
            return sampler.normal(
                loc=0.0,
                scale=self.state_sample_std,
                size=(self.position_dim,),
            ).astype(np.float32)
        return sampler.uniform(
            low=-self.state_limit,
            high=self.state_limit,
            size=(self.position_dim,),
        ).astype(np.float32)

    def sample_goal(self, rng: np.random.Generator | None = None) -> np.ndarray:
        goal_pos = self.sample_position(rng)
        if self.dynamics_model == "first_order":
            return goal_pos
        goal_vel = np.zeros(self.position_dim, dtype=np.float32)
        return np.concatenate([goal_pos, goal_vel], axis=0).astype(np.float32)

    def sample_state(self, rng: np.random.Generator | None = None) -> np.ndarray:
        pos = self.sample_position(rng)
        if self.dynamics_model == "first_order":
            return pos

        sampler = rng if rng is not None else self._rng
        if self.initial_velocity_std > 0.0:
            vel = sampler.normal(
                loc=0.0,
                scale=self.initial_velocity_std,
                size=(self.position_dim,),
            ).astype(np.float32)
        else:
            vel = np.zeros(self.position_dim, dtype=np.float32)
        return np.concatenate([pos, vel], axis=0).astype(np.float32)

    def goal_distance(
        self,
        state: np.ndarray | None = None,
        goal: np.ndarray | None = None,
    ) -> float:
        s = self.state if state is None else np.asarray(state, dtype=np.float32)
        g = self.goal if goal is None else np.asarray(goal, dtype=np.float32)
        return float(np.linalg.norm(s[: self.position_dim] - g[: self.position_dim]))

    def reset(self, seed: int | None = None, goal: np.ndarray | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self.state = self.sample_state(self._rng)
        if goal is None:
            self.goal = self.sample_goal(self._rng)
        else:
            goal_arr = np.asarray(goal, dtype=np.float32)
            if goal_arr.shape == (self.position_dim,) and self.dynamics_model == "double_integrator":
                goal_arr = np.concatenate([goal_arr, np.zeros(self.position_dim, dtype=np.float32)], axis=0)
            self.goal = goal_arr
        return self.state.copy()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self.dynamics_model == "first_order":
            # Deliberately keep unconstrained dynamics to avoid boundary clipping artifacts.
            self.state = (self.state + action).astype(np.float32)
        else:
            dt = self.double_integrator_dt
            pos = self.state[: self.position_dim]
            vel = self.state[self.position_dim :]
            vel_next = vel + action * dt
            if self.velocity_damping > 0.0:
                damp = max(0.0, 1.0 - self.velocity_damping * dt)
                vel_next = vel_next * np.float32(damp)
            if self.velocity_clip > 0.0:
                vel_next = np.clip(vel_next, -self.velocity_clip, self.velocity_clip).astype(np.float32)
            pos_next = pos + vel_next * dt
            self.state = np.concatenate([pos_next, vel_next], axis=0).astype(np.float32)
        self._t += 1

        done = self._t >= self.episode_length
        dist = self.goal_distance(self.state, self.goal)
        info = {
            "dist_to_goal": dist,
            "is_success": float(dist <= self.success_threshold),
        }
        reward = 0.0
        return self.state.copy(), reward, done, info


def _manual_random_rollout() -> None:
    env = PointMass2D()
    rng = np.random.default_rng(7)
    obs = env.reset(seed=11)

    states = [obs]
    actions = []
    done = False
    final_info = {}
    while not done:
        action = rng.uniform(-env.action_limit, env.action_limit, size=(env.act_dim,)).astype(np.float32)
        obs, _, done, final_info = env.step(action)
        states.append(obs)
        actions.append(action)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    print("states shape:", states.shape)
    print("actions shape:", actions.shape)
    print("state min/max:", float(states.min()), float(states.max()))
    print("action min/max:", float(actions.min()), float(actions.max()))
    print("final dist:", float(final_info["dist_to_goal"]))


if __name__ == "__main__":
    _manual_random_rollout()
