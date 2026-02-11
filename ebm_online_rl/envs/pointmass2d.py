from __future__ import annotations

import numpy as np


class PointMass2D:
    """Simple 2D point mass with clipped position and velocity-like actions."""

    def __init__(
        self,
        episode_length: int = 50,
        state_limit: float = 1.0,
        action_limit: float = 0.1,
        success_threshold: float = 0.05,
    ) -> None:
        self.episode_length = int(episode_length)
        self.state_limit = float(state_limit)
        self.action_limit = float(action_limit)
        self.success_threshold = float(success_threshold)

        self.obs_dim = 2
        self.act_dim = 2

        self._rng = np.random.default_rng(0)
        self._t = 0
        self.state = np.zeros(self.obs_dim, dtype=np.float32)
        self.goal = np.zeros(self.obs_dim, dtype=np.float32)

    def sample_goal(self, rng: np.random.Generator | None = None) -> np.ndarray:
        sampler = rng if rng is not None else self._rng
        return sampler.uniform(
            low=-self.state_limit,
            high=self.state_limit,
            size=(self.obs_dim,),
        ).astype(np.float32)

    def reset(self, seed: int | None = None, goal: np.ndarray | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self.state = self.sample_goal(self._rng)
        if goal is None:
            self.goal = self.sample_goal(self._rng)
        else:
            self.goal = np.asarray(goal, dtype=np.float32)
            self.goal = np.clip(self.goal, -self.state_limit, self.state_limit)
        return self.state.copy()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.action_limit, self.action_limit)
        self.state = np.clip(self.state + action, -self.state_limit, self.state_limit).astype(np.float32)
        self._t += 1

        done = self._t >= self.episode_length
        dist = float(np.linalg.norm(self.state - self.goal))
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

