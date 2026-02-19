from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Episode:
    states: np.ndarray  # [T+1, obs_dim]
    actions: np.ndarray  # [T, act_dim]

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class EpisodeReplayBuffer:
    """Stores full episodes and samples fixed-horizon packed trajectories."""

    def __init__(self, obs_dim: int, act_dim: int, max_episodes: int = 20000) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.max_episodes = int(max_episodes)
        self._episodes: List[Episode] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def add_episode(self, states: np.ndarray, actions: np.ndarray) -> None:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if states.ndim != 2 or states.shape[1] != self.obs_dim:
            raise ValueError(f"states must have shape [T+1, {self.obs_dim}], got {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != self.act_dim:
            raise ValueError(f"actions must have shape [T, {self.act_dim}], got {actions.shape}")
        if states.shape[0] != actions.shape[0] + 1:
            raise ValueError(
                f"states length must be actions length + 1, got states={states.shape[0]}, actions={actions.shape[0]}"
            )

        self._episodes.append(Episode(states=states.copy(), actions=actions.copy()))
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes :]

    def _eligible_episode_indices(self, horizon: int) -> np.ndarray:
        idxs = [i for i, ep in enumerate(self._episodes) if ep.length >= horizon]
        return np.asarray(idxs, dtype=np.int64)

    def can_sample(self, horizon: int) -> bool:
        return self._eligible_episode_indices(horizon).size > 0

    def sample_trajectory_segment(
        self,
        batch_size: int,
        horizon: int,
        rng: np.random.Generator,
        action_scale: float,
    ) -> np.ndarray:
        """Returns packed trajectories [B, H+1, obs_dim + act_dim]."""
        eligible = self._eligible_episode_indices(horizon)
        if eligible.size == 0:
            raise RuntimeError(f"No episode has length >= horizon={horizon}.")

        transition_dim = self.obs_dim + self.act_dim
        batch = np.zeros((batch_size, horizon + 1, transition_dim), dtype=np.float32)

        chosen = rng.choice(eligible, size=batch_size, replace=True)
        for b, ep_idx in enumerate(chosen):
            ep = self._episodes[int(ep_idx)]
            start_max = ep.length - horizon
            start = int(rng.integers(0, start_max + 1))

            states = ep.states[start : start + horizon + 1]  # [H+1, obs]
            actions = ep.actions[start : start + horizon]  # [H, act]
            actions_norm = (actions / action_scale).astype(np.float32)

            packed = np.zeros((horizon + 1, transition_dim), dtype=np.float32)
            packed[:-1, : self.obs_dim] = states[:-1]
            packed[:-1, self.obs_dim :] = actions_norm
            packed[-1, : self.obs_dim] = states[-1]
            batch[b] = packed

        return batch

    def sample_achieved_goal(self, rng: np.random.Generator) -> np.ndarray:
        if len(self._episodes) == 0:
            raise RuntimeError("Replay buffer is empty; cannot sample achieved goal.")
        ep_idx = int(rng.integers(0, len(self._episodes)))
        ep = self._episodes[ep_idx]
        t = int(rng.integers(0, ep.states.shape[0]))
        return ep.states[t].copy()

