from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch


def build_anchor_times(horizon: int, stride: int) -> List[int]:
    """Build sparse anchor timesteps over a trajectory of length `horizon`."""
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    times = list(range(0, int(horizon), int(stride)))
    last_t = int(horizon) - 1
    if not times or times[-1] != last_t:
        times.append(last_t)
    return times


def extract_anchor_xy(x: torch.Tensor, act_dim: int, times: Iterable[int]) -> Dict[int, torch.Tensor]:
    """
    Extract position anchors from [act | obs] trajectories.

    Args:
        x: Tensor of shape [B, H, D].
        act_dim: Action dimension; xy are read from x[..., act_dim:act_dim+2].
        times: Timesteps to anchor.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B,H,D], got shape={tuple(x.shape)}")
    _, horizon, transition_dim = x.shape
    if act_dim < 0 or (act_dim + 2) > transition_dim:
        raise ValueError(
            f"act_dim slice invalid for transition_dim={transition_dim}: act_dim={act_dim}"
        )

    anchors: Dict[int, torch.Tensor] = {}
    for t in times:
        ti = int(t)
        if ti < 0 or ti >= horizon:
            raise ValueError(f"anchor timestep out of range: t={ti}, horizon={horizon}")
        anchors[ti] = x[:, ti, act_dim : act_dim + 2].clone()
    return anchors


def apply_pos_only_anchors_(x: torch.Tensor, act_dim: int, anchor_xy: Mapping[int, torch.Tensor]) -> None:
    """In-place clamp of position-only anchors for [act | obs] trajectory tensors."""
    if x.ndim != 3:
        raise ValueError(f"x must be [B,H,D], got shape={tuple(x.shape)}")
    bsz, horizon, transition_dim = x.shape
    if act_dim < 0 or (act_dim + 2) > transition_dim:
        raise ValueError(
            f"act_dim slice invalid for transition_dim={transition_dim}: act_dim={act_dim}"
        )

    for t, xy in anchor_xy.items():
        ti = int(t)
        if ti < 0 or ti >= horizon:
            raise ValueError(f"anchor timestep out of range: t={ti}, horizon={horizon}")

        xy_t = xy.to(device=x.device, dtype=x.dtype)
        if xy_t.ndim == 1:
            xy_t = xy_t.unsqueeze(0)
        if xy_t.ndim != 2 or xy_t.shape[1] != 2:
            raise ValueError(f"anchor xy must be [B,2] or [2], got shape={tuple(xy_t.shape)}")

        if xy_t.shape[0] == 1 and bsz > 1:
            xy_t = xy_t.expand(bsz, -1)
        elif xy_t.shape[0] != bsz:
            raise ValueError(f"anchor batch mismatch: expected {bsz}, got {xy_t.shape[0]}")

        x[:, ti, act_dim : act_dim + 2] = xy_t
