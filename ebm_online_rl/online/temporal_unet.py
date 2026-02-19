from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            _group_norm(out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size=kernel_size)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size=kernel_size)
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(time_dim, out_ch))
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.block2(h)
        return h + self.residual(x)


class TemporalUNet1D(nn.Module):
    """
    Lightweight temporal U-Net over trajectory sequences.

    Input:  [B, T, D]
    Output: [B, T, D]
    """

    def __init__(
        self,
        transition_dim: int,
        base_dim: int = 64,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.transition_dim = int(transition_dim)

        dims = [self.transition_dim, *[base_dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = base_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.Mish(),
            nn.Linear(base_dim * 4, time_dim),
        )

        self.downs = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(
                nn.ModuleDict(
                    {
                        "res1": ResidualTemporalBlock(dim_in, dim_out, time_dim, kernel_size=kernel_size),
                        "res2": ResidualTemporalBlock(dim_out, dim_out, time_dim, kernel_size=kernel_size),
                        "down": nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=2, padding=1)
                        if not is_last
                        else nn.Identity(),
                    }
                )
            )

        mid_dim = dims[-1]
        self.mid1 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim, kernel_size=kernel_size)
        self.mid2 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim, kernel_size=kernel_size)

        self.ups = nn.ModuleList()
        current_ch = mid_dim
        skip_channels = [pair[1] for pair in in_out]
        for skip_ch in reversed(skip_channels):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "res1": ResidualTemporalBlock(current_ch + skip_ch, skip_ch, time_dim, kernel_size=kernel_size),
                        "res2": ResidualTemporalBlock(skip_ch, skip_ch, time_dim, kernel_size=kernel_size),
                    }
                )
            )
            current_ch = skip_ch

        self.final = nn.Sequential(
            Conv1dBlock(current_ch, current_ch, kernel_size=kernel_size),
            nn.Conv1d(current_ch, self.transition_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got {x.shape}")
        if t.ndim != 1:
            raise ValueError(f"t must be [B], got {t.shape}")

        x = x.transpose(1, 2)  # [B, D, T]
        t_emb = self.time_mlp(t.float())

        skips = []
        for block in self.downs:
            x = block["res1"](x, t_emb)
            x = block["res2"](x, t_emb)
            skips.append(x)
            x = block["down"](x)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for block in self.ups:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block["res1"](x, t_emb)
            x = block["res2"](x, t_emb)

        x = self.final(x)
        return x.transpose(1, 2)

