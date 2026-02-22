#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _PositionalEmbedding(nn.Module):
    """
    CleanDiffuser-style positional embedding used by EqNet in diffusion-stitching.
    """

    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.max_positions = int(max_positions)
        self.endpoint = bool(endpoint)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            x = x.reshape(-1)
        freqs = torch.arange(
            start=0, end=self.dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1.0 / self.max_positions) ** freqs
        # Keep the same ger-based form as upstream EqNet.
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class _LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class _MultBiasLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias_layer = nn.Parameter(torch.randn((1,)))
        self.mult_layer = nn.Parameter(torch.randn((1,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.mult_layer) + self.bias_layer


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        kernel_size: int = 3,
        linear: bool = False,
    ):
        super().__init__()
        padding = (int(kernel_size) - 1) // 2
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=int(kernel_size),
            stride=1,
            padding=padding,
            padding_mode="replicate",
        )
        self.layer_norm_1 = _LayerNorm(dim=self.out_channels) if not linear else nn.Identity()
        self.activation = nn.Mish() if not linear else nn.Identity()
        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=int(kernel_size),
            stride=1,
            padding=padding,
            padding_mode="replicate",
        )
        self.layer_norm_2 = _LayerNorm(dim=self.out_channels) if not linear else _MultBiasLayer()
        self.last_activation = nn.Mish() if not linear else nn.Identity()
        self.emb_mlp = nn.Sequential(nn.Mish(), nn.Linear(int(emb_dim), self.out_channels))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer_norm_1(out)
        out = self.activation(out)
        out = out + self.emb_mlp(emb).unsqueeze(-1)
        out = self.conv2(out)
        out = self.layer_norm_2(out)
        if self.in_channels == self.out_channels:
            return self.last_activation(out) + x
        return self.last_activation(out)


class _EqNetCore(nn.Module):
    """
    Minimal port of diffusion-stitching EqNet (d27cf2a), adapted to run standalone.
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 32,
        model_dim: int = 32,
        use_timestep_emb: bool = True,
        n_layers: int = 25,
        kernel_size: int = 3,
        kernel_expansion_rate: int = 5,
        encode_position: bool = False,
        channel_size: int = 64,
    ):
        super().__init__()
        if int(model_dim) != int(emb_dim):
            raise ValueError(
                "EqNet currently requires model_dim == emb_dim for compatibility "
                f"(got model_dim={model_dim}, emb_dim={emb_dim})."
            )
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if int(kernel_expansion_rate) <= 0:
            raise ValueError(f"kernel_expansion_rate must be > 0, got {kernel_expansion_rate}")
        if int(n_layers) < 0:
            raise ValueError(f"n_layers must be >= 0, got {n_layers}")

        self.use_timestep_emb = bool(use_timestep_emb)
        self.channel_size = int(channel_size)
        self.encode_position = bool(encode_position)
        self.map_noise = _PositionalEmbedding(int(emb_dim))

        conv_layers = []
        out_channels = self.channel_size
        conv_layers.append(
            _ResidualBlock(
                in_channels=int(in_dim),
                out_channels=out_channels,
                kernel_size=int(kernel_size),
                emb_dim=int(emb_dim),
            )
        )
        expansion_counter = 0
        k = int(kernel_size)
        for _ in range(int(n_layers)):
            expansion_counter += 1
            in_channels = self.channel_size
            out_channels = self.channel_size
            if expansion_counter >= int(kernel_expansion_rate):
                k += 2
                expansion_counter = 0
            conv_layers.append(
                _ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    emb_dim=int(emb_dim),
                )
            )
        conv_layers.append(
            _ResidualBlock(
                in_channels=out_channels,
                out_channels=int(in_dim),
                kernel_size=k,
                emb_dim=int(emb_dim),
                linear=True,
            )
        )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.map_emb = nn.Sequential(
            nn.Linear(int(emb_dim), int(model_dim) * 4),
            nn.Mish(),
            nn.Linear(int(model_dim) * 4, int(model_dim)),
        )

    @staticmethod
    def _sinusoidal_time_embeddings(
        *,
        size: int,
        length: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        range_param: float = 1000.0,
    ) -> torch.Tensor:
        position_enc = np.array(
            [
                [pos / np.power(range_param, 2 * i / size) for i in range(size)]
                if pos != 0
                else np.zeros(size)
                for pos in range(length)
            ]
        )
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
        position_enc = torch.tensor(position_enc, dtype=torch.float32, device=device).unsqueeze(0)
        position_enc = position_enc.repeat(batch_size, 1, 1)
        return position_enc.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, H, D], got {tuple(x.shape)}")
        horizon = int(x.shape[1])
        if horizon <= 0 or (horizon & (horizon - 1)) != 0:
            raise ValueError(
                f"EqNet requires power-of-two horizon, got horizon={horizon}. "
                "Use horizon in {32, 64, 128, ...}."
            )

        x = x.permute(0, 2, 1)
        emb = self.map_noise(noise.reshape(-1).to(dtype=torch.float32))
        if not self.use_timestep_emb:
            emb = emb * 0.0
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + torch.zeros_like(emb)
        emb = self.map_emb(emb)

        for i, layer in enumerate(self.conv_layers):
            if self.encode_position and i == 1:
                horizon_length = int(x.shape[-1])
                pos_enc = self._sinusoidal_time_embeddings(
                    size=horizon_length,
                    length=self.channel_size,
                    batch_size=int(x.shape[0]),
                    device=x.device,
                    dtype=x.dtype,
                )
                x = x + pos_enc
            x = layer(x, emb)

        return x.permute(0, 2, 1)


class EqNetAdapter(nn.Module):
    """
    Adapter that matches Diffuser TemporalUnet interface: forward(x, cond, time).
    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        emb_dim: int = 32,
        model_dim: int = 32,
        use_timestep_emb: bool = True,
        n_layers: int = 25,
        kernel_size: int = 3,
        kernel_expansion_rate: int = 5,
        encode_position: bool = False,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.transition_dim = int(transition_dim)
        self.cond_dim = int(cond_dim)
        self.eqnet = _EqNetCore(
            in_dim=self.transition_dim,
            emb_dim=int(emb_dim),
            model_dim=int(model_dim),
            use_timestep_emb=bool(use_timestep_emb),
            n_layers=int(n_layers),
            kernel_size=int(kernel_size),
            kernel_expansion_rate=int(kernel_expansion_rate),
            encode_position=bool(encode_position),
        )

    def forward(self, x: torch.Tensor, cond: object, time: torch.Tensor) -> torch.Tensor:
        # cond is a dictionary of inpainting constraints in this codebase; EqNet does not consume it.
        _ = cond
        return self.eqnet(x=x, noise=time, condition=None)
