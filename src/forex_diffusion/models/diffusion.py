"""
Diffusion utilities and model for latent diffusion (v-prediction, cosine schedule).

- cosine_alphas: compute alpha_bar schedule (Nichol & Dhariwal cosine)
- TimeEmbedding: sinusoidal timestep embedding
- DiffusionModel: simple MLP predictor for v (can be substituted with Temporal U-Net/DiT)
- q_sample / z0_from_v_and_zt utilities for v-parametrization
- sampler_ddim: deterministic DDIM sampler (eta controllable; eta=0 deterministic)
- sampler_dpmpp_heun: simple Heun-style integrator placeholder for DPM-++ (2nd order)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alphas(T: int = 1000, s: float = 0.008):
    """
    Compute cosine schedule alpha_bar[t] for t=0..T following Nichol & Dhariwal.
    Returns dict with arrays (torch tensors) for alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar.
    """
    steps = T + 1
    t = torch.linspace(0, T, steps) / float(T)
    f = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f
    alpha_bar = torch.clamp(alpha_bar, min=1e-12, max=1.0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
    return {
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
    }


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time-step embedding similar to Transformer/Imagen.
    Input: scalar timesteps t (LongTensor)
    Output: embedding vector of dim 'dim'
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) long tensor of timesteps
        returns (B, dim) float tensor
        """
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / float(half))
        args = t.unsqueeze(1).float() * freqs.to(t.device).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class SimpleConditioning(nn.Module):
    """
    Small MLP to encode optional conditioning h (multi-scale context) and horizon embedding.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DiffusionModel(nn.Module):
    """
    Simple v-prediction model operating on latent vectors z (B, z_dim).
    This is a placeholder for a Temporal U-Net / DiT architecture.
    The model receives:
      - z_t: (B, z_dim)
      - t: (B,) timesteps
      - cond: optional tensor (B, C_cond) multi-scale conditioning
      - e_h: optional horizon embedding (B, dim)
    Returns:
      - v_hat: (B, z_dim)
    """
    def __init__(self, z_dim: int = 128, time_emb_dim: int = 128, cond_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.z_dim = z_dim
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_proj = SimpleConditioning(cond_dim, time_emb_dim) if cond_dim is not None else None

        # Core MLP that consumes [z_t, time_emb, cond_emb]
        input_dim = z_dim + time_emb_dim + (time_emb_dim if self.cond_proj is not None else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None, e_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass returns v_hat with same shape as z_t.
        """
        B = z_t.shape[0]
        te = self.time_emb(t)  # (B, time_emb_dim)
        if self.cond_proj is not None and cond is not None:
            cemb = self.cond_proj(cond)
            inp = torch.cat([z_t, te, cemb], dim=-1)
        else:
            inp = torch.cat([z_t, te], dim=-1)
        v_hat = self.net(inp)
        return v_hat


# Utility functions for v-parametrization
def q_sample_from_z0(z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor], schedule: dict):
    """
    q(z_t | z0) = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * noise
    z0: (B, D)
    t: (B,) long tensor of timesteps
    noise: (B, D) if None, sample standard normal
    schedule: result of cosine_alphas
    returns z_t, eps (noise used)
    """
    device = z0.device
    alpha_bar = schedule["alpha_bar"].to(device)
    sqrt_alpha_bar = schedule["sqrt_alpha_bar"].to(device)
    sqrt_one_minus_alpha_bar = schedule["sqrt_one_minus_alpha_bar"].to(device)
    # Gather per-sample scalars
    a_t = sqrt_alpha_bar[t].unsqueeze(-1)  # (B,1)
    b_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
    if noise is None:
        noise = torch.randn_like(z0)
    z_t = a_t * z0 + b_t * noise
    return z_t, noise


def z0_from_v_and_zt(v: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, schedule: dict) -> torch.Tensor:
    """
    Recover z0 from predicted v and current z_t using:
    z0 = sqrt(alpha_bar_t) * z_t - sqrt(1 - alpha_bar_t) * v
    """
    device = z_t.device
    sqrt_alpha_bar = schedule["sqrt_alpha_bar"].to(device)
    sqrt_one_minus_alpha_bar = schedule["sqrt_one_minus_alpha_bar"].to(device)
    a_t = sqrt_alpha_bar[t].unsqueeze(-1)
    b_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
    z0 = a_t * z_t - b_t * v
    return z0


def eps_from_z0_and_zt(z0: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, schedule: dict) -> torch.Tensor:
    """
    eps = (z_t - sqrt_alpha_bar * z0) / sqrt(1 - alpha_bar)
    """
    device = z_t.device
    sqrt_alpha_bar = schedule["sqrt_alpha_bar"].to(device)
    sqrt_one_minus_alpha_bar = schedule["sqrt_one_minus_alpha_bar"].to(device)
    a_t = sqrt_alpha_bar[t].unsqueeze(-1)
    b_t = sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
    eps = (z_t - a_t * z0) / b_t
    return eps


def ddim_step(z_t: torch.Tensor, t: int, t_prev: int, v_hat: torch.Tensor, schedule: dict) -> torch.Tensor:
    """
    One DDIM deterministic step from t -> t_prev given predicted v_hat.
    Using conversion: z0_hat = sqrt(alpha_bar_t) * z_t - sqrt(1-alpha_bar_t) * v_hat
    eps_hat = (z_t - sqrt(alpha_bar_t) * z0_hat) / sqrt(1-alpha_bar_t)
    z_{t_prev} = sqrt(alpha_bar_{t_prev}) * z0_hat + sqrt(1 - alpha_bar_{t_prev}) * eps_hat
    """
    device = z_t.device
    # Build tensors for indices
    T_t = torch.full((z_t.shape[0],), t, dtype=torch.long, device=device)
    T_prev = torch.full((z_t.shape[0],), t_prev, dtype=torch.long, device=device)
    z0_hat = z0_from_v_and_zt(v_hat, z_t, T_t, schedule)
    eps_hat = eps_from_z0_and_zt(z0_hat, z_t, T_t, schedule)
    a_prev = schedule["sqrt_alpha_bar"].to(device)[T_prev].unsqueeze(-1)
    b_prev = schedule["sqrt_one_minus_alpha_bar"].to(device)[T_prev].unsqueeze(-1)
    z_prev = a_prev * z0_hat + b_prev * eps_hat
    return z_prev


def sampler_ddim(
    model: nn.Module,
    z_init: Optional[torch.Tensor],
    shape: Tuple[int, int],
    steps: int = 20,
    eta: float = 0.0,
    device: Optional[torch.device] = None,
    schedule: Optional[dict] = None,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    DDIM sampler for v-prediction.
    - model: predicts v_hat given (z_t, t, cond)
    - z_init: optional initial noise (B, D); if None sample standard normal
    - shape: (B, D) used if z_init is None
    - steps: number of timesteps (<= schedule['T'])
    - eta: noise scale (eta=0 deterministic)
    Returns z0 estimate after sampling.
    """
    if steps > 20:
        raise ValueError("Sampler steps must be <= 20 for production safety.")
    device = device or (z_init.device if z_init is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if schedule is None:
        schedule = cosine_alphas(T=1000, s=0.008)
    T = schedule["alpha_bar"].shape[0] - 1
    # Choose timesteps linearly spaced from T..0 with 'steps' points
    timesteps = list(
        reversed([int(x) for x in torch.linspace(0, T, steps + 1).tolist()])  # reversed so start from T downwards
    )
    # timesteps is length steps+1; we iterate pairs (t_i -> t_{i+1})
    if z_init is None:
        z = torch.randn(*shape, device=device)
    else:
        z = z_init.to(device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        # model expects t tensor
        t_tensor = torch.full((z.shape[0],), t, dtype=torch.long, device=device)
        v_hat = model(z, t_tensor, cond)  # predict v
        # deterministic z_prev via DDIM formula (eta ignored if 0)
        z = ddim_step(z, t, t_prev, v_hat, schedule)
        # if eta > 0 add stochasticity (not fully implemented, approximate)
        if eta > 0.0:
            sigma = eta * torch.sqrt(torch.clamp(1.0 - schedule["alpha_bar"][t_prev], min=0.0))
            z = z + sigma.to(device).unsqueeze(-1) * torch.randn_like(z)
    # After loop, estimate z0 from last v prediction at t=0
    t0 = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
    v_final = model(z, t0, cond)
    z0 = z0_from_v_and_zt(v_final, z, t0, schedule)
    return z0


def sampler_dpmpp_heun(
    model: nn.Module,
    z_init: Optional[torch.Tensor],
    shape: Tuple[int, int],
    steps: int = 20,
    device: Optional[torch.device] = None,
    schedule: Optional[dict] = None,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simple Heun-style integrator as a DPM-++ placeholder (2nd order).
    This is a pragmatic implementation providing higher-order integration over the score field.
    It is not a full DPM-++ reference but suitable as MVP.

    steps limited to <= 20.
    """
    if steps > 20:
        raise ValueError("Sampler steps must be <= 20 for production safety.")
    device = device or (z_init.device if z_init is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if schedule is None:
        schedule = cosine_alphas(T=1000, s=0.008)
    T = schedule["alpha_bar"].shape[0] - 1
    timesteps = list(reversed([int(x) for x in torch.linspace(0, T, steps + 1).tolist()]))

    if z_init is None:
        z = torch.randn(*shape, device=device)
    else:
        z = z_init.to(device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        t_tensor = torch.full((z.shape[0],), t, dtype=torch.long, device=device)
        # Predictor step (Euler)
        v_hat1 = model(z, t_tensor, cond)
        z0_hat1 = z0_from_v_and_zt(v_hat1, z, t_tensor, schedule)
        eps_hat1 = eps_from_z0_and_zt(z0_hat1, z, t_tensor, schedule)
        # Euler prediction for z at t_prev
        a_prev = schedule["sqrt_alpha_bar"].to(device)[t_prev]
        b_prev = schedule["sqrt_one_minus_alpha_bar"].to(device)[t_prev]
        a_t = schedule["sqrt_alpha_bar"].to(device)[t]
        b_t = schedule["sqrt_one_minus_alpha_bar"].to(device)[t]
        # single-step Euler approximation of change (not exact): compute z_euler
        z_euler = a_prev * z0_hat1.unsqueeze(0) + b_prev * eps_hat1
        # Corrector step
        t_prev_tensor = torch.full((z.shape[0],), t_prev, dtype=torch.long, device=device)
        v_hat2 = model(z_euler, t_prev_tensor, cond)
        z0_hat2 = z0_from_v_and_zt(v_hat2, z_euler, t_prev_tensor, schedule)
        eps_hat2 = eps_from_z0_and_zt(z0_hat2, z_euler, t_prev_tensor, schedule)
        # Heun update (average slopes)
        z = 0.5 * (a_prev * z0_hat1 + b_prev * eps_hat1 + a_prev * z0_hat2 + b_prev * eps_hat2)
    # final z0
    t0 = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
    v_final = model(z, t0, cond)
    z0 = z0_from_v_and_zt(v_final, z, t0, schedule)
    return z0
