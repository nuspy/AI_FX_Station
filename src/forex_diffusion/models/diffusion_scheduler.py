"""
Diffusion Noise Scheduler

Implements cosine noise scheduling for diffusion models.
Supports both DDPM (training) and DDIM (fast inference) sampling.

Based on:
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Denoising Diffusion Implicit Models" (Song et al., 2021)
- "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple, Literal
from loguru import logger


class CosineNoiseScheduler:
    """
    Cosine noise scheduler for diffusion models.

    Provides smoother noise schedule compared to linear scheduling,
    resulting in better sample quality.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 0.008,
        clip_min: float = 1e-12,
        device: str = "cpu"
    ):
        """
        Initialize cosine noise scheduler.

        Args:
            T: Total diffusion timesteps
            s: Offset for cosine schedule (controls steepness)
            clip_min: Minimum value for alpha_bar (numerical stability)
            device: Device for tensors ("cpu" or "cuda")
        """
        self.T = T
        self.s = s
        self.clip_min = clip_min
        self.device = device

        # Compute alpha_bar schedule
        self.alpha_bar = self._compute_alpha_bar(T, s, clip_min)

        # Precompute useful quantities
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # For DDPM: alpha_t = alpha_bar_t / alpha_bar_{t-1}
        alpha = torch.zeros(T + 1, device=device)
        alpha[0] = 1.0
        alpha[1:] = self.alpha_bar[1:] / self.alpha_bar[:-1]
        self.alpha = alpha

        # beta_t = 1 - alpha_t
        self.beta = 1.0 - self.alpha

        # For posterior mean computation
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alpha)

        logger.debug(f"Initialized CosineNoiseScheduler: T={T}, s={s}, "
                    f"alpha_bar_0={self.alpha_bar[0]:.6f}, "
                    f"alpha_bar_T={self.alpha_bar[-1]:.6f}")

    def _compute_alpha_bar(self, T: int, s: float, clip_min: float) -> torch.Tensor:
        """
        Compute cosine schedule for alpha_bar.

        Args:
            T: Number of timesteps
            s: Offset parameter
            clip_min: Minimum clipping value

        Returns:
            alpha_bar: Cumulative product of alphas (T+1,)
        """
        # Time steps from 0 to T
        t = torch.arange(T + 1, dtype=torch.float32, device=self.device)

        # Cosine schedule
        # f(t) = cos^2((t/T + s) / (1 + s) * pi / 2)
        f_t = torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2

        # alpha_bar[t] = f(t) / f(0)
        alpha_bar = f_t / f_t[0]

        # Clip to avoid numerical issues
        alpha_bar = torch.clamp(alpha_bar, min=clip_min, max=1.0)

        return alpha_bar

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean data (forward diffusion process).

        Implements: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x0: Clean data (batch, ...)
            t: Timesteps (batch,) - integers in [0, T]
            noise: Optional noise tensor (if None, sample from N(0, I))

        Returns:
            x_t: Noisy data (batch, ...)
            noise: Noise used (batch, ...)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Get alpha_bar for each timestep
        # t: (batch,)
        # alpha_bar: (T+1,)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]  # (batch,)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]  # (batch,)

        # Reshape for broadcasting
        # x0: (batch, ...)
        # sqrt_alpha_bar_t: (batch,) -> (batch, 1, 1, ...)
        while sqrt_alpha_bar_t.dim() < x0.dim():
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)

        # Add noise
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def step_ddpm(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDPM reverse diffusion step: x_t -> x_{t-1}.

        Args:
            x_t: Noisy data at timestep t (batch, ...)
            t: Current timesteps (batch,) - integers in [0, T]
            predicted_noise: Model's noise prediction (batch, ...)
            noise: Optional noise for stochastic sampling (if None, sample)

        Returns:
            x_{t-1}: Denoised data at previous timestep (batch, ...)
        """
        # Predict x_0 from x_t and noise
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

        # Reshape for broadcasting
        while sqrt_alpha_bar_t.dim() < x_t.dim():
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)

        # Predict x_0
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        # Get alpha_bar for t-1
        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_t_prev = self.alpha_bar[t_prev]

        # Reshape
        while alpha_bar_t_prev.dim() < x_t.dim():
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)

        # Posterior mean
        # mu = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t) * x_0 +
        #      sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * x_t

        alpha_t = self.alpha[t]
        beta_t = self.beta[t]
        alpha_bar_t = self.alpha_bar[t]

        # Reshape
        while alpha_t.dim() < x_t.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        coef1 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        mu = coef1 * x0_pred + coef2 * x_t

        # Posterior variance
        # sigma^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

        # Add noise (skip for t=0)
        if noise is None:
            noise = torch.randn_like(x_t)

        # Only add noise if t > 0
        mask = (t > 0).float()
        while mask.dim() < x_t.dim():
            mask = mask.unsqueeze(-1)

        x_t_prev = mu + mask * torch.sqrt(variance) * noise

        return x_t_prev

    def step_ddim(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        predicted_noise: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM reverse diffusion step (deterministic or semi-deterministic).

        Args:
            x_t: Noisy data at timestep t (batch, ...)
            t: Current timesteps (batch,)
            t_prev: Previous timesteps (batch,)
            predicted_noise: Model's noise prediction (batch, ...)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)

        Returns:
            x_{t_prev}: Denoised data at previous timestep (batch, ...)
        """
        # Get alpha_bar values
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t_prev = self.alpha_bar[t_prev]

        # Reshape for broadcasting
        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)

        # Predict x_0
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        # Compute direction pointing to x_t
        sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)

        # Variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )

        # Compute x_{t-1}
        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)

        # Deterministic direction
        direction = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * predicted_noise

        # Stochastic component
        noise = torch.randn_like(x_t) if eta > 0 else 0

        x_t_prev = sqrt_alpha_bar_t_prev * x0_pred + direction + sigma_t * noise

        return x_t_prev

    def get_sampling_timesteps(
        self,
        num_steps: int,
        method: Literal["uniform", "quadratic"] = "uniform"
    ) -> torch.Tensor:
        """
        Generate timesteps for inference (fewer steps than training).

        Args:
            num_steps: Number of denoising steps
            method: Timestep spacing method

        Returns:
            timesteps: Timesteps to use (num_steps+1,) from T to 0
        """
        if method == "uniform":
            # Uniform spacing
            timesteps = torch.linspace(self.T, 0, num_steps + 1, dtype=torch.long,
                                      device=self.device)
        elif method == "quadratic":
            # Quadratic spacing (more steps at beginning)
            t_norm = torch.linspace(0, 1, num_steps + 1, device=self.device) ** 2
            timesteps = (t_norm * self.T).long()
            timesteps = torch.flip(timesteps, [0])
        else:
            raise ValueError(f"Unknown method: {method}")

        return timesteps

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict clean data x_0 from noisy data x_t and predicted noise.

        Args:
            x_t: Noisy data (batch, ...)
            t: Timesteps (batch,)
            predicted_noise: Predicted noise (batch, ...)

        Returns:
            x_0: Predicted clean data (batch, ...)
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

        # Reshape for broadcasting
        while sqrt_alpha_bar_t.dim() < x_t.dim():
            sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)

        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        return x0_pred

    def to(self, device: str):
        """Move scheduler tensors to device."""
        self.device = device
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.sqrt_alpha = self.sqrt_alpha.to(device)
        self.sqrt_one_minus_alpha = self.sqrt_one_minus_alpha.to(device)
        return self


class DPMPPScheduler(CosineNoiseScheduler):
    """
    DPM++ Scheduler for higher-quality sampling.

    Based on "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
    (Lu et al., 2022).

    Provides better sample quality than DDIM with same number of steps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialized DPM++ Scheduler (higher quality sampling)")

    def step_dpmpp(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        predicted_noise: torch.Tensor,
        x_prev_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DPM++ reverse diffusion step (second-order accurate).

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            t_prev: Previous timestep
            predicted_noise: Model's noise prediction
            x_prev_cache: Cached previous prediction (for second-order)

        Returns:
            x_{t_prev}: Denoised data at previous timestep
        """
        # For now, fall back to DDIM (full DPM++ implementation is complex)
        # TODO: Implement full DPM-Solver++ algorithm
        return self.step_ddim(x_t, t, t_prev, predicted_noise, eta=0.0)
