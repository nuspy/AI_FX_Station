"""
VAE 1D module for MagicForex.

- Encoder/Decoder operate on input patches X ∈ R^{B, C, L} where L is patch_len (default 64)
- Encoder parametrizes q_phi(z|X) -> mu, logvar
- Reparameterization trick implemented
- Decoder reconstructs X_hat from z
- BetaScheduler implements KL annealing (logistic | linear)
- Utility functions: kl_divergence
"""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Simple 1D convolutional block: Conv1d -> GroupNorm -> SiLU
    Works for downsampling when stride=2.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None, groups: int = 8):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        # GroupNorm requires num_groups <= out_ch
        gn_groups = min(groups, out_ch) if out_ch > 0 else 1
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        return self.act(x)


class UpConvBlock(nn.Module):
    """
    Transposed convolutional block for decoder upsampling.
    ConvTranspose1d -> GroupNorm -> SiLU
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4, stride: int = 2, padding: int = 1, groups: int = 8):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        gn_groups = min(groups, out_ch) if out_ch > 0 else 1
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = self.gn(x)
        return self.act(x)


class BetaScheduler:
    """
    KL weight scheduler (beta) for annealing.
    Supports 'logistic' and 'linear' schedules.
    """
    def __init__(self, kind: str = "logistic", warmup_steps: int = 10000, k: float = 0.002, beta_max: float = 1.0):
        self.kind = kind
        self.warmup_steps = max(1, int(warmup_steps))
        self.k = float(k)
        self.beta_max = float(beta_max)

    def value(self, step: int) -> float:
        if self.kind == "linear":
            return min(self.beta_max, float(step) / float(self.warmup_steps) * self.beta_max)
        # logistic
        step = float(step)
        x = (step - self.warmup_steps / 2.0) * self.k
        beta = float(self.beta_max / (1.0 + math.exp(-x)))
        return beta


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """
    Compute KL divergence between q(z|x)=N(mu, sigma^2) and p(z)=N(0, I).
    Returns scalar (batch-summed) or per-batch if reduction='none'/'mean'.
    """
    # KL per dimension: 0.5 * (mu^2 + var - logvar - 1)
    var = torch.exp(logvar)
    kl = 0.5 * (mu.pow(2) + var - logvar - 1.0)
    if reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.sum(dim=1).mean()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class VAE(nn.Module):
    """
    1D VAE suitable for time-series patches.

    Args:
        in_channels: number of input channels (C)
        patch_len: length L of the input patch (default 64)
        z_dim: latent dimensionality
        hidden_channels: base number of channels for conv stacks
        n_down: number of downsampling stages (each stage halves length if stride=2 used)
    """
    def __init__(
        self,
        in_channels: int = 6,
        patch_len: int = 64,
        z_dim: int = 128,
        hidden_channels: int = 128,
        n_down: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_len = patch_len
        self.z_dim = z_dim
        self.hidden_channels = hidden_channels
        self.n_down = n_down

        # Build encoder
        enc_layers = []
        ch = in_channels
        out_ch = hidden_channels
        for i in range(n_down):
            enc_layers.append(ConvBlock(ch, out_ch, kernel_size=3, stride=2))
            ch = out_ch
            out_ch = min(out_ch * 2, 1024)
        # One extra conv without downsample to increase receptive field
        enc_layers.append(ConvBlock(ch, ch, kernel_size=3, stride=1))
        self.encoder = nn.Sequential(*enc_layers)

        # Compute flattened size after convs by doing a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_len)
            enc_out = self.encoder(dummy)
            _, c_out, l_out = enc_out.shape
            self._enc_c_out = c_out
            self._enc_l_out = l_out
            self._enc_flat = c_out * l_out

        # Map to mu / logvar
        self.fc_mu = nn.Linear(self._enc_flat, z_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, z_dim)

        # Decoder: project z to conv shape then upsample with ConvTranspose
        self.fc_dec = nn.Linear(z_dim, self._enc_flat)

        dec_layers = []
        ch = self._enc_c_out
        # mirror of encoder: upsample n_down times
        for i in range(n_down):
            # upconv doubles length
            out_ch = max(ch // 2, in_channels)
            dec_layers.append(UpConvBlock(ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        # final conv to reconstruct channels
        dec_layers.append(nn.Conv1d(ch, in_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

        # Final activation linear for reconstruction (regression), no activation
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input x (B, C, L) -> (mu, logvar) both shape (B, z_dim)
        """
        h = self.encoder(x)
        h_flat = h.view(h.shape[0], -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * eps
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean for deterministic eval
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z (B, z_dim) -> reconstructed x_hat (B, C, L)
        """
        h = self.fc_dec(z)
        h = h.view(h.shape[0], self._enc_c_out, self._enc_l_out)
        x_hat = self.decoder(h)
        # Ensure output length equals patch_len (may differ by 1 due to rounding); center-crop or pad as needed
        if x_hat.shape[-1] != self.patch_len:
            diff = x_hat.shape[-1] - self.patch_len
            if diff > 0:
                # crop
                start = diff // 2
                x_hat = x_hat[..., start : start + self.patch_len]
            else:
                # pad
                pad = (-diff) // 2
                x_hat = F.pad(x_hat, (pad, pad), mode="replicate")
                if x_hat.shape[-1] != self.patch_len:
                    x_hat = x_hat[..., : self.patch_len]
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: returns (x_hat, mu, logvar, z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def sample_prior(self, n: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from prior p(z)=N(0,I) and decode to data space.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=device)
        return self.decode(z)

    def latent_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mu, logvar for input x without sampling.
        """
        return self.encode(x)


if __name__ == "__main__":
    # Quick smoke test
    model = VAE(in_channels=6, patch_len=64, z_dim=64, hidden_channels=64, n_down=3)
    x = torch.randn(2, 6, 64)
    x_hat, mu, logvar, z = model(x)
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    kl = kl_divergence(mu, logvar, reduction="mean")
    print("x_hat.shape", x_hat.shape, "mu.shape", mu.shape, "z.shape", z.shape, "recon", recon_loss.item(), "kl", kl.item())
