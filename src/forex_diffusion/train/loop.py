"""
PyTorch-Lightning training loop for MagicForex.

- LightningModule implements combined loss:
    L = lambda_v * MSE(v, v_hat) + lambda_crps * CRPS(decoded_samples, y) + lambda_kl * KL
- Uses VAE encoder/decoder for latent z, DiffusionModel for v-prediction (v-parametrization).
- CRPS computed by sample estimator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from loguru import logger

from ..utils.config import get_config
from ..models.vae import VAE, kl_divergence
from ..models.diffusion import cosine_alphas, q_sample_from_z0, z0_from_v_and_zt

# CRPS sample estimator
def crps_sample_estimator(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Sample-based CRPS estimator.
    samples: (N, B, D) or (N, B) -- N samples
    y: (B, D) or (B,)
    Returns scalar average CRPS across batch and dims.
    CRPS = mean_i |S_i - y| - 0.5 * mean_{i,j} |S_i - S_j|
    """
    if samples.dim() == 2:
        N, B = samples.shape
        D = 1
        samples_ = samples.unsqueeze(-1)  # (N,B,1)
        y_ = y.unsqueeze(-1)
    elif samples.dim() == 3:
        N, B, D = samples.shape
        samples_ = samples
        y_ = y
    else:
        raise ValueError("samples tensor must be 2D or 3D")

    # abs differences to y
    # shape (N, B, D)
    abs_diff = torch.abs(samples_ - y_.unsqueeze(0))
    term1 = abs_diff.mean(dim=0)  # (B,D) mean over samples
    term1 = term1.mean()  # scalar

    # pairwise absolute differences between samples
    # compute efficiently: for each pair i<j sum |S_i - S_j|
    # we compute full matrix then average
    # shape expand for broadcasting
    s1 = samples_.unsqueeze(0)  # (1, N, B, D)
    s2 = samples_.unsqueeze(1)  # (N, 1, B, D)
    # compute |si - sj| -> (N, N, B, D)
    pair_abs = torch.abs(s1 - s2)
    # average over pairs (including i=j zeros) => mean over first two dims
    term2 = pair_abs.mean(dim=(0, 1)).mean()  # scalar

    crps = term1 - 0.5 * term2
    return crps


class ForexDiffusionLit(pl.LightningModule):
    """
    LightningModule that coordinates training of VAE and Diffusion model.
    Expects training batches with:
      - x: past patch tensor shape (B, C, L)
      - y: target values for CRPS (B, D) -- e.g., close at future horizon(s)
      - optional cond: conditioning tensor for diffusion model
    """
    def __init__(self, cfg: Optional[Any] = None):
        super().__init__()
        self.cfg = cfg or get_config()
        # model hyperparams from config
        model_cfg = getattr(self.cfg, "model", {})
        vae_cfg = getattr(self.cfg, "vae", {}) if hasattr(self.cfg, "vae") else {}
        diff_cfg = getattr(self.cfg, "diffusion", {}) if hasattr(self.cfg, "diffusion") else {}
        training_cfg = getattr(self.cfg, "training", {})

        in_channels = len(getattr(self.cfg, "features", {}).get("standardization", {}).get("cols", [])) if False else vae_cfg.get("channels", ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"])
        in_ch = len(in_channels) if isinstance(in_channels, list) else int(vae_cfg.get("in_channels", len(vae_cfg.get("channels", ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"]))))
        patch_len = int(vae_cfg.get("patch_len", model_cfg.get("patch_len", 64)))
        z_dim = int(vae_cfg.get("z_dim", model_cfg.get("z_dim", 128)))

        # instantiate VAE and DiffusionModel (simple MLP predictor)
        self.vae = VAE(in_channels=in_ch, patch_len=patch_len, z_dim=z_dim, hidden_channels=int(vae_cfg.get("encoder", {}).get("hidden_channels", 256)), n_down=int(vae_cfg.get("encoder", {}).get("n_layers", 6)))
        # Build a simple diffusion model (z_dim -> v)
        from ..models.diffusion import DiffusionModel
        time_emb_dim = int(diff_cfg.get("conditioning", {}).get("horizon_embedding_dim", 64))
        cond_dim = int(diff_cfg.get("conditioning", {}).get("symbol_embedding_dim", 32)) if diff_cfg.get("conditioning", {}).get("symbol_embedding_dim", None) else None
        self.diffusion_model = DiffusionModel(z_dim=z_dim, time_emb_dim=time_emb_dim, cond_dim=cond_dim, hidden_dim=512)

        # schedule
        T = int(diff_cfg.get("T", 1000))
        s = float(diff_cfg.get("schedule", {}).get("s", 0.008))
        self.schedule = cosine_alphas(T=T, s=s)

        # training weights
        self.lambda_v = float(diff_cfg.get("training", {}).get("lambda_v", 1.0))
        self.lambda_crps = float(diff_cfg.get("training", {}).get("lambda_crps", 1.0))
        self.lambda_kl = float(diff_cfg.get("training", {}).get("lambda_kl", 0.01))

        # number of samples to compute CRPS
        self.crps_samples = int(getattr(self.cfg, "inference", {}).get("n_samples", 100))
        # small safety cap
        self.crps_samples = min(max(8, self.crps_samples), 512)

        # beta scheduler for KL (VAE)
        from ..models.vae import BetaScheduler
        kl_anneal_cfg = vae_cfg.get("loss", {}).get("kl_anneal", {})
        self.beta_scheduler = BetaScheduler(kind=kl_anneal_cfg.get("type", "logistic"), warmup_steps=int(kl_anneal_cfg.get("warmup_steps", 10000)), k=float(kl_anneal_cfg.get("k", 0.002)), beta_max=float(vae_cfg.get("loss", {}).get("kl_weight_max", 1.0)))

        # optimizer params
        self.lr = float(training_cfg.get("learning_rate", 2e-4))
        self.weight_decay = float(training_cfg.get("weight_decay", 1e-6))

        # example inputs for logging shapes
        self.example_input_array = torch.randn(2, in_ch, patch_len)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use training_step/validation_step")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Batch must contain:
         - x: (B, C, L) input patch
         - y: (B, D) target(s) for CRPS (e.g. future close(s))
         - optional cond: conditioning tensor for diffusion model
        """
        x = batch["x"]
        y = batch.get("y", None)
        cond = batch.get("cond", None)

        # Encode to latent z0 (mu, logvar)
        mu, logvar = self.vae.encode(x)
        # sample z0 from q_phi
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std  # (B, z_dim)

        B = z0.shape[0]
        device = z0.device

        # sample timestep t uniformly between 1 and T (avoid t=0)
        T = self.schedule["alpha_bar"].shape[0] - 1
        t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)

        # sample noise for forward q
        noise = torch.randn_like(z0)
        # compute z_t from z0
        z_t, used_noise = q_sample_from_z0(z0, t, noise=noise, schedule=self.schedule)

        # compute true v per v-prediction definition: v = sqrt_alpha_t * eps - sqrt_one_minus_alpha_t * z0
        sqrt_alpha_bar = self.schedule["sqrt_alpha_bar"].to(device)
        sqrt_one_minus = self.schedule["sqrt_one_minus_alpha_bar"].to(device)
        a_t = sqrt_alpha_bar[t].unsqueeze(-1)
        b_t = sqrt_one_minus[t].unsqueeze(-1)
        v_true = a_t * used_noise - b_t * z0

        # predict v
        v_hat = self.diffusion_model(z_t, t, cond)

        # loss on v (MSE)
        loss_v = nn.functional.mse_loss(v_hat, v_true, reduction="mean")

        # KL divergence
        loss_kl = kl_divergence(mu, logvar, reduction="mean")

        # produce z0_hat via predicted v and z_t
        z0_hat = z0_from_v_and_zt(v_hat, z_t, t, self.schedule)

        # For CRPS: draw multiple samples from posterior q_phi(z|x) and decode
        # We'll sample num_samples from posterior by sampling eps ~ N(0,1) and computing z = mu + sigma * eps, then decode
        n_samples = min(self.crps_samples, 128)  # limit per-batch for memory reasons
        samples = []
        with torch.no_grad():
            for i in range(n_samples):
                eps_s = torch.randn_like(std)
                z_s = mu + eps_s * std
                x_rec = self.vae.decode(z_s)  # (B, C, L)
                # choose target dimension: close at last timestep of patch (index -1)
                # or other aggregator; here we take reconstructed close last element in channel order
                # we assume input channels order includes 'close' at index 3 (common). If not, caller must provide mapping.
                # For safety, try to take center channel as close
                c_index = min(3, x_rec.shape[1] - 1)
                close_pred = x_rec[:, c_index, -1].unsqueeze(-1)  # (B,1)
                samples.append(close_pred)
        samples = torch.stack(samples, dim=0)  # (N, B, 1)

        crps = torch.tensor(0.0, device=device)
        if y is not None:
            # ensure y shape matches (B,1)
            y_target = y if y.dim() == 2 else y.unsqueeze(-1)
            crps = crps_sample_estimator(samples, y_target)

        loss = self.lambda_v * loss_v + self.lambda_crps * crps + self.lambda_kl * loss_kl

        # logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_v", loss_v, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss_kl", loss_kl, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/crps", crps, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["x"]
        y = batch.get("y", None)
        cond = batch.get("cond", None)

        mu, logvar = self.vae.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std

        B = z0.shape[0]
        device = z0.device
        T = self.schedule["alpha_bar"].shape[0] - 1
        t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        z_t, used_noise = q_sample_from_z0(z0, t, noise=noise, schedule=self.schedule)

        sqrt_alpha_bar = self.schedule["sqrt_alpha_bar"].to(device)
        sqrt_one_minus = self.schedule["sqrt_one_minus_alpha_bar"].to(device)
        a_t = sqrt_alpha_bar[t].unsqueeze(-1)
        b_t = sqrt_one_minus[t].unsqueeze(-1)
        v_true = a_t * used_noise - b_t * z0
        v_hat = self.diffusion_model(z_t, t, cond)
        loss_v = nn.functional.mse_loss(v_hat, v_true, reduction="mean")
        loss_kl = kl_divergence(mu, logvar, reduction="mean")

        # CRPS with limited samples
        n_samples = min(self.crps_samples, 64)
        samples = []
        with torch.no_grad():
            for i in range(n_samples):
                eps_s = torch.randn_like(std)
                z_s = mu + eps_s * std
                x_rec = self.vae.decode(z_s)
                c_index = min(3, x_rec.shape[1] - 1)
                close_pred = x_rec[:, c_index, -1].unsqueeze(-1)
                samples.append(close_pred)
        samples = torch.stack(samples, dim=0)
        crps = torch.tensor(0.0, device=device)
        if y is not None:
            y_target = y if y.dim() == 2 else y.unsqueeze(-1)
            crps = crps_sample_estimator(samples, y_target)

        loss = self.lambda_v * loss_v + self.lambda_crps * crps + self.lambda_kl * loss_kl

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/loss_v", loss_v, prog_bar=False, on_epoch=True)
        self.log("val/loss_kl", loss_kl, prog_bar=False, on_epoch=True)
        self.log("val/crps", crps, prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optional LR scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
