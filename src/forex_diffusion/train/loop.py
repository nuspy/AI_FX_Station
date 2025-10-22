"""
PyTorch-Lightning training loop for MagicForex.

- LightningModule implements combined loss:
    L = lambda_v * MSE(v, v_hat) + lambda_crps * CRPS(decoded_samples, y) + lambda_kl * KL
- Uses VAE encoder/decoder for latent z, DiffusionModel for v-prediction (v-parametrization).
- CRPS computed by sample estimator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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

        # Helper function to safely extract nested config values
        def safe_get(obj, *keys, default=None):
            """Safely navigate nested config attributes/dicts, including Pydantic extra fields"""
            for key in keys:
                if obj is None:
                    return default
                # Try attribute access first (Pydantic models)
                if hasattr(obj, key):
                    obj = getattr(obj, key)
                # Try Pydantic model_dump for extra fields
                elif hasattr(obj, 'model_dump'):
                    try:
                        obj_dict = obj.model_dump()
                        obj = obj_dict.get(key)
                        if obj is None:
                            return default
                    except Exception:
                        return default
                # Fall back to dict access
                elif isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    return default
            return obj if obj is not None else default

        # Extract config sections safely
        model_cfg = getattr(self.cfg, "model", None)
        vae_cfg = getattr(self.cfg, "vae", None) if hasattr(self.cfg, "vae") else None
        diff_cfg = getattr(self.cfg, "diffusion", None) if hasattr(self.cfg, "diffusion") else None
        training_cfg = getattr(self.cfg, "training", None)

        # Default channels
        default_channels = ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"]
        channels = safe_get(vae_cfg, "channels", default=default_channels) or default_channels
        in_ch = len(channels) if isinstance(channels, list) else safe_get(vae_cfg, "in_channels", default=len(default_channels))
        in_ch = int(in_ch)  # Ensure it's an integer

        logger.debug(f"[ForexDiffusionLit] Channels: {channels}, in_ch: {in_ch}")

        patch_len = int(safe_get(vae_cfg, "patch_len", default=None) or safe_get(model_cfg, "patch_len", default=64))
        z_dim = int(safe_get(vae_cfg, "z_dim", default=None) or safe_get(model_cfg, "z_dim", default=128))

        # instantiate VAE and DiffusionModel (simple MLP predictor)
        hidden_channels = int(safe_get(vae_cfg, "encoder", "hidden_channels", default=256))
        n_down = int(safe_get(vae_cfg, "encoder", "n_layers", default=6))

        logger.debug(f"[ForexDiffusionLit] Initializing VAE with: in_ch={in_ch}, patch_len={patch_len}, z_dim={z_dim}, hidden_channels={hidden_channels}, n_down={n_down}")

        self.vae = VAE(in_channels=in_ch, patch_len=patch_len, z_dim=z_dim, hidden_channels=hidden_channels, n_down=n_down)

        logger.debug(f"[ForexDiffusionLit] VAE encoder flattened size: {self.vae._enc_flat}")

        # Build a simple diffusion model (z_dim -> v)
        from ..models.diffusion import DiffusionModel
        time_emb_dim = int(safe_get(diff_cfg, "conditioning", "horizon_embedding_dim", default=64))
        # Only use conditioning if dataset provides it (cond_* columns exist)
        # For now, disable conditioning since dataset doesn't include it
        # TODO: Add multi-scale pooling and symbol embeddings to dataset
        cond_dim = None  # Disable conditioning for now
        self.diffusion_model = DiffusionModel(z_dim=z_dim, time_emb_dim=time_emb_dim, cond_dim=cond_dim, hidden_dim=512)

        logger.debug(f"[ForexDiffusionLit] DiffusionModel input_dim: z_dim={z_dim} + time_emb={time_emb_dim} + cond={0 if cond_dim is None else time_emb_dim} = {z_dim + time_emb_dim + (0 if cond_dim is None else time_emb_dim)}")
        
        # Multi-Horizon Prediction Head
        # Predicts different values for each horizon from latent z
        # Architecture: z_dim -> hidden -> num_horizons
        num_horizons = int(safe_get(model_cfg, "num_horizons", default=1))

        # Fallback for flat hparams from checkpoint
        if num_horizons == 1 and hasattr(self.cfg, 'num_horizons'):
            try:
                num_horizons = int(self.cfg.num_horizons)
                logger.info(f"[ForexDiffusionLit] Overriding num_horizons from flat hparams: {num_horizons}")
            except (TypeError, ValueError):
                pass
        if num_horizons > 1:
            self.multi_horizon_head = torch.nn.Sequential(
                torch.nn.Linear(z_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, num_horizons)
            )
            logger.info(f"[ForexDiffusionLit] Multi-horizon head initialized: z_dim={z_dim} -> {num_horizons} horizons")
        else:
            self.multi_horizon_head = None
            logger.debug("[ForexDiffusionLit] Single-horizon mode (no prediction head)")

        # schedule
        T = int(safe_get(diff_cfg, "T", default=1000))
        s = float(safe_get(diff_cfg, "schedule", "s", default=0.008))
        self.schedule = cosine_alphas(T=T, s=s)

        # training weights
        self.lambda_v = float(safe_get(diff_cfg, "training", "lambda_v", default=1.0))
        self.lambda_crps = float(safe_get(diff_cfg, "training", "lambda_crps", default=1.0))
        self.lambda_kl = float(safe_get(diff_cfg, "training", "lambda_kl", default=0.01))

        # number of samples to compute CRPS
        inference_cfg = getattr(self.cfg, "inference", None) if hasattr(self.cfg, "inference") else None
        self.crps_samples = int(safe_get(inference_cfg, "n_samples", default=100))
        # small safety cap
        self.crps_samples = min(max(8, self.crps_samples), 512)

        # beta scheduler for KL (VAE)
        from ..models.vae import BetaScheduler
        kl_anneal_type = safe_get(vae_cfg, "loss", "kl_anneal", "type", default="logistic")
        kl_warmup = int(safe_get(vae_cfg, "loss", "kl_anneal", "warmup_steps", default=10000))
        kl_k = float(safe_get(vae_cfg, "loss", "kl_anneal", "k", default=0.002))
        kl_beta_max = float(safe_get(vae_cfg, "loss", "kl_weight_max", default=1.0))
        self.beta_scheduler = BetaScheduler(kind=kl_anneal_type, warmup_steps=kl_warmup, k=kl_k, beta_max=kl_beta_max)

        # optimizer params
        self.lr = float(safe_get(training_cfg, "learning_rate", default=2e-4))
        self.weight_decay = float(safe_get(training_cfg, "weight_decay", default=1e-6))

        # example inputs for logging shapes
        self.example_input_array = torch.randn(2, in_ch, patch_len)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        Forward pass for inference/summary.
        Encodes input to latent space and returns VAE reconstruction.

        Args:
            x: (B, C, L) input patch
            cond: optional conditioning tensor

        Returns:
            Reconstructed input tensor (B, C, L)
        """
        mu, logvar = self.vae.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu  # Use mean for deterministic forward (no sampling)
        x_rec = self.vae.decode(z)
        return x_rec

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

        # persist latents async if db_writer and meta info present in batch
        try:
            if getattr(self, "db_writer", None) is not None and "meta" in batch and batch["meta"] is not None:
                meta = batch["meta"]
                symbol = meta.get("symbol")
                timeframe = meta.get("timeframe")
                # ts list: if batch contains ts_utc vector use it else None
                ts_arr = meta.get("ts_utc")  # expected list/array aligned with batch
                # convert z0 to python lists and enqueue by sample
                z_list = z0.detach().cpu().numpy().tolist()
                for i, z_vec in enumerate(z_list):
                    ts_val = int(ts_arr[i]) if ts_arr is not None and i < len(ts_arr) else None
                    self.db_writer.write_latents_async(symbol=symbol, timeframe=timeframe, ts_utc=ts_val or int(time.time() * 1000), latent=z_vec, model_version=self.pipeline_version)
        except Exception:
            # non-blocking, ignore failures in persist
            pass

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
        
        # Determine number of horizons from target shape
        num_horizons = 1
        if y is not None:
            y_target = y if y.dim() == 2 else y.unsqueeze(-1)  # (B, H) or (B, 1)
            num_horizons = y_target.shape[1]
        
        # Multi-horizon prediction head
        # If model has explicit multi-horizon decoder, use it; otherwise use single decoder
        # For now, we implement a simple approach: use z0_hat to predict all horizons
        # In future, could add a dedicated MLP head: z0 -> (H,) predictions
        
        samples_all_horizons = []  # Will be list of tensors (N, B, H)
        
        with torch.no_grad():
            for i in range(n_samples):
                eps_s = torch.randn_like(std)
                z_s = mu + eps_s * std
                x_rec = self.vae.decode(z_s)  # (B, C, L)
                
                # Multi-horizon prediction
                if num_horizons > 1 and self.multi_horizon_head is not None:
                    # Use dedicated MLP head to predict all horizons from latent
                    horizon_preds = self.multi_horizon_head(z_s)  # (B, H)
                    samples_all_horizons.append(horizon_preds)
                else:
                    # Fallback: extract close from reconstruction
                    # Assume channel 3 is 'close' (common convention)
                    c_index = min(3, x_rec.shape[1] - 1)
                    close_pred = x_rec[:, c_index, -1]  # (B,) - last timestep
                    
                    if num_horizons > 1:
                        # Replicate for multi-horizon (legacy fallback)
                        horizon_preds = close_pred.unsqueeze(-1).repeat(1, num_horizons)  # (B, H)
                        samples_all_horizons.append(horizon_preds)
                    else:
                        # Single horizon
                        samples_all_horizons.append(close_pred.unsqueeze(-1))  # (B, 1)
        
        samples = torch.stack(samples_all_horizons, dim=0)  # (N, B, H)

        crps = torch.tensor(0.0, device=device)
        if y is not None:
            # Compute CRPS for each horizon and average
            if num_horizons > 1:
                # Multi-horizon CRPS: average across all horizons
                crps_per_horizon = []
                for h in range(num_horizons):
                    samples_h = samples[:, :, h:h+1]  # (N, B, 1)
                    y_h = y_target[:, h:h+1]  # (B, 1)
                    crps_h = crps_sample_estimator(samples_h, y_h)
                    crps_per_horizon.append(crps_h)
                
                # Average CRPS across horizons
                crps = torch.stack(crps_per_horizon).mean()
                
                # Log per-horizon CRPS for monitoring
                for h, crps_h in enumerate(crps_per_horizon):
                    self.log(f"train/crps_h{h}", crps_h, on_step=False, on_epoch=True, prog_bar=False)
            else:
                # Single horizon
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
