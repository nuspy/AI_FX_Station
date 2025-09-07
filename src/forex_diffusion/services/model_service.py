"""
ModelService: load model artifacts (VAE + Diffusion) and provide forecast sampling API.
"""
ModelService: load model artifacts (VAE + Diffusion) and provide forecast sampling API.

- Tries to load latest checkpoint from artifacts dir (supports checkpoints with keys 'vae' and 'diffusion'
  or separate files vae.pt / diffusion.pt).
- If model artifacts not found or loading fails, falls back to Random-Walk (log-normal) sampler.
- Exposes forecast(symbol, timeframe, horizons, N_samples, apply_conformal) -> dict same contract as inference.service expects.
"""

from __future__ import annotations

import os
import glob
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from ..utils.config import get_config
from ..models.vae import VAE
from ..models.diffusion import DiffusionModel, sampler_ddim, sampler_dpmpp_heun, cosine_alphas


class ModelService:
    """
    Higher-level ModelService that can load trained artifacts and perform sampling.

    If no artifact found, provides RW fallback.
    """
    def __init__(self, engine: Any, artifacts_dir: Optional[str] = None, device: Optional[str] = None):
        self.engine = engine
        self.cfg = get_config()
        self.artifacts_dir = artifacts_dir or getattr(self.cfg, "model", {}).get("artifacts_dir", "./artifacts/models")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vae: Optional[VAE] = None
        self.diffusion: Optional[DiffusionModel] = None
        self.schedule = None
        self._loaded = False

        # Attempt to load latest checkpoint at init
        try:
            self._try_load_latest()
        except Exception as e:
            logger.warning("ModelService: no model artifact loaded: {}", e)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find latest .pt/.pth file in artifacts dir."""
        patterns = ["*.pt", "*.pth"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(self.artifacts_dir, p)))
        if not files:
            return None
        files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]

    def _try_load_latest(self) -> bool:
        ckpt = self._find_latest_checkpoint()
        if ckpt is None:
            return False
        logger.info("ModelService: attempting to load checkpoint {}", ckpt)
        try:
            data = torch.load(ckpt, map_location="cpu")
            # data may be dict with keys 'vae', 'diffusion' or state_dicts
            vae_state = None
            diff_state = None
            if isinstance(data, dict):
                if "vae" in data:
                    vae_state = data["vae"]
                if "diffusion" in data:
                    diff_state = data["diffusion"]
                # allow flat state_dicts
                if any(k.startswith("encoder") or k.startswith("fc_mu") for k in data.keys()):
                    vae_state = data
            # Fallback: attempt to load separate files
            # Instantiate models using config params
            vae_cfg = getattr(self.cfg, "vae", {}) if hasattr(self.cfg, "vae") else {}
            in_channels = len(vae_cfg.get("channels", ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"]))
            patch_len = int(vae_cfg.get("patch_len", 64))
            z_dim = int(vae_cfg.get("z_dim", 128))
            hidden = int(vae_cfg.get("encoder", {}).get("hidden_channels", 256))
            n_layers = int(vae_cfg.get("encoder", {}).get("n_layers", 6))
            self.vae = VAE(in_channels=in_channels, patch_len=patch_len, z_dim=z_dim, hidden_channels=hidden, n_down=n_layers)
            # Diffusion model
            diff_cfg = getattr(self.cfg, "diffusion", {}) if hasattr(self.cfg, "diffusion") else {}
            time_emb = int(diff_cfg.get("conditioning", {}).get("horizon_embedding_dim", 64))
            cond_dim = int(diff_cfg.get("conditioning", {}).get("symbol_embedding_dim", 32)) if diff_cfg.get("conditioning", {}).get("symbol_embedding_dim", None) else None
            self.diffusion = DiffusionModel(z_dim=z_dim, time_emb_dim=time_emb, cond_dim=cond_dim, hidden_dim=512)

            # load states if present
            if vae_state is not None:
                try:
                    self.vae.load_state_dict(vae_state)
                except Exception:
                    # if the saved dict is nested, try 'state_dict'
                    if "state_dict" in vae_state:
                        self.vae.load_state_dict(vae_state["state_dict"])
            if diff_state is not None:
                try:
                    self.diffusion.load_state_dict(diff_state)
                except Exception:
                    if "state_dict" in diff_state:
                        self.diffusion.load_state_dict(diff_state["state_dict"])

            # Move to device and eval
            self.vae.to(self.device).eval()
            self.diffusion.to(self.device).eval()

            # schedule
            T = int(diff_cfg.get("T", 1000))
            s = float(diff_cfg.get("schedule", {}).get("s", 0.008))
            self.schedule = cosine_alphas(T=T, s=s)

            self._loaded = True
            logger.info("ModelService: loaded model artifacts successfully")
            return True
        except Exception as e:
            logger.exception("Failed loading model artifact {}: {}", ckpt, e)
            return False

    def is_model_loaded(self) -> bool:
        return self._loaded and (self.vae is not None and self.diffusion is not None)

    # Utility to get last close / recent closes (similar to inference.service earlier)
    def _get_last_close_and_recent(self, symbol: str, timeframe: str, n: int = 1024):
        from sqlalchemy import select, MetaData
        meta = MetaData()
        meta.reflect(bind=self.engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            raise RuntimeError("market_data_candles table not found")
        with self.engine.connect() as conn:
            stmt = select(tbl.c.ts_utc, tbl.c.close).where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe).order_by(tbl.c.ts_utc.desc()).limit(n)
            rows = conn.execute(stmt).fetchall()
            if not rows:
                raise RuntimeError(f"No historical data for {symbol} {timeframe}")
            closes = np.array([float(r[1]) for r in rows[::-1]])
            last_close = float(rows[0][1])
            ts = np.array([int(r[0]) for r in rows[::-1]])
        return {"last_close": last_close, "recent_closes": closes, "recent_ts": ts}

    def _estimate_sigma(self, recent_closes: np.ndarray, window: int = 100) -> float:
        if recent_closes is None or len(recent_closes) < 2:
            return 1e-6
        r = np.diff(np.log(recent_closes))
        if r.size > window:
            r = r[-window:]
        sigma = float(np.nanstd(r, ddof=1))
        if sigma <= 0:
            sigma = 1e-6
        return sigma

    def _rw_forecast_quantiles(self, last_close: float, sigma_1: float, h_min: int, N_samples: int):
        factor = np.sqrt(max(1.0, float(h_min)))
        Z = np.random.randn(N_samples)
        sim_prices = last_close * np.exp(sigma_1 * factor * Z)
        qs = np.quantile(sim_prices, [0.05, 0.5, 0.95])
        return sim_prices, qs

    def forecast(self, symbol: str, timeframe: str, horizons: List[str], N_samples: int = 200, apply_conformal: bool = True) -> Dict[str, Any]:
        """
        Forecast API: attempts to use loaded models; if not available uses RW fallback.
        Returns dict: {quantiles, bands_conformal, credibility, diagnostics, samples(optional)}
        """
        label2min = {}
        cfg_h = getattr(self.cfg, "horizons", None)
        if isinstance(cfg_h, dict):
            for h in cfg_h.get("list", []):
                label2min[str(h.get("label"))] = int(h.get("minutes"))
        else:
            try:
                for item in self.cfg.horizons.list:
                    label2min[item["label"]] = int(item["minutes"])
            except Exception:
                pass

        # gather recent closes
        try:
            rec = self._get_last_close_and_recent(symbol, timeframe, n=1024)
        except Exception as e:
            logger.error("ModelService.forecast: failed to read recent closes: {}", e)
            raise

        last_close = rec["last_close"]
        recent_closes = rec["recent_closes"]
        sigma_1 = self._estimate_sigma(recent_closes, window=100)

        quantiles_out = {}
        bands = {}
        credibility = {}
        diagnostics = {"model_loaded": self.is_model_loaded(), "model_sigma_estimate": float(sigma_1)}

        # If model loaded, attempt to sample trajectories via diffusion in latent space
        if self.is_model_loaded():
            device = self.device
            # sample shape (N_samples, z_dim)
            z_shape = (N_samples, self.vae.z_dim)
            # Use sampler (DDIM by default)
            sampler_name = getattr(self.cfg, "sampler", {}).get("default", "ddim")
            steps = int(getattr(self.cfg, "sampler", {}).get(sampler_name, {}).get("steps", 20))
            steps = min(steps, int(getattr(self.cfg, "sampler", {}).get("max_steps", 20)))
            schedule = self.schedule or cosine_alphas()
            z_init = torch.randn(*z_shape, device=device)
            try:
                if sampler_name == "ddim":
                    z0 = sampler_ddim(self.diffusion, z_init, shape=z_shape, steps=steps, eta=float(getattr(self.cfg, "sampler", {}).get("ddim", {}).get("eta", 0.0)), device=device, schedule=schedule, cond=None)
                else:
                    z0 = sampler_dpmpp_heun(self.diffusion, z_init, shape=z_shape, steps=steps, device=device, schedule=schedule, cond=None)
                # decode all z0 to trajectories in data space via vae.decode -> (N, C, L)
                with torch.no_grad():
                    x_recon = self.vae.decode(z0.to(device)).cpu().numpy()  # (N, C, L)
                # extract last close from channel index assumed to be 3 (best-effort)
                c_index = min(3, x_recon.shape[1] - 1)
                last_close_preds = x_recon[:, c_index, -1]  # (N,)
                # For each horizon, use same samples (model-level should produce multi-horizon; MVP uses same)
                for h_label in horizons:
                    h_min = label2min.get(h_label)
                    if h_min is None:
                        if h_label.endswith("m"):
                            h_min = int(h_label[:-1])
                        elif h_label.endswith("h"):
                            h_min = int(h_label[:-1]) * 60
                        elif h_label.endswith("d"):
                            h_min = int(h_label[:-1]) * 1440
                        else:
                            h_min = 1
                    # For MVP use last_close_preds as proxy for horizon predictions
                    qs = np.quantile(last_close_preds, [0.05, 0.5, 0.95])
                    quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}
                    bands[h_label] = {"low": float(qs[0]), "high": float(qs[2])}
                    credibility[h_label] = 1.0  # placeholder high because model sampling used
                diagnostics["samples"] = last_close_preds.tolist()
            except Exception as e:
                logger.exception("ModelService: sampling using model failed, falling back to RW: {}", e)
                diagnostics["sampling_error"] = str(e)
                # fallback to RW for all horizons
                for h_label in horizons:
                    h_min = label2min.get(h_label) or (int(h_label[:-1]) if h_label.endswith("m") else 1)
                    samples, qs = self._rw_forecast_quantiles(last_close, sigma_1, h_min, N_samples)
                    quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}
                    bands[h_label] = {"low": float(qs[0]), "high": float(qs[2])}
                    credibility[h_label] = 0.5
                    diagnostics.setdefault("fallback_rw", True)
        else:
            # RW fallback
            for h_label in horizons:
                h_min = label2min.get(h_label)
                if h_min is None:
                    if h_label.endswith("m"):
                        h_min = int(h_label[:-1])
                    elif h_label.endswith("h"):
                        h_min = int(h_label[:-1]) * 60
                    elif h_label.endswith("d"):
                        h_min = int(h_label[:-1]) * 1440
                    else:
                        h_min = 1
                samples, qs = self._rw_forecast_quantiles(last_close, sigma_1, h_min, N_samples)
                quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}
                bands[h_label] = {"low": float(qs[0]), "high": float(qs[2])}
                credibility[h_label] = float(1.0 / (1.0 + (np.ptp(samples) / (sigma_1 + 1e-9))))

        # Conformal calibration TODO: integrate historical predicted quantiles and apply weighted ICP
        # For now delta=0
        result = {"quantiles": quantiles_out, "bands_conformal": bands, "credibility": credibility, "diagnostics": diagnostics}
        return result
- Tries to load latest checkpoint from artifacts dir (supports checkpoints with keys 'vae' and 'diffusion'
  or separate files vae.pt / diffusion.pt).
- If model artifacts not found or loading fails, falls back to Random-Walk (log-normal) sampler.
- Exposes forecast(symbol, timeframe, horizons, N_samples, apply_conformal) -> dict same contract as inference.service expects.
"""

from __future__ import annotations

import os
import glob
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from ..utils.config import get_config
from ..models.vae import VAE
from ..models.diffusion import DiffusionModel, sampler_ddim, sampler_dpmpp_heun, cosine_alphas


class ModelService:
    """
    Higher-level ModelService that can load trained artifacts and perform sampling.

    If no artifact found, provides RW fallback.
    """
    def __init__(self, engine: Any, artifacts_dir: Optional[str] = None, device: Optional[str] = None):
        self.engine = engine
        self.cfg = get_config()
        self.artifacts_dir = artifacts_dir or getattr(self.cfg, "model", {}).get("artifacts_dir", "./artifacts/models")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vae: Optional[VAE] = None
        self.diffusion: Optional[DiffusionModel] = None
        self.schedule = None
        self._loaded = False

        # Attempt to load latest checkpoint at init
        try:
            self._try_load_latest()
        except Exception as e:
            logger.warning("ModelService: no model artifact loaded: {}", e)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find latest .pt/.pth file in artifacts dir."""
        patterns = ["*.pt", "*.pth"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(self.artifacts_dir, p)))
        if not files:
            return None
        files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]

    def _try_load_latest(self) -> bool:
        ckpt = self._find_latest_checkpoint()
        if ckpt is None:
            return False
        logger.info("ModelService: attempting to load checkpoint {}", ckpt)
        try:
            data = torch.load(ckpt, map_location="cpu")
            # data may be dict with keys 'vae', 'diffusion' or state_dicts
            vae_state = None
            diff_state = None
            if isinstance(data, dict):
                if "vae" in data:
                    vae_state = data["vae"]
                if "diffusion" in data:
                    diff_state = data["diffusion"]
                # allow flat state_dicts
                if any(k.startswith("encoder") or k.startswith("fc_mu") for k in data.keys()):
                    vae_state = data
            # Fallback: attempt to load separate files
            # Instantiate models using config params
            vae_cfg = getattr(self.cfg, "vae", {}) if hasattr(self.cfg, "vae") else {}
            in_channels = len(vae_cfg.get("channels", ["open", "high", "low", "close", "volume", "hour_sin", "hour_cos"]))
            patch_len = int(vae_cfg.get("patch_len", 64))
            z_dim = int(vae_cfg.get("z_dim", 128))
            hidden = int(vae_cfg.get("encoder", {}).get("hidden_channels", 256))
            n_layers = int(vae_cfg.get("encoder", {}).get("n_layers", 6))
            self.vae = VAE(in_channels=in_channels, patch_len=patch_len, z_dim=z_dim, hidden_channels=hidden, n_down=n_layers)
            # Diffusion model
            diff_cfg = getattr(self.cfg, "diffusion", {}) if hasattr(self.cfg, "diffusion") else {}
            time_emb = int(diff_cfg.get("conditioning", {}).get("horizon_embedding_dim", 64))
            cond_dim = int(diff_cfg.get("conditioning", {}).get("symbol_embedding_dim", 32)) if diff_cfg.get("conditioning", {}).get("symbol_embedding_dim", None) else None
            self.diffusion = DiffusionModel(z_dim=z_dim, time_emb_dim=time_emb, cond_dim=cond_dim, hidden_dim=512)

            # load states if present
            if vae_state is not None:
                try:
                    self.vae.load_state_dict(vae_state)
                except Exception:
                    # if the saved dict is nested, try 'state_dict'
                    if "state_dict" in vae_state:
                        self.vae.load_state_dict(vae_state["state_dict"])
            if diff_state is not None:
                try:
                    self.diffusion.load_state_dict(diff_state)
                except Exception:
                    if "state_dict" in diff_state:
                        self.diffusion.load_state_dict(diff_state["state_dict"])

            # Move to device and eval
            self.vae.to(self.device).eval()
            self.diffusion.to(self.device).eval()

            # schedule
            T = int(diff_cfg.get("T", 1000))
            s = float(diff_cfg.get("schedule", {}).get("s", 0.008))
            self.schedule = cosine_alphas(T=T, s=s)

            self._loaded = True
            logger.info("ModelService: loaded model artifacts successfully")
            return True
        except Exception as e:
            logger.exception("Failed loading model artifact {}: {}", ckpt, e)
            return False

    def is_model_loaded(self) -> bool:
        return self._loaded and (self.vae is not None and self.diffusion is not None)

    # Utility to get last close / recent closes (similar to inference.service earlier)
    def _get_last_close_and_recent(self, symbol: str, timeframe: str, n: int = 1024):
        from sqlalchemy import select, MetaData
        meta = MetaData()
        meta.reflect(bind=self.engine, only=["market_data_candles"])
        tbl = meta.tables.get("market_data_candles")
        if tbl is None:
            raise RuntimeError("market_data_candles table not found")
        with self.engine.connect() as conn:
            stmt = select(tbl.c.ts_utc, tbl.c.close).where(tbl.c.symbol == symbol).where(tbl.c.timeframe == timeframe).order_by(tbl.c.ts_utc.desc()).limit(n)
            rows = conn.execute(stmt).fetchall()
            if not rows:
                raise RuntimeError(f"No historical data for {symbol} {timeframe}")
            closes = np.array([float(r[1]) for r in rows[::-1]])
            last_close = float(rows[0][1])
            ts = np.array([int(r[0]) for r in rows[::-1]])
        return {"last_close": last_close, "recent_closes": closes, "recent_ts": ts}

    def _estimate_sigma(self, recent_closes: np.ndarray, window: int = 100) -> float:
        if recent_closes is None or len(recent_closes) < 2:
            return 1e-6
        r = np.diff(np.log(recent_closes))
        if r.size > window:
            r = r[-window:]
        sigma = float(np.nanstd(r, ddof=1))
        if sigma <= 0:
            sigma = 1e-6
        return sigma

    def _rw_forecast_quantiles(self, last_close: float, sigma_1: float, h_min: int, N_samples: int):
        factor = np.sqrt(max(1.0, float(h_min)))
        Z = np.random.randn(N_samples)
        sim_prices = last_close * np.exp(sigma_1 * factor * Z)
        qs = np.quantile(sim_prices, [0.05, 0.5, 0.95])
        return sim_prices, qs

    def forecast(self, symbol: str, timeframe: str, horizons: List[str], N_samples: int = 200, apply_conformal: bool = True) -> Dict[str, Any]:
        """
        Forecast API: attempts to use loaded models; if not available uses RW fallback.
        Returns dict: {quantiles, bands_conformal, credibility, diagnostics, samples(optional)}
        """
        label2min = {}
        cfg_h = getattr(self.cfg, "horizons", None)
        if isinstance(cfg_h, dict):
            for h in cfg_h.get("list", []):
                label2min[str(h.get("label"))] = int(h.get("minutes"))
        else:
            try:
                for item in self.cfg.horizons.list:
                    label2min[item["label"]] = int(item["minutes"])
            except Exception:
                pass

        # gather recent closes
        try:
            rec = self._get_last_close_and_recent(symbol, timeframe, n=1024)
        except Exception as e:
            logger.error("ModelService.forecast: failed to read recent closes: {}", e)
            raise

        last_close = rec["last_close"]
        recent_closes = rec["recent_closes"]
        sigma_1 = self._estimate_sigma(recent_closes, window=100)

        quantiles_out = {}
        bands = {}
        credibility = {}
        diagnostics = {"model_loaded": self.is_model_loaded(), "model_sigma_estimate": float(sigma_1)}

        # If model loaded, attempt to sample trajectories via diffusion in latent space
        if self.is_model_loaded():
            device = self.device
            # sample shape (N_samples, z_dim)
            z_shape = (N_samples, self.vae.z_dim)
            # Use sampler (DDIM by default)
            sampler_name = getattr(self.cfg, "sampler", {}).get("default", "ddim")
            steps = int(getattr(self.cfg, "sampler", {}).get(sampler_name, {}).get("steps", 20))
            steps = min(steps, int(getattr(self.cfg, "sampler", {}).get("max_steps", 20)))
            schedule = self.schedule or cosine_alphas()
            z_init = torch.randn(*z_shape, device=device)
            try:
                if sampler_name == "ddim":
                    z0 = sampler_ddim(self.diffusion, z_init, shape=z_shape, steps=steps, eta=float(getattr(self.cfg, "sampler", {}).get("ddim", {}).get("eta", 0.0)), device=device, schedule=schedule, cond=None)
                else:
                    z0 = sampler_dpmpp_heun(self.diffusion, z_init, shape=z_shape, steps=steps, device=device, schedule=schedule, cond=None)
                # decode all z0 to trajectories in data space via vae.decode -> (N, C, L)
                with torch.no_grad():
                    x_recon = self.vae.decode(z0.to(device)).cpu().numpy()  # (N, C, L)
                # extract last close from channel index assumed to be 3 (best-effort)
                c_index = min(3, x_recon.shape[1] - 1)
                last_close_preds = x_recon[:, c_index, -1]  # (N,)
                # For each horizon, use same samples (model-level should produce multi-horizon; MVP uses same)
                for h_label in horizons:
                    h_min = label2min.get(h_label)
                    if h_min is None:
                        if h_label.endswith("m"):
                            h_min = int(h_label[:-1])
                        elif h_label.endswith("h"):
                            h_min = int(h_label[:-1]) * 60
                        elif h_label.endswith("d"):
                            h_min = int(h_label[:-1]) * 1440
                        else:
                            h_min = 1
                    # For MVP use last_close_preds as proxy for horizon predictions
                    qs = np.quantile(last_close_preds, [0.05, 0.5, 0.95])
                    quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}
                    bands[h_label] = {"low": float(qs[0]), "high": float(qs[2])}
                    credibility[h_label] = 1.0  # placeholder high because model sampling used
                diagnostics["samples"] = last_close_preds.tolist()
            except Exception as e:
                logger.exception("ModelService: sampling using model failed, falling back to RW: {}", e)
                diagnostics["sampling_error"] = str(e)
                # fallback to RW for all horizons
                for h_label in horizons:
                    h_min = label2min.get(h_label) or (int(h_label[:-1]) if h_label.endswith("m") else 1)
                    samples, qs = self._rw_forecast_quantiles(last_close, sigma_1, h_min, N_samples)
                    quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}
                    bands[h_label] = {"low": float(qs[0]), "high": float(qs[2])}
                    credibility[h_label] = 0.5
                    diagnostics.setdefault("fallback_rw", True)
        else:
            # RW fallback
            for h_label in horizons:
                h_min = label2min.get(h_label)
                if h_min is None:
                    if h_label.endswith("m"):
                        h_min = int(h_label[:-1])
                    elif h_label.endswith("h"):
                        h_min = int(h_label[:-1]) * 60
                    elif h_label.endswith("d"):
                        h_min = int(h_label[:-1]) * 1440
                    else:
                        h_min = 1
                samples, qs = self._rw_forecast_quantiles(last_close, sigma_1, h_min, N_samples)
                quantiles_out[h_label] = {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}
                bands[h_label] = {"low": float(qs[0]), "high": float(qs[2])}
                credibility[h_label] = float(1.0 / (1.0 + (np.ptp(samples) / (sigma_1 + 1e-9))))

        # Conformal calibration TODO: integrate historical predicted quantiles and apply weighted ICP
        # For now delta=0
        result = {"quantiles": quantiles_out, "bands_conformal": bands, "credibility": credibility, "diagnostics": diagnostics}
        return result
