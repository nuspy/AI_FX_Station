"""
Configuration loader for MagicForex.

- Loads YAML config (default: ./configs/default.yaml)
- Merges relevant environment variables (e.g. ALPHAVANTAGE_KEY, DUKASCOPY_KEY)
- Validates important invariants (e.g. sampler.max_steps <= 20)
- Exposes a singleton get_config() for global access
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# load .env if present (python-dotenv)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # populates os.environ from .env at repo root
except Exception:
    # dotenv is optional; if not available we continue using env vars directly
    pass

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict, ValidationError


class GenericConfig(BaseModel):
    """
    Generic container allowing arbitrary nested config fields.
    Uses pydantic v2 ConfigDict with extra='allow' instead of __root__.
    """
    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> Dict[str, Any]:
        # model_dump returns a dict of all fields (including arbitrary)
        return self.model_dump()

    # Provide compatibility alias for older code expecting dict-like access
    def __getitem__(self, item):
        d = self.to_dict()
        return d.get(item)

    def get(self, item, default=None):
        d = self.to_dict()
        return d.get(item, default)


class AppConfig(BaseModel):
    name: str = "magicforex"
    debug: bool = False
    seed: int = 42
    alembic_upgrade_on_start: bool = True
    artifacts_dir: str = "./artifacts"

    model_config = ConfigDict(extra="ignore")


class DBConfig(BaseModel):
    dialect: str = "sqlite"
    database_url: str = "sqlite:///./data/forex_diffusion.db"
    pool_size: int = 5

    model_config = ConfigDict(extra="ignore")


class ProvidersConfig(BaseModel):
    default: str = "alpha_vantage"
    alpha_vantage: Dict[str, Any] = Field(default_factory=dict)
    dukascopy: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class SamplerConfig(BaseModel):
    default: str = "ddim"
    ddim: Dict[str, Any] = Field(default_factory=lambda: {"steps": 20, "eta": 0.0})
    dpmpp: Dict[str, Any] = Field(default_factory=lambda: {"steps": 20, "order": 2})
    max_steps: int = 20

    model_config = ConfigDict(extra="ignore")


class ModelConfig(BaseModel):
    artifacts_dir: str = "./artifacts/models"
    max_saved: int = 10
    versioning: bool = True
    z_dim: int = 128
    patch_len: int = 64

    model_config = ConfigDict(extra="allow")


class TrainingConfig(BaseModel):
    device: str = "auto"
    batch_size: int = 64
    num_workers: int = 8
    max_epochs: int = 200
    learning_rate: float = 2e-4

    model_config = ConfigDict(extra="allow")


class GUIConfig(BaseModel):
    provider_default: str = "alpha_vantage"
    update_interval_ms: int = 250
    batch_update_size: int = 500

    model_config = ConfigDict(extra="allow")


class Settings(BaseModel):
    """
    Top-level settings model. Unknown sections are preserved under 'extra' because
    nested custom fields are allowed via ConfigDict(extra="allow").
    """
    app: AppConfig = Field(default_factory=AppConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    gui: GUIConfig = Field(default_factory=GUIConfig)

    # allow arbitrary additional fields (data, diffusion, vae, features, etc.)
    model_config = ConfigDict(extra="allow", frozen=True)


# Prefer config file located under the repository root: <repo>/configs/default.yaml
# This avoids failures when script is run from a nested working directory (eg. src/forex_diffusion/utils).
_repo_root = Path(__file__).resolve().parents[3]  # <repo>/src/.. -> repo root
_default_config_path = _repo_root / "configs" / "default.yaml"

# Fallback: if for some reason the repo-root config doesn't exist, also consider local ./configs/default.yaml
# at runtime (preserve original behavior for callers that put configs next to the script).
if not _default_config_path.exists():
    _cwd_candidate = Path("./configs/default.yaml")
    if _cwd_candidate.exists():
        _default_config_path = _cwd_candidate

_config_singleton: Optional[Settings] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.error("Configuration file not found: {}", path)
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        logger.error("Configuration file root must be a mapping (dict).")
        raise ValueError("Configuration file root must be a mapping (dict).")
    return raw


def _merge_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides for sensitive keys and common options.
    """
    providers = cfg.get("providers", {})

    # AlphaVantage
    alpha = providers.get("alpha_vantage", {}) or {}
    av_key = os.getenv("ALPHAVANTAGE_KEY")
    if av_key:
        alpha["key"] = av_key
        logger.debug("Overriding Alpha Vantage key from environment")
    providers["alpha_vantage"] = alpha

    # Dukascopy
    duk = providers.get("dukascopy", {}) or {}
    duk_key = os.getenv("DUKASCOPY_KEY")
    if duk_key:
        duk["key"] = duk_key
        logger.debug("Overriding Dukascopy key from environment")
    providers["dukascopy"] = duk

    # Tiingo (API key)
    ti = providers.get("tiingo", {}) or {}
    ti_key = os.getenv("TIINGO_API_KEY") or os.getenv("TIINGO_KEY")
    if ti_key:
        ti["key"] = ti_key
        logger.debug("Overriding Tiingo key from environment")
    providers["tiingo"] = ti

    cfg["providers"] = providers
    return cfg


def _validate_business_rules(parsed: Settings) -> None:
    # Enforce sampler max steps safety
    try:
        max_steps = parsed.sampler.max_steps
    except Exception:
        max_steps = None

    if max_steps is not None and max_steps > 20:
        logger.error("Sampler max_steps is > 20 ({}). This is forbidden.", max_steps)
        raise ValueError("sampler.max_steps must be <= 20 for production safety.")

    # Ensure database URL is present
    if not parsed.db.database_url:
        logger.error("db.database_url is empty in configuration.")
        raise ValueError("db.database_url must be set.")


def load_config(path: Optional[str] = None, validate: bool = True) -> Settings:
    """
    Load configuration from YAML and environment, return Settings instance.

    Args:
        path: optional path to YAML config (defaults to ./configs/default.yaml)
        validate: if True, run business rule validations

    Returns:
        Settings: validated, frozen settings object
    """
    p = Path(path) if path else _default_config_path
    raw = _load_yaml(p)
    raw = _merge_env_overrides(raw)

    try:
        settings = Settings(**raw)
    except ValidationError as e:
        logger.exception("Configuration validation failed: {}", e)
        raise

    if validate:
        _validate_business_rules(settings)

    logger.info("Configuration loaded from {} (debug={})", p, settings.app.debug)
    return settings


def get_config() -> Settings:
    """
    Return a singleton Settings instance (lazy-loaded).
    """
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = load_config()
    return _config_singleton


__all__ = ["Settings", "load_config", "get_config"]
