import pytest
from src.forex_diffusion.utils.config import get_config


def test_config_load():
    cfg = get_config()
    assert hasattr(cfg, "app")
    assert cfg.app.name == "magicforex"
    # sampler max steps enforced
    assert cfg.sampler.max_steps <= 20
