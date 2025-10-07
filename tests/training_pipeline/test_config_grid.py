"""
Unit tests for config_grid module.

Tests configuration grid generation, hashing, and deduplication.
"""

import pytest
from src.forex_diffusion.training.training_pipeline.config_grid import (
    compute_config_hash,
    generate_config_grid,
    add_config_hashes,
    deduplicate_configs,
    filter_already_trained,
    validate_config,
    get_config_summary
)


class TestConfigHashing:
    """Tests for configuration hashing."""

    def test_compute_config_hash_deterministic(self):
        """Test that same config produces same hash."""
        config1 = {
            'model_type': 'xgboost',
            'symbol': 'EURUSD',
            'base_timeframe': '1H',
            'days_history': 60
        }

        config2 = {
            'model_type': 'xgboost',
            'symbol': 'EURUSD',
            'base_timeframe': '1H',
            'days_history': 60
        }

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_config_hash_different(self):
        """Test that different configs produce different hashes."""
        config1 = {
            'model_type': 'xgboost',
            'symbol': 'EURUSD'
        }

        config2 = {
            'model_type': 'lightgbm',
            'symbol': 'EURUSD'
        }

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 != hash2

    def test_compute_config_hash_order_independent(self):
        """Test that field order doesn't affect hash."""
        config1 = {
            'model_type': 'xgboost',
            'symbol': 'EURUSD',
            'days_history': 60
        }

        config2 = {
            'days_history': 60,
            'symbol': 'EURUSD',
            'model_type': 'xgboost'
        }

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2


class TestGridGeneration:
    """Tests for configuration grid generation."""

    def test_generate_config_grid_single_values(self):
        """Test grid with single values."""
        grid_params = {
            'model_type': ['xgboost'],
            'symbol': ['EURUSD'],
            'days_history': [60]
        }

        configs = generate_config_grid(grid_params)

        assert len(configs) == 1
        assert configs[0]['model_type'] == 'xgboost'
        assert configs[0]['symbol'] == 'EURUSD'
        assert configs[0]['days_history'] == 60

    def test_generate_config_grid_cartesian_product(self):
        """Test Cartesian product generation."""
        grid_params = {
            'model_type': ['xgboost', 'lightgbm'],
            'symbol': ['EURUSD', 'GBPUSD'],
            'days_history': [30, 60]
        }

        configs = generate_config_grid(grid_params)

        # 2 * 2 * 2 = 8 combinations
        assert len(configs) == 8

        # Check all combinations exist
        model_types = set(c['model_type'] for c in configs)
        symbols = set(c['symbol'] for c in configs)
        days = set(c['days_history'] for c in configs)

        assert model_types == {'xgboost', 'lightgbm'}
        assert symbols == {'EURUSD', 'GBPUSD'}
        assert days == {30, 60}

    def test_generate_config_grid_empty(self):
        """Test empty grid params."""
        grid_params = {}
        configs = generate_config_grid(grid_params)

        assert len(configs) == 1  # One empty config
        assert configs[0] == {}


class TestConfigHashing:
    """Tests for adding hashes to configs."""

    def test_add_config_hashes(self):
        """Test adding hashes to config list."""
        configs = [
            {'model_type': 'xgboost', 'symbol': 'EURUSD'},
            {'model_type': 'lightgbm', 'symbol': 'EURUSD'}
        ]

        configs_with_hash = add_config_hashes(configs)

        assert len(configs_with_hash) == 2
        assert all('config_hash' in c for c in configs_with_hash)
        assert configs_with_hash[0]['config_hash'] != configs_with_hash[1]['config_hash']

    def test_add_config_hashes_preserves_data(self):
        """Test that original config data is preserved."""
        configs = [
            {'model_type': 'xgboost', 'symbol': 'EURUSD', 'days': 60}
        ]

        configs_with_hash = add_config_hashes(configs)

        assert configs_with_hash[0]['model_type'] == 'xgboost'
        assert configs_with_hash[0]['symbol'] == 'EURUSD'
        assert configs_with_hash[0]['days'] == 60


class TestDeduplication:
    """Tests for configuration deduplication."""

    def test_deduplicate_configs_no_duplicates(self):
        """Test deduplication with no duplicates."""
        configs = [
            {'config_hash': 'hash1', 'model_type': 'xgboost'},
            {'config_hash': 'hash2', 'model_type': 'lightgbm'}
        ]

        deduped = deduplicate_configs(configs)

        assert len(deduped) == 2

    def test_deduplicate_configs_with_duplicates(self):
        """Test deduplication removes duplicates."""
        configs = [
            {'config_hash': 'hash1', 'model_type': 'xgboost'},
            {'config_hash': 'hash1', 'model_type': 'xgboost'},  # Duplicate
            {'config_hash': 'hash2', 'model_type': 'lightgbm'}
        ]

        deduped = deduplicate_configs(configs)

        assert len(deduped) == 2
        hashes = [c['config_hash'] for c in deduped]
        assert 'hash1' in hashes
        assert 'hash2' in hashes

    def test_deduplicate_configs_keeps_first(self):
        """Test that first occurrence is kept."""
        configs = [
            {'config_hash': 'hash1', 'model_type': 'xgboost', 'order': 1},
            {'config_hash': 'hash1', 'model_type': 'xgboost', 'order': 2}
        ]

        deduped = deduplicate_configs(configs)

        assert len(deduped) == 1
        assert deduped[0]['order'] == 1  # First one kept


class TestFilterAlreadyTrained:
    """Tests for filtering already-trained configurations."""

    def test_filter_already_trained_none_trained(self):
        """Test with no already-trained configs."""
        configs = [
            {'config_hash': 'hash1'},
            {'config_hash': 'hash2'}
        ]

        trained_hashes = set()

        remaining, already_trained = filter_already_trained(configs, trained_hashes)

        assert len(remaining) == 2
        assert len(already_trained) == 0

    def test_filter_already_trained_some_trained(self):
        """Test with some already-trained configs."""
        configs = [
            {'config_hash': 'hash1'},
            {'config_hash': 'hash2'},
            {'config_hash': 'hash3'}
        ]

        trained_hashes = {'hash2'}

        remaining, already_trained = filter_already_trained(configs, trained_hashes)

        assert len(remaining) == 2
        assert len(already_trained) == 1
        assert already_trained[0]['config_hash'] == 'hash2'

    def test_filter_already_trained_all_trained(self):
        """Test when all configs are already trained."""
        configs = [
            {'config_hash': 'hash1'},
            {'config_hash': 'hash2'}
        ]

        trained_hashes = {'hash1', 'hash2'}

        remaining, already_trained = filter_already_trained(configs, trained_hashes)

        assert len(remaining) == 0
        assert len(already_trained) == 2


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_config_valid(self):
        """Test validation of valid config."""
        config = {
            'model_type': 'xgboost',
            'symbol': 'EURUSD',
            'base_timeframe': '1H',
            'days_history': 60,
            'horizon': 1
        }

        # Should not raise
        validate_config(config)

    def test_validate_config_missing_required(self):
        """Test validation fails with missing required fields."""
        config = {
            'model_type': 'xgboost'
            # Missing symbol, etc.
        }

        with pytest.raises(ValueError):
            validate_config(config)

    def test_validate_config_invalid_model_type(self):
        """Test validation fails with invalid model type."""
        config = {
            'model_type': 'invalid_model',
            'symbol': 'EURUSD',
            'base_timeframe': '1H',
            'days_history': 60,
            'horizon': 1
        }

        with pytest.raises(ValueError):
            validate_config(config)


class TestConfigSummary:
    """Tests for configuration summary generation."""

    def test_get_config_summary(self):
        """Test summary generation."""
        configs = [
            {'model_type': 'xgboost', 'symbol': 'EURUSD'},
            {'model_type': 'lightgbm', 'symbol': 'EURUSD'},
            {'model_type': 'xgboost', 'symbol': 'GBPUSD'}
        ]

        summary = get_config_summary(configs)

        assert summary['total_configs'] == 3
        assert 'xgboost' in summary['model_types']
        assert 'lightgbm' in summary['model_types']
        assert 'EURUSD' in summary['symbols']
        assert 'GBPUSD' in summary['symbols']

    def test_get_config_summary_empty(self):
        """Test summary with empty list."""
        configs = []
        summary = get_config_summary(configs)

        assert summary['total_configs'] == 0
        assert summary['model_types'] == set()
        assert summary['symbols'] == set()
