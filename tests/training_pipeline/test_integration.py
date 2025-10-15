"""
Integration tests for training pipeline.

Tests end-to-end workflows of the two-phase training system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np


class TestTrainingPipelineIntegration:
    """Integration tests for complete training pipeline."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.test_dir / "artifacts"
        self.checkpoints_dir = self.test_dir / "checkpoints"

        self.artifacts_dir.mkdir()
        self.checkpoints_dir.mkdir()

    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def create_test_data(self, n_samples=100):
        """Create test training data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
            'open': np.random.randn(n_samples) + 1.1000,
            'high': np.random.randn(n_samples) + 1.1010,
            'low': np.random.randn(n_samples) + 1.0990,
            'close': np.random.randn(n_samples) + 1.1000,
            'volume': np.random.randint(1000, 10000, n_samples)
        })

    @pytest.mark.integration
    def test_queue_creation_and_loading(self):
        """Test creating and loading training queue."""
        from src.forex_diffusion.training.training_pipeline.training_orchestrator import TrainingOrchestrator

        orchestrator = TrainingOrchestrator(
            artifacts_dir=str(self.artifacts_dir),
            checkpoints_dir=str(self.checkpoints_dir)
        )

        # Create small grid
        grid_params = {
            'model_type': ['random_forest'],
            'symbol': ['EURUSD'],
            'base_timeframe': ['1H'],
            'days_history': [30],
            'horizon': [1],
            'encoder': ['time_delta']
        }

        # Mock database operations
        with patch('src.forex_diffusion.training.training_pipeline.database.session_scope'):
            queue_id = orchestrator.create_training_queue(
                grid_params=grid_params,
                skip_existing=False
            )

            # Verify queue was created
            assert queue_id is not None

    @pytest.mark.integration
    def test_checkpoint_save_and_load(self):
        """Test checkpoint save and load functionality."""
        from src.forex_diffusion.training.training_pipeline.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(str(self.checkpoints_dir))

        # Create test checkpoint data
        checkpoint_data = {
            'queue_id': 1,
            'queue_name': 'test_queue',
            'current_index': 5,
            'total_configs': 10,
            'status': 'running',
            'models_kept': 2,
            'models_deleted': 3
        }

        # Save checkpoint
        checkpoint_file = manager.create_checkpoint(
            queue_id=1,
            queue_name='test_queue',
            current_index=5,
            total_configs=10,
            configs=[],
            status='running',
            models_kept=2,
            models_deleted=3
        )

        assert checkpoint_file.exists()

        # Load checkpoint
        loaded_data = manager.load_checkpoint(str(checkpoint_file))

        assert loaded_data['queue_id'] == checkpoint_data['queue_id']
        assert loaded_data['current_index'] == checkpoint_data['current_index']
        assert loaded_data['models_kept'] == checkpoint_data['models_kept']

    @pytest.mark.integration
    def test_config_hash_persistence(self):
        """Test that config hashes are consistent across sessions."""
        from src.forex_diffusion.training.training_pipeline.config_grid import (
            compute_config_hash,
            generate_config_grid,
            add_config_hashes
        )

        grid_params = {
            'model_type': ['xgboost'],
            'symbol': ['EURUSD'],
            'days_history': [60]
        }

        # Generate twice
        configs1 = generate_config_grid(grid_params)
        configs1 = add_config_hashes(configs1)

        configs2 = generate_config_grid(grid_params)
        configs2 = add_config_hashes(configs2)

        # Hashes should be identical
        assert configs1[0]['config_hash'] == configs2[0]['config_hash']

    @pytest.mark.integration
    def test_model_file_manager_cleanup(self):
        """Test model file cleanup functionality."""
        from src.forex_diffusion.training.training_pipeline.model_file_manager import ModelFileManager

        manager = ModelFileManager(str(self.artifacts_dir))

        # Create dummy model files
        model1 = self.artifacts_dir / "model1.pkl"
        model2 = self.artifacts_dir / "model2.pkl"

        model1.write_text("dummy")
        model2.write_text("dummy")

        assert model1.exists()
        assert model2.exists()

        # Test deletion
        model1.unlink()
        assert not model1.exists()
        assert model2.exists()  # Other file unaffected


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    @pytest.mark.integration
    def test_config_loader_integration(self):
        """Test ConfigLoader with actual YAML file."""
        from src.forex_diffusion.training.training_pipeline.config_loader import ConfigLoader

        # Create temporary config file
        config_dir = Path(tempfile.mkdtemp())
        config_file = config_dir / "test_config.yaml"

        config_content = """
storage:
  artifacts_dir: "./test_artifacts"
  checkpoints_dir: "./test_checkpoints"

queue:
  auto_checkpoint_interval: 5
  max_inference_workers: 2

model_management:
  delete_non_best_models: true
  min_improvement_threshold: 0.05
"""

        config_file.write_text(config_content)

        try:
            loader = ConfigLoader(config_file)

            assert loader.artifacts_dir == "./test_artifacts"
            assert loader.auto_checkpoint_interval == 5
            assert loader.max_inference_workers == 2
            assert loader.delete_non_best_models is True
            assert loader.min_improvement_threshold == 0.05

        finally:
            shutil.rmtree(config_dir)

    @pytest.mark.integration
    def test_config_integration_with_orchestrator(self):
        """Test that orchestrator uses config values."""
        from src.forex_diffusion.training.training_pipeline.training_orchestrator import TrainingOrchestrator
        from src.forex_diffusion.training.training_pipeline.config_loader import get_config

        # Create orchestrator (should load from config)
        orchestrator = TrainingOrchestrator()

        config = get_config()

        # Verify orchestrator uses config values
        assert str(orchestrator.checkpoints_dir) == config.checkpoints_dir
        assert orchestrator.auto_checkpoint_interval == config.auto_checkpoint_interval


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.integration
    @pytest.mark.database
    def test_training_run_crud_operations(self):
        """Test complete CRUD cycle for training runs."""
        from src.forex_diffusion.training.training_pipeline.database import (
            session_scope,
            create_training_run,
            get_training_run_by_id,
            update_training_run_status,
            delete_training_run
        )

        with session_scope() as session:
            # Create
            run = create_training_run(
                session=session,
                model_type='xgboost',
                encoder='time_delta',
                symbol='EURUSD',
                base_timeframe='1H',
                days_history=60,
                horizon=1,
                config_hash='test_hash_123'
            )

            run_id = run.id
            assert run_id is not None

            # Read
            loaded_run = get_training_run_by_id(session, run_id)
            assert loaded_run is not None
            assert loaded_run.model_type == 'xgboost'

            # Update
            update_training_run_status(session, run_id, 'completed')
            updated_run = get_training_run_by_id(session, run_id)
            assert updated_run.status == 'completed'

            # Delete
            deleted = delete_training_run(session, run_id)
            assert deleted is True

            # Verify deletion
            deleted_run = get_training_run_by_id(session, run_id)
            assert deleted_run is None


class TestCrashRecoveryIntegration:
    """Integration tests for crash recovery system."""

    @pytest.mark.integration
    def test_crash_detection_workflow(self):
        """Test complete crash detection workflow."""
        from src.forex_diffusion.training.training_pipeline.crash_recovery import CrashRecoveryManager
        from datetime import datetime, timedelta

        # Mock database with crashed queue
        with patch('src.forex_diffusion.training.training_pipeline.database.session_scope'):
            with patch('src.forex_diffusion.training.training_pipeline.database.get_all_training_queues') as mock_get:
                # Create mock crashed queue
                mock_queue = Mock()
                mock_queue.id = 1
                mock_queue.queue_name = 'test_queue'
                mock_queue.status = 'running'
                mock_queue.created_at = datetime.now() - timedelta(hours=1)
                mock_queue.updated_at = datetime.now() - timedelta(minutes=10)  # No activity for 10 min
                mock_queue.current_index = 5
                mock_queue.total_configs = 10
                mock_queue.models_kept = 2
                mock_queue.models_deleted = 1

                mock_get.return_value = [mock_queue]

                manager = CrashRecoveryManager()

                # Detect crashed queues
                crashed = manager.detect_crashed_queues()

                assert len(crashed) == 1
                assert crashed[0]['id'] == 1
                assert crashed[0]['progress_pct'] == 50.0  # 5/10

                # Get recommendations
                recommendations = manager.get_recovery_recommendations(crashed)

                assert len(recommendations) == 1
                assert recommendations[0]['priority'] == 'medium'  # 50% progress
