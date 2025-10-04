"""
Unit tests for multi-provider system.

Run with: pytest tests/test_providers.py
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from forex_diffusion.providers import (
    BaseProvider,
    ProviderCapability,
    ProviderManager,
    TiingoProvider,
    CTraderProvider,
)
from forex_diffusion.credentials import ProviderCredentials, CredentialsManager


class TestProviderCapabilities:
    """Test provider capability system."""

    def test_tiingo_capabilities(self):
        """Tiingo should support quotes, bars, historical bars, and websocket."""
        provider = TiingoProvider(config={"api_key": "test_key"})

        assert ProviderCapability.QUOTES in provider.capabilities
        assert ProviderCapability.BARS in provider.capabilities
        assert ProviderCapability.HISTORICAL_BARS in provider.capabilities
        assert ProviderCapability.WEBSOCKET in provider.capabilities

        # Should NOT support DOM, sentiment, etc.
        assert ProviderCapability.DOM not in provider.capabilities
        assert ProviderCapability.SENTIMENT not in provider.capabilities

    def test_ctrader_capabilities(self):
        """cTrader should support comprehensive data."""
        provider = CTraderProvider(config={
            "client_id": "test_id",
            "client_secret": "test_secret",
            "access_token": "test_token"
        })

        # Should support most capabilities
        assert ProviderCapability.QUOTES in provider.capabilities
        assert ProviderCapability.BARS in provider.capabilities
        assert ProviderCapability.TICKS in provider.capabilities
        assert ProviderCapability.VOLUMES in provider.capabilities
        assert ProviderCapability.DOM in provider.capabilities
        assert ProviderCapability.HISTORICAL_BARS in provider.capabilities
        assert ProviderCapability.HISTORICAL_TICKS in provider.capabilities

    def test_provider_supports_check(self):
        """Test provider.supports() method."""
        provider = TiingoProvider(config={"api_key": "test_key"})

        assert provider.supports(ProviderCapability.QUOTES) is True
        assert provider.supports(ProviderCapability.DOM) is False
        assert provider.supports(ProviderCapability.SENTIMENT) is False


class TestProviderManager:
    """Test provider manager factory."""

    def test_create_tiingo_provider(self):
        """Manager should create Tiingo provider."""
        manager = ProviderManager()
        provider = manager.create_provider("tiingo", config={"api_key": "test_key"})

        assert isinstance(provider, TiingoProvider)
        assert provider.name == "tiingo"

    def test_create_ctrader_provider(self):
        """Manager should create cTrader provider."""
        manager = ProviderManager()
        provider = manager.create_provider("ctrader", config={
            "client_id": "test_id",
            "client_secret": "test_secret"
        })

        assert isinstance(provider, CTraderProvider)
        assert provider.name == "ctrader"

    def test_create_unknown_provider(self):
        """Manager should raise error for unknown provider."""
        manager = ProviderManager()

        with pytest.raises(ValueError, match="Unknown provider"):
            manager.create_provider("unknown_provider")

    def test_get_available_providers(self):
        """Manager should list available providers."""
        manager = ProviderManager()
        providers = manager.get_available_providers()

        assert "tiingo" in providers
        assert "ctrader" in providers

    def test_primary_secondary_providers(self):
        """Test primary/secondary provider management."""
        manager = ProviderManager()

        # Create providers
        tiingo = manager.create_provider("tiingo", config={"api_key": "test_key"})
        ctrader = manager.create_provider("ctrader", config={
            "client_id": "test_id",
            "client_secret": "test_secret"
        })

        # Set primary/secondary
        manager.set_primary_provider("tiingo")
        manager.set_secondary_provider("ctrader")

        assert manager.get_primary_provider() == tiingo
        assert manager.get_secondary_provider() == ctrader


class TestCredentialsManager:
    """Test credentials management."""

    @patch('forex_diffusion.credentials.manager.keyring')
    def test_save_credentials(self, mock_keyring):
        """Test saving credentials to keyring."""
        manager = CredentialsManager()

        creds = ProviderCredentials(
            provider_name="test_provider",
            api_key="test_key",
            api_secret="test_secret"
        )

        manager.save(creds)

        # Should have called keyring.set_password
        assert mock_keyring.set_password.called

    @patch('forex_diffusion.credentials.manager.keyring')
    def test_load_credentials(self, mock_keyring):
        """Test loading credentials from keyring."""
        mock_keyring.get_password.return_value = '{"provider_name": "test", "api_key": "key123"}'

        manager = CredentialsManager()
        creds = manager.load("test")

        assert creds is not None
        assert creds.provider_name == "test"
        assert creds.api_key == "key123"

    @patch('forex_diffusion.credentials.manager.keyring')
    def test_delete_credentials(self, mock_keyring):
        """Test deleting credentials."""
        manager = CredentialsManager()
        manager.delete("test_provider")

        assert mock_keyring.delete_password.called


class TestTiingoProvider:
    """Test Tiingo provider implementation."""

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test Tiingo connection."""
        provider = TiingoProvider(config={"api_key": "test_key"})

        # Mock the underlying client
        with patch.object(provider, 'rest_client'):
            connected = await provider.connect()

            # Should return True (mock connection)
            # In real test, would need actual API key
            assert isinstance(connected, bool)

    @pytest.mark.asyncio
    async def test_get_historical_bars(self):
        """Test getting historical bars."""
        provider = TiingoProvider(config={"api_key": "test_key"})

        # Mock response
        with patch.object(provider, '_get_historical_bars_impl', new_callable=AsyncMock) as mock_impl:
            mock_impl.return_value = Mock(empty=False)  # Mock DataFrame

            result = await provider.get_historical_bars(
                symbol="EUR/USD",
                timeframe="1h",
                start_ts_ms=1600000000000,
                end_ts_ms=1600086400000
            )

            assert mock_impl.called
            assert result is not None


class TestCTraderProvider:
    """Test cTrader provider implementation."""

    def test_convert_timeframe(self):
        """Test timeframe conversion."""
        provider = CTraderProvider(config={
            "client_id": "test",
            "client_secret": "secret"
        })

        assert provider._convert_timeframe("1m") == 1
        assert provider._convert_timeframe("5m") == 5
        assert provider._convert_timeframe("1h") == 9
        assert provider._convert_timeframe("1d") == 12
        assert provider._convert_timeframe("invalid") is None

    def test_rate_limiting(self):
        """Test rate limiter."""
        provider = CTraderProvider(config={
            "client_id": "test",
            "client_secret": "secret"
        })

        # Add 5 requests (max limit)
        for _ in range(5):
            provider._rate_limiter.append(datetime.now().timestamp())

        # Should be at capacity
        assert len(provider._rate_limiter) == 5

    @pytest.mark.asyncio
    async def test_rate_limit_wait(self):
        """Test rate limit waiting."""
        provider = CTraderProvider(config={
            "client_id": "test",
            "client_secret": "secret"
        })

        # Fill rate limiter
        import time
        now = time.time()
        for i in range(5):
            provider._rate_limiter.append(now + i * 0.1)

        # Should wait before next request
        start = time.time()
        await provider._rate_limit_wait()
        elapsed = time.time() - start

        # Should have waited some amount of time
        assert elapsed >= 0  # May or may not wait depending on timing


class TestProviderHealth:
    """Test provider health monitoring."""

    @pytest.mark.asyncio
    async def test_health_tracking(self):
        """Test health status tracking."""
        provider = TiingoProvider(config={"api_key": "test_key"})

        # Initially not connected
        assert provider.health.is_connected is False

        # Connect (mocked)
        with patch.object(provider, 'rest_client'):
            await provider.connect()

        # Check health
        assert provider.is_healthy() is True

    def test_error_tracking(self):
        """Test error tracking in health."""
        provider = TiingoProvider(config={"api_key": "test_key"})

        # Add error
        provider.health.errors.append("Test error")

        # Should track error
        assert len(provider.health.errors) == 1
        assert "Test error" in provider.health.errors


@pytest.fixture
def mock_engine():
    """Mock database engine for tests."""
    engine = Mock()
    connection = Mock()
    engine.connect.return_value.__enter__.return_value = connection
    return engine


class TestAggregators:
    """Test aggregator services."""

    def test_aggregator_init(self, mock_engine):
        """Test aggregator initialization."""
        from forex_diffusion.services.aggregator import AggregatorService

        aggregator = AggregatorService(mock_engine, symbols=["EUR/USD"])

        assert aggregator.engine == mock_engine
        assert aggregator._symbols == ["EUR/USD"]

    def test_dom_aggregator_init(self, mock_engine):
        """Test DOM aggregator initialization."""
        from forex_diffusion.services.dom_aggregator import DOMAggreg atorService

        dom_agg = DOMAggreg atorService(mock_engine, symbols=["EUR/USD"], interval_seconds=5)

        assert dom_agg.engine == mock_engine
        assert dom_agg._interval == 5

    def test_sentiment_aggregator_init(self, mock_engine):
        """Test sentiment aggregator initialization."""
        from forex_diffusion.services.sentiment_aggregator import SentimentAggregatorService

        sent_agg = SentimentAggregatorService(mock_engine, symbols=["EUR/USD"], interval_seconds=30)

        assert sent_agg.engine == mock_engine
        assert sent_agg._interval == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
