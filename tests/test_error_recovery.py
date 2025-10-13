"""
Unit tests for Error Recovery System

Tests error handling, retry logic, and emergency procedures.
Part of PROC-001 - BUG-001 (CRITICAL) testing.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from forex_diffusion.trading.error_recovery import (
    ErrorRecoveryManager,
    BrokerConnectionError,
    InsufficientFundsError,
    InvalidOrderError,
    CriticalSystemError
)


class TestErrorRecoveryBasic:
    """Basic error recovery tests"""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        manager = ErrorRecoveryManager()
        
        assert manager.max_retries == 3
        assert manager.base_delay == 1.0
        assert len(manager.error_log) == 0
    
    def test_error_logging(self):
        """Test error is logged correctly"""
        manager = ErrorRecoveryManager()
        
        error = BrokerConnectionError("Connection failed")
        manager.log_error(error, "test_operation")
        
        assert len(manager.error_log) == 1
        assert manager.error_log[0]['error_type'] == 'BrokerConnectionError'
        assert manager.error_log[0]['operation'] == 'test_operation'


class TestRetryLogic:
    """Retry and exponential backoff tests"""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful operation on first try"""
        manager = ErrorRecoveryManager()
        
        mock_func = Mock(return_value="success")
        
        result = manager.retry_with_backoff(
            func=mock_func,
            max_retries=3,
            error_types=(BrokerConnectionError,),
            operation_name="test"
        )
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test success after some failures"""
        manager = ErrorRecoveryManager()
        
        # Fail twice, then succeed
        mock_func = Mock(side_effect=[
            BrokerConnectionError("fail 1"),
            BrokerConnectionError("fail 2"),
            "success"
        ])
        
        result = manager.retry_with_backoff(
            func=mock_func,
            max_retries=3,
            error_types=(BrokerConnectionError,),
            operation_name="test"
        )
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_max_attempts_exceeded(self):
        """Test failure after max retries"""
        manager = ErrorRecoveryManager()
        
        # Always fail
        mock_func = Mock(side_effect=BrokerConnectionError("always fails"))
        
        with pytest.raises(BrokerConnectionError):
            manager.retry_with_backoff(
                func=mock_func,
                max_retries=3,
                error_types=(BrokerConnectionError,),
                operation_name="test"
            )
        
        # Should try 1 + 3 retries = 4 total
        assert mock_func.call_count == 4
    
    @patch('time.sleep')
    def test_exponential_backoff(self, mock_sleep):
        """Test exponential backoff delays"""
        manager = ErrorRecoveryManager(base_delay=1.0, max_delay=10.0)
        
        mock_func = Mock(side_effect=[
            BrokerConnectionError("fail"),
            BrokerConnectionError("fail"),
            BrokerConnectionError("fail"),
            "success"
        ])
        
        result = manager.retry_with_backoff(
            func=mock_func,
            max_retries=3,
            error_types=(BrokerConnectionError,),
            operation_name="test"
        )
        
        # Check delays: 1s, 2s, 4s
        assert mock_sleep.call_count == 3
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays == pytest.approx([1.0, 2.0, 4.0], rel=0.1)


class TestBrokerConnectionRecovery:
    """Broker connection error recovery tests"""
    
    def test_handle_broker_connection_error(self):
        """Test broker connection error handling"""
        manager = ErrorRecoveryManager()
        
        error = BrokerConnectionError("Connection lost")
        
        result = manager.handle_error(error, {"operation": "place_order"})
        
        assert result['action'] == 'retry'
        assert result['recommended_action'] == 'reconnect'
    
    def test_broker_reconnection_success(self):
        """Test successful reconnection"""
        manager = ErrorRecoveryManager()
        
        mock_broker = Mock()
        mock_broker.reconnect.return_value = True
        
        success = manager.attempt_broker_reconnection(mock_broker)
        
        assert success is True
        assert mock_broker.reconnect.called


class TestInsufficientFundsRecovery:
    """Insufficient funds error recovery tests"""
    
    def test_handle_insufficient_funds(self):
        """Test insufficient funds handling"""
        manager = ErrorRecoveryManager()
        
        error = InsufficientFundsError("Insufficient margin")
        
        result = manager.handle_error(error, {
            "operation": "place_order",
            "size": 2.0
        })
        
        assert result['action'] == 'reduce_size'
        assert result['new_size'] == pytest.approx(1.0)  # 50% reduction
    
    def test_position_size_reduction(self):
        """Test position size reduction logic"""
        manager = ErrorRecoveryManager(size_reduction_factor=0.5)
        
        original_size = 2.0
        reduced_size = manager.reduce_position_size(original_size)
        
        assert reduced_size == 1.0
    
    def test_minimum_size_constraint(self):
        """Test minimum size after reduction"""
        manager = ErrorRecoveryManager(
            size_reduction_factor=0.5,
            min_position_size=0.01
        )
        
        # Very small size
        reduced_size = manager.reduce_position_size(0.015)
        
        # Should not go below minimum
        assert reduced_size >= 0.01


class TestEmergencyProcedures:
    """Emergency shutdown and position closing tests"""
    
    def test_emergency_close_all_positions(self):
        """Test emergency close all positions"""
        manager = ErrorRecoveryManager()
        
        mock_broker = Mock()
        mock_broker.close_position.return_value = True
        
        positions = [
            {'id': '001', 'symbol': 'EUR/USD', 'size': 1.0},
            {'id': '002', 'symbol': 'GBP/USD', 'size': 0.5}
        ]
        
        failed = manager.emergency_close_all_positions(
            broker_api=mock_broker,
            positions=positions
        )
        
        assert len(failed) == 0
        assert mock_broker.close_position.call_count == 2
    
    def test_emergency_close_with_failures(self):
        """Test emergency close with some failures"""
        manager = ErrorRecoveryManager()
        
        mock_broker = Mock()
        # First succeeds, second fails
        mock_broker.close_position.side_effect = [True, False]
        
        positions = [
            {'id': '001', 'symbol': 'EUR/USD'},
            {'id': '002', 'symbol': 'GBP/USD'}
        ]
        
        failed = manager.emergency_close_all_positions(
            broker_api=mock_broker,
            positions=positions
        )
        
        assert len(failed) == 1
        assert failed[0]['id'] == '002'
    
    def test_emergency_close_multiple_attempts(self):
        """Test emergency close retries failed positions"""
        manager = ErrorRecoveryManager(emergency_close_retries=3)
        
        mock_broker = Mock()
        # Fail twice, succeed on third
        mock_broker.close_position.side_effect = [False, False, True]
        
        positions = [{'id': '001', 'symbol': 'EUR/USD'}]
        
        failed = manager.emergency_close_all_positions(
            broker_api=mock_broker,
            positions=positions
        )
        
        # Should succeed after retries
        assert len(failed) == 0
        assert mock_broker.close_position.call_count == 3


class TestErrorStatistics:
    """Error tracking and statistics tests"""
    
    def test_get_error_statistics(self):
        """Test error statistics generation"""
        manager = ErrorRecoveryManager()
        
        # Log some errors
        manager.log_error(BrokerConnectionError("error 1"), "op1")
        manager.log_error(BrokerConnectionError("error 2"), "op2")
        manager.log_error(InsufficientFundsError("error 3"), "op3")
        
        stats = manager.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['by_type']['BrokerConnectionError'] == 2
        assert stats['by_type']['InsufficientFundsError'] == 1
    
    def test_error_rate_calculation(self):
        """Test error rate over time"""
        manager = ErrorRecoveryManager()
        
        # Log errors
        for i in range(5):
            manager.log_error(BrokerConnectionError(f"error {i}"), "test")
        
        stats = manager.get_error_statistics(time_window_hours=24)
        
        assert stats['error_rate'] >= 0.0


class TestCriticalErrors:
    """Critical system error tests"""
    
    def test_critical_error_handling(self):
        """Test critical error triggers emergency procedures"""
        manager = ErrorRecoveryManager()
        
        error = CriticalSystemError("System failure")
        
        result = manager.handle_error(error, {})
        
        assert result['action'] == 'emergency_shutdown'
        assert result['severity'] == 'critical'
    
    def test_critical_error_alert(self):
        """Test critical error sends alert"""
        manager = ErrorRecoveryManager()
        
        mock_alert_callback = Mock()
        manager.set_alert_callback(mock_alert_callback)
        
        error = CriticalSystemError("System failure")
        manager.handle_error(error, {})
        
        assert mock_alert_callback.called


@pytest.mark.integration
class TestIntegration:
    """Integration tests with trading system"""
    
    def test_integration_with_trading_engine(self):
        """Test integration with trading engine"""
        # TODO: Implement when trading engine integration is complete
        pass
    
    def test_recovery_during_live_trading(self):
        """Test recovery procedures during live trading"""
        # TODO: Implement with live trading simulation
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
