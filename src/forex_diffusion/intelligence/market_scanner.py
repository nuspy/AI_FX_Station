# src/forex_diffusion/intelligence/market_scanner.py
"""
Real-time Market Intelligence and Scanning System for ForexGPT Phase 3
Monitors multiple currency pairs simultaneously and generates intelligent alerts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import asyncio
import threading
import queue
import json
from dataclasses import dataclass, asdict

try:
    import websockets
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MarketAlert:
    """Market alert data structure"""
    alert_id: str
    timestamp: datetime
    currency_pair: str
    alert_type: str
    message: str
    confidence: float
    urgency: str  # low, medium, high, critical
    data: Dict[str, Any]
    expiry: Optional[datetime] = None


@dataclass
class ScannerConfig:
    """Scanner configuration"""
    currency_pairs: List[str]
    scan_interval_seconds: int = 60
    alert_threshold: float = 0.75
    max_alerts_per_hour: int = 10
    enable_pattern_detection: bool = True
    enable_volatility_alerts: bool = True
    enable_trend_alerts: bool = True
    enable_news_integration: bool = False


class RealTimeMarketScanner:
    """
    Real-time market scanner that monitors multiple currency pairs
    and generates intelligent trading alerts
    """

    def __init__(self, config: ScannerConfig):
        self.config = config
        self.is_running = False
        self.alerts_queue = queue.Queue()
        self.market_data = {}
        self.alert_history = []
        self.scan_thread = None
        self.last_scan_time = None

        # Alert counters for rate limiting
        self.hourly_alert_count = 0
        self.last_hour_reset = datetime.now()

        # Initialize pattern engine if available
        self.pattern_engine = None
        try:
            from ..ml.advanced_pattern_engine import AdvancedPatternEngine
            self.pattern_engine = AdvancedPatternEngine()
            logger.info("Pattern engine initialized for market scanning")
        except ImportError:
            logger.warning("Pattern engine not available - using statistical analysis")

    def start_scanning(self):
        """Start real-time market scanning"""
        if self.is_running:
            logger.warning("Scanner already running")
            return

        logger.info(f"Starting market scanner for {len(self.config.currency_pairs)} pairs")
        self.is_running = True

        # Start scanning thread
        self.scan_thread = threading.Thread(target=self._scanning_loop, daemon=True)
        self.scan_thread.start()

        logger.info("Market scanner started successfully")

    def stop_scanning(self):
        """Stop market scanning"""
        if not self.is_running:
            return

        logger.info("Stopping market scanner...")
        self.is_running = False

        if self.scan_thread:
            self.scan_thread.join(timeout=5)

        logger.info("Market scanner stopped")

    def _scanning_loop(self):
        """Main scanning loop"""
        while self.is_running:
            try:
                # Reset hourly alert counter
                self._reset_hourly_counter()

                # Scan all currency pairs
                scan_results = self._scan_currency_pairs()

                # Process scan results and generate alerts
                self._process_scan_results(scan_results)

                # Update last scan time
                self.last_scan_time = datetime.now()

                # Wait for next scan
                threading.Event().wait(self.config.scan_interval_seconds)

            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
                threading.Event().wait(30)  # Wait 30 seconds on error

    def _reset_hourly_counter(self):
        """Reset hourly alert counter if needed"""
        now = datetime.now()
        if (now - self.last_hour_reset).total_seconds() >= 3600:
            self.hourly_alert_count = 0
            self.last_hour_reset = now

    def _scan_currency_pairs(self) -> Dict[str, Dict[str, Any]]:
        """Scan all configured currency pairs"""
        scan_results = {}

        for pair in self.config.currency_pairs:
            try:
                # Get market data for the pair
                market_data = self._get_market_data(pair)

                if market_data is not None:
                    # Perform analysis
                    analysis = self._analyze_pair(pair, market_data)
                    scan_results[pair] = analysis

                    # Store market data
                    self.market_data[pair] = market_data

            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
                scan_results[pair] = {'error': str(e)}

        return scan_results

    def _get_market_data(self, currency_pair: str) -> Optional[pd.DataFrame]:
        """Get market data for a currency pair"""
        try:
            # For demo purposes, generate realistic forex data
            # In production, this would connect to real data feeds
            return self._generate_realistic_market_data(currency_pair)

        except Exception as e:
            logger.error(f"Error getting market data for {currency_pair}: {e}")
            return None

    def _generate_realistic_market_data(self, currency_pair: str, periods: int = 100) -> pd.DataFrame:
        """Generate realistic market data for demonstration"""
        # Use pair name as seed for consistent but different data per pair
        pair_seed = sum(ord(c) for c in currency_pair)
        np.random.seed(pair_seed + int(datetime.now().timestamp()) % 1000)

        # Base prices for different pairs
        base_prices = {
            'EURUSD': 1.0950, 'GBPUSD': 1.2650, 'USDJPY': 149.50,
            'AUDUSD': 0.6420, 'USDCHF': 0.8890, 'USDCAD': 1.3720,
            'NZDUSD': 0.5890, 'EURJPY': 163.20, 'GBPJPY': 189.10,
            'EURGBP': 0.8650
        }

        base_price = base_prices.get(currency_pair, 1.1000)

        # Generate time series
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=periods)
        dates = pd.date_range(start=start_time, end=end_time, periods=periods)

        # Generate price movement with forex characteristics
        prices = [base_price]
        for i in range(1, periods):
            # Market hours effect
            hour = dates[i].hour
            if 6 <= hour <= 18:  # Active trading hours
                volatility = 1.0
                volume_base = 800000
            else:
                volatility = 0.3
                volume_base = 200000

            # Price movement
            change = np.random.normal(0, 0.0003 * volatility)
            trend = 0.00002 * np.sin(i / 20)  # Subtle trend
            new_price = prices[-1] * (1 + change + trend)
            prices.append(new_price)

        # Create OHLC data
        highs = []
        lows = []
        opens = []
        closes = []
        volumes = []

        for i in range(len(prices)):
            base = prices[i]
            spread = 0.00015  # 1.5 pips

            # Generate candle
            open_price = base
            close_price = base + np.random.normal(0, 0.0002)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.0005)
            low_price = min(open_price, close_price) - np.random.uniform(0, 0.0005)
            volume = np.random.uniform(200000, 1000000)

            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)

        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)

        return df

    def _analyze_pair(self, currency_pair: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a currency pair for trading opportunities"""
        analysis = {
            'pair': currency_pair,
            'timestamp': datetime.now(),
            'price': data['close'].iloc[-1],
            'alerts': [],
            'signals': {},
            'risk_level': 'medium'
        }

        try:
            # 1. Pattern Analysis
            if self.config.enable_pattern_detection:
                pattern_signals = self._detect_patterns(data)
                analysis['signals']['patterns'] = pattern_signals

                # Generate pattern alerts
                for pattern in pattern_signals:
                    if pattern['confidence'] >= self.config.alert_threshold:
                        alert = self._create_pattern_alert(currency_pair, pattern)
                        analysis['alerts'].append(alert)

            # 2. Volatility Analysis
            if self.config.enable_volatility_alerts:
                volatility_signals = self._analyze_volatility(data)
                analysis['signals']['volatility'] = volatility_signals

                # Generate volatility alerts
                if volatility_signals['alert_level'] >= self.config.alert_threshold:
                    alert = self._create_volatility_alert(currency_pair, volatility_signals)
                    analysis['alerts'].append(alert)

            # 3. Trend Analysis
            if self.config.enable_trend_alerts:
                trend_signals = self._analyze_trends(data)
                analysis['signals']['trends'] = trend_signals

                # Generate trend alerts
                if trend_signals['strength'] >= self.config.alert_threshold:
                    alert = self._create_trend_alert(currency_pair, trend_signals)
                    analysis['alerts'].append(alert)

            # 4. Technical Indicators
            technical_signals = self._analyze_technical_indicators(data)
            analysis['signals']['technical'] = technical_signals

            # 5. Risk Assessment
            analysis['risk_level'] = self._assess_risk_level(analysis['signals'])

        except Exception as e:
            logger.error(f"Error analyzing {currency_pair}: {e}")
            analysis['error'] = str(e)

        return analysis

    def _detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect chart patterns"""
        patterns = []

        try:
            # Use advanced pattern engine if available
            if self.pattern_engine:
                pattern_types = ['support', 'resistance', 'uptrend', 'downtrend']

                for pattern_type in pattern_types:
                    result = self.pattern_engine.predict_pattern_evolution(data, pattern_type)
                    if result.get('ensemble_prediction', {}).get('confidence', 0) > 0.6:
                        patterns.append({
                            'type': pattern_type,
                            'confidence': result['ensemble_prediction']['confidence'],
                            'direction': result['ensemble_prediction']['direction'],
                            'target_levels': result.get('target_levels', {}),
                            'time_horizon': result.get('time_horizon', {}),
                            'method': 'ml_ensemble'
                        })
            else:
                # Fallback pattern detection
                patterns.extend(self._simple_pattern_detection(data))

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        return patterns

    def _simple_pattern_detection(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Simple pattern detection fallback"""
        patterns = []

        try:
            close = data['close']
            high = data['high']
            low = data['low']

            # Support/Resistance detection
            if len(data) >= 20:
                recent_lows = low.rolling(20).min()
                recent_highs = high.rolling(20).max()
                current_price = close.iloc[-1]

                # Support level
                support_level = recent_lows.iloc[-1]
                support_distance = (current_price - support_level) / support_level

                if support_distance < 0.001:  # Very close to support
                    patterns.append({
                        'type': 'support',
                        'confidence': 0.75,
                        'direction': 'bullish',
                        'level': support_level,
                        'method': 'statistical'
                    })

                # Resistance level
                resistance_level = recent_highs.iloc[-1]
                resistance_distance = (resistance_level - current_price) / current_price

                if resistance_distance < 0.001:  # Very close to resistance
                    patterns.append({
                        'type': 'resistance',
                        'confidence': 0.75,
                        'direction': 'bearish',
                        'level': resistance_level,
                        'method': 'statistical'
                    })

        except Exception as e:
            logger.error(f"Error in simple pattern detection: {e}")

        return patterns

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility levels"""
        try:
            close = data['close']

            # Calculate volatility measures
            returns = close.pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1]
            historical_vol = returns.std()

            # Volatility percentile
            vol_series = returns.rolling(20).std()
            vol_percentile = (vol_series <= current_vol).mean()

            # Alert level based on volatility spike
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            alert_level = min(1.0, max(0.0, (vol_ratio - 1) * 2))

            return {
                'current_volatility': float(current_vol),
                'historical_volatility': float(historical_vol),
                'volatility_percentile': float(vol_percentile),
                'volatility_ratio': float(vol_ratio),
                'alert_level': float(alert_level),
                'status': 'high' if vol_ratio > 1.5 else 'normal' if vol_ratio > 0.8 else 'low'
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {'alert_level': 0.0, 'status': 'unknown', 'error': str(e)}

    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend strength and direction"""
        try:
            close = data['close']

            # Calculate trend indicators
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()

            # Trend direction
            if len(close) >= 50:
                trend_direction = 'up' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'down'

                # Trend strength based on angle
                recent_slope = (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10] if len(sma_20) >= 10 else 0
                strength = min(1.0, abs(recent_slope) * 1000)  # Scale to 0-1

            else:
                # Short-term trend
                recent_slope = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(close) >= 10 else 0
                trend_direction = 'up' if recent_slope > 0 else 'down'
                strength = min(1.0, abs(recent_slope) * 500)

            # Momentum
            momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0

            return {
                'direction': trend_direction,
                'strength': float(strength),
                'momentum': float(momentum),
                'slope': float(recent_slope),
                'status': 'strong' if strength > 0.7 else 'medium' if strength > 0.4 else 'weak'
            }

        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'strength': 0.0, 'status': 'unknown', 'error': str(e)}

    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators"""
        try:
            close = data['close']

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain / loss))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal

            return {
                'rsi': float(current_rsi),
                'rsi_signal': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral',
                'macd': float(macd.iloc[-1]) if not macd.empty else 0,
                'macd_signal': float(macd_signal.iloc[-1]) if not macd_signal.empty else 0,
                'macd_histogram': float(macd_histogram.iloc[-1]) if not macd_histogram.empty else 0,
                'macd_trend': 'bullish' if macd.iloc[-1] > macd_signal.iloc[-1] else 'bearish'
            }

        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return {'rsi': 50, 'rsi_signal': 'neutral', 'error': str(e)}

    def _assess_risk_level(self, signals: Dict[str, Any]) -> str:
        """Assess overall risk level based on signals"""
        try:
            risk_factors = []

            # Volatility risk
            if 'volatility' in signals:
                vol_status = signals['volatility'].get('status', 'normal')
                if vol_status == 'high':
                    risk_factors.append('high_volatility')

            # Trend risk
            if 'trends' in signals:
                trend_strength = signals['trends'].get('strength', 0)
                if trend_strength < 0.3:
                    risk_factors.append('weak_trend')

            # Technical risk
            if 'technical' in signals:
                rsi_signal = signals['technical'].get('rsi_signal', 'neutral')
                if rsi_signal in ['overbought', 'oversold']:
                    risk_factors.append('extreme_rsi')

            # Overall assessment
            if len(risk_factors) >= 2:
                return 'high'
            elif len(risk_factors) == 1:
                return 'medium'
            else:
                return 'low'

        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'medium'

    def _create_pattern_alert(self, currency_pair: str, pattern: Dict[str, Any]) -> MarketAlert:
        """Create pattern-based alert"""
        alert_id = f"pattern_{currency_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        message = f"{pattern['type'].title()} pattern detected in {currency_pair} "
        message += f"with {pattern['confidence']:.1%} confidence. "
        message += f"Expected direction: {pattern['direction']}"

        urgency = 'high' if pattern['confidence'] > 0.8 else 'medium'

        return MarketAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            currency_pair=currency_pair,
            alert_type='pattern',
            message=message,
            confidence=pattern['confidence'],
            urgency=urgency,
            data=pattern,
            expiry=datetime.now() + timedelta(hours=6)
        )

    def _create_volatility_alert(self, currency_pair: str, volatility: Dict[str, Any]) -> MarketAlert:
        """Create volatility-based alert"""
        alert_id = f"volatility_{currency_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        message = f"Volatility spike detected in {currency_pair}. "
        message += f"Current volatility is {volatility['volatility_ratio']:.1f}x normal levels"

        urgency = 'high' if volatility['volatility_ratio'] > 2.0 else 'medium'

        return MarketAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            currency_pair=currency_pair,
            alert_type='volatility',
            message=message,
            confidence=volatility['alert_level'],
            urgency=urgency,
            data=volatility,
            expiry=datetime.now() + timedelta(hours=2)
        )

    def _create_trend_alert(self, currency_pair: str, trend: Dict[str, Any]) -> MarketAlert:
        """Create trend-based alert"""
        alert_id = f"trend_{currency_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        message = f"Strong {trend['direction']} trend detected in {currency_pair}. "
        message += f"Trend strength: {trend['strength']:.1%}"

        urgency = 'high' if trend['strength'] > 0.8 else 'medium'

        return MarketAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            currency_pair=currency_pair,
            alert_type='trend',
            message=message,
            confidence=trend['strength'],
            urgency=urgency,
            data=trend,
            expiry=datetime.now() + timedelta(hours=12)
        )

    def _process_scan_results(self, scan_results: Dict[str, Dict[str, Any]]):
        """Process scan results and queue alerts"""
        for pair, analysis in scan_results.items():
            if 'alerts' in analysis:
                for alert in analysis['alerts']:
                    # Check rate limiting
                    if self.hourly_alert_count >= self.config.max_alerts_per_hour:
                        logger.warning("Hourly alert limit reached - skipping alert")
                        continue

                    # Queue alert
                    self.alerts_queue.put(alert)
                    self.alert_history.append(alert)
                    self.hourly_alert_count += 1

                    logger.info(f"Alert generated: {alert.alert_type} for {alert.currency_pair}")

    def get_active_alerts(self) -> List[MarketAlert]:
        """Get all active (non-expired) alerts"""
        now = datetime.now()
        active_alerts = []

        # Get alerts from queue
        alerts_to_check = []
        try:
            while True:
                alert = self.alerts_queue.get_nowait()
                alerts_to_check.append(alert)
        except queue.Empty:
            pass

        # Check expiry and return active alerts
        for alert in alerts_to_check:
            if alert.expiry is None or alert.expiry > now:
                active_alerts.append(alert)
            # Re-queue non-expired alerts
            if alert.expiry is None or alert.expiry > now:
                self.alerts_queue.put(alert)

        return active_alerts

    def get_scanner_status(self) -> Dict[str, Any]:
        """Get scanner status information"""
        active_alerts = self.get_active_alerts()

        return {
            'is_running': self.is_running,
            'currency_pairs': self.config.currency_pairs,
            'scan_interval_seconds': self.config.scan_interval_seconds,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'active_alerts_count': len(active_alerts),
            'total_alerts_generated': len(self.alert_history),
            'hourly_alert_count': self.hourly_alert_count,
            'alert_rate_limit': self.config.max_alerts_per_hour,
            'market_data_pairs': len(self.market_data),
            'features_enabled': {
                'pattern_detection': self.config.enable_pattern_detection,
                'volatility_alerts': self.config.enable_volatility_alerts,
                'trend_alerts': self.config.enable_trend_alerts,
                'news_integration': self.config.enable_news_integration
            },
            'ml_engine_available': self.pattern_engine is not None
        }


# Test the market scanner
def test_market_scanner():
    """Test the real-time market scanner"""
    print("Testing Real-time Market Scanner...")

    # Create configuration
    config = ScannerConfig(
        currency_pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        scan_interval_seconds=5,  # Fast scanning for testing
        alert_threshold=0.6,
        max_alerts_per_hour=20
    )

    # Create scanner
    scanner = RealTimeMarketScanner(config)

    # Get initial status
    status = scanner.get_scanner_status()
    print(f"Scanner initialized: {status}")

    # Start scanning
    print("Starting market scanner...")
    scanner.start_scanning()

    # Let it run for a short time
    import time
    print("Scanning for 15 seconds...")
    time.sleep(15)

    # Get status and alerts
    status = scanner.get_scanner_status()
    alerts = scanner.get_active_alerts()

    print(f"Scanner status after 15 seconds:")
    print(f"  Active alerts: {status['active_alerts_count']}")
    print(f"  Total alerts generated: {status['total_alerts_generated']}")
    print(f"  Last scan: {status['last_scan_time']}")

    # Show some alerts
    for i, alert in enumerate(alerts[:3]):  # Show first 3 alerts
        print(f"\nAlert {i+1}:")
        print(f"  Pair: {alert.currency_pair}")
        print(f"  Type: {alert.alert_type}")
        print(f"  Message: {alert.message}")
        print(f"  Confidence: {alert.confidence:.2f}")
        print(f"  Urgency: {alert.urgency}")

    # Stop scanner
    print("\nStopping scanner...")
    scanner.stop_scanning()

    print("✓ Market scanner test completed")

if __name__ == "__main__":
    test_market_scanner()