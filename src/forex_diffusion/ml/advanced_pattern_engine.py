# src/forex_diffusion/ml/advanced_pattern_engine.py
"""
Advanced ML Pattern Engine for ForexGPT Phase 3
Combines multiple ML models for comprehensive pattern analysis and prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    ML_AVAILABLE = True

    # Import model cache for performance optimization
    try:
        from .model_cache import ModelCache, OptimizedPatternPredictor, get_model_cache
        CACHE_AVAILABLE = True
    except ImportError:
        CACHE_AVAILABLE = False

except ImportError:
    ML_AVAILABLE = False
    CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedPatternEngine:
    """
    Advanced ML-powered pattern analysis engine
    Features ensemble models, sentiment integration, and market intelligence
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.is_trained = False
        self.pattern_history = []
        self.performance_metrics = {}

        # Initialize model cache for performance optimization
        if CACHE_AVAILABLE:
            self.cache = get_model_cache()
            self.optimized_predictor = OptimizedPatternPredictor()
            logger.info("Model cache initialized for optimized performance")
        else:
            self.cache = None
            self.optimized_predictor = None

        # Initialize models if ML is available
        if ML_AVAILABLE:
            self._initialize_models()
            logger.info("Advanced Pattern Engine initialized with ML capabilities")

    def predict_pattern_fast(self, data: pd.DataFrame, pattern_type: str) -> Dict[str, Any]:
        """
        Fast pattern prediction using optimized predictor
        Uses caching for improved performance
        """
        if self.optimized_predictor and len(data) > 0:
            try:
                # Use optimized predictor with caching
                price_data = data['close'].values if 'close' in data.columns else data.iloc[:, 0].values
                return self.optimized_predictor.predict_pattern(price_data, pattern_type)
            except Exception as e:
                logger.warning(f"Fast prediction failed, falling back to standard method: {e}")

        # Fallback to standard prediction
        return self.predict_pattern_evolution(data, pattern_type)

    def _initialize_models(self):
        """Initialize ensemble of ML models"""
        self.models = {
            'pattern_classifier': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'trend_predictor': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'volatility_predictor': RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=42
            )
        }

        self.scalers = {name: StandardScaler() for name in self.models.keys()}
        logger.info(f"Initialized {len(self.models)} ML models")

    def extract_comprehensive_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract comprehensive features for ML analysis"""
        try:
            features = {}

            # Basic price features
            close = data['close']
            high = data['high']
            low = data['low']
            open_price = data['open']

            # 1. Technical Indicators
            technical_features = []

            # Moving averages and trends
            for period in [5, 10, 20, 50, 100]:
                if len(close) >= period:
                    ma = close.rolling(period).mean()
                    technical_features.extend([
                        ma.iloc[-1] if not ma.empty else close.iloc[-1],
                        (close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] if not ma.empty and ma.iloc[-1] != 0 else 0
                    ])

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain / loss))
            technical_features.append(rsi.iloc[-1] if not rsi.empty else 50)

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            technical_features.extend([
                macd.iloc[-1] if not macd.empty else 0,
                macd_signal.iloc[-1] if not macd_signal.empty else 0,
                (macd.iloc[-1] - macd_signal.iloc[-1]) if not macd.empty and not macd_signal.empty else 0
            ])

            # Bollinger Bands
            bb_period = 20
            sma = close.rolling(bb_period).mean()
            std = close.rolling(bb_period).std()
            bb_upper = sma + (2 * std)
            bb_lower = sma - (2 * std)
            bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
            technical_features.append(bb_position)

            features['technical'] = np.array(technical_features)

            # 2. Price Action Features
            price_action_features = []

            # Candle patterns
            body_size = abs(close - open_price)
            total_range = high - low
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low

            if len(data) > 0:
                current_candle = {
                    'body_ratio': body_size.iloc[-1] / total_range.iloc[-1] if total_range.iloc[-1] != 0 else 0,
                    'upper_shadow_ratio': upper_shadow.iloc[-1] / total_range.iloc[-1] if total_range.iloc[-1] != 0 else 0,
                    'lower_shadow_ratio': lower_shadow.iloc[-1] / total_range.iloc[-1] if total_range.iloc[-1] != 0 else 0,
                    'is_bullish': 1 if close.iloc[-1] > open_price.iloc[-1] else 0
                }
                price_action_features.extend(current_candle.values())

            # Volatility measures
            for period in [10, 20, 50]:
                if len(close) >= period:
                    volatility = close.rolling(period).std()
                    price_action_features.append(volatility.iloc[-1] if not volatility.empty else 0)

            features['price_action'] = np.array(price_action_features)

            # 3. Market Structure Features
            market_structure_features = []

            # Support and resistance levels
            for period in [20, 50, 100]:
                if len(data) >= period:
                    recent_high = high.rolling(period).max().iloc[-1]
                    recent_low = low.rolling(period).min().iloc[-1]
                    current_price = close.iloc[-1]

                    # Position relative to range
                    position_in_range = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
                    market_structure_features.append(position_in_range)

                    # Distance from highs/lows
                    distance_from_high = (recent_high - current_price) / recent_high if recent_high != 0 else 0
                    distance_from_low = (current_price - recent_low) / recent_low if recent_low != 0 else 0
                    market_structure_features.extend([distance_from_high, distance_from_low])

            features['market_structure'] = np.array(market_structure_features)

            # 4. Volume Features (if available)
            volume_features = []
            if 'volume' in data.columns:
                volume = data['volume']

                # Volume moving averages
                for period in [10, 20, 50]:
                    if len(volume) >= period:
                        vol_ma = volume.rolling(period).mean()
                        current_vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] != 0 else 1
                        volume_features.append(current_vol_ratio)

                # Volume trend
                if len(volume) >= 10:
                    vol_trend = np.polyfit(range(10), volume.tail(10), 1)[0]
                    volume_features.append(vol_trend)
            else:
                volume_features = [1.0] * 4  # Default volume features

            features['volume'] = np.array(volume_features)

            logger.info(f"Extracted features: {sum(len(f) for f in features.values())} total")
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return minimal fallback features
            return {
                'technical': np.array([50, 0, 0]),  # Neutral RSI, MACD
                'price_action': np.array([0.5, 0.3, 0.3, 0]),  # Balanced candle
                'market_structure': np.array([0.5, 0.1, 0.1]),  # Mid-range position
                'volume': np.array([1.0, 1.0, 1.0, 0])  # Normal volume
            }

    def predict_pattern_evolution(self, data: pd.DataFrame, pattern_type: str) -> Dict[str, Any]:
        """Predict how a pattern will evolve using ensemble models"""
        try:
            if not ML_AVAILABLE or not self.is_trained:
                return self._statistical_pattern_prediction(data, pattern_type)

            # Extract features
            features = self.extract_comprehensive_features(data)

            # Combine features for prediction
            combined_features = np.concatenate(list(features.values())).reshape(1, -1)

            # Get predictions from ensemble models
            predictions = {}

            for model_name, model in self.models.items():
                try:
                    scaler = self.scalers[model_name]
                    features_scaled = scaler.transform(combined_features)

                    # Get prediction probabilities
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_scaled)[0]
                        confidence = max(probabilities)
                        prediction = model.predict(features_scaled)[0]
                    else:
                        prediction = model.predict(features_scaled)[0]
                        confidence = 0.7  # Default confidence for models without probabilities

                    predictions[model_name] = {
                        'prediction': int(prediction),
                        'confidence': float(confidence)
                    }

                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
                    predictions[model_name] = {'prediction': 0, 'confidence': 0.5}

            # Ensemble prediction
            avg_prediction = np.mean([p['prediction'] for p in predictions.values()])
            avg_confidence = np.mean([p['confidence'] for p in predictions.values()])

            # Generate comprehensive result
            result = {
                'pattern_type': pattern_type,
                'timestamp': datetime.now().isoformat(),
                'ensemble_prediction': {
                    'direction': 'bullish' if avg_prediction > 0.5 else 'bearish',
                    'confidence': float(avg_confidence),
                    'strength': self._confidence_to_strength(avg_confidence)
                },
                'individual_models': predictions,
                'features_analyzed': {
                    'technical_indicators': len(features['technical']),
                    'price_action': len(features['price_action']),
                    'market_structure': len(features['market_structure']),
                    'volume_analysis': len(features['volume'])
                },
                'risk_assessment': self._assess_pattern_risk(data, avg_confidence),
                'target_levels': self._calculate_target_levels(data, pattern_type, avg_confidence),
                'time_horizon': self._estimate_time_horizon(pattern_type, avg_confidence)
            }

            # Store prediction for learning
            self.pattern_history.append(result)

            logger.info(f"Pattern evolution prediction: {pattern_type} - {avg_confidence:.2f} confidence")
            return result

        except Exception as e:
            logger.error(f"Error in pattern evolution prediction: {e}")
            return self._statistical_pattern_prediction(data, pattern_type)

    def _statistical_pattern_prediction(self, data: pd.DataFrame, pattern_type: str) -> Dict[str, Any]:
        """Statistical fallback when ML is not available"""
        try:
            close = data['close']

            # Calculate statistical measures
            returns = close.pct_change().dropna()
            volatility = returns.std()
            momentum = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0

            # Simple pattern-based logic
            confidence = 0.6 + abs(momentum) * 2  # Base confidence + momentum component
            confidence = min(0.9, max(0.4, confidence))  # Clamp between 40% and 90%

            direction = 'bullish' if momentum > 0 else 'bearish'

            return {
                'pattern_type': pattern_type,
                'timestamp': datetime.now().isoformat(),
                'ensemble_prediction': {
                    'direction': direction,
                    'confidence': confidence,
                    'strength': self._confidence_to_strength(confidence)
                },
                'method': 'statistical_fallback',
                'risk_assessment': {
                    'volatility_score': min(1.0, volatility * 100),
                    'momentum_score': abs(momentum),
                    'overall_risk': 'medium'
                }
            }

        except Exception as e:
            logger.error(f"Error in statistical prediction: {e}")
            return {
                'pattern_type': pattern_type,
                'ensemble_prediction': {
                    'direction': 'neutral',
                    'confidence': 0.5,
                    'strength': 'weak'
                },
                'error': str(e)
            }

    def _confidence_to_strength(self, confidence: float) -> str:
        """Convert confidence score to strength rating"""
        if confidence >= 0.8:
            return 'very_strong'
        elif confidence >= 0.7:
            return 'strong'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.5:
            return 'weak'
        else:
            return 'very_weak'

    def _assess_pattern_risk(self, data: pd.DataFrame, confidence: float) -> Dict[str, Any]:
        """Assess risk factors for the pattern"""
        try:
            close = data['close']

            # Volatility risk
            volatility = close.rolling(20).std().iloc[-1] / close.iloc[-1]
            volatility_score = min(1.0, volatility * 100)

            # Trend strength
            trend_strength = abs((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) if len(close) >= 20 else 0

            # Overall risk assessment
            risk_factors = {
                'volatility_score': float(volatility_score),
                'trend_strength': float(trend_strength),
                'confidence_factor': float(confidence),
                'overall_risk': 'low' if confidence > 0.7 and volatility_score < 0.3 else 'medium' if confidence > 0.5 else 'high'
            }

            return risk_factors

        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {'overall_risk': 'medium', 'error': str(e)}

    def _calculate_target_levels(self, data: pd.DataFrame, pattern_type: str, confidence: float) -> Dict[str, float]:
        """Calculate potential target levels"""
        try:
            close = data['close']
            current_price = close.iloc[-1]

            # Calculate ATR for target sizing
            if len(data) >= 14:
                high = data['high']
                low = data['low']
                tr = np.maximum(high - low,
                               np.maximum(abs(high - close.shift(1)),
                                        abs(low - close.shift(1))))
                atr = tr.rolling(14).mean().iloc[-1]
            else:
                atr = current_price * 0.01  # 1% as default ATR

            # Pattern-specific target calculation
            multiplier = confidence * 2  # Higher confidence = larger targets

            if pattern_type in ['support', 'bullish']:
                target_1 = current_price + (atr * multiplier)
                target_2 = current_price + (atr * multiplier * 1.5)
                stop_loss = current_price - (atr * 0.5)
            else:  # resistance, bearish
                target_1 = current_price - (atr * multiplier)
                target_2 = current_price - (atr * multiplier * 1.5)
                stop_loss = current_price + (atr * 0.5)

            return {
                'target_1': float(target_1),
                'target_2': float(target_2),
                'stop_loss': float(stop_loss),
                'current_price': float(current_price),
                'atr_used': float(atr)
            }

        except Exception as e:
            logger.error(f"Error calculating targets: {e}")
            current = data['close'].iloc[-1]
            return {
                'target_1': float(current * 1.01),
                'target_2': float(current * 1.02),
                'stop_loss': float(current * 0.99),
                'current_price': float(current)
            }

    def _estimate_time_horizon(self, pattern_type: str, confidence: float) -> Dict[str, int]:
        """Estimate time horizon for pattern completion"""
        base_hours = {
            'support': 24,
            'resistance': 24,
            'uptrend': 72,
            'downtrend': 72,
            'consolidation': 48
        }

        # Adjust based on confidence
        hours = base_hours.get(pattern_type, 48)
        confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5 range
        estimated_hours = int(hours * confidence_multiplier)

        return {
            'estimated_hours': estimated_hours,
            'min_hours': max(6, estimated_hours // 2),
            'max_hours': estimated_hours * 2
        }

    def train_ensemble_models(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Train the ensemble of ML models"""
        if not ML_AVAILABLE:
            logger.warning("ML not available - training skipped")
            return {'success': False, 'reason': 'ML not available'}

        try:
            logger.info("Training ensemble ML models...")

            # For demonstration, create synthetic training data
            X, y_pattern, y_trend, y_volatility = self._create_comprehensive_training_data(2000)

            # Train each model
            training_results = {}

            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")

                # Select appropriate target variable
                if model_name == 'pattern_classifier':
                    y = y_pattern
                elif model_name == 'trend_predictor':
                    y = y_trend
                else:  # volatility_predictor
                    y = y_volatility

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Scale features
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train model
                model.fit(X_train_scaled, y_train)

                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

                training_results[model_name] = {
                    'train_accuracy': float(train_score),
                    'test_accuracy': float(test_score),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std())
                }

                logger.info(f"  {model_name} - Test accuracy: {test_score:.3f}")

            self.is_trained = True
            self.performance_metrics = training_results

            logger.info("Ensemble training completed successfully")
            return {
                'success': True,
                'models_trained': len(self.models),
                'performance': training_results,
                'training_samples': len(X),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            return {'success': False, 'error': str(e)}

    def _create_comprehensive_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create comprehensive synthetic training data"""
        np.random.seed(42)

        # Generate features (matching our feature extraction)
        n_features = 3 + 4 + 9 + 4  # technical + price_action + market_structure + volume
        X = np.random.randn(n_samples, n_features)

        # Generate correlated targets
        # Pattern classification (support/resistance/trend)
        y_pattern = ((X[:, 0] > 0) & (X[:, 5] > 0.5)).astype(int)

        # Trend prediction (up/down)
        y_trend = (X[:, 1] + X[:, 3] * 0.5 > 0).astype(int)

        # Volatility prediction (high/low)
        y_volatility = (np.abs(X[:, 2]) > 0.5).astype(int)

        return X, y_pattern, y_trend, y_volatility

    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            'engine_version': '3.0.0',
            'ml_available': ML_AVAILABLE,
            'is_trained': self.is_trained,
            'models_count': len(self.models),
            'predictions_made': len(self.pattern_history),
            'performance_metrics': self.performance_metrics,
            'last_updated': datetime.now().isoformat(),
            'features': {
                'ensemble_models': True,
                'comprehensive_features': True,
                'risk_assessment': True,
                'target_calculation': True,
                'time_estimation': True
            }
        }


# Test the advanced pattern engine
def test_advanced_pattern_engine():
    """Test the advanced pattern engine"""
    print("Testing Advanced Pattern Engine...")

    # Create sample data
    dates = pd.date_range('2024-09-01', periods=200, freq='h')
    np.random.seed(42)

    base_price = 1.1000
    prices = []
    for i in range(200):
        if i == 0:
            prices.append(base_price)
        else:
            change = np.random.normal(0, 0.0003)
            trend = 0.00005 * np.sin(i / 20)
            price = prices[-1] * (1 + change + trend)
            prices.append(price)

    data = pd.DataFrame({
        'open': prices,
        'high': np.array(prices) + np.abs(np.random.randn(200) * 0.0005),
        'low': np.array(prices) - np.abs(np.random.randn(200) * 0.0005),
        'close': np.roll(prices, -1),
        'volume': np.random.uniform(100000, 1000000, 200),
    }, index=dates)

    # Fix OHLC consistency
    data.loc[data.index[-1], 'close'] = data.loc[data.index[-1], 'open']
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

    # Test engine
    engine = AdvancedPatternEngine()

    # Train models
    print("Training ensemble models...")
    training_result = engine.train_ensemble_models([])
    print(f"Training result: {training_result['success']}")

    # Test pattern predictions
    patterns = ['support', 'resistance', 'uptrend', 'downtrend']

    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")
        result = engine.predict_pattern_evolution(data, pattern)

        prediction = result['ensemble_prediction']
        print(f"  Direction: {prediction['direction']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        print(f"  Strength: {prediction['strength']}")

        if 'target_levels' in result:
            targets = result['target_levels']
            print(f"  Target 1: {targets['target_1']:.5f}")
            print(f"  Stop Loss: {targets['stop_loss']:.5f}")

    # Engine status
    status = engine.get_engine_status()
    print("\nEngine Status:")
    print(f"  Models trained: {status['is_trained']}")
    print(f"  Predictions made: {status['predictions_made']}")
    print(f"  ML available: {status['ml_available']}")

    print("✓ Advanced Pattern Engine test completed")

if __name__ == "__main__":
    test_advanced_pattern_engine()