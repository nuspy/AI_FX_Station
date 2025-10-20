"""
AI Trading-specific 3D visualization methods
Extension for visualization_3d.py with 20 additional report types
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .data_provider import ReportDataProvider


class AITradingReports:
    """Additional AI trading-specific 3D visualization methods"""

    def __init__(self, output_dir: Path, data_provider: Optional[ReportDataProvider] = None):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required")
        self.output_dir = output_dir
        self.data_provider = data_provider or ReportDataProvider()

    # ====================
    # AI FORECASTING PERFORMANCE (5)
    # ====================

    def create_forecast_accuracy_timeline(self, symbol: str = "EUR/USD", days: int = 30) -> Dict[str, Any]:
        """
        3D timeline showing forecast accuracy over time across multiple horizons.
        X: Time, Y: Forecast horizon, Z: Accuracy metric
        """
        try:
            # Get real forecast data
            df = self.data_provider.get_forecast_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No forecast data available'}

            # Prepare data for 3D visualization
            x_data = list(range(len(df)))  # Time index
            y_data = df['horizon'].tolist()
            z_data = df['accuracy'].tolist()
            colors = df['confidence'].tolist()

            fig = go.Figure(data=[go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Confidence")
                ),
                text=[f"Time: {i}<br>Horizon: {h}min<br>Accuracy: {a:.2%}<br>Confidence: {c:.2%}"
                      for i, h, a, c in zip(x_data, y_data, z_data, colors)],
                hoverinfo='text'
            )])

            fig.update_layout(
                title=f"Forecast Accuracy 3D Timeline - {symbol}",
                scene=dict(
                    xaxis_title="Time Period",
                    yaxis_title="Forecast Horizon (min)",
                    zaxis_title="Accuracy"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_accuracy_timeline_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating forecast accuracy timeline: {e}")
            return {'success': False, 'error': str(e)}

    def create_prediction_confidence_sphere(self, symbol: str = "EUR/USD", days: int = 30) -> Dict[str, Any]:
        """
        Sphere showing prediction confidence distribution across different market conditions.
        Points on sphere represent predictions, color shows confidence level.
        """
        try:
            # Get real forecast data
            df = self.data_provider.get_forecast_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No forecast data available'}

            n_predictions = len(df)
            # Map predictions to sphere using horizon and accuracy as angles
            theta = (df['horizon'].values / df['horizon'].max()) * 2 * np.pi
            phi = df['accuracy'].values * np.pi

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            confidence = df['confidence'].values

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=confidence,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Confidence")
                ),
                text=[f"Horizon: {h}min<br>Accuracy: {a:.2%}<br>Confidence: {c:.2%}"
                      for h, a, c in zip(df['horizon'], df['accuracy'], confidence)],
                hoverinfo='text'
            )])

            fig.update_layout(
                title=f"Prediction Confidence Sphere - {symbol}",
                scene=dict(
                    xaxis_title="Market Condition X",
                    yaxis_title="Market Condition Y",
                    zaxis_title="Market Condition Z"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_confidence_sphere_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating prediction confidence sphere: {e}")
            return {'success': False, 'error': str(e)}

    def create_model_performance_terrain(self, symbol: str = "EUR/USD", days: int = 30) -> Dict[str, Any]:
        """
        3D terrain map showing multiple model performance metrics.
        Peaks = best performing models/parameters.
        """
        try:
            # Get real forecast data
            df = self.data_provider.get_forecast_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No forecast data available'}

            # Create grid of horizons vs time with accuracy as Z
            unique_horizons = sorted(df['horizon'].unique())
            if len(unique_horizons) < 2:
                return {'success': False, 'error': 'Insufficient horizon diversity'}

            # Group by time bins and horizon
            df['time_bin'] = pd.cut(range(len(df)), bins=min(20, len(df)//5), labels=False)

            # Create pivot table for surface
            pivot = df.pivot_table(
                values='accuracy',
                index='time_bin',
                columns='horizon',
                aggfunc='mean'
            ).fillna(0)

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Jet',
                colorbar=dict(title="Accuracy")
            )])

            fig.update_layout(
                title=f"Model Performance Terrain - {symbol}",
                scene=dict(
                    xaxis_title="Forecast Horizon (min)",
                    yaxis_title="Time Period",
                    zaxis_title="Accuracy"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_performance_terrain_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating model performance terrain: {e}")
            return {'success': False, 'error': str(e)}

    def create_quantile_adherence_surface(self, symbol: str = "EUR/USD", days: int = 30) -> Dict[str, Any]:
        """
        Surface showing how well actual prices stay within predicted quantile bands.
        X: Time, Y: Quantile level, Z: Adherence score.
        """
        try:
            # Get forecast data (using accuracy as proxy for adherence)
            df = self.data_provider.get_forecast_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No forecast data available'}

            # Create synthetic quantile levels from confidence
            df['quantile'] = pd.cut(df['confidence'], bins=5, labels=[0.1, 0.3, 0.5, 0.7, 0.9])
            df['quantile'] = df['quantile'].astype(float)
            df['time_bin'] = pd.cut(range(len(df)), bins=min(30, len(df)//3), labels=False)

            # Create pivot table
            pivot = df.pivot_table(
                values='accuracy',
                index='time_bin',
                columns='quantile',
                aggfunc='mean'
            ).fillna(df['accuracy'].mean())

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='RdYlGn',
                colorbar=dict(title="Adherence")
            )])

            fig.update_layout(
                title=f"Quantile Adherence Surface - {symbol}",
                scene=dict(
                    xaxis_title="Quantile Level",
                    yaxis_title="Time Period",
                    zaxis_title="Adherence Score"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantile_adherence_surface_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating quantile adherence surface: {e}")
            return {'success': False, 'error': str(e)}

    def create_forecast_error_distribution_3d(self, symbol: str = "EUR/USD", days: int = 30) -> Dict[str, Any]:
        """
        3D histogram/distribution of forecast errors across time and horizons.
        """
        try:
            # Get real forecast data
            df = self.data_provider.get_forecast_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No forecast data available'}

            # Calculate errors as deviation from perfect accuracy
            errors = 1.0 - df['accuracy'].values
            horizons = df['horizon'].values
            times = np.arange(len(df))

            fig = go.Figure(data=[go.Scatter3d(
                x=times,
                y=horizons,
                z=errors,
                mode='markers',
                marker=dict(
                    size=3,
                    color=errors,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Error")
                ),
                text=[f"Time: {t}<br>Horizon: {h}min<br>Error: {e:.3f}<br>Confidence: {c:.2%}"
                      for t, h, e, c in zip(times, horizons, errors, df['confidence'])],
                hoverinfo='text'
            )])

            fig.update_layout(
                title=f"Forecast Error Distribution 3D - {symbol}",
                scene=dict(
                    xaxis_title="Time",
                    yaxis_title="Forecast Horizon (min)",
                    zaxis_title="Forecast Error"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forecast_error_distribution_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating forecast error distribution: {e}")
            return {'success': False, 'error': str(e)}

    # ====================
    # PATTERN RECOGNITION (5)
    # ====================

    def create_pattern_success_landscape(self, symbol: str = "EUR/USD", days: int = 90) -> Dict[str, Any]:
        """
        3D landscape of pattern success rates over time.
        Peaks = high success patterns, valleys = low success.
        """
        try:
            # Get real pattern data
            df = self.data_provider.get_pattern_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No pattern data available'}

            # Get unique patterns and create numeric mapping
            unique_patterns = sorted(df['pattern_type'].unique())
            pattern_map = {p: i for i, p in enumerate(unique_patterns)}

            df['pattern_id'] = df['pattern_type'].map(pattern_map)
            df['time_bin'] = pd.cut(range(len(df)), bins=min(30, len(df)//3), labels=False)

            # Create pivot table
            pivot = df.pivot_table(
                values='success',
                index='time_bin',
                columns='pattern_id',
                aggfunc='mean'
            ).fillna(0)

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='RdYlGn',
                colorbar=dict(title="Success Rate")
            )])

            fig.update_layout(
                title=f"Pattern Success Rate Landscape - {symbol}",
                scene=dict(
                    xaxis_title="Pattern Type",
                    yaxis_title="Time Period",
                    zaxis_title="Success Rate",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(unique_patterns))),
                        ticktext=unique_patterns
                    )
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_success_landscape_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating pattern success landscape: {e}")
            return {'success': False, 'error': str(e)}

    def create_chart_patterns_cluster(self, symbol: str = "EUR/USD", days: int = 90) -> Dict[str, Any]:
        """
        3D cluster analysis of detected chart patterns.
        Similar patterns clustered together in 3D space.
        """
        try:
            # Get real pattern data
            df = self.data_provider.get_pattern_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No pattern data available'}

            # Get unique patterns
            pattern_types = df['pattern_type'].unique()
            pattern_map = {p: i for i, p in enumerate(pattern_types)}

            # Create clusters for each pattern type using confidence and success
            x, y, z, colors, labels = [], [], [], [], []

            for _, row in df.iterrows():
                pattern_id = pattern_map[row['pattern_type']]
                # Use pattern characteristics to position in 3D
                cx, cy, cz = pattern_id * 2, pattern_id * 1.5, pattern_id * 2.5

                # Add jitter based on confidence and success
                x.append(cx + (row['confidence'] - 0.5) * 2)
                y.append(cy + (row['success'] - 0.5) * 2)
                z.append(cz + np.random.randn() * 0.3)
                colors.append(pattern_id)
                labels.append(f"{row['pattern_type']}<br>Confidence: {row['confidence']:.2%}<br>Success: {row['success']}")

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Pattern Type")
                ),
                text=labels,
                hoverinfo='text'
            )])

            fig.update_layout(
                title=f"Chart Patterns 3D Cluster - {symbol}",
                scene=dict(
                    xaxis_title="Feature 1 (Confidence)",
                    yaxis_title="Feature 2 (Success)",
                    zaxis_title="Feature 3"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_patterns_cluster_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating chart patterns cluster: {e}")
            return {'success': False, 'error': str(e)}

    def create_candlestick_patterns_timeline(self, symbol: str = "EUR/USD", days: int = 90) -> Dict[str, Any]:
        """
        3D timeline of candlestick pattern occurrences and their outcomes.
        """
        try:
            # Get real pattern data
            df = self.data_provider.get_pattern_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No pattern data available'}

            # Get unique patterns and create mapping
            unique_patterns = sorted(df['pattern_type'].unique())
            pattern_map = {p: i for i, p in enumerate(unique_patterns)}

            # Create scatter points for each pattern occurrence
            x = list(range(len(df)))
            y = [pattern_map[p] for p in df['pattern_type']]
            # Use success and confidence to create outcome metric
            z = (df['success'].values * 2 - 1) * df['confidence'].values  # Range: -1 to 1
            colors = z

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Outcome")
                ),
                text=[f"Pattern: {p}<br>Success: {s}<br>Confidence: {c:.2%}<br>Outcome: {o:.3f}"
                      for p, s, c, o in zip(df['pattern_type'], df['success'], df['confidence'], z)],
                hoverinfo='text'
            )])

            fig.update_layout(
                title=f"Candlestick Patterns Timeline 3D - {symbol}",
                scene=dict(
                    xaxis_title="Time",
                    yaxis_title="Pattern Type",
                    zaxis_title="Outcome Score",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(unique_patterns))),
                        ticktext=unique_patterns
                    )
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"candlestick_patterns_timeline_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating candlestick patterns timeline: {e}")
            return {'success': False, 'error': str(e)}

    def create_pattern_frequency_heatmap(self, symbol: str = "EUR/USD", days: int = 90) -> Dict[str, Any]:
        """
        3D heatmap showing pattern frequency across time periods and market conditions.
        """
        try:
            # Get real pattern data
            df = self.data_provider.get_pattern_data(symbol=symbol, days=days)

            if df.empty:
                return {'success': False, 'error': 'No pattern data available'}

            # Extract hour from timestamp
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

            # Get unique patterns
            unique_patterns = sorted(df['pattern_type'].unique())
            pattern_map = {p: i for i, p in enumerate(unique_patterns)}

            df['pattern_id'] = df['pattern_type'].map(pattern_map)

            # Create pivot table for frequency count
            pivot = df.pivot_table(
                values='confidence',
                index='pattern_id',
                columns='hour',
                aggfunc='count',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Hot',
                colorbar=dict(title="Frequency")
            )])

            fig.update_layout(
                title=f"Pattern Frequency Heatmap 3D - {symbol}",
                scene=dict(
                    xaxis_title="Hour of Day",
                    yaxis_title="Pattern Type",
                    zaxis_title="Frequency",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(unique_patterns))),
                        ticktext=unique_patterns
                    )
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_frequency_heatmap_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating pattern frequency heatmap: {e}")
            return {'success': False, 'error': str(e)}

    def create_support_resistance_strength_map(self, symbol: str = "EUR/USD", days: int = 30, timeframe: str = "1h") -> Dict[str, Any]:
        """
        3D map showing support/resistance level strengths over time.
        Height = strength of level, position = price level.
        """
        try:
            # Get market data to calculate S/R levels
            df = self.data_provider.get_market_data(symbol=symbol, timeframe=timeframe, days=days)

            if df.empty:
                return {'success': False, 'error': 'No market data available'}

            # Use high/low prices to identify S/R levels
            price_range = df['high'].max() - df['low'].min()
            num_levels = 20
            price_levels = np.linspace(df['low'].min(), df['high'].max(), num_levels)

            # Create time bins
            time_bins = min(30, len(df) // 10)
            df['time_bin'] = pd.cut(range(len(df)), bins=time_bins, labels=False)

            # Calculate strength for each price level and time bin
            X_data, Y_data, Z_data = [], [], []
            for time_bin in sorted(df['time_bin'].unique()):
                bin_df = df[df['time_bin'] == time_bin]
                for price_level in price_levels:
                    # Strength based on proximity of highs/lows to this level
                    high_touches = np.sum(np.abs(bin_df['high'] - price_level) < price_range * 0.01)
                    low_touches = np.sum(np.abs(bin_df['low'] - price_level) < price_range * 0.01)
                    strength = high_touches + low_touches

                    X_data.append(time_bin)
                    Y_data.append(price_level)
                    Z_data.append(strength)

            # Convert to meshgrid format
            unique_times = sorted(set(X_data))
            unique_prices = sorted(set(Y_data))
            X, Y = np.meshgrid(unique_times, unique_prices)
            Z = np.zeros_like(X, dtype=float)

            for x_val, y_val, z_val in zip(X_data, Y_data, Z_data):
                i = unique_prices.index(y_val)
                j = unique_times.index(x_val)
                Z[i, j] = z_val

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Plasma',
                colorbar=dict(title="Strength")
            )])

            fig.update_layout(
                title=f"Support/Resistance Strength Map - {symbol}",
                scene=dict(
                    xaxis_title="Time Period",
                    yaxis_title="Price Level",
                    zaxis_title="Level Strength"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"support_resistance_map_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating S/R strength map: {e}")
            return {'success': False, 'error': str(e)}

    # ====================
    # TRADING OPERATIONS (5)
    # ====================

    def create_trade_pnl_trajectory(self, days: int = 60) -> Dict[str, Any]:
        """
        3D trajectory of trade P&L over time.
        X: Time, Y: Trade number, Z: Cumulative P&L.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            trade_nums = np.arange(len(df))
            cumulative_pnl = np.cumsum(df['pnl'].values)
            times = np.arange(len(df))

            fig = go.Figure(data=[go.Scatter3d(
                x=times,
                y=trade_nums,
                z=cumulative_pnl,
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=cumulative_pnl,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Cumulative P&L")
                ),
                line=dict(color='blue', width=2),
                text=[f"Trade #{i}<br>Symbol: {s}<br>P&L: ${p:.2f}<br>Cumulative: ${c:.2f}"
                      for i, s, p, c in zip(trade_nums, df['symbol'], df['pnl'], cumulative_pnl)],
                hoverinfo='text'
            )])

            fig.update_layout(
                title="Trade P&L Trajectory 3D",
                scene=dict(
                    xaxis_title="Time",
                    yaxis_title="Trade Number",
                    zaxis_title="Cumulative P&L ($)"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_pnl_trajectory_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating trade P&L trajectory: {e}")
            return {'success': False, 'error': str(e)}

    def create_position_risk_exposure_map(self, days: int = 60) -> Dict[str, Any]:
        """
        3D map showing risk exposure across different positions and time.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Extract hour from timestamp
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

            # Calculate risk as volume * abs(pnl)
            df['risk'] = df['volume'] * np.abs(df['pnl'])

            # Get unique symbols
            unique_symbols = sorted(df['symbol'].unique())
            symbol_map = {s: i for i, s in enumerate(unique_symbols)}
            df['symbol_id'] = df['symbol'].map(symbol_map)

            # Create pivot table
            pivot = df.pivot_table(
                values='risk',
                index='symbol_id',
                columns='hour',
                aggfunc='sum',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Reds',
                colorbar=dict(title="Risk ($)")
            )])

            fig.update_layout(
                title="Position Risk Exposure Map",
                scene=dict(
                    xaxis_title="Hour of Day",
                    yaxis_title="Currency Pair",
                    zaxis_title="Risk Exposure",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(unique_symbols))),
                        ticktext=unique_symbols
                    )
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"position_risk_exposure_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating position risk exposure map: {e}")
            return {'success': False, 'error': str(e)}

    def create_drawdown_analysis_terrain(self, days: int = 60) -> Dict[str, Any]:
        """
        3D terrain showing drawdown depth and duration analysis.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Sort by timestamp and calculate equity curve
            df = df.sort_values('timestamp').reset_index(drop=True)
            equity = 10000 + np.cumsum(df['pnl'].values)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak * 100

            # Limit data for surface plot
            n_points = min(50, len(equity))
            time = np.arange(n_points)
            drawdown_subset = drawdown[:n_points]

            # Create duration dimension
            duration = np.arange(20)
            X, Y = np.meshgrid(time, duration)
            Z = np.outer(drawdown_subset, np.exp(-duration / 5))

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z.T,
                colorscale='Reds_r',
                colorbar=dict(title="Drawdown %")
            )])

            fig.update_layout(
                title="Drawdown Analysis Terrain",
                scene=dict(
                    xaxis_title="Time",
                    yaxis_title="Drawdown Duration",
                    zaxis_title="Drawdown Depth (%)"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawdown_analysis_terrain_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating drawdown analysis terrain: {e}")
            return {'success': False, 'error': str(e)}

    def create_win_loss_distribution_sphere(self, days: int = 60) -> Dict[str, Any]:
        """
        Sphere showing distribution of winning and losing trades across market conditions.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            n_trades = len(df)
            # Map trades to sphere based on duration and volume
            theta = (df['duration_min'].values / df['duration_min'].max()) * 2 * np.pi
            phi = (df['volume'].values / df['volume'].max()) * np.pi

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            pnl = df['pnl'].values

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=pnl,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="P&L ($)")
                ),
                text=[f"Symbol: {s}<br>P&L: ${p:.2f}<br>Duration: {d:.1f}min<br>Volume: {v:.2f}"
                      for s, p, d, v in zip(df['symbol'], pnl, df['duration_min'], df['volume'])],
                hoverinfo='text'
            )])

            fig.update_layout(
                title="Win/Loss Distribution Sphere",
                scene=dict(
                    xaxis_title="Market Condition X",
                    yaxis_title="Market Condition Y",
                    zaxis_title="Market Condition Z"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"win_loss_distribution_sphere_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating win/loss distribution sphere: {e}")
            return {'success': False, 'error': str(e)}

    def create_trade_duration_profit_surface(self, days: int = 60) -> Dict[str, Any]:
        """
        Surface showing relationship between trade duration and profit.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Extract entry hour from timestamp
            df['entry_hour'] = pd.to_datetime(df['timestamp']).dt.hour

            # Bin durations
            df['duration_bin'] = pd.cut(df['duration_min'], bins=20, labels=False)

            # Create pivot table
            pivot = df.pivot_table(
                values='pnl',
                index='duration_bin',
                columns='entry_hour',
                aggfunc='mean',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='RdYlGn',
                colorbar=dict(title="Avg Profit ($)")
            )])

            fig.update_layout(
                title="Trade Duration vs Profit Surface",
                scene=dict(
                    xaxis_title="Entry Hour",
                    yaxis_title="Duration Bin",
                    zaxis_title="Average Profit ($)"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_duration_profit_surface_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating trade duration/profit surface: {e}")
            return {'success': False, 'error': str(e)}

    # ====================
    # STRATEGY EFFICIENCY (5)
    # ====================

    def create_strategy_performance_comparison(self, days: int = 60) -> Dict[str, Any]:
        """
        3D comparison of multiple strategies across different metrics.
        """
        try:
            # Get real trade data grouped by symbol (as proxy for strategy)
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Group by symbol and calculate metrics
            symbols = sorted(df['symbol'].unique())
            metrics = ['Avg Return', 'Win Rate', 'Avg Duration']

            x, y, z, colors = [], [], [], []

            for i, symbol in enumerate(symbols):
                symbol_df = df[df['symbol'] == symbol]

                # Calculate metrics
                avg_return = symbol_df['pnl'].mean()
                win_rate = (symbol_df['pnl'] > 0).sum() / len(symbol_df) if len(symbol_df) > 0 else 0
                avg_duration = symbol_df['duration_min'].mean()

                metric_values = [avg_return / 100, win_rate, avg_duration / 60]  # Normalize

                for j, (metric, value) in enumerate(zip(metrics, metric_values)):
                    x.append(i)
                    y.append(j)
                    z.append(value)
                    colors.append(value)

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=15,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                text=[f"{symbols[int(i)]}: {metrics[int(j)]}<br>Value: {z_val:.3f}"
                      for i, j, z_val in zip(x, y, z)],
                hoverinfo='text'
            )])

            fig.update_layout(
                title="Strategy Performance Comparison 3D",
                scene=dict(
                    xaxis_title="Symbol",
                    yaxis_title="Metric",
                    zaxis_title="Score",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(symbols))),
                        ticktext=symbols
                    ),
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(metrics))),
                        ticktext=metrics
                    )
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_performance_comparison_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating strategy performance comparison: {e}")
            return {'success': False, 'error': str(e)}

    def create_risk_adjusted_returns_landscape(self, days: int = 60) -> Dict[str, Any]:
        """
        Landscape showing risk-adjusted returns across different market regimes.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Calculate volatility (std of pnl) and returns over time
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Create rolling windows
            window_size = max(5, len(df) // 20)
            df['volatility'] = df['pnl'].rolling(window=window_size, min_periods=1).std()
            df['return'] = df['pnl'].rolling(window=window_size, min_periods=1).mean()
            df['risk_adj_return'] = df['return'] / (df['volatility'] + 1e-6)

            # Bin data
            df['vol_bin'] = pd.cut(df['volatility'], bins=15, labels=False)
            df['ret_bin'] = pd.cut(df['return'], bins=15, labels=False)

            # Create pivot table
            pivot = df.pivot_table(
                values='risk_adj_return',
                index='ret_bin',
                columns='vol_bin',
                aggfunc='mean',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='RdYlGn',
                colorbar=dict(title="Risk-Adj Return")
            )])

            fig.update_layout(
                title="Risk-Adjusted Returns Landscape",
                scene=dict(
                    xaxis_title="Volatility Bin",
                    yaxis_title="Return Bin",
                    zaxis_title="Risk-Adjusted Return"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_adjusted_returns_landscape_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating risk-adjusted returns landscape: {e}")
            return {'success': False, 'error': str(e)}

    def create_sharpe_ratio_evolution_surface(self, days: int = 60) -> Dict[str, Any]:
        """
        Surface showing evolution of Sharpe ratio over time and across parameters.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculate Sharpe ratio with different lookback windows
            lookback_windows = list(range(5, min(30, len(df) // 2)))
            time_bins = min(30, len(df) // 5)

            df['time_bin'] = pd.cut(range(len(df)), bins=time_bins, labels=False)

            sharpe_data = []
            for window in lookback_windows:
                for time_bin in sorted(df['time_bin'].unique()):
                    bin_df = df[df['time_bin'] == time_bin]
                    if len(bin_df) >= window:
                        returns = bin_df['pnl'].tail(window)
                        sharpe = returns.mean() / (returns.std() + 1e-6)
                        sharpe_data.append({
                            'time_bin': time_bin,
                            'window': window,
                            'sharpe': sharpe
                        })

            if not sharpe_data:
                return {'success': False, 'error': 'Insufficient data for Sharpe calculation'}

            sharpe_df = pd.DataFrame(sharpe_data)

            # Create pivot table
            pivot = sharpe_df.pivot_table(
                values='sharpe',
                index='window',
                columns='time_bin',
                aggfunc='mean',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio")
            )])

            fig.update_layout(
                title="Sharpe Ratio Evolution Surface",
                scene=dict(
                    xaxis_title="Time Period",
                    yaxis_title="Lookback Window",
                    zaxis_title="Sharpe Ratio"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sharpe_ratio_evolution_surface_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating Sharpe ratio evolution surface: {e}")
            return {'success': False, 'error': str(e)}

    def create_trade_execution_quality_map(self, days: int = 60) -> Dict[str, Any]:
        """
        Map showing trade execution quality (fill price vs expected) over time and size.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Calculate slippage as price difference normalized by volume
            df['slippage'] = np.abs(df['exit_price'] - df['entry_price']) / (df['volume'] + 0.01)
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

            # Bin volume
            df['volume_bin'] = pd.cut(df['volume'], bins=15, labels=False)

            # Create pivot table
            pivot = df.pivot_table(
                values='slippage',
                index='volume_bin',
                columns='hour',
                aggfunc='mean',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Reds',
                colorbar=dict(title="Avg Slippage")
            )])

            fig.update_layout(
                title="Trade Execution Quality Map",
                scene=dict(
                    xaxis_title="Hour of Day",
                    yaxis_title="Volume Bin",
                    zaxis_title="Slippage"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_execution_quality_map_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating trade execution quality map: {e}")
            return {'success': False, 'error': str(e)}

    def create_slippage_spread_analysis(self, days: int = 60) -> Dict[str, Any]:
        """
        3D analysis of slippage and spread costs across different conditions.
        """
        try:
            # Get real trade data
            df = self.data_provider.get_trade_data(days=days)

            if df.empty:
                return {'success': False, 'error': 'No trade data available'}

            # Calculate volatility (rolling std of pnl as proxy)
            df = df.sort_values('timestamp').reset_index(drop=True)
            window = max(5, len(df) // 20)
            df['volatility'] = df['pnl'].rolling(window=window, min_periods=1).std()

            # Use volume as proxy for liquidity (higher volume = higher liquidity)
            df['liquidity'] = df['volume']

            # Calculate execution cost (slippage)
            df['execution_cost'] = np.abs(df['exit_price'] - df['entry_price'])

            # Bin data
            df['vol_bin'] = pd.cut(df['volatility'], bins=20, labels=False)
            df['liq_bin'] = pd.cut(df['liquidity'], bins=20, labels=False)

            # Create pivot table
            pivot = df.pivot_table(
                values='execution_cost',
                index='liq_bin',
                columns='vol_bin',
                aggfunc='mean',
                fill_value=0
            )

            X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
            Z = pivot.values

            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Hot',
                colorbar=dict(title="Execution Cost")
            )])

            fig.update_layout(
                title="Slippage & Spread Analysis 3D",
                scene=dict(
                    xaxis_title="Volatility Bin",
                    yaxis_title="Liquidity Bin",
                    zaxis_title="Execution Cost"
                ),
                width=1200,
                height=800
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slippage_spread_analysis_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {'success': True, 'html_file': str(filepath)}

        except Exception as e:
            logger.error(f"Error creating slippage/spread analysis: {e}")
            return {'success': False, 'error': str(e)}
