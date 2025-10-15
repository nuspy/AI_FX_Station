"""
Advanced 3D Visualizer for ForexGPT
Creates interactive 3D visualizations using Plotly
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")


class Advanced3DVisualizer:
    """Advanced 3D visualization generator for forex market data"""

    def __init__(self, data_provider=None):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for 3D visualizations. Install with: pip install plotly")

        self.output_dir = Path("reports_3d")
        self.output_dir.mkdir(exist_ok=True)

        # Import AI trading reports extension and data provider
        from .ai_trading_reports import AITradingReports
        from .data_provider import ReportDataProvider

        if data_provider is None:
            data_provider = ReportDataProvider()

        self.data_provider = data_provider
        self.ai_reports = AITradingReports(self.output_dir, data_provider=data_provider)

    def create_3d_market_surface(self, pairs: List[str], data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create a 3D surface plot showing price movements across currency pairs over time

        Args:
            pairs: List of currency pair symbols
            data: Dictionary mapping pair symbols to DataFrames with OHLCV data

        Returns:
            Dict with success status and file path
        """
        try:
            # Prepare data for surface plot
            time_points = []
            pair_indices = []
            prices = []

            for i, pair in enumerate(pairs[:10]):  # Limit to 10 pairs
                if pair not in data:
                    continue

                df = data[pair].tail(100)  # Last 100 periods

                for j, (idx, row) in enumerate(df.iterrows()):
                    time_points.append(j)
                    pair_indices.append(i)
                    prices.append(row['close'])

            # Create 3D surface
            fig = go.Figure(data=[go.Scatter3d(
                x=time_points,
                y=pair_indices,
                z=prices,
                mode='markers',
                marker=dict(
                    size=3,
                    color=prices,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Price")
                )
            )])

            fig.update_layout(
                title="3D Market Surface - Price Movements Across Pairs",
                scene=dict(
                    xaxis_title="Time Period",
                    yaxis_title="Currency Pair Index",
                    zaxis_title="Price",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(pairs[:10]))),
                        ticktext=pairs[:10]
                    )
                ),
                width=1200,
                height=800
            )

            # Save to HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"3d_market_surface_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {
                'success': True,
                'html_file': str(filepath),
                'message': f"3D Market Surface created successfully"
            }

        except Exception as e:
            logger.error(f"Error creating 3D market surface: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_correlation_sphere(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a 3D sphere visualization showing correlations between pairs

        Args:
            corr_matrix: Correlation matrix DataFrame

        Returns:
            Dict with success status and file path
        """
        try:
            # Generate sphere coordinates for pairs
            n_pairs = len(corr_matrix)
            theta = np.linspace(0, 2 * np.pi, n_pairs)
            phi = np.linspace(0, np.pi, n_pairs)

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            # Create figure
            fig = go.Figure()

            # Add points for each pair
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(size=10, color='blue'),
                text=corr_matrix.columns.tolist(),
                textposition='top center',
                name='Currency Pairs'
            ))

            # Add correlation lines
            for i in range(n_pairs):
                for j in range(i + 1, n_pairs):
                    corr = corr_matrix.iloc[i, j]

                    # Only show significant correlations
                    if abs(corr) > 0.5:
                        color = 'red' if corr > 0 else 'blue'
                        width = abs(corr) * 5

                        fig.add_trace(go.Scatter3d(
                            x=[x[i], x[j]],
                            y=[y[i], y[j]],
                            z=[z[i], z[j]],
                            mode='lines',
                            line=dict(color=color, width=width),
                            showlegend=False,
                            hoverinfo='text',
                            text=f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr:.2f}"
                        ))

            fig.update_layout(
                title="Correlation Sphere - Currency Pair Relationships",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z"
                ),
                width=1200,
                height=800,
                showlegend=True
            )

            # Save to HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlation_sphere_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {
                'success': True,
                'html_file': str(filepath),
                'message': f"Correlation Sphere created successfully"
            }

        except Exception as e:
            logger.error(f"Error creating correlation sphere: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_volatility_landscape(self, volatility_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a 3D landscape showing volatility patterns

        Args:
            volatility_data: DataFrame with volatility data (pairs as columns, time as rows)

        Returns:
            Dict with success status and file path
        """
        try:
            # Prepare data for surface plot
            z_data = volatility_data.values.T  # Transpose: pairs as rows, time as columns
            x_data = list(range(len(volatility_data)))  # Time points
            y_data = volatility_data.columns.tolist()  # Pair names

            # Create 3D surface
            fig = go.Figure(data=[go.Surface(
                z=z_data,
                x=x_data,
                y=list(range(len(y_data))),
                colorscale='RdYlBu_r',
                colorbar=dict(title="Volatility")
            )])

            fig.update_layout(
                title="Volatility Landscape - 3D Terrain Map",
                scene=dict(
                    xaxis_title="Time Period",
                    yaxis_title="Currency Pair",
                    zaxis_title="Volatility",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(y_data))),
                        ticktext=y_data
                    )
                ),
                width=1200,
                height=800
            )

            # Save to HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"volatility_landscape_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {
                'success': True,
                'html_file': str(filepath),
                'message': f"Volatility Landscape created successfully"
            }

        except Exception as e:
            logger.error(f"Error creating volatility landscape: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def create_heat_map_analytics(self, data: pd.DataFrame, analysis_type: str = 'correlation') -> Dict[str, Any]:
        """
        Create an interactive heatmap for various analytics

        Args:
            data: DataFrame with market data
            analysis_type: Type of analysis ('correlation', 'covariance', 'returns')

        Returns:
            Dict with success status and file path
        """
        try:
            # Calculate analysis matrix
            if analysis_type == 'correlation':
                matrix = data.corr()
                title = "Correlation Heat Map"
                colorscale = 'RdBu_r'
            elif analysis_type == 'covariance':
                matrix = data.cov()
                title = "Covariance Heat Map"
                colorscale = 'Viridis'
            else:  # returns
                matrix = data.pct_change().corr()
                title = "Returns Correlation Heat Map"
                colorscale = 'RdBu_r'

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.columns.tolist(),
                colorscale=colorscale,
                text=matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Value")
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Currency Pair",
                yaxis_title="Currency Pair",
                width=1000,
                height=900
            )

            # Save to HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_{analysis_type}_{timestamp}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))

            return {
                'success': True,
                'html_file': str(filepath),
                'message': f"{title} created successfully"
            }

        except Exception as e:
            logger.error(f"Error creating heat map: {e}")
            return {
                'success': False,
                'error': str(e)
            }
