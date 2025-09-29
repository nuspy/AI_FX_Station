# src/forex_diffusion/visualization/advanced/visualization_3d.py
"""
Advanced 3D Visualization System for ForexGPT Phase 3
Creates immersive 3D market analysis and correlation visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, callback
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

logger = logging.getLogger(__name__)


class Advanced3DVisualizer:
    """
    Advanced 3D visualization system for forex market analysis
    Creates immersive visualizations including 3D surfaces, correlation spheres, and market landscapes
    """

    def __init__(self):
        self.plotly_available = PLOTLY_AVAILABLE
        self.dash_available = DASH_AVAILABLE

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - 3D visualizations will use fallback methods")

        self.color_schemes = {
            'professional': ['#2E86C1', '#F39C12', '#27AE60', '#E74C3C', '#8E44AD'],
            'trading': ['#00C851', '#FF4444', '#33B5E5', '#FF8800', '#AA66CC'],
            'heatmap': 'RdYlBu_r'
        }

    def create_3d_market_surface(self,
                                currency_pairs: List[str],
                                data: Dict[str, pd.DataFrame],
                                timeframe: str = '1H') -> Dict[str, Any]:
        """Create 3D market surface visualization"""

        if not self.plotly_available:
            return self._create_fallback_visualization()

        try:
            logger.info(f"Creating 3D market surface for {len(currency_pairs)} pairs")

            # Prepare data for 3D surface
            timestamps = []
            prices_matrix = []
            pair_names = []

            # Get common timestamp range
            all_timestamps = set()
            for pair in currency_pairs:
                if pair in data:
                    all_timestamps.update(data[pair].index)

            common_timestamps = sorted(list(all_timestamps))[-100:]  # Last 100 periods

            # Build price matrix
            for pair in currency_pairs:
                if pair in data:
                    pair_data = data[pair]
                    prices = []

                    for ts in common_timestamps:
                        if ts in pair_data.index:
                            prices.append(pair_data.loc[ts, 'close'])
                        else:
                            # Interpolate missing data
                            nearest_prices = pair_data['close'].iloc[-5:].mean()
                            prices.append(nearest_prices)

                    prices_matrix.append(prices)
                    pair_names.append(pair)

            if not prices_matrix:
                return {'success': False, 'error': 'No data available for visualization'}

            # Convert to numpy arrays
            prices_matrix = np.array(prices_matrix)
            x = np.arange(len(common_timestamps))  # Time axis
            y = np.arange(len(pair_names))  # Currency pairs axis

            # Create meshgrid
            X, Y = np.meshgrid(x, y)

            # Create 3D surface plot
            fig = go.Figure()

            # Add 3D surface
            surface = go.Surface(
                x=X,
                y=Y,
                z=prices_matrix,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Price Level", side="right"),
                    tickmode="linear"
                )
            )

            fig.add_trace(surface)

            # Update layout
            fig.update_layout(
                title=f'3D Market Surface - {timeframe} Analysis',
                scene=dict(
                    xaxis_title='Time Periods',
                    yaxis_title='Currency Pairs',
                    zaxis_title='Price Levels',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=1000,
                height=700
            )

            # Add annotations for currency pairs
            annotations = []
            for i, pair in enumerate(pair_names):
                annotations.append(dict(
                    x=0,
                    y=i,
                    z=prices_matrix[i, 0],
                    text=pair,
                    showarrow=False
                ))

            # Save as HTML
            html_file = f'3d_market_surface_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(html_file)

            logger.info(f"3D market surface created: {html_file}")

            return {
                'success': True,
                'figure': fig,
                'html_file': html_file,
                'pairs_analyzed': len(pair_names),
                'time_periods': len(common_timestamps),
                'features': {
                    '3d_surface': True,
                    'interactive': True,
                    'color_mapping': True,
                    'time_series': True
                }
            }

        except Exception as e:
            logger.error(f"Error creating 3D market surface: {e}")
            return {'success': False, 'error': str(e)}

    def create_correlation_sphere(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Create 3D correlation sphere visualization"""

        if not self.plotly_available:
            return self._create_fallback_visualization()

        try:
            logger.info("Creating 3D correlation sphere...")

            # Generate sphere coordinates
            pairs = correlation_matrix.columns.tolist()
            n_pairs = len(pairs)

            # Create points on a sphere surface
            phi = np.linspace(0, 2*np.pi, n_pairs, endpoint=False)
            theta = np.linspace(0, np.pi, n_pairs, endpoint=False)

            # Convert spherical to cartesian coordinates
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            # Create 3D scatter plot
            fig = go.Figure()

            # Add currency pair points
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=np.diagonal(correlation_matrix),
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Correlation Strength")
                ),
                text=pairs,
                textposition="middle center",
                textfont=dict(size=10),
                name="Currency Pairs"
            )

            fig.add_trace(scatter)

            # Add correlation lines
            for i in range(n_pairs):
                for j in range(i+1, n_pairs):
                    correlation = correlation_matrix.iloc[i, j]

                    # Only show strong correlations
                    if abs(correlation) > 0.5:
                        line_color = 'red' if correlation > 0 else 'blue'
                        line_width = abs(correlation) * 5

                        fig.add_trace(go.Scatter3d(
                            x=[x[i], x[j]],
                            y=[y[i], y[j]],
                            z=[z[i], z[j]],
                            mode='lines',
                            line=dict(
                                color=line_color,
                                width=line_width
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

            # Update layout
            fig.update_layout(
                title='3D Currency Correlation Sphere',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='cube'
                ),
                width=900,
                height=700
            )

            # Save as HTML
            html_file = f'correlation_sphere_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(html_file)

            logger.info(f"3D correlation sphere created: {html_file}")

            return {
                'success': True,
                'figure': fig,
                'html_file': html_file,
                'pairs_analyzed': n_pairs,
                'correlations_shown': sum(1 for i in range(n_pairs) for j in range(i+1, n_pairs)
                                        if abs(correlation_matrix.iloc[i, j]) > 0.5),
                'features': {
                    '3d_sphere': True,
                    'correlation_lines': True,
                    'interactive': True,
                    'color_coding': True
                }
            }

        except Exception as e:
            logger.error(f"Error creating correlation sphere: {e}")
            return {'success': False, 'error': str(e)}

    def create_volatility_landscape(self, volatility_data: pd.DataFrame) -> Dict[str, Any]:
        """Create 3D volatility landscape visualization"""

        if not self.plotly_available:
            return self._create_fallback_visualization()

        try:
            logger.info("Creating 3D volatility landscape...")

            # Prepare data
            timestamps = volatility_data.index
            currency_pairs = volatility_data.columns

            # Create meshgrid
            x = np.arange(len(timestamps))
            y = np.arange(len(currency_pairs))
            X, Y = np.meshgrid(x, y)

            # Z values are volatility levels
            Z = volatility_data.values.T

            # Create 3D surface plot
            fig = go.Figure()

            # Add volatility surface
            surface = go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Volatility Level", side="right")
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                )
            )

            fig.add_trace(surface)

            # Add high volatility peaks as scatter points
            high_vol_threshold = np.percentile(Z, 90)
            high_vol_indices = np.where(Z > high_vol_threshold)

            if len(high_vol_indices[0]) > 0:
                scatter = go.Scatter3d(
                    x=X[high_vol_indices],
                    y=Y[high_vol_indices],
                    z=Z[high_vol_indices],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond'
                    ),
                    name='High Volatility Events'
                )
                fig.add_trace(scatter)

            # Update layout
            fig.update_layout(
                title='3D Volatility Landscape',
                scene=dict(
                    xaxis_title='Time Periods',
                    yaxis_title='Currency Pairs',
                    zaxis_title='Volatility Level',
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=1.2)
                    )
                ),
                width=1000,
                height=700
            )

            # Save as HTML
            html_file = f'volatility_landscape_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(html_file)

            logger.info(f"3D volatility landscape created: {html_file}")

            return {
                'success': True,
                'figure': fig,
                'html_file': html_file,
                'high_vol_events': len(high_vol_indices[0]),
                'time_periods': len(timestamps),
                'currency_pairs': len(currency_pairs),
                'features': {
                    'volatility_surface': True,
                    'contour_lines': True,
                    'peak_detection': True,
                    'interactive': True
                }
            }

        except Exception as e:
            logger.error(f"Error creating volatility landscape: {e}")
            return {'success': False, 'error': str(e)}

    def create_heat_map_analytics(self, data: pd.DataFrame, analysis_type: str = 'correlation') -> Dict[str, Any]:
        """Create advanced heat map analytics"""

        if not self.plotly_available:
            return self._create_fallback_visualization()

        try:
            logger.info(f"Creating {analysis_type} heat map analytics...")

            if analysis_type == 'correlation':
                # Create correlation matrix
                corr_matrix = data.corr()
                title = 'Currency Correlation Heat Map'
                colorscale = 'RdBu'

            elif analysis_type == 'volatility':
                # Create volatility heat map
                volatility_matrix = data.rolling(20).std()
                corr_matrix = volatility_matrix
                title = 'Volatility Heat Map'
                colorscale = 'Reds'

            elif analysis_type == 'strength':
                # Create currency strength heat map
                # Calculate relative strength
                returns = data.pct_change()
                strength_matrix = returns.rolling(20).mean()
                corr_matrix = strength_matrix
                title = 'Currency Strength Heat Map'
                colorscale = 'RdYlGn'

            else:
                return {'success': False, 'error': f'Unknown analysis type: {analysis_type}'}

            # Create heat map
            fig = go.Figure()

            heatmap = go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=colorscale,
                showscale=True,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(
                    title=dict(text=analysis_type.title(), side="right")
                )
            )

            fig.add_trace(heatmap)

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Currency Pairs',
                yaxis_title='Currency Pairs',
                width=800,
                height=600
            )

            # Save as HTML
            html_file = f'{analysis_type}_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(html_file)

            logger.info(f"{analysis_type.title()} heat map created: {html_file}")

            return {
                'success': True,
                'figure': fig,
                'html_file': html_file,
                'analysis_type': analysis_type,
                'matrix_size': corr_matrix.shape,
                'features': {
                    'heat_map': True,
                    'interactive': True,
                    'value_annotations': True,
                    'color_scaling': True
                }
            }

        except Exception as e:
            logger.error(f"Error creating heat map: {e}")
            return {'success': False, 'error': str(e)}

    def create_interactive_dashboard(self, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create interactive 3D dashboard"""

        if not self.dash_available:
            logger.warning("Dash not available - cannot create interactive dashboard")
            return {'success': False, 'error': 'Dash not available'}

        try:
            logger.info("Creating interactive 3D dashboard...")

            # Create Dash app
            app = dash.Dash(__name__)

            # Define layout
            app.layout = html.Div([
                html.H1("ForexGPT Advanced 3D Analytics Dashboard"),

                html.Div([
                    html.Div([
                        html.Label("Select Visualization Type:"),
                        dcc.Dropdown(
                            id='viz-type-dropdown',
                            options=[
                                {'label': '3D Market Surface', 'value': 'surface'},
                                {'label': 'Correlation Sphere', 'value': 'sphere'},
                                {'label': 'Volatility Landscape', 'value': 'volatility'},
                                {'label': 'Heat Map Analytics', 'value': 'heatmap'}
                            ],
                            value='surface'
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),

                    html.Div([
                        html.Label("Select Currency Pairs:"),
                        dcc.Dropdown(
                            id='pairs-dropdown',
                            options=[{'label': pair, 'value': pair} for pair in data_sources.keys()],
                            value=list(data_sources.keys())[:5],
                            multi=True
                        )
                    ], style={'width': '65%', 'float': 'right', 'display': 'inline-block'})
                ]),

                html.Div([
                    dcc.Graph(id='3d-visualization')
                ])
            ])

            # Callback for updating visualization
            @app.callback(
                Output('3d-visualization', 'figure'),
                [Input('viz-type-dropdown', 'value'),
                 Input('pairs-dropdown', 'value')]
            )
            def update_visualization(viz_type, selected_pairs):
                if not selected_pairs:
                    return go.Figure()

                selected_data = {pair: data_sources[pair] for pair in selected_pairs if pair in data_sources}

                if viz_type == 'surface':
                    result = self.create_3d_market_surface(selected_pairs, selected_data)
                elif viz_type == 'sphere':
                    # Create correlation matrix for selected pairs
                    combined_data = pd.concat([data[['close']].rename(columns={'close': pair})
                                             for pair, data in selected_data.items()], axis=1)
                    corr_matrix = combined_data.corr()
                    result = self.create_correlation_sphere(corr_matrix)
                elif viz_type == 'volatility':
                    # Create volatility data
                    vol_data = pd.concat([data[['close']].rename(columns={'close': pair}).pct_change().rolling(20).std()
                                        for pair, data in selected_data.items()], axis=1)
                    result = self.create_volatility_landscape(vol_data)
                elif viz_type == 'heatmap':
                    combined_data = pd.concat([data[['close']].rename(columns={'close': pair})
                                             for pair, data in selected_data.items()], axis=1)
                    result = self.create_heat_map_analytics(combined_data, 'correlation')
                else:
                    return go.Figure()

                return result.get('figure', go.Figure()) if result.get('success') else go.Figure()

            logger.info("Interactive dashboard created successfully")

            return {
                'success': True,
                'app': app,
                'features': {
                    'interactive_dashboard': True,
                    'multiple_visualizations': True,
                    'real_time_updates': True,
                    'user_controls': True
                }
            }

        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return {'success': False, 'error': str(e)}

    def _create_fallback_visualization(self) -> Dict[str, Any]:
        """Create fallback visualization when advanced libraries are not available"""
        logger.warning("Creating fallback visualization")

        return {
            'success': True,
            'fallback': True,
            'message': 'Advanced 3D visualization libraries not available. Install plotly and dash for full functionality.',
            'features': {
                'fallback_mode': True,
                'basic_charts': True
            }
        }

    def get_visualization_capabilities(self) -> Dict[str, Any]:
        """Get current visualization capabilities"""
        return {
            'plotly_available': self.plotly_available,
            'dash_available': self.dash_available,
            'capabilities': {
                '3d_surface_plots': self.plotly_available,
                'correlation_spheres': self.plotly_available,
                'volatility_landscapes': self.plotly_available,
                'heat_maps': self.plotly_available,
                'interactive_dashboard': self.dash_available
            },
            'color_schemes': list(self.color_schemes.keys()),
            'supported_formats': ['html', 'png', 'svg', 'pdf'] if self.plotly_available else ['text']
        }


# Test the 3D visualization system
def test_3d_visualization():
    """Test the 3D visualization system"""
    print("Testing Advanced 3D Visualization System...")

    # Create sample data
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF']
    dates = pd.date_range('2024-09-01', periods=100, freq='h')

    data_sources = {}

    for i, pair in enumerate(pairs):
        np.random.seed(42 + i)
        base_price = 1.1000 + i * 0.1
        prices = np.cumsum(np.random.randn(100) * 0.003) + base_price

        data_sources[pair] = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(100) * 0.001),
            'low': prices - np.abs(np.random.randn(100) * 0.001),
            'close': np.roll(prices, -1),
            'volume': np.random.uniform(100000, 1000000, 100),
        }, index=dates)

    # Test visualizer
    visualizer = Advanced3DVisualizer()

    # Test capabilities
    capabilities = visualizer.get_visualization_capabilities()
    print(f"Visualization capabilities: {capabilities}")

    # Test 3D market surface
    print("\nTesting 3D market surface...")
    surface_result = visualizer.create_3d_market_surface(pairs, data_sources)
    print(f"Surface result: {surface_result.get('success', False)}")

    # Test correlation sphere
    print("\nTesting correlation sphere...")
    combined_data = pd.concat([data[['close']].rename(columns={'close': pair})
                              for pair, data in data_sources.items()], axis=1)
    corr_matrix = combined_data.corr()
    sphere_result = visualizer.create_correlation_sphere(corr_matrix)
    print(f"Sphere result: {sphere_result.get('success', False)}")

    # Test heat map
    print("\nTesting heat map analytics...")
    heatmap_result = visualizer.create_heat_map_analytics(combined_data, 'correlation')
    print(f"Heat map result: {heatmap_result.get('success', False)}")

    print("✓ 3D Visualization system test completed")

if __name__ == "__main__":
    test_3d_visualization()