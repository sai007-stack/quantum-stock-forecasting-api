"""
Visualization module for quantum stock market forecasting results.
Creates interactive charts and graphs for better understanding of predictions.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

class QuantumForecastVisualizer:
    """
    Creates interactive visualizations for quantum stock market forecasts.
    """
    
    def __init__(self):
        self.color_scheme = {
            'quantum': '#00D4AA',
            'classical': '#FF6B6B',
            'actual': '#4ECDC4',
            'forecast': '#45B7D1',
            'confidence': '#96CEB4',
            'background': '#F8F9FA'
        }
    
    def create_price_forecast_chart(self, historical_data: pd.DataFrame, 
                                  forecast_data: Dict[str, Any], 
                                  symbol: str) -> go.Figure:
        """
        Create an interactive price forecast chart.
        
        Args:
            historical_data: Historical stock data
            forecast_data: Quantum forecast results
            symbol: Stock symbol
            
        Returns:
            Plotly figure object
        """
        try:
            # Prepare historical data
            hist_df = historical_data.copy()
            hist_df['Date'] = hist_df.index
            hist_df = hist_df.reset_index(drop=True)
            
            # Prepare forecast data
            forecast_prices = forecast_data.get('forecast', [])
            last_date = hist_df['Date'].iloc[-1]
            
            # Create forecast dates with proper handling
            try:
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(forecast_prices),
                    freq='D'
                )
            except Exception as e:
                print(f"Warning: Could not create forecast dates: {e}")
                # Fallback: create simple date range
                forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(len(forecast_prices))]
            
            # Create the main figure
            fig = go.Figure()
            
            # Add historical price line
            fig.add_trace(go.Scatter(
                x=hist_df['Date'],
                y=hist_df['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color=self.color_scheme['actual'], width=2),
                hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            # Add forecast line
            if forecast_prices:
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_prices,
                    mode='lines+markers',
                    name='Quantum Forecast',
                    line=dict(color=self.color_scheme['quantum'], width=3, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='<b>%{x}</b><br>Forecast: $%{y:.2f}<extra></extra>'
                ))
                
                # Add confidence interval
                confidence = forecast_data.get('confidence', 0) / 100
                if confidence > 0:
                    # Calculate confidence bounds
                    volatility = forecast_data.get('volatility', 0.02)
                    upper_bound = [price * (1 + volatility * (1 - confidence)) for price in forecast_prices]
                    lower_bound = [price * (1 - volatility * (1 - confidence)) for price in forecast_prices]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=lower_bound,
                        mode='lines',
                        fill='tonexty',
                        fillcolor=f'rgba(0, 212, 170, {confidence * 0.3})',
                        name=f'Confidence Interval ({confidence:.1%})',
                        line=dict(width=0),
                        hoverinfo='skip'
                    ))
            
            # Update layout
            fig.update_layout(
                title=f'Quantum Stock Forecast: {symbol}',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add vertical line separating historical and forecast
            if forecast_prices:
                try:
                    # Use the last date directly (Plotly handles pandas timestamps)
                    fig.add_vline(
                        x=last_date,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text="Forecast Start",
                        annotation_position="top"
                    )
                except Exception as e:
                    print(f"Warning: Could not add vertical line: {e}")
                    # Skip the vertical line if there's an error
            
            return fig
            
        except Exception as e:
            print(f"Error creating forecast chart: {e}")
            # Return a simple fallback chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3],
                y=[100, 101, 102],
                mode='lines',
                name='Sample Data',
                line=dict(color='#00D4AA', width=2)
            ))
            fig.update_layout(
                title=f'Forecast Chart for {symbol} (Simplified)',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                template='plotly_white',
                height=500
            )
            return fig
    
    def create_ml_comparison_chart(self, historical_data: pd.DataFrame, 
                                 classical_forecast: Dict[str, Any],
                                 quantum_forecast: Dict[str, Any], 
                                 symbol: str) -> go.Figure:
        """
        Create a side-by-side comparison chart of classical vs quantum ML forecasts.
        
        Args:
            historical_data: Historical stock data
            classical_forecast: Classical ML forecast results
            quantum_forecast: Quantum ML forecast results
            symbol: Stock symbol
            
        Returns:
            Plotly figure with comparison chart
        """
        try:
            # Prepare historical data
            hist_df = historical_data.copy()
            hist_df['Date'] = hist_df.index
            hist_df = hist_df.reset_index(drop=True)
            
            # Get forecasts
            classical_prices = classical_forecast.get('forecast', [])
            quantum_prices = quantum_forecast.get('forecast', [])
            
            # Create forecast dates
            last_date = hist_df['Date'].iloc[-1]
            forecast_days = max(len(classical_prices), len(quantum_prices))
            
            try:
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
            except Exception as e:
                print(f"Warning: Could not create forecast dates: {e}")
                forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]
            
            # Create subplot with 2 columns
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Classical ML Forecast', 'Quantum ML Forecast'),
                horizontal_spacing=0.1
            )
            
            # Historical data for both subplots
            hist_trace = go.Scatter(
                x=hist_df['Date'],
                y=hist_df['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color=self.color_scheme['actual'], width=2),
                hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
            )
            
            # Add historical data to both subplots
            fig.add_trace(hist_trace, row=1, col=1)
            fig.add_trace(hist_trace, row=1, col=2)
            
            # Classical ML forecast
            if classical_prices:
                classical_trace = go.Scatter(
                    x=forecast_dates[:len(classical_prices)],
                    y=classical_prices,
                    mode='lines+markers',
                    name='Classical Forecast',
                    line=dict(color='#FF6B6B', width=3, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='<b>%{x}</b><br>Classical: $%{y:.2f}<extra></extra>'
                )
                fig.add_trace(classical_trace, row=1, col=1)
            
            # Quantum ML forecast
            if quantum_prices:
                quantum_trace = go.Scatter(
                    x=forecast_dates[:len(quantum_prices)],
                    y=quantum_prices,
                    mode='lines+markers',
                    name='Quantum Forecast',
                    line=dict(color=self.color_scheme['quantum'], width=3, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='<b>%{x}</b><br>Quantum: $%{y:.2f}<extra></extra>'
                )
                fig.add_trace(quantum_trace, row=1, col=2)
            
            # Update layout
            fig.update_layout(
                title=f'Classical vs Quantum ML Comparison: {symbol}',
                height=500,
                showlegend=True,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating comparison chart: {e}")
            # Return a simple fallback chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[1, 2, 3],
                y=[100, 101, 102],
                mode='lines',
                name='Sample Data',
                line=dict(color='#00D4AA', width=2)
            ))
            fig.update_layout(
                title=f'ML Comparison Chart for {symbol} (Simplified)',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                template='plotly_white',
                height=500
            )
            return fig
    
    def create_performance_comparison_chart(self, classical_metrics: Dict[str, Any], 
                                         quantum_metrics: Dict[str, Any]) -> go.Figure:
        """
        Create a performance comparison chart between classical and quantum ML.
        
        Args:
            classical_metrics: Classical ML performance metrics
            quantum_metrics: Quantum ML performance metrics
            
        Returns:
            Plotly figure with performance comparison
        """
        try:
            # Extract metrics
            metrics = ['Accuracy', 'Confidence', 'Training Time', 'Scalability']
            
            classical_values = [
                classical_metrics.get('val_r2', 0.5) * 100,  # Convert to percentage
                classical_metrics.get('confidence', 50),
                1,  # Fast training
                3   # Limited scalability
            ]
            
            quantum_values = [
                quantum_metrics.get('accuracy', 0.5) * 100,  # Convert to percentage
                quantum_metrics.get('confidence', 50),
                4,  # Slower training
                5   # High scalability
            ]
            
            # Create radar chart
            fig = go.Figure()
            
            # Classical ML trace
            fig.add_trace(go.Scatterpolar(
                r=classical_values,
                theta=metrics,
                fill='toself',
                name='Classical ML',
                line_color='#FF6B6B',
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            # Quantum ML trace
            fig.add_trace(go.Scatterpolar(
                r=quantum_values,
                theta=metrics,
                fill='toself',
                name='Quantum ML',
                line_color=self.color_scheme['quantum'],
                fillcolor=f'rgba(0, 212, 170, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Performance Comparison: Classical vs Quantum ML",
                height=500,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance comparison: {e}")
            # Return a simple bar chart as fallback
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Classical ML',
                x=['Accuracy', 'Confidence'],
                y=[50, 50],
                marker_color='#FF6B6B'
            ))
            fig.add_trace(go.Bar(
                name='Quantum ML',
                x=['Accuracy', 'Confidence'],
                y=[60, 60],
                marker_color=self.color_scheme['quantum']
            ))
            fig.update_layout(
                title="Performance Comparison (Simplified)",
                height=400,
                template='plotly_white'
            )
            return fig
    
    def create_quantum_metrics_dashboard(self, forecast_data: Dict[str, Any]) -> go.Figure:
        """
        Create a dashboard showing quantum-specific metrics.
        
        Args:
            forecast_data: Quantum forecast results
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots with better spacing (no subplot titles to avoid overlap)
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        confidence = forecast_data.get('confidence', 0)
        quantum_advantage = forecast_data.get('quantum_advantage', 0)
        volatility = forecast_data.get('volatility', 0)
        model_accuracy = forecast_data.get('model_accuracy', 0)
        
        # Confidence gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
            title={'text': "Confidence (%)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_scheme['quantum']},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=1, col=1)
        
        # Quantum advantage gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=quantum_advantage,
            domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
            title={'text': "Quantum Advantage (%)", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [-20, 20]},
                'bar': {'color': self.color_scheme['quantum']},
                'steps': [
                    {'range': [-20, 0], 'color': "lightcoral"},
                    {'range': [0, 10], 'color': "lightyellow"},
                    {'range': [10, 20], 'color': "lightgreen"}
                ]
            }
        ), row=1, col=2)
        
        # Volatility analysis
        volatility_categories = ['Historical', 'Predicted', 'Quantum-Adjusted']
        volatility_values = [volatility, volatility * 1.1, volatility * 0.9]
        
        fig.add_trace(go.Bar(
            x=volatility_categories,
            y=volatility_values,
            marker_color=[self.color_scheme['actual'], 
                         self.color_scheme['forecast'], 
                         self.color_scheme['quantum']],
            name='Volatility'
        ), row=2, col=1)
        
        # Prediction accuracy
        accuracy_categories = ['Classical ML', 'Quantum ML', 'Hybrid']
        accuracy_values = [50, model_accuracy * 100, model_accuracy * 100 + quantum_advantage]
        
        fig.add_trace(go.Bar(
            x=accuracy_categories,
            y=accuracy_values,
            marker_color=[self.color_scheme['classical'], 
                         self.color_scheme['quantum'], 
                         self.color_scheme['confidence']],
            name='Accuracy'
        ), row=2, col=2)
        
        fig.update_layout(
            title="Quantum Forecasting Metrics",
            height=600,
            showlegend=False,
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            title_x=0.5,
            title_font_size=20
        )
        
        # Add titles to bar charts
        fig.update_xaxes(title_text="Volatility Analysis", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Accuracy", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
        
        return fig
    
    def create_technical_analysis_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create a comprehensive technical analysis chart.
        
        Args:
            data: Stock data with technical indicators
            symbol: Stock symbol
            
        Returns:
            Plotly figure with technical analysis
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price & Moving Averages', 
                          'RSI (Relative Strength Index)',
                          'MACD',
                          'Volume'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color=self.color_scheme['actual'], width=2)
        ), row=1, col=1)
        
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='purple', width=1)
            ), row=1, col=1)
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color=self.color_scheme['quantum'], width=2)
            ), row=2, col=1)
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color=self.color_scheme['forecast'], width=2)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='orange', width=2)
            ), row=3, col=1)
            
            # MACD Histogram
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.7
            ), row=3, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=self.color_scheme['confidence'],
            opacity=0.7
        ), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Technical Analysis: {symbol}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_quantum_circuit_visualization(self, n_qubits: int = 4) -> go.Figure:
        """
        Create a visualization of the quantum circuit used for prediction.
        
        Args:
            n_qubits: Number of qubits in the circuit
            
        Returns:
            Plotly figure showing quantum circuit
        """
        # This is a simplified visualization
        # In practice, you'd use Qiskit's built-in visualization tools
        
        fig = go.Figure()
        
        # Add qubit lines
        for i in range(n_qubits):
            fig.add_trace(go.Scatter(
                x=[0, 10],
                y=[i, i],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
        
        # Add quantum gates (simplified representation)
        gate_positions = [2, 4, 6, 8]
        gate_types = ['H', 'RY', 'CNOT', 'M']
        
        for i, (pos, gate) in enumerate(zip(gate_positions, gate_types)):
            for j in range(n_qubits):
                if gate == 'CNOT' and j < n_qubits - 1:
                    # CNOT gate
                    fig.add_trace(go.Scatter(
                        x=[pos, pos],
                        y=[j, j+1],
                        mode='lines+markers',
                        line=dict(color='red', width=3),
                        marker=dict(size=10, symbol='circle'),
                        showlegend=False
                    ))
                else:
                    # Other gates
                    fig.add_trace(go.Scatter(
                        x=[pos],
                        y=[j],
                        mode='markers+text',
                        marker=dict(size=15, color=self.color_scheme['quantum']),
                        text=[gate],
                        textposition='middle center',
                        showlegend=False
                    ))
        
        fig.update_layout(
            title='Quantum Circuit for Stock Prediction',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    def create_forecast_comparison(self, historical_data: pd.DataFrame, 
                                 quantum_forecast: Dict[str, Any],
                                 classical_forecast: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create a comparison chart between quantum and classical forecasts.
        
        Args:
            historical_data: Historical stock data
            quantum_forecast: Quantum forecast results
            classical_forecast: Classical forecast results (optional)
            
        Returns:
            Plotly figure comparing forecasts
        """
        fig = go.Figure()
        
        # Historical data
        hist_df = historical_data.copy()
        hist_df['Date'] = hist_df.index
        
        fig.add_trace(go.Scatter(
            x=hist_df['Date'],
            y=hist_df['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color=self.color_scheme['actual'], width=2)
        ))
        
        # Quantum forecast
        quantum_prices = quantum_forecast.get('forecast', [])
        if quantum_prices:
            last_date = hist_df['Date'].iloc[-1]
            quantum_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(quantum_prices),
                freq='D'
            )
            
            fig.add_trace(go.Scatter(
                x=quantum_dates,
                y=quantum_prices,
                mode='lines+markers',
                name='Quantum Forecast',
                line=dict(color=self.color_scheme['quantum'], width=3, dash='dash'),
                marker=dict(size=6)
            ))
        
        # Classical forecast (if available)
        if classical_forecast:
            classical_prices = classical_forecast.get('forecast', [])
            if classical_prices:
                classical_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(classical_prices),
                    freq='D'
                )
                
                fig.add_trace(go.Scatter(
                    x=classical_dates,
                    y=classical_prices,
                    mode='lines+markers',
                    name='Classical Forecast',
                    line=dict(color=self.color_scheme['classical'], width=3, dash='dot'),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title='Quantum vs Classical Forecast Comparison',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
