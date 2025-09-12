"""
Configuration file for Quantum Stock Market Forecasting application.
Contains all configurable parameters and settings.
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration class."""
    
    # Quantum computing parameters
    QUANTUM_PARAMS = {
        'default_qubits': 4,
        'default_layers': 2,
        'max_qubits': 8,
        'max_layers': 5,
        'optimization_iterations': 100,
        'quantum_simulator': 'qasm_simulator'
    }
    
    # Data fetching parameters
    DATA_PARAMS = {
        'default_period': '1y',
        'default_interval': '1d',
        'cache_duration': 300,  # 5 minutes
        'max_data_points': 1000,
        'min_data_points': 50
    }
    
    # Forecasting parameters
    FORECAST_PARAMS = {
        'max_forecast_days': 30,
        'default_forecast_days': 7,
        'confidence_threshold': 0.6,
        'volatility_window': 20
    }
    
    # Visualization parameters
    VISUALIZATION_PARAMS = {
        'chart_height': 500,
        'dashboard_height': 600,
        'color_scheme': {
            'quantum': '#00D4AA',
            'classical': '#FF6B6B',
            'actual': '#4ECDC4',
            'forecast': '#45B7D1',
            'confidence': '#96CEB4',
            'background': '#F8F9FA'
        }
    }
    
    # API settings
    API_PARAMS = {
        'yfinance_timeout': 10,
        'max_retries': 3,
        'retry_delay': 1
    }
    
    # Streamlit settings
    STREAMLIT_PARAMS = {
        'page_title': 'Quantum Stock Market Forecasting',
        'page_icon': '🔮',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Popular stock symbols
    POPULAR_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
        'UNH', 'JNJ', 'V', 'PG', 'JPM', 'MA', 'HD', 'DIS', 'PYPL', 'ADBE',
        'NFLX', 'CRM', 'INTC', 'CMCSA', 'PFE', 'TMO', 'ABT', 'COST', 'PEP',
        'WMT', 'MRK', 'ACN', 'CSCO', 'ABBV', 'VZ', 'T', 'NKE', 'DHR', 'UNP',
        'QCOM', 'PM', 'RTX', 'HON', 'SPGI', 'LOW', 'IBM', 'AMGN', 'CVX',
        'BA', 'GS', 'CAT', 'AXP', 'DE', 'BKNG', 'SBUX', 'MDT', 'ISRG',
        'GILD', 'ADP', 'TJX', 'LMT', 'BLK', 'SYK', 'ZTS', 'TXN', 'INTU'
    ]
    
    # Technical indicators parameters
    TECHNICAL_INDICATORS = {
        'rsi_period': 14,
        'sma_periods': [5, 20, 50],
        'ema_periods': [12, 26],
        'bollinger_period': 20,
        'bollinger_std': 2,
        'stochastic_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    # Quantum algorithms configuration
    QUANTUM_ALGORITHMS = {
        'vqc': {
            'name': 'Variational Quantum Classifier',
            'description': 'Uses variational quantum circuits for classification',
            'optimizer': 'COBYLA',
            'max_iterations': 100
        },
        'pennylane_qnn': {
            'name': 'PennyLane Quantum Neural Network',
            'description': 'Quantum neural network using PennyLane framework',
            'optimizer': 'GradientDescent',
            'learning_rate': 0.1
        },
        'hybrid': {
            'name': 'Hybrid Quantum-Classical',
            'description': 'Combines quantum and classical approaches',
            'quantum_weight': 0.7,
            'classical_weight': 0.3
        }
    }
    
    # Error messages
    ERROR_MESSAGES = {
        'invalid_symbol': 'Invalid stock symbol. Please enter a valid symbol.',
        'no_data': 'No data available for the selected symbol and period.',
        'forecast_error': 'Error generating quantum forecast. Please try again.',
        'network_error': 'Network error. Please check your connection.',
        'quantum_error': 'Quantum computation error. Please try with different parameters.'
    }
    
    # Success messages
    SUCCESS_MESSAGES = {
        'data_fetched': 'Stock data fetched successfully!',
        'forecast_generated': 'Quantum forecast generated successfully!',
        'model_trained': 'Quantum model trained successfully!'
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return {
            'quantum': cls.QUANTUM_PARAMS,
            'data': cls.DATA_PARAMS,
            'forecast': cls.FORECAST_PARAMS,
            'visualization': cls.VISUALIZATION_PARAMS,
            'api': cls.API_PARAMS,
            'streamlit': cls.STREAMLIT_PARAMS,
            'popular_stocks': cls.POPULAR_STOCKS,
            'technical_indicators': cls.TECHNICAL_INDICATORS,
            'quantum_algorithms': cls.QUANTUM_ALGORITHMS,
            'error_messages': cls.ERROR_MESSAGES,
            'success_messages': cls.SUCCESS_MESSAGES
        }
    
    @classmethod
    def get_quantum_params(cls) -> Dict[str, Any]:
        """Get quantum computing parameters."""
        return cls.QUANTUM_PARAMS
    
    @classmethod
    def get_data_params(cls) -> Dict[str, Any]:
        """Get data fetching parameters."""
        return cls.DATA_PARAMS
    
    @classmethod
    def get_forecast_params(cls) -> Dict[str, Any]:
        """Get forecasting parameters."""
        return cls.FORECAST_PARAMS
