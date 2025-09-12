"""
Quantum Stock Market Forecasting Application
Main Streamlit application for quantum-enhanced stock predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_fetcher import StockDataFetcher
from quantum_ml import QuantumStockPredictor, PennyLaneQuantumPredictor
from real_quantum_ml import RealQuantumStockPredictor, RealPennyLaneQuantumPredictor
from classical_ml import ClassicalMLPredictor, MLComparison
from visualizer import QuantumForecastVisualizer

# Page configuration
st.set_page_config(
    page_title="Quantum Stock Market Forecasting",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .quantum-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00D4AA;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00D4AA, #00A8CC);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00A8CC, #00D4AA);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔮 Quantum Stock Market Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'stock_info' not in st.session_state:
        st.session_state.stock_info = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Stock symbol input
        st.markdown("### 📈 Stock Selection")
        
        # Initialize session state symbol if not exists
        if 'symbol' not in st.session_state:
            st.session_state.symbol = "AAPL"
        
        # Use session state symbol as default value
        symbol = st.text_input(
            "Enter Stock Symbol",
            value=st.session_state.symbol,
            help="Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Update session state when text input changes
        st.session_state.symbol = symbol
        
        # Show current selected symbol
        st.markdown(f"**Current Symbol:** {symbol}")
        
        # Data period selection
        period = st.selectbox(
            "Data Period",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=3,
            help="Select the historical data period"
        )
        
        # Forecast parameters
        st.markdown("### 🔮 Quantum Parameters")
        days_ahead = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to forecast ahead"
        )
        
        n_qubits = st.selectbox(
            "Number of Qubits",
            options=[2, 4, 6, 8],
            index=1,
            help="Number of qubits for quantum circuit"
        )
        
        # Comparison mode selection
        comparison_mode = st.selectbox(
            "Analysis Mode",
            options=["Quantum Only", "Classical vs Quantum Comparison"],
            index=0,
            help="Choose between quantum-only analysis or comparison with classical ML"
        )
        
        algorithm = st.selectbox(
            "Quantum Algorithm",
            options=["Real VQC (Qiskit)", "Real PennyLane QNN", "Simulated VQC", "Simulated PennyLane"],
            index=0,
            help="Select the quantum machine learning algorithm"
        )
        
        # Classical ML algorithm selection (only show if comparison mode is selected)
        classical_algorithm = None
        if comparison_mode == "Classical vs Quantum Comparison":
            classical_algorithm = st.selectbox(
                "Classical ML Algorithm",
                options=["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"],
                index=0,
                help="Select the classical machine learning algorithm for comparison"
            )
        
        # Action buttons
        st.markdown("### ⚡ Actions")
        fetch_data_btn = st.button("📊 Fetch Stock Data", type="primary")
        generate_forecast_btn = st.button("🔮 Generate Quantum Forecast", type="secondary")
        
        # Educational section
        st.markdown("### 📚 ML Comparison Guide")
        with st.expander("🔍 Classical vs Quantum ML Differences"):
            st.markdown("""
            **Classical Machine Learning:**
            - Uses traditional algorithms (Random Forest, Linear Regression)
            - Limited by classical computational complexity
            - Good interpretability and fast training
            - Limited scalability with large datasets
            
            **Quantum Machine Learning:**
            - Leverages quantum superposition and entanglement
            - Potential exponential speedup for certain problems
            - Can process exponentially more features
            - Better suited for complex, high-dimensional data
            
            **Key Differences:**
            - **Computational Power**: Quantum can explore multiple solutions simultaneously
            - **Feature Space**: Quantum can handle exponentially more features
            - **Training Time**: Classical is faster for small datasets, Quantum scales better
            - **Interpretability**: Classical is more interpretable, Quantum is more complex
            """)
        
        # Quick access to popular stocks
        st.markdown("### 🏆 Popular Stocks")
        
        # US Stocks
        st.markdown("**🇺🇸 US Stocks:**")
        us_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"]
        cols = st.columns(2)
        for i, stock in enumerate(us_stocks):
            with cols[i % 2]:
                if st.button(stock, key=f"us_stock_{stock}"):
                    st.session_state.symbol = stock
                    # Clear previous data to trigger fresh fetch
                    st.session_state.stock_data = None
                    st.session_state.stock_info = None
                    st.rerun()
        
        # Indian Stocks
        st.markdown("**🇮🇳 Indian Stocks:**")
        indian_stocks = ["TCS", "RELIANCE", "INFY", "HDFCBANK", "HINDUNILVR", "ICICIBANK", "KOTAKBANK", "BHARTIARTL"]
        cols = st.columns(2)
        for i, stock in enumerate(indian_stocks):
            with cols[i % 2]:
                if st.button(stock, key=f"in_stock_{stock}"):
                    st.session_state.symbol = stock
                    # Clear previous data to trigger fresh fetch
                    st.session_state.stock_data = None
                    st.session_state.stock_info = None
                    st.rerun()
        
        # Add note about data availability
        st.markdown("""
        <div style="padding: 10px; border-radius: 5px; margin-top: 10px;">
            <small>
            💡 <strong>Data Sources:</strong><br>
            🇮🇳 Indian stocks: NSE (National Stock Exchange)<br>
            🇺🇸 US stocks: Yahoo Finance<br>
            📊 Fallback: Realistic mock data for demonstration
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Market Analysis")
        
        # Fetch data section
        if fetch_data_btn or st.session_state.stock_data is not None:
            with st.spinner("Fetching stock data..."):
                try:
                    data_fetcher = StockDataFetcher()
                    
                    # Validate symbol
                    if not data_fetcher.validate_symbol(symbol):
                        st.error(f"❌ Invalid stock symbol: {symbol}")
                        st.stop()
                    
                    # Fetch data
                    stock_data = data_fetcher.fetch_stock_data(symbol, period=period)
                    stock_info = data_fetcher.get_stock_info(symbol)
                    
                    # Calculate technical indicators
                    stock_data = data_fetcher.calculate_technical_indicators(stock_data)
                    
                    # Store in session state
                    st.session_state.stock_data = stock_data
                    st.session_state.stock_info = stock_info
                    
                    st.success(f"✅ Successfully fetched data for {symbol}")
                    
                except Exception as e:
                    st.error(f"❌ Error fetching data: {str(e)}")
                    st.stop()
        
        # Display stock data if available
        if st.session_state.stock_data is not None:
            stock_data = st.session_state.stock_data
            stock_info = st.session_state.stock_info
            
            # Basic price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00D4AA', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} - Historical Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("## ℹ️ Stock Information")
        
        if st.session_state.stock_info is not None:
            info = st.session_state.stock_info
            
            # Stock info cards
            st.markdown(f"""
            <div class="quantum-card">
                <h3>{info.get('name', 'Unknown')}</h3>
                <p><strong>Symbol:</strong> {info.get('symbol', 'N/A')}</p>
                <p><strong>Sector:</strong> {info.get('sector', 'N/A')}</p>
                <p><strong>Industry:</strong> {info.get('industry', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Current metrics
            st.markdown("### 📈 Current Metrics")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Current Price",
                    f"${info.get('current_price', 0):.2f}",
                    f"{info.get('day_change', 0):.2f}"
                )
                st.metric(
                    "Market Cap",
                    f"${info.get('market_cap', 0):,.0f}" if info.get('market_cap') else "N/A"
                )
            
            with col_b:
                st.metric(
                    "P/E Ratio",
                    f"{info.get('pe_ratio', 0):.2f}" if info.get('pe_ratio') else "N/A"
                )
                st.metric(
                    "Beta",
                    f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A"
                )
    
    # Forecast section
    if generate_forecast_btn and st.session_state.stock_data is not None:
        st.markdown("---")
        
        if comparison_mode == "Classical vs Quantum Comparison":
            st.markdown("## 🔬 Classical vs Quantum ML Comparison")
        else:
            st.markdown("## 🔮 Quantum Forecast Results")
        
        with st.spinner("Generating forecasts..."):
            try:
                # Initialize quantum predictor
                if algorithm == "Real VQC (Qiskit)":
                    predictor = RealQuantumStockPredictor(n_qubits=n_qubits)
                elif algorithm == "Real PennyLane QNN":
                    predictor = RealPennyLaneQuantumPredictor(n_qubits=n_qubits)
                elif algorithm == "Simulated VQC":
                    predictor = QuantumStockPredictor(n_qubits=n_qubits)
                elif algorithm == "Simulated PennyLane":
                    predictor = PennyLaneQuantumPredictor(n_qubits=n_qubits)
                else:
                    predictor = RealQuantumStockPredictor(n_qubits=n_qubits)
                
                # Train quantum model first
                if algorithm in ["Real VQC (Qiskit)", "Real PennyLane QNN"]:
                    # Create training features for real quantum models
                    features = predictor._create_stock_features(st.session_state.stock_data)
                    if len(features) > 0:
                        # Create binary labels (up/down)
                        price_changes = np.diff(st.session_state.stock_data['Close'].values)
                        labels = (price_changes > 0).astype(int)
                        
                        # Train the model
                        if algorithm == "Real VQC (Qiskit)":
                            training_result = predictor.train_real_vqc(features, labels)
                        else:  # Real PennyLane QNN
                            training_result = predictor.train_real_quantum_model(features, labels)
                        
                        print(f"Real quantum model trained with accuracy: {training_result.get('accuracy', 0):.3f}")
                
                # Generate quantum forecast
                quantum_forecast = predictor.quantum_forecast(
                    st.session_state.stock_data, 
                    days_ahead=days_ahead
                )
                
                # Generate classical forecast if comparison mode is selected
                classical_forecast = None
                classical_predictor = None
                if comparison_mode == "Classical vs Quantum Comparison":
                    # Map algorithm names to classical ML types
                    classical_ml_map = {
                        "Random Forest": "random_forest",
                        "Gradient Boosting": "gradient_boosting", 
                        "Linear Regression": "linear",
                        "Ridge Regression": "ridge"
                    }
                    
                    classical_ml_type = classical_ml_map.get(classical_algorithm, "random_forest")
                    classical_predictor = ClassicalMLPredictor(model_type=classical_ml_type)
                    
                    # Train classical model
                    classical_training = classical_predictor.train_model(st.session_state.stock_data)
                    
                    # Generate classical forecast
                    classical_forecast = classical_predictor.predict_forecast(
                        st.session_state.stock_data, 
                        forecast_days=days_ahead
                    )
                
                # Store forecast data
                forecast_data = quantum_forecast
                
                # Store forecast data
                st.session_state.forecast_data = forecast_data
                
                if forecast_data.get('error'):
                    st.error(f"❌ Forecast error: {forecast_data['error']}")
                else:
                    st.success("✅ Quantum forecast generated successfully!")
                    
            except Exception as e:
                st.error(f"❌ Error generating forecast: {str(e)}")
    
    # Display forecast results
    if st.session_state.forecast_data is not None and not st.session_state.forecast_data.get('error'):
        forecast_data = st.session_state.forecast_data
        stock_data = st.session_state.stock_data
        
        # Initialize visualizer
        visualizer = QuantumForecastVisualizer()
        
        # Check if we have comparison data
        has_comparison = (comparison_mode == "Classical vs Quantum Comparison" and 
                         'classical_forecast' in locals() and 
                         classical_forecast is not None and 
                         not classical_forecast.get('error'))
        
        if has_comparison:
            # Comparison chart
            st.markdown("### 📊 Classical vs Quantum Forecast Comparison")
            comparison_chart = visualizer.create_ml_comparison_chart(
                stock_data, classical_forecast, forecast_data, symbol
            )
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Performance comparison
            st.markdown("### 🎯 Performance Comparison")
            performance_chart = visualizer.create_performance_comparison_chart(
                classical_forecast.get('metrics', {}), 
                forecast_data
            )
            st.plotly_chart(performance_chart, use_container_width=True)
            
        else:
            # Single quantum forecast chart
            st.markdown("### 📈 Price Forecast")
            forecast_chart = visualizer.create_price_forecast_chart(
                stock_data, forecast_data, symbol
            )
            st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Metrics dashboard
        if has_comparison:
            # Comparison metrics
            st.markdown("### 📊 Comparison Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔴 Classical ML")
                classical_metrics = classical_forecast.get('metrics', {})
                st.metric("Accuracy (R²)", f"{classical_metrics.get('val_r2', 0):.3f}")
                st.metric("Confidence", f"{classical_forecast.get('confidence', 0):.1f}%")
                st.metric("Training Samples", classical_forecast.get('training_samples', 'N/A'))
            
            with col2:
                st.markdown("#### 🔵 Quantum ML")
                st.metric("Accuracy", f"{forecast_data.get('model_accuracy', 0):.3f}")
                st.metric("Confidence", f"{forecast_data.get('confidence', 0):.1f}%")
                st.metric("Quantum Advantage", f"{forecast_data.get('quantum_advantage', 0):.1f}%")
            
            with col3:
                st.markdown("#### ⚡ Performance")
                # Calculate improvements
                classical_acc = classical_metrics.get('val_r2', 0.5)
                quantum_acc = forecast_data.get('model_accuracy', 0.5)
                accuracy_improvement = ((quantum_acc - classical_acc) / classical_acc) * 100 if classical_acc > 0 else 0
                
                st.metric("Accuracy Improvement", f"{accuracy_improvement:+.1f}%")
                st.metric("Confidence Difference", f"{forecast_data.get('confidence', 0) - classical_forecast.get('confidence', 0):+.1f}%")
                st.metric("Model Type", f"{classical_algorithm} vs {algorithm}")
        else:
            # Single quantum metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Forecast Metrics")
                
                # Key metrics
                metrics_data = {
                    "Model Confidence": f"{forecast_data.get('confidence', 0):.1f}%",
                    "Quantum Advantage": f"{forecast_data.get('quantum_advantage', 0):.1f}%",
                    "Last Price": f"${forecast_data.get('last_price', 0):.2f}",
                    "Volatility": f"{forecast_data.get('volatility', 0):.3f}",
                    "Model Accuracy": f"{forecast_data.get('model_accuracy', 0):.1%}"
                }
                
                for metric, value in metrics_data.items():
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #1f77b4;">
                        <strong style="color: #1f77b4;">{metric}:</strong> <span style="color: #2c3e50;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔮 Quantum Circuit")
                circuit_fig = visualizer.create_quantum_circuit_visualization(n_qubits)
                st.plotly_chart(circuit_fig, use_container_width=True)
        
        # Technical analysis
        st.markdown("### 📊 Technical Analysis")
        tech_analysis_fig = visualizer.create_technical_analysis_chart(stock_data, symbol)
        st.plotly_chart(tech_analysis_fig, use_container_width=True)
        
        # Quantum metrics dashboard
        st.markdown("### 🎛️ Quantum Metrics Dashboard")
        metrics_dashboard = visualizer.create_quantum_metrics_dashboard(forecast_data)
        st.plotly_chart(metrics_dashboard, use_container_width=True)
        
        # Quantum Stock Forecast Table
        st.markdown("### 📈 Quantum Stock Forecast Table")
        
        # Create forecast table data
        forecast_table_data = []
        last_price = forecast_data.get('last_price', 0)
        forecast_values = forecast_data.get('forecast', [])
        
        # Add historical data (last 5 days)
        if len(stock_data) >= 5:
            for i in range(-5, 0):
                date = stock_data.index[i].strftime('%Y-%m-%d')
                price = stock_data['Close'].iloc[i]
                forecast_table_data.append({
                    'Date': date,
                    'Price': f"${price:.2f}",
                    'Type': 'Historical',
                    'Change': f"{((price - last_price) / last_price * 100):+.2f}%" if i == -1 else f"{((price - stock_data['Close'].iloc[i-1]) / stock_data['Close'].iloc[i-1] * 100):+.2f}%" if i > -5 else "N/A",
                    'Confidence': 'N/A',
                    'Status': '✅ Actual'
                })
        
        # Add forecast data
        import pandas as pd
        from datetime import datetime, timedelta
        
        last_date = stock_data.index[-1]
        for i, forecast_price in enumerate(forecast_values):
            forecast_date = (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            price_change = ((forecast_price - last_price) / last_price * 100)
            
            # Determine confidence level
            confidence = forecast_data.get('confidence', 0)
            if confidence >= 80:
                status = "🟢 High Confidence"
                status_color = "#28a745"
            elif confidence >= 60:
                status = "🟡 Medium Confidence"
                status_color = "#ffc107"
            else:
                status = "🔴 Low Confidence"
                status_color = "#dc3545"
            
            forecast_table_data.append({
                'Date': forecast_date,
                'Price': f"${forecast_price:.2f}",
                'Type': 'Quantum Forecast',
                'Change': f"{price_change:+.2f}%",
                'Confidence': f"{confidence:.1f}%",
                'Status': status
            })
        
        # Display the table using Streamlit's native dataframe
        if forecast_table_data:
            # Convert to DataFrame for better alignment
            df_forecast = pd.DataFrame(forecast_table_data)
            
            # Style the dataframe with custom CSS
            st.markdown("""
            <style>
            .stDataFrame {
                font-family: Arial, sans-serif;
            }
            .stDataFrame table {
                border-collapse: collapse;
                width: 100%;
            }
            .stDataFrame th {
                background-color: #1f77b4 !important;
                color: white !important;
                font-weight: bold !important;
                padding: 12px !important;
                text-align: center !important;
            }
            .stDataFrame td {
                padding: 10px !important;
                border-bottom: 1px solid #ddd !important;
                text-align: center !important;
            }
            .stDataFrame tr:nth-child(even) {
                background-color: #f2f2f2 !important;
            }
            .stDataFrame tr:hover {
                background-color: #e6f3ff !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display the dataframe with proper formatting
            st.dataframe(
                df_forecast,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn(
                        "Date",
                        help="Forecast date",
                        width="medium"
                    ),
                    "Price": st.column_config.TextColumn(
                        "Price",
                        help="Stock price",
                        width="medium"
                    ),
                    "Type": st.column_config.TextColumn(
                        "Type",
                        help="Data type",
                        width="large"
                    ),
                    "Change": st.column_config.TextColumn(
                        "Change",
                        help="Price change percentage",
                        width="medium"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "Confidence",
                        help="Model confidence",
                        width="medium"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Prediction status",
                        width="large"
                    )
                }
            )
            
            # Add summary statistics
            st.markdown("### 📊 Forecast Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Forecast Period", f"{len(forecast_values)} days")
            
            with col2:
                avg_change = sum([float(row['Change'].replace('%', '')) for row in forecast_table_data if row['Type'] == 'Quantum Forecast']) / len(forecast_values)
                st.metric("Avg Daily Change", f"{avg_change:+.2f}%")
            
            with col3:
                max_price = max([float(row['Price'].replace('$', '')) for row in forecast_table_data])
                st.metric("Max Forecast Price", f"${max_price:.2f}")
            
            with col4:
                min_price = min([float(row['Price'].replace('$', '')) for row in forecast_table_data])
                st.metric("Min Forecast Price", f"${min_price:.2f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🔮 <strong>Quantum Stock Market Forecasting</strong> | Powered by Qiskit & PennyLane</p>
        <p><em>Disclaimer: This application is for educational purposes only. 
        Stock market predictions are inherently uncertain and should not be used as investment advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
