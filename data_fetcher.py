"""
Data fetching module for stock market data.
Handles real-time and historical data retrieval from various sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import requests
from datetime import datetime, timedelta
import time
import random
import json

# Try to import nsepy for Indian stock data
try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    print("⚠️ nsepy module not available. Install with: pip install nsepy")

# Try to import alpha_vantage for reliable stock data
try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    print("⚠️ alpha_vantage module not available. Install with: pip install alpha_vantage")

class StockDataFetcher:
    """
    Fetches and processes stock market data for quantum ML analysis.
    """
    
    def __init__(self, alpha_vantage_key: str = "4RY2ZIVSJDH8OJTZ"):
        self.cache = {}
        self.cache_duration = 600  # 10 minutes cache (increased)
        self.rate_limit_delay = 2  # 2 seconds between requests
        self.last_request_time = 0
        self.alpha_vantage_key = alpha_vantage_key
        self.popular_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'LYFT', 'SQ',
            'SPOT', 'TWTR', 'SNAP', 'PINS', 'ZM', 'DOCU', 'OKTA', 'SNOW'
        ]
        self.indian_symbols = [
            'TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK',
            'KOTAKBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'LT', 'ASIANPAINT',
            'MARUTI', 'AXISBANK', 'NESTLEIND', 'ULTRACEMCO', 'TITAN', 'POWERGRID'
        ]
        self.mock_data_cache = {}  # Cache for mock data
        
        # Initialize Alpha Vantage if available
        if ALPHA_VANTAGE_AVAILABLE:
            self.alpha_vantage_ts = TimeSeries(key=self.alpha_vantage_key, output_format="pandas")
            print("✅ Alpha Vantage API initialized successfully")
        else:
            self.alpha_vantage_ts = None
            print("⚠️ Alpha Vantage not available, falling back to Yahoo Finance")
    
    def _rate_limit(self):
        """Implement rate limiting to avoid 429 errors"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_yahoo_finance_available(self) -> bool:
        """Check if Yahoo Finance is accessible"""
        try:
            response = requests.get('https://finance.yahoo.com', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _is_indian_symbol(self, symbol: str) -> bool:
        """Check if the symbol is an Indian stock"""
        symbol_clean = symbol.upper().replace('.NS', '').replace('.BO', '')
        return symbol_clean in self.indian_symbols or symbol.endswith('.NS') or symbol.endswith('.BO')
    
    def _fetch_alpha_vantage_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API"""
        if not ALPHA_VANTAGE_AVAILABLE or self.alpha_vantage_ts is None:
            return None
            
        try:
            print(f"🔄 Fetching data from Alpha Vantage for {symbol}...")
            
            # Determine output size based on period
            outputsize = "compact"  # Last 100 data points
            if period in ["2y", "5y", "10y", "max"]:
                outputsize = "full"  # Full historical data
            
            # Try different symbol formats for Indian stocks
            symbol_variants = [symbol]
            if self._is_indian_symbol(symbol):
                # Add BSE and NSE variants
                symbol_clean = symbol.replace('.NS', '').replace('.BO', '')
                symbol_variants.extend([f"{symbol_clean}.BSE", f"{symbol_clean}.NSE"])
            
            for symbol_variant in symbol_variants:
                try:
                    data, meta = self.alpha_vantage_ts.get_daily(symbol=symbol_variant, outputsize=outputsize)
                    
                    if data is not None and not data.empty:
                        # Store original data before filtering
                        data_original = data.copy()
                        
                        # Rename columns to match expected format
                        data = data.rename(columns={
                            '1. open': 'Open',
                            '2. high': 'High', 
                            '3. low': 'Low',
                            '4. close': 'Close',
                            '5. volume': 'Volume'
                        })
                        
                        # Filter data based on period
                        if period != "max":
                            end_date = datetime.now()
                            if period == "1y":
                                start_date = end_date - timedelta(days=365)
                            elif period == "6mo":
                                start_date = end_date - timedelta(days=180)
                            elif period == "3mo":
                                start_date = end_date - timedelta(days=90)
                            elif period == "1mo":
                                start_date = end_date - timedelta(days=30)
                            elif period == "2y":
                                start_date = end_date - timedelta(days=730)
                            elif period == "5y":
                                start_date = end_date - timedelta(days=1825)
                            else:
                                start_date = end_date - timedelta(days=365)
                            
                            # Convert index to datetime if it's not already
                            if not isinstance(data.index, pd.DatetimeIndex):
                                data.index = pd.to_datetime(data.index)
                            
                            # Filter data
                            data = data[data.index >= start_date]
                            
                            # If no data after filtering, return original data
                            if data.empty:
                                print(f"⚠️ No data found for {symbol_variant} in the specified period, returning available data")
                                data = data_original  # Use original data without filtering
                        
                        print(f"✅ Successfully fetched Alpha Vantage data for {symbol_variant}")
                        return data
                        
                except Exception as e:
                    print(f"⚠️ Alpha Vantage failed for {symbol_variant}: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Alpha Vantage API error for {symbol}: {str(e)}")
            return None
    
    def _fetch_nse_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from NSE using yfinance with .NS suffix"""
        try:
            # Add .NS suffix if not present
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol_ns = symbol + '.NS'
            else:
                symbol_ns = symbol
            
            print(f"🔄 Trying yfinance with {symbol_ns}...")
            
            # Use yfinance with .NS suffix
            ticker = yf.Ticker(symbol_ns)
            data = ticker.history(period=period)
            
            if data is not None and not data.empty:
                print(f"✅ Successfully fetched NSE data for {symbol_ns}")
                return data
            else:
                # Try with date range if period method fails
                end_date = datetime.now()
                if period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = end_date - timedelta(days=365)
                
                data = ticker.history(start=start_date, end=end_date)
                if data is not None and not data.empty:
                    print(f"✅ Successfully fetched NSE data for {symbol_ns} using date range")
                    return data
            
            return None
            
        except Exception as e:
            print(f"⚠️ NSE data fetch failed for {symbol}: {str(e)}")
            return None
    
    def fetch_stock_data(self, symbol: str, period: str = "1y", 
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Validate symbol
            symbol = symbol.upper().strip()
            if not symbol:
                raise ValueError("Stock symbol cannot be empty")
            
            # Check cache first
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_data.copy()
            
            # Try Alpha Vantage first (most reliable)
            if ALPHA_VANTAGE_AVAILABLE and self.alpha_vantage_ts is not None:
                print(f"🔄 Trying Alpha Vantage for {symbol}...")
                data = self._fetch_alpha_vantage_data(symbol, period)
                if data is not None and not data.empty:
                    # Clean and validate data
                    data = self._clean_data(data)
                    # Cache the data
                    self.cache[cache_key] = (data.copy(), time.time())
                    return data
                else:
                    print(f"⚠️ Alpha Vantage data not available for {symbol}, trying other sources...")
            
            # Try NSE data for Indian stocks
            if self._is_indian_symbol(symbol):
                print(f"🇮🇳 Detected Indian stock: {symbol}")
                data = self._fetch_nse_data(symbol, period)
                if data is not None and not data.empty:
                    # Clean and validate data
                    data = self._clean_data(data)
                    # Cache the data
                    self.cache[cache_key] = (data.copy(), time.time())
                    return data
                else:
                    print(f"⚠️ NSE data not available for {symbol}, trying Yahoo Finance...")
            
            # Fetch data from Yahoo Finance with improved error handling
            print(f"🔄 Trying Yahoo Finance for {symbol}...")
            
            # Method 1: Try direct download (most reliable)
            try:
                end_date = datetime.now()
                if period == "1y":
                    start_date = end_date - timedelta(days=365)
                elif period == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif period == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif period == "1mo":
                    start_date = end_date - timedelta(days=30)
                elif period == "2y":
                    start_date = end_date - timedelta(days=730)
                elif period == "5y":
                    start_date = end_date - timedelta(days=1825)
                else:
                    start_date = end_date - timedelta(days=365)
                
                # Use direct download method (works better than Ticker)
                data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                                 end=end_date.strftime('%Y-%m-%d'), progress=False)
                
                if data is not None and not data.empty:
                    print(f"✅ Successfully fetched data for {symbol} using direct download")
                else:
                    data = None
                    
            except Exception as e:
                print(f"⚠️ Direct download method failed for {symbol}: {str(e)}")
                data = None
            
            # Method 2: Try Ticker approach if direct download fails
            if data is None or data.empty:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, interval=interval)
                    if not data.empty:
                        print(f"✅ Successfully fetched data for {symbol} using Ticker method")
                except Exception as e:
                    print(f"⚠️ Ticker method failed for {symbol}: {str(e)}")
            
            # Method 3: Try with different interval if both methods fail
            if data is None or data.empty:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, interval="1d")
                    if not data.empty:
                        print(f"✅ Successfully fetched data for {symbol} using 1d interval")
                except Exception as e:
                    print(f"⚠️ 1d interval method failed for {symbol}: {str(e)}")
            
            # If still no data, generate mock data
            if data is None or data.empty:
                print(f"⚠️ No data from Yahoo Finance for {symbol}. Generating mock data for demonstration.")
                print(f"💡 Try these popular symbols: {', '.join(self.popular_symbols[:8])}")
                data = self._generate_mock_data(symbol, period)
            
            # Clean and validate data
            data = self._clean_data(data)
            
            # Cache the data
            self.cache[cache_key] = (data.copy(), time.time())
            
            return data
            
        except Exception as e:
            # If all else fails, generate mock data
            print(f"⚠️ Error fetching data for {symbol}: {str(e)}. Generating mock data for demonstration.")
            return self._generate_mock_data(symbol, period)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data.
        
        Args:
            data: Raw stock data
            
        Returns:
            Cleaned data
        """
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure all values are positive
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data = data[data[col] > 0]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def _generate_mock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Generate mock stock data for demonstration purposes.
        
        Args:
            symbol: Stock symbol
            period: Data period
            
        Returns:
            DataFrame with mock OHLCV data
        """
        # Determine number of days based on period
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'ytd': 300, 'max': 3650
        }
        days = period_days.get(period, 365)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic stock price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
        
        # Base price varies by symbol
        base_prices = {
            'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 'TSLA': 200.0,
            'AMZN': 3000.0, 'META': 300.0, 'NVDA': 400.0, 'BRK-B': 300.0
        }
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Generate OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(np.random.normal(0, 0.01))
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close
            
            # Generate volume
            volume = int(np.random.normal(1000000, 200000))
            volume = max(volume, 100000)  # Minimum volume
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get additional stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'day_change': info.get('regularMarketChange', 0),
                'day_change_percent': info.get('regularMarketChangePercent', 0)
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'name': 'Unknown',
                'error': str(e)
            }
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try Alpha Vantage first (most reliable)
            if ALPHA_VANTAGE_AVAILABLE and self.alpha_vantage_ts is not None:
                try:
                    data, meta = self.alpha_vantage_ts.get_daily(symbol=symbol, outputsize="compact")
                    if data is not None and not data.empty:
                        return True
                except:
                    pass
            
            # Try yfinance as fallback
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                if not data.empty:
                    return True
            except:
                pass
            
            # If both fail, accept common symbols for mock data
            common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'BRK-B', 
                            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'MA', 'HD', 'DIS', 'PYPL', 'ADBE',
                            'INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK',
                            'KOTAKBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'LT', 'ASIANPAINT',
                            'MARUTI', 'AXISBANK', 'NESTLEIND', 'ULTRACEMCO', 'TITAN', 'POWERGRID']
            return symbol.upper() in common_symbols
            
        except:
            # If all else fails, accept common symbols for mock data
            common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'BRK-B', 
                            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'MA', 'HD', 'DIS', 'PYPL', 'ADBE',
                            'INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK',
                            'KOTAKBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'LT', 'ASIANPAINT',
                            'MARUTI', 'AXISBANK', 'NESTLEIND', 'ULTRACEMCO', 'TITAN', 'POWERGRID']
            return symbol.upper() in common_symbols
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of popular stock symbols for reference.
        
        Returns:
            List of popular stock symbols
        """
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'MA', 'HD', 'DIS', 'PYPL', 'ADBE',
            'NFLX', 'CRM', 'INTC', 'CMCSA', 'PFE', 'TMO', 'ABT', 'COST', 'PEP',
            'WMT', 'MRK', 'ACN', 'CSCO', 'ABBV', 'VZ', 'T', 'NKE', 'DHR', 'UNP',
            'QCOM', 'PM', 'RTX', 'HON', 'SPGI', 'LOW', 'IBM', 'AMGN', 'CVX',
            'BA', 'GS', 'CAT', 'AXP', 'DE', 'BKNG', 'SBUX', 'MDT', 'ISRG',
            'GILD', 'ADP', 'TJX', 'LMT', 'BLK', 'SYK', 'ZTS', 'TXN', 'INTU'
        ]
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional technical indicators for the stock data.
        
        Args:
            data: OHLCV data
            
        Returns:
            Data with additional technical indicators
        """
        df = data.copy()
        
        # Price-based indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_val = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_val * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_val * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        df['Stoch_K'] = self._calculate_stochastic_k(df)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K."""
        lowest_low = df['Low'].rolling(window=period).min()
        highest_high = df['High'].rolling(window=period).max()
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get market sentiment indicators for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with sentiment indicators
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get recent news sentiment (simplified)
            news = ticker.news[:10]  # Last 10 news items
            
            sentiment_scores = []
            for article in news:
                # Simple sentiment analysis based on title keywords
                title = article.get('title', '').lower()
                positive_words = ['up', 'rise', 'gain', 'positive', 'bullish', 'strong', 'growth']
                negative_words = ['down', 'fall', 'drop', 'negative', 'bearish', 'weak', 'decline']
                
                pos_count = sum(1 for word in positive_words if word in title)
                neg_count = sum(1 for word in negative_words if word in title)
                
                if pos_count + neg_count > 0:
                    sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                    sentiment_scores.append(sentiment)
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            return {
                'sentiment_score': avg_sentiment,
                'news_count': len(news),
                'recent_news': news[:3],  # Top 3 recent news
                'recommendation': info.get('recommendationKey', 'unknown'),
                'target_price': info.get('targetMeanPrice', 0),
                'current_price': info.get('currentPrice', 0)
            }
            
        except Exception as e:
            return {
                'sentiment_score': 0,
                'news_count': 0,
                'recent_news': [],
                'error': str(e)
            }
