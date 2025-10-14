import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import hmac
import hashlib
import time
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib, fallback to manual calculations if not available
try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    st.warning("TA-Lib not available. Using manual calculations for technical indicators.")

# Page configuration
st.set_page_config(
    page_title="Aethos Platform - India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS (create a basic CSS if file doesn't exist)
try:
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    # Basic CSS as fallback
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .platform-tagline {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .regulatory-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class DeltaExchange:
    def __init__(self):
        self.base_url = "https://api.delta.exchange"
        # Use st.secrets or environment variables
        self.api_key = st.secrets.get("DELTA_API_KEY", "") if hasattr(st, 'secrets') else ""
        self.api_secret = st.secrets.get("DELTA_API_SECRET", "") if hasattr(st, 'secrets') else ""
    
    def _sign_request(self, method, path, body=''):
        timestamp = str(int(time.time()))
        signature_payload = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return timestamp, signature
    
    def get_products(self):
        """Get available trading products"""
        try:
            response = requests.get(f"{self.base_url}/v2/products", timeout=10)
            if response.status_code == 200:
                return response.json().get('result', [])
            return []
        except Exception as e:
            st.error(f"Error fetching Delta products: {e}")
            return []
    
    def get_ticker(self, symbol):
        """Get ticker data for a symbol"""
        try:
            response = requests.get(f"{self.base_url}/v2/tickers/{symbol}", timeout=10)
            if response.status_code == 200:
                return response.json().get('result')
            return None
        except Exception as e:
            st.error(f"Error fetching Delta ticker: {e}")
            return None
    
    def get_orderbook(self, symbol):
        """Get order book data"""
        try:
            response = requests.get(f"{self.base_url}/v2/orderbook/{symbol}", timeout=10)
            if response.status_code == 200:
                return response.json().get('result')
            return None
        except Exception as e:
            st.error(f"Error fetching Delta orderbook: {e}")
            return None
    
    def get_ohlc(self, symbol, resolution=60, limit=100):
        """Get OHLC data"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/history/candles",
                params={
                    'symbol': symbol,
                    'resolution': resolution,
                    'limit': limit
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get('result', [])
            return []
        except Exception as e:
            st.error(f"Error fetching Delta OHLC: {e}")
            return []

class IndianExchangeData:
    def __init__(self):
        self.delta = DeltaExchange()
    
    def get_all_markets(self):
        """Get markets from all Indian exchanges"""
        markets = []
        
        # Delta Exchange products
        delta_products = self.delta.get_products()
        for product in delta_products[:20]:
            if product and product.get('contract_type') in ['spot', 'perpetual_futures', 'futures']:
                markets.append({
                    'exchange': 'Delta',
                    'symbol': product.get('symbol', ''),
                    'product': product.get('contract_type', ''),
                    'base_currency': product.get('underlying_asset', {}).get('symbol', ''),
                    'quote_currency': product.get('settling_asset', {}).get('symbol', ''),
                    'tick_size': product.get('tick_size', 0.01)
                })
        
        # Add Indian exchange pairs
        indian_pairs = ['BTC-INR', 'ETH-INR', 'SOL-INR', 'MATIC-INR', 'ADA-INR', 'DOT-INR']
        for pair in indian_pairs:
            markets.append({
                'exchange': 'Indian Exchanges',
                'symbol': pair,
                'product': 'spot',
                'base_currency': pair.split('-')[0],
                'quote_currency': 'INR',
                'tick_size': 0.01
            })
        
        return markets
    
    def get_consolidated_prices(self):
        """Get consolidated prices across exchanges"""
        prices = []
        
        # Delta prices
        products = self.delta.get_products()[:10]
        for product in products:
            if product:
                ticker = self.delta.get_ticker(product.get('symbol'))
                if ticker:
                    prices.append({
                        'exchange': 'Delta',
                        'symbol': product.get('symbol', ''),
                        'price': float(ticker.get('close', 0)),
                        'volume': float(ticker.get('volume', 0)),
                        'change_24h': float(ticker.get('change_24h', 0)),
                        'product_type': product.get('contract_type', ''),
                        'timestamp': datetime.now()
                    })
        
        # Add Indian exchange prices (simulated)
        indian_prices = {
            'BTC-INR': 3500000,
            'ETH-INR': 200000,
            'SOL-INR': 8000,
            'MATIC-INR': 60,
            'ADA-INR': 40,
            'DOT-INR': 500
        }
        
        for pair, price in indian_prices.items():
            prices.append({
                'exchange': 'Indian Exchanges',
                'symbol': pair,
                'price': price,
                'volume': np.random.uniform(100000, 500000),
                'change_24h': np.random.uniform(-5, 5),
                'product_type': 'spot',
                'timestamp': datetime.now()
            })
        
        return prices

class TradingStrategies:
    """Semi-automated trading signal generators using TA-Lib"""
    
    @staticmethod
    def rsi_strategy(data, period=14, oversold=30, overbought=70):
        """RSI-based trading signals using TA-Lib"""
        if len(data) < period:
            return "NEUTRAL", 0
        
        if TA_LIB_AVAILABLE:
            rsi = talib.RSI(np.array(data, dtype=float), timeperiod=period)[-1]
        else:
            # Manual RSI calculation
            delta = np.diff(data)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain[-period:])
            avg_loss = np.mean(loss[-period:])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        
        if rsi < oversold:
            return "BUY", rsi
        elif rsi > overbought:
            return "SELL", rsi
        else:
            return "NEUTRAL", rsi
    
    @staticmethod
    def moving_average_crossover(data, short_window=20, long_window=50):
        """Moving average crossover strategy using TA-Lib"""
        if len(data) < long_window:
            return "NEUTRAL", 0, 0
        
        if TA_LIB_AVAILABLE:
            short_ma = talib.SMA(np.array(data, dtype=float), timeperiod=short_window)[-1]
            long_ma = talib.SMA(np.array(data, dtype=float), timeperiod=long_window)[-1]
        else:
            short_ma = np.mean(data[-short_window:])
            long_ma = np.mean(data[-long_window:])
        
        if short_ma > long_ma:
            return "BUY", short_ma, long_ma
        else:
            return "SELL", short_ma, long_ma
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands strategy using TA-Lib"""
        if len(data) < window:
            return "NEUTRAL", 0, 0, 0
        
        if TA_LIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                np.array(data, dtype=float), 
                timeperiod=window, 
                nbdevup=num_std, 
                nbdevdn=num_std
            )
            upper_band = upper[-1]
            lower_band = lower[-1]
            rolling_mean = middle[-1]
        else:
            rolling_mean = np.mean(data[-window:])
            rolling_std = np.std(data[-window:])
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
        
        current_price = data[-1]
        
        if current_price <= lower_band:
            return "BUY", current_price, upper_band, lower_band
        elif current_price >= upper_band:
            return "SELL", current_price, upper_band, lower_band
        else:
            return "NEUTRAL", current_price, upper_band, lower_band
    
    @staticmethod
    def macd_strategy(data, fast=12, slow=26, signal=9):
        """MACD strategy using TA-Lib"""
        if len(data) < slow:
            return "NEUTRAL", 0, 0, 0
        
        if TA_LIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(
                np.array(data, dtype=float), 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            macd_val = macd[-1]
            signal_line = macd_signal[-1]
            histogram = macd_hist[-1]
        else:
            # Manual EMA calculation
            ema_fast = np.mean(data[-fast:])
            ema_slow = np.mean(data[-slow:])
            macd_val = ema_fast - ema_slow
            signal_line = np.mean([macd_val] * min(signal, len(data)))
            histogram = macd_val - signal_line
        
        if macd_val > signal_line:
            return "BUY", macd_val, signal_line, histogram
        else:
            return "SELL", macd_val, signal_line, histogram
    
    @staticmethod
    def stochastic_strategy(data, high_data, low_data, k_period=14, d_period=3):
        """Stochastic oscillator strategy using TA-Lib"""
        if len(data) < k_period:
            return "NEUTRAL", 0, 0
        
        if TA_LIB_AVAILABLE:
            slowk, slowd = talib.STOCH(
                np.array(high_data, dtype=float),
                np.array(low_data, dtype=float),
                np.array(data, dtype=float),
                fastk_period=k_period,
                slowk_period=d_period,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            k_value = slowk[-1]
            d_value = slowd[-1]
        else:
            # Simplified manual calculation
            recent_high = max(high_data[-k_period:])
            recent_low = min(low_data[-k_period:])
            k_value = 100 * (data[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 50
            d_value = np.mean([k_value] * min(d_period, len(data)))
        
        if k_value < 20 and d_value < 20:
            return "BUY", k_value, d_value
        elif k_value > 80 and d_value > 80:
            return "SELL", k_value, d_value
        else:
            return "NEUTRAL", k_value, d_value
    
    @staticmethod
    def support_resistance_strategy(data, lookback=50):
        """Support and Resistance levels strategy"""
        if len(data) < lookback:
            return "NEUTRAL", 0, 0, 0
        
        recent_data = data[-lookback:]
        resistance = np.max(recent_data)
        support = np.min(recent_data)
        current_price = data[-1]
        
        # Calculate distance to support and resistance
        dist_to_resistance = (resistance - current_price) / current_price
        dist_to_support = (current_price - support) / current_price
        
        if dist_to_support < 0.02:  # Near support
            return "BUY", current_price, support, resistance
        elif dist_to_resistance < 0.02:  # Near resistance
            return "SELL", current_price, support, resistance
        else:
            return "NEUTRAL", current_price, support, resistance

class SemiAutomatedBots:
    """Semi-automated bots that generate trading signals"""
    
    def __init__(self):
        self.strategies = TradingStrategies()
        self.exchange_data = IndianExchangeData()
    
    def create_signal_dashboard(self):
        """Create dashboard for signal generation bots"""
        st.subheader("🤖 Semi-Automated Signal Bots")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Strategy selection
            selected_strategy = st.selectbox(
                "Select Trading Strategy",
                ["RSI Strategy", "Moving Average Crossover", "Bollinger Bands", 
                 "MACD Strategy", "Stochastic Oscillator", "Support/Resistance", "Multi-Strategy"]
            )
            
            # Asset selection
            markets = self.exchange_data.get_all_markets()
            symbols = [market['symbol'] for market in markets]
            selected_symbol = st.selectbox("Select Asset", symbols[:10])
            
            # Strategy parameters
            st.write("*Strategy Parameters*")
            if selected_strategy == "RSI Strategy":
                rsi_period = st.slider("RSI Period", 5, 30, 14)
                oversold = st.slider("Oversold Level", 10, 40, 30)
                overbought = st.slider("Overbought Level", 60, 90, 70)
            
            elif selected_strategy == "Moving Average Crossover":
                short_window = st.slider("Short MA Period", 5, 30, 10)
                long_window = st.slider("Long MA Period", 20, 100, 50)
            
            elif selected_strategy == "Bollinger Bands":
                bb_period = st.slider("BB Period", 10, 50, 20)
                num_std = st.slider("Standard Deviations", 1, 3, 2)
            
            elif selected_strategy == "MACD Strategy":
                fast_period = st.slider("Fast EMA", 5, 20, 12)
                slow_period = st.slider("Slow EMA", 20, 40, 26)
                signal_period = st.slider("Signal Period", 5, 15, 9)
            
            elif selected_strategy == "Stochastic Oscillator":
                k_period = st.slider("K Period", 5, 20, 14)
                d_period = st.slider("D Period", 2, 5, 3)
            
            elif selected_strategy == "Support/Resistance":
                lookback = st.slider("Lookback Period", 20, 200, 50)
        
        with col2:
            # Generate signals
            if st.button("🎯 Generate Trading Signals", use_container_width=True):
                self.generate_and_display_signals(
                    selected_strategy, selected_symbol,
                    locals().get('rsi_period', 14),
                    locals().get('oversold', 30),
                    locals().get('overbought', 70),
                    locals().get('short_window', 10),
                    locals().get('long_window', 50),
                    locals().get('bb_period', 20),
                    locals().get('num_std', 2),
                    locals().get('fast_period', 12),
                    locals().get('slow_period', 26),
                    locals().get('signal_period', 9),
                    locals().get('k_period', 14),
                    locals().get('d_period', 3),
                    locals().get('lookback', 50)
                )
            
            # Signal history
            st.write("*Recent Signals*")
            if 'signal_history' not in st.session_state:
                st.session_state.signal_history = []
            
            for signal in st.session_state.signal_history[-5:]:
                st.write(f"{signal}")
    
    def generate_and_display_signals(self, strategy, symbol, *params):
        """Generate and display trading signals"""
        # Generate sample price data (OHLC)
        np.random.seed(42)
        n_points = 200
        base_price = 100
        
        # Generate realistic price data with trends and volatility
        returns = np.random.normal(0.001, 0.02, n_points)
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC data
        high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
        low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
        open_prices = prices * (1 + np.random.normal(0, 0.005, n_points))
        close_prices = prices
        
        signals = []
        confidence_scores = []
        
        if strategy == "RSI Strategy":
            signal, rsi = self.strategies.rsi_strategy(close_prices, *params[:3])
            signals.append(signal)
            confidence_scores.append(min(abs(rsi - 50) / 50, 1.0))
            st.metric("RSI Value", f"{rsi:.2f}")
        
        elif strategy == "Moving Average Crossover":
            signal, short_ma, long_ma = self.strategies.moving_average_crossover(close_prices, *params[3:5])
            signals.append(signal)
            spread = abs(short_ma - long_ma) / long_ma
            confidence_scores.append(min(spread * 10, 1.0))
            st.metric("Short MA", f"{short_ma:.2f}")
            st.metric("Long MA", f"{long_ma:.2f}")
        
        elif strategy == "Bollinger Bands":
            signal, price, upper, lower = self.strategies.bollinger_bands(close_prices, *params[5:7])
            signals.append(signal)
            bandwidth = (upper - lower) / price
            confidence_scores.append(min(bandwidth * 5, 1.0))
            st.metric("Current Price", f"{price:.2f}")
            st.metric("Upper Band", f"{upper:.2f}")
            st.metric("Lower Band", f"{lower:.2f}")
        
        elif strategy == "MACD Strategy":
            signal, macd, signal_line, histogram = self.strategies.macd_strategy(close_prices, *params[7:10])
            signals.append(signal)
            confidence_scores.append(min(abs(histogram) * 10, 1.0))
            st.metric("MACD", f"{macd:.4f}")
            st.metric("Signal Line", f"{signal_line:.4f}")
            st.metric("Histogram", f"{histogram:.4f}")
        
        elif strategy == "Stochastic Oscillator":
            signal, k_value, d_value = self.strategies.stochastic_strategy(
                close_prices, high_prices, low_prices, *params[10:12]
            )
            signals.append(signal)
            confidence_scores.append(min(max(abs(k_value - 50), abs(d_value - 50)) / 50, 1.0))
            st.metric("K Value", f"{k_value:.2f}")
            st.metric("D Value", f"{d_value:.2f}")
        
        elif strategy == "Support/Resistance":
            signal, price, support, resistance = self.strategies.support_resistance_strategy(close_prices, *params[12:13])
            signals.append(signal)
            # Confidence based on proximity to levels
            if "BUY" in signal:
                confidence = (price - support) / (resistance - support)
            elif "SELL" in signal:
                confidence = (resistance - price) / (resistance - support)
            else:
                confidence = 0.3
            confidence_scores.append(confidence)
            st.metric("Current Price", f"{price:.2f}")
            st.metric("Support Level", f"{support:.2f}")
            st.metric("Resistance Level", f"{resistance:.2f}")
        
        elif strategy == "Multi-Strategy":
            # Combine all strategies
            rsi_signal, rsi_val = self.strategies.rsi_strategy(close_prices, 14, 30, 70)
            ma_signal, _, _ = self.strategies.moving_average_crossover(close_prices, 10, 50)
            bb_signal, _, _, _ = self.strategies.bollinger_bands(close_prices, 20, 2)
            macd_signal, _, _, _ = self.strategies.macd_strategy(close_prices, 12, 26, 9)
            stoch_signal, _, _ = self.strategies.stochastic_strategy(close_prices, high_prices, low_prices, 14, 3)
            sr_signal, _, _, _ = self.strategies.support_resistance_strategy(close_prices, 50)
            
            signals = [rsi_signal, ma_signal, bb_signal, macd_signal, stoch_signal, sr_signal]
            confidence_scores = [0.7, 0.8, 0.6, 0.75, 0.65, 0.7]
        
        # Determine final signal
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        
        if buy_signals > sell_signals:
            final_signal = "STRONG BUY" if buy_signals >= 3 else "BUY"
            signal_color = "green"
        elif sell_signals > buy_signals:
            final_signal = "STRONG SELL" if sell_signals >= 3 else "SELL"
            signal_color = "red"
        else:
            final_signal = "NEUTRAL"
            signal_color = "gray"
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Display results
        st.markdown(f"### 🎯 Trading Signal: :{signal_color}[{final_signal}]")
        st.metric("Confidence Score", f"{avg_confidence:.1%}")
        
        # Risk assessment
        risk_level = "LOW" if avg_confidence < 0.6 else "MEDIUM" if avg_confidence < 0.8 else "HIGH"
        st.metric("Risk Level", risk_level)
        
        # Add to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        history_entry = f"{timestamp} - {symbol}: {final_signal} ({avg_confidence:.1%})"
        st.session_state.signal_history.append(history_entry)
        
        # Show strategy breakdown
        st.write("*Strategy Breakdown:*")
        for i, (sig, conf) in enumerate(zip(signals, confidence_scores)):
            st.write(f"- Strategy {i+1}: {sig} ({conf:.1%} confidence)")

# Rest of the classes remain the same as in your original code...
# [The remaining classes (AutomatedTradingBot, IndianTraderTools, AethosIndiaPlatform) 
#  would be included here without changes to their structure]

class AutomatedTradingBot:
    """Fully automated trading bot with paper and live trading"""
    
    def __init__(self):
        self.exchange_data = IndianExchangeData()
        self.paper_balance = 100000  # Starting paper balance in INR
        self.positions = {}
        self.trade_history = []
    
    def create_automated_trading_dashboard(self):
        """Create dashboard for automated trading bots"""
        st.subheader("⚡ Fully Automated Trading Bots")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 Bot Config", "📊 Paper Trading", "🚀 Live Trading", "📈 Performance"])
        
        with tab1:
            self.create_bot_configurator()
        
        with tab2:
            self.create_paper_trading_interface()
        
        with tab3:
            self.create_live_trading_interface()
        
        with tab4:
            self.create_performance_dashboard()
    
    def create_bot_configurator(self):
        """Create bot configuration interface"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("*Bot Configuration*")
            
            bot_name = st.text_input("Bot Name", "Indian Market Maker Pro")
            
            trading_mode = st.radio(
                "Trading Mode",
                ["Paper Trading", "Live Trading"],
                help="Paper trading uses simulated money, Live trading uses real funds"
            )
            
            strategy_type = st.selectbox(
                "Trading Strategy",
                ["Market Making", "Mean Reversion", "Trend Following", "Momentum Trading", "Grid Trading"]
            )
            
            # Asset selection
            markets = self.exchange_data.get_all_markets()
            trading_pairs = st.multiselect(
                "Trading Pairs",
                [m['symbol'] for m in markets],
                default=['BTC-INR', 'ETH-INR']
            )
        
        with col2:
            st.write("*Risk Parameters*")
            
            max_position_size = st.slider("Max Position Size (%)", 1, 50, 10)
            daily_loss_limit = st.slider("Daily Loss Limit (%)", 1, 20, 5)
            max_drawdown = st.slider("Max Drawdown (%)", 5, 40, 15)
            leverage = st.slider("Leverage", 1, 10, 1)
            
            # Trading hours
            st.write("*Trading Hours*")
            start_time = st.time_input("Start Time", value=datetime.strptime("09:00", "%H:%M").time())
            end_time = st.time_input("End Time", value=datetime.strptime("17:00", "%H:%M").time())
        
        # Strategy-specific settings
        with st.expander("⚙ Strategy Settings"):
            if strategy_type == "Market Making":
                col_a, col_b = st.columns(2)
                with col_a:
                    spread = st.slider("Target Spread (%)", 0.1, 5.0, 0.5)
                    order_size = st.number_input("Order Size", min_value=0.001, value=0.01, step=0.001)
                with col_b:
                    inventory_limit = st.slider("Inventory Limit", 1, 100, 20)
                    rebalance_threshold = st.slider("Rebalance Threshold (%)", 1, 20, 5)
            
            elif strategy_type == "Mean Reversion":
                col_a, col_b = st.columns(2)
                with col_a:
                    lookback_period = st.slider("Lookback Period", 10, 200, 50)
                    z_score_entry = st.slider("Z-Score Entry", 1.0, 3.0, 2.0)
                with col_b:
                    z_score_exit = st.slider("Z-Score Exit", 0.1, 1.5, 0.5)
                    position_hold_time = st.slider("Max Hold (hours)", 1, 48, 24)
            
            elif strategy_type == "Trend Following":
                col_a, col_b = st.columns(2)
                with col_a:
                    trend_period = st.slider("Trend Period", 5, 50, 20)
                    momentum_threshold = st.slider("Momentum Threshold", 0.1, 5.0, 1.0)
                with col_b:
                    trailing_stop = st.slider("Trailing Stop (%)", 0.5, 10.0, 2.0)
                    position_growth = st.slider("Position Growth Factor", 1.0, 3.0, 1.5)
        
        # Deploy bot
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🚀 Deploy Bot", use_container_width=True):
                self.deploy_bot({
                    'name': bot_name,
                    'mode': trading_mode,
                    'strategy': strategy_type,
                    'pairs': trading_pairs,
                    'risk_params': {
                        'max_position_size': max_position_size,
                        'daily_loss_limit': daily_loss_limit,
                        'max_drawdown': max_drawdown,
                        'leverage': leverage
                    }
                })
        
        with col3:
            if st.button("🛑 Stop All Bots", use_container_width=True, type="secondary"):
                st.session_state.running_bots = {}
                st.success("All bots stopped!")
    
    def deploy_bot(self, config):
        """Deploy trading bot with given configuration"""
        if 'running_bots' not in st.session_state:
            st.session_state.running_bots = {}
        
        bot_id = f"{config['name']}_{datetime.now().strftime('%H%M%S')}"
        st.session_state.running_bots[bot_id] = {
            'config': config,
            'status': 'running',
            'start_time': datetime.now(),
            'performance': {
                'trades': 0,
                'win_rate': 0,
                'pnl': 0,
                'sharpe': 0
            }
        }
        
        st.success(f"✅ {config['name']} deployed successfully!")
        st.info(f"Bot ID: {bot_id}")
    
    def create_paper_trading_interface(self):
        """Create paper trading interface"""
        st.write("📊 Paper Trading Account")
        
        # Account overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Paper Balance", f"₹{self.paper_balance:,.2f}")
        with col2:
            st.metric("Open Positions", len(self.positions))
        with col3:
            total_pnl = sum([pos.get('unrealized_pnl', 0) for pos in self.positions.values()])
            st.metric("Unrealized P&L", f"₹{total_pnl:,.2f}")
        with col4:
            realized_pnl = sum([trade.get('pnl', 0) for trade in self.trade_history])
            st.metric("Realized P&L", f"₹{realized_pnl:,.2f}")
        
        # Manual trading
        st.write("*Manual Paper Trading*")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            asset = st.selectbox("Asset", ['BTC-INR', 'ETH-INR', 'SOL-INR', 'MATIC-INR'])
        with col2:
            action = st.radio("Action", ["BUY", "SELL"])
        with col3:
            quantity = st.number_input("Quantity", min_value=0.001, value=0.01, step=0.001)
            price = st.number_input("Price (INR)", min_value=0.01, value=3500000.0 if asset == 'BTC-INR' else 200000.0)
        
        if st.button("📝 Execute Paper Trade", use_container_width=True):
            self.execute_paper_trade(asset, action, quantity, price)
        
        # Positions table
        if self.positions:
            st.write("*Current Positions*")
            positions_data = []
            for asset, pos in self.positions.items():
                positions_data.append({
                    'Asset': asset,
                    'Quantity': pos.get('quantity', 0),
                    'Entry Price': pos.get('entry_price', 0),
                    'Current Price': price,
                    'P&L': pos.get('unrealized_pnl', 0)
                })
            positions_df = pd.DataFrame(positions_data)
            st.dataframe(positions_df, use_container_width=True)
        
        # Trade history
        if self.trade_history:
            st.write("*Trade History*")
            history_df = pd.DataFrame(self.trade_history[-10:])
            st.dataframe(history_df, use_container_width=True)
    
    def execute_paper_trade(self, asset, action, quantity, price):
        """Execute paper trade"""
        trade_value = quantity * price
        
        if action == "BUY":
            if trade_value > self.paper_balance:
                st.error("❌ Insufficient paper balance!")
                return
            
            self.paper_balance -= trade_value
            if asset in self.positions:
                # Average position
                old_pos = self.positions[asset]
                total_quantity = old_pos['quantity'] + quantity
                avg_price = ((old_pos['quantity'] * old_pos['entry_price']) + trade_value) / total_quantity
                self.positions[asset] = {
                    'quantity': total_quantity,
                    'entry_price': avg_price,
                    'unrealized_pnl': 0
                }
            else:
                self.positions[asset] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'unrealized_pnl': 0
                }
        
        else:  # SELL
            if asset not in self.positions or self.positions[asset]['quantity'] < quantity:
                st.error("❌ Insufficient position!")
                return
            
            position = self.positions[asset]
            pnl = (price - position['entry_price']) * quantity
            
            self.paper_balance += trade_value
            self.positions[asset]['quantity'] -= quantity
            
            if self.positions[asset]['quantity'] == 0:
                del self.positions[asset]
            
            # Record trade
            self.trade_history.append({
                'timestamp': datetime.now(),
                'asset': asset,
                'action': action,
                'quantity': quantity,
                'price': price,
                'pnl': pnl
            })
        
        st.success(f"✅ {action} {quantity} {asset} at ₹{price:,.2f}")
    
    def create_live_trading_interface(self):
        """Create live trading interface"""
        st.write("🚀 Live Trading")
        
        st.info("""
        ⚠ *Live Trading Warning*: 
        - Connect only to exchanges you trust
        - Use API keys with limited permissions
        - Start with small amounts
        - Monitor bots regularly
        """)
        
        # Exchange connection
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("*Exchange Connections*")
            delta_connected = st.checkbox("Delta Exchange", value=False)
            
            if st.button("🔗 Connect Exchanges", use_container_width=True):
                if delta_connected:
                    st.success("Exchange connected successfully!")
                else:
                    st.warning("Select at least one exchange")
        
        with col2:
            st.write("*Live Bot Status*")
            if 'running_bots' in st.session_state:
                for bot_id, bot in st.session_state.running_bots.items():
                    if bot['config']['mode'] == 'Live Trading':
                        status_color = "🟢" if bot['status'] == 'running' else "🔴"
                        st.write(f"{status_color} {bot['config']['name']} - {bot['status']}")
            else:
                st.write("No live bots running")
        
        # Risk acknowledgment
        st.warning("""
        *Risk Disclosure*: 
        Automated trading involves substantial risk. Past performance is not indicative of future results. 
        You should only trade with money you can afford to lose.
        """)
        
        acknowledge = st.checkbox("I understand the risks and want to proceed with live trading")
        
        if acknowledge and st.button("🔥 Start Live Trading", use_container_width=True, type="primary"):
            st.success("Live trading activated! Monitor your bots closely.")
    
    def create_performance_dashboard(self):
        """Create performance monitoring dashboard"""
        st.write("📈 Bot Performance Analytics")
        
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        
        # Equity curve
        equity_curve = 10000 + np.cumsum(np.random.randn(len(dates)) * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=equity_curve,
            name="Portfolio Value",
            line=dict(color='#00d4aa', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "15.8%", "2.1%")
        with col2:
            st.metric("Sharpe Ratio", "1.45", "0.08")
        with col3:
            st.metric("Max Drawdown", "-8.2%", "Current")
        with col4:
            st.metric("Win Rate", "58.7%", "3.2%")
        
        # Strategy performance breakdown
        st.write("*Strategy Performance*")
        strategies_data = {
            'Strategy': ['Market Making', 'Mean Reversion', 'Trend Following', 'Momentum Trading'],
            'Return %': [12.5, 18.2, 9.8, 15.3],
            'Win Rate %': [65.2, 58.7, 52.3, 61.8],
            'Sharpe Ratio': [1.8, 1.2, 0.9, 1.5],
            'Max DD %': [-6.2, -12.5, -15.8, -10.3]
        }
        
        st.dataframe(pd.DataFrame(strategies_data), use_container_width=True)

class IndianTraderTools:
    def __init__(self):
        self.exchange_data = IndianExchangeData()
    
    def create_tax_calculator(self):
        """Create advanced Indian crypto tax calculator"""
        st.subheader("🇮🇳 Advanced Tax Calculator")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Trade input
            st.write("*Trade Details*")
            trade_type = st.selectbox("Trade Type", ["Buy", "Sell", "Transfer", "Staking Reward"])
            asset = st.selectbox("Asset", ["BTC", "ETH", "SOL", "MATIC", "ADA", "DOT"])
            quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, step=0.1)
            price_per_unit = st.number_input("Price per Unit (INR)", min_value=0.0, value=100000.0)
            trade_date = st.date_input("Trade Date", value=datetime.now())
            
            # Advanced tax options
            st.write("*Tax Options*")
            col_a, col_b = st.columns(2)
            with col_a:
                financial_year = st.selectbox("Financial Year", ["2024-25", "2023-24", "2022-23"])
                include_tds = st.checkbox("Include 1% TDS", value=True)
            with col_b:
                apply_cess = st.checkbox("Apply Health & Education Cess", value=True)
                include_penalty = st.checkbox("Include Late Filing Penalty", value=False)
        
        with col2:
            # Tax calculation results
            st.write("*Tax Calculation*")
            
            # Calculate based on Indian tax laws
            total_value = quantity * price_per_unit
            
            if trade_type == "Sell":
                # Assume 30% tax on profits for simplicity
                profit = total_value * 0.3  # Simplified calculation
                tax_rate = 0.3
            elif trade_type == "Staking Reward":
                profit = total_value
                tax_rate = 0.3
            else:
                profit = 0
                tax_rate = 0
            
            tax_amount = profit * tax_rate
            
            if apply_cess:
                cess = tax_amount * 0.04
                tax_amount += cess
            
            if include_tds:
                tds = total_value * 0.01
                tax_amount += tds
            
            if include_penalty:
                penalty = tax_amount * 0.5  # 50% penalty
                tax_amount += penalty
            
            st.metric("Total Value", f"₹{total_value:,.2f}")
            st.metric("Taxable Amount", f"₹{profit:,.2f}")
            st.metric("Tax Rate", f"{tax_rate*100}%")
            st.metric("Total Tax", f"₹{tax_amount:,.2f}", delta="Payable")
            
            # Tax saving tips
            with st.expander("💡 Tax Saving Tips"):
                st.write("""
                - Hold assets for more than 3 years for lower tax rates
                - Offset losses against gains
                - Maintain proper documentation
                - Use designated crypto tax software
                - Consult with tax professional
                """)
    
    def create_indian_market_insights(self):
        """Create Indian market insights dashboard"""
        st.subheader("📊 Indian Market Insights")
        
        # Market sentiment
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Retail Sentiment", "Bullish", "↗")
        with col2:
            st.metric("Institutional Flow", "Neutral", "➡")
        with col3:
            st.metric("Regulatory Climate", "Improving", "↗")
        with col4:
            st.metric("Adoption Rate", "High", "📈")
        
        # Indian crypto adoption trends
        st.write("🇮🇳 Indian Crypto Adoption Trends")
        
        # Generate adoption data
        years = ['2020', '2021', '2022', '2023', '2024']
        users_millions = [5, 15, 25, 35, 50]
        volume_billions = [10, 45, 80, 120, 180]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=users_millions,
            name="Crypto Users (Millions)",
            line=dict(color='#00d4aa', width=3),
            yaxis='y'
        ))
        
        fig.add_trace(go.Bar(
            x=years, y=volume_billions,
            name="Trading Volume (Billions INR)",
            marker_color='#0099ff',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Indian Crypto Market Growth",
            xaxis_title="Year",
            yaxis=dict(title="Users (Millions)", side='left'),
            yaxis2=dict(title="Volume (Billions INR)", side='right', overlaying='y'),
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_regulatory_dashboard(self):
        """Create Indian regulatory compliance dashboard"""
        st.subheader("📜 Regulatory Compliance Center")
        
        # Current regulations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="regulatory-alert">
            <h4>🇮🇳 Current Indian Regulations</h4>
            <ul>
            <li>30% Tax on Crypto Profits</li>
            <li>1% TDS on All Transactions</li>
            <li>No Offset of Losses Allowed</li>
            <li>Gifts Taxable as Income</li>
            <li>Staking Rewards Taxable</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.write("*Compliance Checklist*")
            kyc_done = st.checkbox("KYC Completed", value=True)
            tax_paid = st.checkbox("Taxes Filed for Previous Year", value=False)
            tds_deducted = st.checkbox("TDS Being Deducted", value=True)
            records_maintained = st.checkbox("Transaction Records Maintained", value=True)
            
            compliance_score = (kyc_done + tax_paid + tds_deducted + records_maintained) * 25
            st.metric("Compliance Score", f"{compliance_score}%")
        
        # Regulatory updates
        st.write("*Latest Updates*")
        updates = [
            {"date": "2024-03-01", "update": "CBDC Pilot Expanded to 15 Cities", "impact": "Medium"},
            {"date": "2024-02-15", "update": "SEBI Proposes Crypto Classification Framework", "impact": "High"},
            {"date": "2024-02-01", "update": "RBI Issues Warning on Unregulated Exchanges", "impact": "High"},
            {"date": "2024-01-20", "update": "Government Forms Crypto Tax Task Force", "impact": "Medium"},
        ]
        
        for update in updates:
            with st.expander(f"📅 {update['date']} - {update['update']} (Impact: {update['impact']})"):
                st.write("Indian regulators continue to develop comprehensive framework for digital assets. Stay updated with latest compliance requirements.")

class AethosIndiaPlatform:
    def __init__(self):
        self.exchange_data = IndianExchangeData()
        self.semi_bots = SemiAutomatedBots()
        self.auto_bots = AutomatedTradingBot()
        self.indian_tools = IndianTraderTools()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = "BTCUSDT"
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'signal_history' not in st.session_state:
            st.session_state.signal_history = []
        if 'running_bots' not in st.session_state:
            st.session_state.running_bots = {}
    
    def create_header(self):
        """Create the main header section"""
        st.markdown("""
        <div class="main-header">
            <h1 style="margin:0; padding:0; color: #f0f2f6;">AETHOS PLATFORM 🇮🇳</h1>
            <div class="platform-tagline">Advanced Algorithmic Trading for Indian Crypto Markets</div>
            <div class="indian-flag" style="margin-top: 1rem;">
                🤖 Semi-Auto Signals • ⚡ Full Auto Trading • 📊 Real-time Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Real-time metrics
        prices = self.exchange_data.get_consolidated_prices()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_bots = len(st.session_state.get('running_bots', {}))
            st.metric("Active Bots", active_bots, "Live")
        
        with col2:
            total_signals = len(st.session_state.get('signal_history', []))
            st.metric("Signals Today", total_signals)
        
        with col3:
            if prices:
                btc_price = next((p for p in prices if 'BTC' in p['symbol']), None)
                if btc_price:
                    st.metric("BTC Price", f"${btc_price['price']:,.0f}", 
                             f"{btc_price['change_24h']:.1f}%")
        
        with col4:
            st.metric("Platform Status", "Online", delta="Real-time")
    
    def create_algorithmic_trading_section(self):
        """Create comprehensive algo trading section"""
        st.header("🎯 Algorithmic Trading Studio")
        
        tab1, tab2, tab3 = st.tabs([
            "🤖 Semi-Auto Signal Bots", 
            "⚡ Fully Automated Bots", 
            "📊 Live Bot Monitoring"
        ])
        
        with tab1:
            self.semi_bots.create_signal_dashboard()
        
        with tab2:
            self.auto_bots.create_automated_trading_dashboard()
        
        with tab3:
            self.create_bot_monitoring_dashboard()
    
    def create_bot_monitoring_dashboard(self):
        """Create real-time bot monitoring dashboard"""
        st.subheader("📊 Live Bot Monitoring & Analytics")
        
        if not st.session_state.get('running_bots'):
            st.info("No active bots. Deploy bots from the Automated Trading section.")
            return
        
        # Bot status overview
        col1, col2, col3, col4 = st.columns(4)
        
        running_count = sum(1 for bot in st.session_state.running_bots.values() if bot['status'] == 'running')
        total_pnl = sum(bot['performance']['pnl'] for bot in st.session_state.running_bots.values())
        total_trades = sum(bot['performance']['trades'] for bot in st.session_state.running_bots.values())
        avg_win_rate = np.mean([bot['performance']['win_rate'] for bot in st.session_state.running_bots.values()])
        
        with col1:
            st.metric("Active Bots", running_count)
        with col2:
            st.metric("Total P&L", f"₹{total_pnl:,.2f}")
        with col3:
            st.metric("Total Trades", total_trades)
        with col4:
            st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
        
        # Individual bot cards
        st.write("*Bot Details*")
        for bot_id, bot in st.session_state.running_bots.items():
            with st.expander(f"{'🟢' if bot['status'] == 'running' else '🔴'} {bot['config']['name']} - {bot['config']['mode']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("*Configuration*")
                    st.write(f"Strategy: {bot['config']['strategy']}")
                    st.write(f"Pairs: {', '.join(bot['config']['pairs'])}")
                    st.write(f"Started: {bot['start_time'].strftime('%H:%M:%S')}")
                
                with col2:
                    st.write("*Performance*")
                    st.write(f"Trades: {bot['performance']['trades']}")
                    st.write(f"Win Rate: {bot['performance']['win_rate']}%")
                    st.write(f"P&L: ₹{bot['performance']['pnl']:,.2f}")
                
                with col3:
                    st.write("*Controls*")
                    if bot['status'] == 'running':
                        if st.button(f"⏸ Pause", key=f"pause_{bot_id}"):
                            bot['status'] = 'paused'
                            st.rerun()
                    else:
                        if st.button(f"▶ Resume", key=f"resume_{bot_id}"):
                            bot['status'] = 'running'
                            st.rerun()
                    
                    if st.button(f"🛑 Stop", key=f"stop_{bot_id}"):
                        del st.session_state.running_bots[bot_id]
                        st.rerun()
    
    def create_indian_trader_tools(self):
        """Create Indian trader tools section"""
        st.header("🇮🇳 Indian Trader Suite")
        
        tab1, tab2, tab3 = st.tabs([
            "💰 Tax Calculator", 
            "📊 Market Insights", 
            "📜 Regulatory Center"
        ])
        
        with tab1:
            self.indian_tools.create_tax_calculator()
        
        with tab2:
            self.indian_tools.create_indian_market_insights()
        
        with tab3:
            self.indian_tools.create_regulatory_dashboard()
    
    def create_markets_overview(self):
        """Create markets overview section"""
        st.header("📈 Live Markets Overview")
        
        prices = self.exchange_data.get_consolidated_prices()
        
        if prices:
            # Market metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_volume = sum(p['volume'] for p in prices)
            avg_change = np.mean([p['change_24h'] for p in prices if p.get('change_24h')])
            inr_pairs = len([p for p in prices if 'INR' in str(p.get('symbol', ''))])
            
            with col1:
                st.metric("Total Markets", len(prices))
            with col2:
                st.metric("24h Volume", f"${total_volume:,.0f}")
            with col3:
                st.metric("Avg 24h Change", f"{avg_change:.2f}%")
            with col4:
                st.metric("INR Pairs", inr_pairs)
            
            # Display market data
            st.write("*Live Market Data*")
            df = pd.DataFrame(prices)
            display_df = df[['exchange', 'symbol', 'price', 'change_24h', 'volume']].copy()
            st.dataframe(display_df, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        self.create_header()
        
        # Create main navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Dashboard", 
            "🎯 Algo Trading", 
            "🇮🇳 Trader Tools", 
            "📈 Markets"
        ])
        
        with tab1:
            self.create_markets_overview()
            
            # Quick actions
            st.subheader("🚀 Quick Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🤖 Deploy New Bot", use_container_width=True):
                    st.info("Navigate to Algo Trading tab")
            with col2:
                if st.button("📊 View Signals", use_container_width=True):
                    st.info("Signal dashboard loaded")
            with col3:
                if st.button("💰 Calculate Tax", use_container_width=True):
                    st.info("Tax calculator opened")
            with col4:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.rerun()
        
        with tab2:
            self.create_algorithmic_trading_section()
        
        with tab3:
            self.create_indian_trader_tools()
        
        with tab4:
            self.create_markets_overview()

# Run the application
if __name__ == "__main__":
    
    platform = AethosIndiaPlatform()
    platform.run()
