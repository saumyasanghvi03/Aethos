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

# Load custom CSS
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
.exchange-button {
    padding: 0.5rem 1rem;
    margin: 0.2rem;
    border-radius: 5px;
    border: 1px solid #ddd;
    background: white;
    cursor: pointer;
}
.exchange-button.active {
    background: #667eea;
    color: white;
    border-color: #667eea;
}
.token-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}
.indicator-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    text-align: center;
}
.positive { color: #00d4aa; }
.negative { color: #ff4b4b; }
.neutral { color: #ffa500; }
</style>
""", unsafe_allow_html=True)

class CoinMarketCapAPI:
    """CoinMarketCap API integration for real fundamental data"""
    
    def __init__(self):
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.api_key = st.secrets.get("COIN_MARKET_CAP_API_KEY", "")
        
    def make_request(self, endpoint, params=None):
        """Make authenticated request to CMC API"""
        if not self.api_key:
            st.error("❌ CoinMarketCap API key not found. Please add COIN_MARKET_CAP_API_KEY to Streamlit secrets.")
            return None
            
        headers = {
            'X-CMC_PRO_API_KEY': self.api_key,
            'Accept': 'application/json'
        }
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ CMC API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"❌ Error fetching CMC data: {e}")
            return None
    
    def get_crypto_listings(self, limit=100, convert='USD'):
        """Get top cryptocurrency listings"""
        params = {
            'limit': limit,
            'convert': convert,
            'sort': 'market_cap',
            'sort_dir': 'desc'
        }
        return self.make_request('cryptocurrency/listings/latest', params)
    
    def get_crypto_info(self, symbol):
        """Get detailed information for a specific cryptocurrency"""
        params = {
            'symbol': symbol.upper(),
            'convert': 'INR,USD'
        }
        return self.make_request('cryptocurrency/quotes/latest', params)
    
    def get_metadata(self, symbol):
        """Get metadata for a cryptocurrency"""
        params = {
            'symbol': symbol.upper()
        }
        return self.make_request('cryptocurrency/info', params)
    
    def get_global_metrics(self):
        """Get global cryptocurrency metrics"""
        params = {
            'convert': 'INR,USD'
        }
        return self.make_request('global-metrics/quotes/latest', params)

class DeltaExchange:
    def __init__(self):
        self.base_url = "https://api.delta.exchange"
        self.api_key = st.secrets.get("DELTA_API_KEY", "") if hasattr(st, 'secrets') else ""
        self.api_secret = st.secrets.get("DELTA_API_SECRET", "") if hasattr(st, 'secrets') else ""
    
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

class IndianExchanges:
    """Indian exchange data for CoinDCX, Coinswitch, Mudrex, and ZebPay"""
    def __init__(self):
        self.exchanges = {
            'CoinDCX': {
                'pairs': ['BTCINR', 'ETHINR', 'USDTINR', 'MATICINR', 'ADAINR', 'DOTINR', 'SOLINR', 'XRPINR', 'BNBINR']
            },
            'Coinswitch': {
                'pairs': ['BTC-INR', 'ETH-INR', 'USDT-INR', 'MATIC-INR', 'ADA-INR', 'DOT-INR']
            },
            'Mudrex': {
                'pairs': ['BTC-INR', 'ETH-INR', 'USDT-INR', 'MATIC-INR', 'SOL-INR', 'ADA-INR']
            },
            'ZebPay': {
                'pairs': ['BTC-INR', 'ETH-INR', 'USDT-INR', 'MATIC-INR', 'ADA-INR', 'SOL-INR']
            }
        }
    
    def get_ticker(self, exchange, symbol):
        """Get ticker data from Indian exchange (simulated)"""
        # Base prices for simulation
        base_prices = {
            'BTCINR': 3500000,
            'ETHINR': 200000,
            'USDTINR': 83,
            'MATICINR': 60,
            'ADAINR': 40,
            'DOTINR': 500,
            'SOLINR': 8000,
            'XRPINR': 50,
            'BNBINR': 25000
        }
        
        # Clean symbol for lookup
        symbol_upper = symbol.upper().replace('-', '').replace('_', '')
        base_price = base_prices.get(symbol_upper, 100)
        
        # Exchange-specific variations
        exchange_factors = {
            'CoinDCX': 1.0, 
            'Coinswitch': 0.998, 
            'Mudrex': 1.001,
            'ZebPay': 1.002
        }
        
        variation = np.random.uniform(-0.02, 0.02)
        price = base_price * exchange_factors.get(exchange, 1.0) * (1 + variation)
        
        return {
            'symbol': symbol,
            'price': price,
            'volume': np.random.uniform(100000, 500000),
            'change_24h': np.random.uniform(-5, 5),
            'high_24h': price * (1 + abs(np.random.uniform(0, 0.1))),
            'low_24h': price * (1 - abs(np.random.uniform(0, 0.1))),
            'exchange': exchange
        }

class MarketDataProvider:
    def __init__(self):
        self.delta = DeltaExchange()
        self.indian = IndianExchanges()
        self.cmc = CoinMarketCapAPI()
        self.available_exchanges = ['Delta', 'CoinDCX', 'Coinswitch', 'Mudrex', 'ZebPay']
    
    def get_exchange_symbols(self, exchange):
        """Get available symbols for an exchange"""
        if exchange == 'Delta':
            products = self.delta.get_products()
            return [p['symbol'] for p in products[:20] if p]
        else:
            return self.indian.exchanges[exchange]['pairs']
    
    def get_token_data(self, exchange, symbol):
        """Get comprehensive token data"""
        if exchange == 'Delta':
            ticker = self.delta.get_ticker(symbol)
            if ticker:
                return {
                    'symbol': symbol,
                    'price': float(ticker.get('close', 0)),
                    'volume': float(ticker.get('volume', 0)),
                    'change_24h': float(ticker.get('change_24h', 0)),
                    'high_24h': float(ticker.get('high', 0)),
                    'low_24h': float(ticker.get('low', 0)),
                    'exchange': exchange
                }
        else:
            return self.indian.get_ticker(exchange, symbol)
        return None
    
    def get_technical_indicators(self, symbol, exchange):
        """Calculate technical indicators for a symbol"""
        np.random.seed(hash(symbol) % 1000)
        n_points = 200
        
        returns = np.random.normal(0.001, 0.02, n_points)
        prices = 100 * np.cumprod(1 + returns)
        
        if TA_LIB_AVAILABLE:
            rsi = talib.RSI(prices, timeperiod=14)[-1]
            macd, macd_signal, macd_hist = talib.MACD(prices)
            upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            sma_20 = talib.SMA(prices, timeperiod=20)[-1]
            sma_50 = talib.SMA(prices, timeperiod=50)[-1]
            high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
            low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
            slowk, slowd = talib.STOCH(high_prices, low_prices, prices)
            stoch_k = slowk[-1]
            stoch_d = slowd[-1]
        else:
            rsi = 50 + np.random.uniform(-20, 20)
            macd_val = np.random.uniform(-2, 2)
            macd_signal_val = np.random.uniform(-2, 2)
            bb_upper = prices[-1] * 1.1
            bb_lower = prices[-1] * 0.9
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:])
            stoch_k = 50 + np.random.uniform(-30, 30)
            stoch_d = 50 + np.random.uniform(-30, 30)
        
        return {
            'RSI': rsi,
            'MACD': macd_val if TA_LIB_AVAILABLE else macd_val,
            'MACD_Signal': macd_signal_val if TA_LIB_AVAILABLE else macd_signal_val,
            'BB_Upper': bb_upper,
            'BB_Lower': bb_lower,
            'SMA_20': sma_20,
            'SMA_50': sma_50,
            'Stoch_K': stoch_k,
            'Stoch_D': stoch_d,
            'Current_Price': prices[-1]
        }
    
    def get_real_fundamental_data(self, symbol):
        """Get REAL fundamental data from CoinMarketCap"""
        # Extract base symbol for CMC lookup
        base_symbol = self._extract_base_symbol(symbol)
        
        # Get quotes data
        quotes_data = self.cmc.get_crypto_info(base_symbol)
        if not quotes_data:
            return self._get_fallback_data(base_symbol)
        
        # Get metadata
        metadata = self.cmc.get_metadata(base_symbol)
        
        return self._parse_cmc_data(quotes_data, metadata, base_symbol)
    
    def _extract_base_symbol(self, symbol):
        """Extract base symbol from trading pair"""
        # Remove exchange-specific suffixes and quote currencies
        symbol_clean = symbol.upper().replace('-', '').replace('_', '').replace('INR', '').replace('USDT', '')
        
        # Common cryptocurrency symbols
        crypto_symbols = ['BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 
                         'DOT', 'MATIC', 'LINK', 'LTC', 'BCH', 'XLM', 'ATOM']
        
        for crypto in crypto_symbols:
            if crypto in symbol_clean:
                return crypto
        
        return symbol_clean[:4] if symbol_clean else symbol
    
    def _parse_cmc_data(self, quotes_data, metadata, symbol):
        """Parse CMC API response into structured data"""
        try:
            data = quotes_data.get('data', {})
            if not data:
                return self._get_fallback_data(symbol)
            
            # Get the first cryptocurrency in response
            crypto_key = list(data.keys())[0]
            crypto_data = data[crypto_key]
            quote_data = crypto_data.get('quote', {}).get('USD', {})
            
            # Parse metadata if available
            meta_info = {}
            if metadata and 'data' in metadata:
                meta_data = metadata['data'].get(crypto_key, {})
                meta_info = {
                    'description': meta_data.get('description', ''),
                    'logo': meta_data.get('logo', ''),
                    'urls': meta_data.get('urls', {}),
                    'tags': meta_data.get('tags', []),
                    'date_added': meta_data.get('date_added', ''),
                    'category': meta_data.get('category', '')
                }
            
            # Calculate additional metrics
            market_cap = quote_data.get('market_cap', 0)
            volume_24h = quote_data.get('volume_24h', 0)
            circulating_supply = crypto_data.get('circulating_supply', 0)
            total_supply = crypto_data.get('total_supply', 0)
            max_supply = crypto_data.get('max_supply', 0)
            
            # Calculate volume to market cap ratio
            volume_mcap_ratio = (volume_24h / market_cap) * 100 if market_cap > 0 else 0
            
            # Determine market sentiment
            percent_change_24h = quote_data.get('percent_change_24h', 0)
            percent_change_7d = quote_data.get('percent_change_7d', 0)
            sentiment = self._calculate_sentiment(percent_change_24h, percent_change_7d)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(
                market_cap, 
                volume_mcap_ratio,
                percent_change_24h
            )
            
            return {
                'name': crypto_data.get('name', symbol),
                'symbol': crypto_data.get('symbol', symbol),
                'price_usd': quote_data.get('price', 0),
                'price_inr': quote_data.get('price', 0) * 83,  # Approximate conversion
                'market_cap': market_cap,
                'market_cap_display': f"${market_cap:,.0f}" if market_cap else "N/A",
                'volume_24h': volume_24h,
                'volume_24h_display': f"${volume_24h:,.0f}" if volume_24h else "N/A",
                'circulating_supply': circulating_supply,
                'circulating_supply_display': f"{circulating_supply:,.0f}" if circulating_supply else "N/A",
                'total_supply': total_supply,
                'total_supply_display': f"{total_supply:,.0f}" if total_supply else "N/A",
                'max_supply': max_supply,
                'max_supply_display': f"{max_supply:,.0f}" if max_supply else "Unlimited",
                'percent_change_1h': quote_data.get('percent_change_1h', 0),
                'percent_change_24h': percent_change_24h,
                'percent_change_7d': percent_change_7d,
                'percent_change_30d': quote_data.get('percent_change_30d', 0),
                'market_cap_rank': crypto_data.get('cmc_rank', 'N/A'),
                'volume_mcap_ratio': volume_mcap_ratio,
                'sentiment': sentiment,
                'risk_level': risk_level,
                'description': meta_info.get('description', 'No description available'),
                'website': meta_info.get('urls', {}).get('website', [''])[0] if meta_info.get('urls') else '',
                'explorer': meta_info.get('urls', {}).get('explorer', [''])[0] if meta_info.get('urls') else '',
                'tags': meta_info.get('tags', []),
                'category': meta_info.get('category', 'cryptocurrency'),
                'fully_diluted_market_cap': quote_data.get('fully_diluted_market_cap', 0),
                'fully_diluted_market_cap_display': f"${quote_data.get('fully_diluted_market_cap', 0):,.0f}" if quote_data.get('fully_diluted_market_cap') else "N/A"
            }
            
        except Exception as e:
            st.error(f"Error parsing CMC data: {e}")
            return self._get_fallback_data(symbol)
    
    def _calculate_sentiment(self, change_24h, change_7d):
        """Calculate market sentiment based on price changes"""
        if change_24h > 5 and change_7d > 10:
            return "Very Bullish"
        elif change_24h > 2 and change_7d > 5:
            return "Bullish"
        elif change_24h < -5 and change_7d < -10:
            return "Very Bearish"
        elif change_24h < -2 and change_7d < -5:
            return "Bearish"
        else:
            return "Neutral"
    
    def _calculate_risk_level(self, market_cap, volume_ratio, change_24h):
        """Calculate risk level based on multiple factors"""
        risk_score = 0
        
        # Market cap factor (higher cap = lower risk)
        if market_cap > 10000000000:  # $10B+
            risk_score += 1
        elif market_cap > 1000000000:  # $1B-$10B
            risk_score += 2
        elif market_cap > 100000000:   # $100M-$1B
            risk_score += 3
        else:                          # <$100M
            risk_score += 4
        
        # Volume ratio factor (higher ratio = lower risk)
        if volume_ratio > 10:
            risk_score += 1
        elif volume_ratio > 5:
            risk_score += 2
        elif volume_ratio > 2:
            risk_score += 3
        else:
            risk_score += 4
        
        # Volatility factor
        if abs(change_24h) < 2:
            risk_score += 1
        elif abs(change_24h) < 5:
            risk_score += 2
        elif abs(change_24h) < 10:
            risk_score += 3
        else:
            risk_score += 4
        
        # Determine risk level
        if risk_score <= 6:
            return "Low"
        elif risk_score <= 9:
            return "Medium"
        else:
            return "High"
    
    def _get_fallback_data(self, symbol):
        """Provide fallback data when CMC API fails"""
        return {
            'name': symbol,
            'symbol': symbol,
            'price_usd': 0,
            'price_inr': 0,
            'market_cap': 0,
            'market_cap_display': "N/A",
            'volume_24h': 0,
            'volume_24h_display': "N/A",
            'circulating_supply': 0,
            'circulating_supply_display': "N/A",
            'total_supply': 0,
            'total_supply_display': "N/A",
            'max_supply': 0,
            'max_supply_display': "N/A",
            'percent_change_1h': 0,
            'percent_change_24h': 0,
            'percent_change_7d': 0,
            'percent_change_30d': 0,
            'market_cap_rank': "N/A",
            'volume_mcap_ratio': 0,
            'sentiment': "Neutral",
            'risk_level': "Medium",
            'description': "Data temporarily unavailable",
            'website': '',
            'explorer': '',
            'tags': [],
            'category': 'cryptocurrency',
            'fully_diluted_market_cap': 0,
            'fully_diluted_market_cap_display': "N/A"
        }

class TechnicalAnalysis:
    """Technical analysis utilities"""
    
    @staticmethod
    def get_signal_from_indicators(indicators):
        """Generate trading signal from technical indicators"""
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if indicators['RSI'] < 30:
            buy_signals += 1
        elif indicators['RSI'] > 70:
            sell_signals += 1
        
        # MACD signals
        if indicators['MACD'] > indicators['MACD_Signal']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Bollinger Bands signals
        if indicators['Current_Price'] <= indicators['BB_Lower']:
            buy_signals += 1
        elif indicators['Current_Price'] >= indicators['BB_Upper']:
            sell_signals += 1
        
        # Moving Average signals
        if indicators['SMA_20'] > indicators['SMA_50']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Stochastic signals
        if indicators['Stoch_K'] < 20 and indicators['Stoch_D'] < 20:
            buy_signals += 1
        elif indicators['Stoch_K'] > 80 and indicators['Stoch_D'] > 80:
            sell_signals += 1
        
        if buy_signals > sell_signals:
            return "BUY", buy_signals / (buy_signals + sell_signals)
        elif sell_signals > buy_signals:
            return "SELL", sell_signals / (buy_signals + sell_signals)
        else:
            return "NEUTRAL", 0.5

class MarketsPage:
    def __init__(self):
        self.market_data = MarketDataProvider()
        self.technical_analysis = TechnicalAnalysis()
    
    def create_markets_page(self):
        """Create the dedicated markets page"""
        st.header("📈 Advanced Markets Analysis")
        
        # Exchange selection
        st.subheader("🏢 Select Exchange")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        exchanges = ['Delta', 'CoinDCX', 'Coinswitch', 'Mudrex', 'ZebPay']
        
        for i, exchange in enumerate(exchanges):
            with [col1, col2, col3, col4, col5][i]:
                if st.button(f"🔗 {exchange}", use_container_width=True,
                           type="primary" if st.session_state.get('selected_exchange') == exchange else "secondary"):
                    st.session_state.selected_exchange = exchange
        
        # Initialize default exchange
        if 'selected_exchange' not in st.session_state:
            st.session_state.selected_exchange = 'Delta'
        
        st.markdown(f"### 🔍 Analyzing: {st.session_state.selected_exchange}")
        
        # Symbol search and selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get available symbols for selected exchange
            symbols = self.market_data.get_exchange_symbols(st.session_state.selected_exchange)
            
            # Symbol search
            search_term = st.text_input("🔎 Search Symbol", placeholder="e.g., BTC, ETH, MATIC...")
            
            if search_term:
                filtered_symbols = [s for s in symbols if search_term.upper() in s.upper()]
            else:
                filtered_symbols = symbols
            
            selected_symbol = st.selectbox(
                "Select Symbol",
                filtered_symbols,
                index=0 if filtered_symbols else None
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🚀 Analyze Token", use_container_width=True, type="primary"):
                st.session_state.analyze_token = True
                st.session_state.selected_symbol = selected_symbol
        
        # Token analysis section
        if st.session_state.get('analyze_token') and st.session_state.get('selected_symbol'):
            self.display_token_analysis(
                st.session_state.selected_exchange,
                st.session_state.selected_symbol
            )
        
        # Quick market overview
        st.markdown("---")
        self.display_market_overview()
    
    def display_token_analysis(self, exchange, symbol):
        """Display comprehensive token analysis"""
        st.markdown("---")
        st.subheader(f"📊 Detailed Analysis: {symbol} on {exchange}")
        
        # Get token data
        token_data = self.market_data.get_token_data(exchange, symbol)
        
        if not token_data:
            st.error(f"Could not fetch data for {symbol} on {exchange}")
            return
        
        # Layout for token analysis
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price", 
                f"₹{token_data['price']:,.2f}" if 'INR' in symbol.upper() else f"${token_data['price']:,.2f}",
                f"{token_data['change_24h']:.2f}%"
            )
        
        with col2:
            st.metric("24h Volume", f"₹{token_data['volume']:,.0f}")
        
        with col3:
            st.metric("24h High", f"₹{token_data['high_24h']:,.2f}")
        
        with col4:
            st.metric("24h Low", f"₹{token_data['low_24h']:,.2f}")
        
        # Technical Analysis and Fundamental Info tabs
        tab1, tab2, tab3 = st.tabs(["📈 Technical Analysis", "📊 Fundamental Info", "🎯 Trading Signals"])
        
        with tab1:
            self.display_technical_analysis(symbol, exchange)
        
        with tab2:
            self.display_fundamental_info(symbol)
        
        with tab3:
            self.display_trading_signals(symbol, exchange)
    
    def display_technical_analysis(self, symbol, exchange):
        """Display technical indicators"""
        indicators = self.market_data.get_technical_indicators(symbol, exchange)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            rsi_color = "red" if indicators['RSI'] > 70 else "green" if indicators['RSI'] < 30 else "orange"
            st.metric("RSI (14)", f"{indicators['RSI']:.2f}", delta_color="off")
            st.progress(indicators['RSI'] / 100)
            st.caption(f"Oversold < 30, Overbought > 70")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            macd_signal = "Bullish" if indicators['MACD'] > indicators['MACD_Signal'] else "Bearish"
            st.metric("MACD", f"{indicators['MACD']:.4f}")
            st.metric("MACD Signal", f"{indicators['MACD_Signal']:.4f}")
            st.caption(f"Signal: {macd_signal}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            bb_position = ((indicators['Current_Price'] - indicators['BB_Lower']) / 
                         (indicators['BB_Upper'] - indicators['BB_Lower'])) * 100
            st.metric("Bollinger Position", f"{bb_position:.1f}%")
            st.progress(bb_position / 100)
            st.caption("Lower Band: Buy, Upper Band: Sell")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            st.metric("Stoch K", f"{indicators['Stoch_K']:.1f}")
            st.metric("Stoch D", f"{indicators['Stoch_D']:.1f}")
            stoch_signal = "Oversold" if indicators['Stoch_K'] < 20 else "Overbought" if indicators['Stoch_K'] > 80 else "Neutral"
            st.caption(f"Signal: {stoch_signal}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Moving Averages")
            ma_data = {
                'Period': ['SMA 20', 'SMA 50'],
                'Value': [indicators['SMA_20'], indicators['SMA_50']],
                'Signal': ['Short-term', 'Long-term']
            }
            st.dataframe(pd.DataFrame(ma_data), use_container_width=True)
            
            # MA Crossover analysis
            if indicators['SMA_20'] > indicators['SMA_50']:
                st.success("✅ Golden Cross: SMA 20 > SMA 50 (Bullish)")
            else:
                st.warning("❌ Death Cross: SMA 20 < SMA 50 (Bearish)")
        
        with col2:
            st.subheader("Price Levels")
            price_levels = {
                'Level': ['Current', 'Support (BB Lower)', 'Resistance (BB Upper)'],
                'Value': [
                    indicators['Current_Price'],
                    indicators['BB_Lower'],
                    indicators['BB_Upper']
                ]
            }
            st.dataframe(pd.DataFrame(price_levels), use_container_width=True)
            
            # Price position analysis
            bb_middle = (indicators['BB_Upper'] + indicators['BB_Lower']) / 2
            if indicators['Current_Price'] > bb_middle:
                st.info("📈 Price above middle band")
            else:
                st.info("📉 Price below middle band")
    
    def display_fundamental_info(self, symbol):
        """Display REAL fundamental information about the token"""
        st.subheader("💰 Real Fundamental Data (CoinMarketCap)")
        
        with st.spinner("Fetching real-time data from CoinMarketCap..."):
            fundamental_data = self.market_data.get_real_fundamental_data(symbol)
        
        st.markdown('<div class="token-card">', unsafe_allow_html=True)
        
        # Price and Market Data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change_1h = fundamental_data['percent_change_1h']
            price_change_24h = fundamental_data['percent_change_24h']
            price_change_7d = fundamental_data['percent_change_7d']
            
            st.metric(
                "USD Price", 
                f"${fundamental_data['price_usd']:,.2f}",
                f"{price_change_24h:.2f}%"
            )
            st.metric("INR Price", f"₹{fundamental_data['price_inr']:,.2f}")
        
        with col2:
            st.metric("Market Cap", fundamental_data['market_cap_display'])
            st.metric("Market Cap Rank", f"#{fundamental_data['market_cap_rank']}")
            st.metric("Fully Diluted MCap", fundamental_data['fully_diluted_market_cap_display'])
        
        with col3:
            st.metric("24h Volume", fundamental_data['volume_24h_display'])
            st.metric("Volume/MCap Ratio", f"{fundamental_data['volume_mcap_ratio']:.2f}%")
            
            # Sentiment indicator
            sentiment_color = {
                'Very Bullish': 'green',
                'Bullish': 'lightgreen',
                'Neutral': 'orange',
                'Bearish': 'lightcoral',
                'Very Bearish': 'red'
            }.get(fundamental_data['sentiment'], 'gray')
            
            st.metric("Market Sentiment", fundamental_data['sentiment'])
        
        with col4:
            st.metric("Circulating Supply", fundamental_data['circulating_supply_display'])
            st.metric("Total Supply", fundamental_data['total_supply_display'])
            st.metric("Max Supply", fundamental_data['max_supply_display'])
        
        # Price performance chart
        st.subheader("📈 Price Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal"
            if price_change_1h > 0:
                delta_color = "normal"
            st.metric("1h Change", f"{price_change_1h:.2f}%", delta_color=delta_color)
        
        with col2:
            delta_color = "normal"
            if price_change_24h > 0:
                delta_color = "normal"
            st.metric("24h Change", f"{price_change_24h:.2f}%", delta_color=delta_color)
        
        with col3:
            delta_color = "normal"
            if price_change_7d > 0:
                delta_color = "normal"
            st.metric("7d Change", f"{price_change_7d:.2f}%", delta_color=delta_color)
        
        with col4:
            price_change_30d = fundamental_data['percent_change_30d']
            delta_color = "normal"
            if price_change_30d > 0:
                delta_color = "normal"
            st.metric("30d Change", f"{price_change_30d:.2f}%", delta_color=delta_color)
        
        # Risk Assessment
        st.subheader("⚠️ Risk Assessment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = {
                'Low': 'green',
                'Medium': 'orange',
                'High': 'red'
            }.get(fundamental_data['risk_level'], 'gray')
            
            st.metric("Risk Level", fundamental_data['risk_level'])
            
            # Risk factors
            if fundamental_data['risk_level'] == 'High':
                st.error("High volatility and risk. Trade with caution.")
            elif fundamental_data['risk_level'] == 'Medium':
                st.warning("Moderate risk. Use proper risk management.")
            else:
                st.success("Lower risk profile. Still monitor positions.")
        
        with col2:
            # Liquidity score based on volume/mcap ratio
            volume_ratio = fundamental_data['volume_mcap_ratio']
            if volume_ratio > 10:
                liquidity_score = "High"
                st.success("💰 High Liquidity")
            elif volume_ratio > 5:
                liquidity_score = "Medium"
                st.warning("💰 Medium Liquidity")
            else:
                liquidity_score = "Low"
                st.error("💰 Low Liquidity")
            
            st.metric("Liquidity Score", liquidity_score)
        
        with col3:
            # Market cap category
            market_cap = fundamental_data['market_cap']
            if market_cap > 10000000000:  # $10B+
                cap_category = "Large Cap"
                st.success("🏢 Large Cap")
            elif market_cap > 1000000000:  # $1B-$10B
                cap_category = "Mid Cap"
                st.warning("🏢 Mid Cap")
            else:
                cap_category = "Small Cap"
                st.error("🏢 Small Cap")
            
            st.metric("Market Cap Category", cap_category)
        
        # Project Information
        st.subheader("🔗 Project Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {fundamental_data['name']}")
            st.write(f"**Symbol:** {fundamental_data['symbol']}")
            st.write(f"**Category:** {fundamental_data['category']}")
            
            if fundamental_data['website']:
                st.write(f"**Website:** [{fundamental_data['website']}]({fundamental_data['website']})")
            if fundamental_data['explorer']:
                st.write(f"**Blockchain Explorer:** [{fundamental_data['explorer']}]({fundamental_data['explorer']})")
        
        with col2:
            if fundamental_data['tags']:
                st.write("**Tags:**")
                for tag in fundamental_data['tags'][:5]:  # Show first 5 tags
                    st.write(f"• {tag}")
        
        # Project Description
        st.subheader("📝 Project Description")
        st.info(fundamental_data['description'][:500] + "..." if len(fundamental_data['description']) > 500 else fundamental_data['description'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_trading_signals(self, symbol, exchange):
        """Display trading signals based on technical analysis"""
        indicators = self.market_data.get_technical_indicators(symbol, exchange)
        signal, confidence = self.technical_analysis.get_signal_from_indicators(indicators)
        
        st.subheader("🎯 Trading Signals")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if signal == "BUY":
                st.success(f"## 🟢 BUY SIGNAL")
                st.metric("Confidence Level", f"{confidence*100:.1f}%")
            elif signal == "SELL":
                st.error(f"## 🔴 SELL SIGNAL")
                st.metric("Confidence Level", f"{confidence*100:.1f}%")
            else:
                st.warning(f"## 🟡 NEUTRAL SIGNAL")
                st.metric("Confidence Level", f"{confidence*100:.1f}%")
        
        # Signal breakdown
        st.subheader("Signal Components")
        
        components = [
            ("RSI", "Oversold/Bought", indicators['RSI'] < 30 or indicators['RSI'] > 70),
            ("MACD", "Crossover", indicators['MACD'] > indicators['MACD_Signal']),
            ("Bollinger Bands", "Price Position", 
             indicators['Current_Price'] <= indicators['BB_Lower'] or indicators['Current_Price'] >= indicators['BB_Upper']),
            ("Moving Averages", "Crossover", indicators['SMA_20'] > indicators['SMA_50']),
            ("Stochastic", "Oversold/Bought", indicators['Stoch_K'] < 20 or indicators['Stoch_K'] > 80)
        ]
        
        for component, description, condition in components:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write(f"**{component}**")
            with col2:
                st.write(description)
            with col3:
                if condition:
                    st.success("✅ Active")
                else:
                    st.info("⚪ Inactive")
        
        # Trading recommendations
        st.subheader("💡 Trading Recommendations")
        
        if signal == "BUY":
            st.success("""
            **Recommended Actions:**
            - Consider entering long positions
            - Set stop-loss below recent support
            - Target resistance levels for profit-taking
            - Monitor for trend confirmation
            """)
        elif signal == "SELL":
            st.error("""
            **Recommended Actions:**
            - Consider exiting long positions
            - Potential short entry opportunities
            - Set stop-loss above recent resistance
            - Monitor for trend reversal signs
            """)
        else:
            st.warning("""
            **Recommended Actions:**
            - Wait for clearer market direction
            - Consider range-bound strategies
            - Monitor key support/resistance levels
            - Look for breakout/breakdown signals
            """)
    
    def display_market_overview(self):
        """Display market overview at the bottom"""
        st.subheader("📊 Market Overview")
        
        # Simulated market data
        markets = [
            {'symbol': 'BTC-INR', 'price': 3500000, 'change': 2.5, 'volume': 25000000},
            {'symbol': 'ETH-INR', 'price': 200000, 'change': 1.8, 'volume': 15000000},
            {'symbol': 'USDT-INR', 'price': 83.2, 'change': 0.1, 'volume': 50000000},
            {'symbol': 'MATIC-INR', 'price': 60.5, 'change': -0.5, 'volume': 5000000},
            {'symbol': 'ADA-INR', 'price': 40.2, 'change': 3.2, 'volume': 3000000},
            {'symbol': 'DOT-INR', 'price': 502.3, 'change': -1.2, 'volume': 2000000},
        ]
        
        for market in markets:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"**{market['symbol']}**")
            with col2:
                st.write(f"₹{market['price']:,.2f}")
            with col3:
                change_color = "green" if market['change'] > 0 else "red"
                st.write(f":{change_color}[{market['change']}%]")
            with col4:
                st.write(f"₹{market['volume']:,.0f}")

# Simplified version of other classes to avoid errors
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

# Simplified TradingStrategies class
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

# Simplified SemiAutomatedBots class
class SemiAutomatedBots:
    """Semi-automated bots that generate trading signals"""
    
    def __init__(self):
        self.strategies = TradingStrategies()
        self.exchange_data = IndianExchangeData()
    
    def create_signal_dashboard(self):
        """Create dashboard for signal generation bots"""
        st.subheader("🤖 Semi-Automated Signal Bots")
        st.info("Signal bot functionality - Simplified version")

# Simplified AutomatedTradingBot class
class AutomatedTradingBot:
    """Fully automated trading bot with paper and live trading"""
    
    def __init__(self):
        self.exchange_data = IndianExchangeData()
        self.paper_balance = 100000
    
    def create_automated_trading_dashboard(self):
        """Create dashboard for automated trading bots"""
        st.subheader("⚡ Fully Automated Trading Bots")
        st.info("Automated trading functionality - Simplified version")

# Simplified IndianTraderTools class
class IndianTraderTools:
    def __init__(self):
        self.exchange_data = IndianExchangeData()
    
    def create_tax_calculator(self):
        """Create advanced Indian crypto tax calculator"""
        st.subheader("🇮🇳 Advanced Tax Calculator")
        st.info("Tax calculator functionality - Simplified version")

# Main Platform Class
class AethosIndiaPlatform:
    def __init__(self):
        self.exchange_data = IndianExchangeData()
        self.semi_bots = SemiAutomatedBots()
        self.auto_bots = AutomatedTradingBot()
        self.indian_tools = IndianTraderTools()
        self.markets_page = MarketsPage()
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
        if 'selected_exchange' not in st.session_state:
            st.session_state.selected_exchange = 'Delta'
        if 'analyze_token' not in st.session_state:
            st.session_state.analyze_token = False
    
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
    
    def create_dashboard(self):
        """Create the main dashboard page"""
        st.header("🏠 Platform Dashboard")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Balance", "₹1,25,000", "₹5,000")
        with col2:
            st.metric("Active Strategies", "3", "1")
        with col3:
            st.metric("Today's P&L", "₹2,500", "₹500")
        with col4:
            st.metric("Success Rate", "72%", "3%")
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Portfolio Performance")
            # Simple portfolio chart
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
            portfolio_value = 100000 + np.cumsum(np.random.randn(len(dates)) * 1000)
            
            fig = px.line(x=dates, y=portfolio_value, title="Portfolio Value Over Time")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔔 Recent Alerts")
            alerts = [
                {"time": "10:30 AM", "message": "BTC strong buy signal detected", "type": "success"},
                {"time": "09:15 AM", "message": "ETH approaching resistance level", "type": "warning"},
                {"time": "Yesterday", "message": "New tax regulations announced", "type": "info"},
            ]
            
            for alert in alerts:
                if alert["type"] == "success":
                    st.success(f"**{alert['time']}**: {alert['message']}")
                elif alert["type"] == "warning":
                    st.warning(f"**{alert['time']}**: {alert['message']}")
                else:
                    st.info(f"**{alert['time']}**: {alert['message']}")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🤖 Deploy New Bot", use_container_width=True):
                st.info("Navigate to Algo Trading tab")
        with col2:
            if st.button("📊 Analyze Market", use_container_width=True):
                st.info("Navigate to Markets tab")
        with col3:
            if st.button("💰 Calculate Tax", use_container_width=True):
                st.info("Tax calculator opened")
        with col4:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()

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
            st.subheader("📊 Live Bot Monitoring & Analytics")
            st.info("Bot monitoring functionality - Simplified version")

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
            st.subheader("📊 Indian Market Insights")
            st.info("Market insights functionality - Simplified version")
        
        with tab3:
            st.subheader("📜 Regulatory Compliance Center")
            st.info("Regulatory information - Simplified version")

    def create_markets_page(self):
        """Create the dedicated markets page"""
        self.markets_page.create_markets_page()

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
            self.create_dashboard()
        
        with tab2:
            self.create_algorithmic_trading_section()
        
        with tab3:
            self.create_indian_trader_tools()
        
        with tab4:
            self.create_markets_page()

# Run the application
if __name__ == "__main__":
    # Display TA-Lib status
    if not TA_LIB_AVAILABLE:
        st.sidebar.warning("""
        ⚠ TA-Lib not installed. 
        For optimal performance, install TA-Lib:
        ```
        # On Ubuntu/Debian
        sudo apt-get install ta-lib
        
        # Then install Python package
        pip install TA-Lib
        ```
        """)
    
    # Check for CMC API key
    if not st.secrets.get("COIN_MARKET_CAP_API_KEY"):
        st.sidebar.error("""
        ❌ CoinMarketCap API key not found.
        Add COIN_MARKET_CAP_API_KEY to your Streamlit secrets for real fundamental data.
        """)
    
    platform = AethosIndiaPlatform()
    platform.run()
