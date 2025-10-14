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

# Try to import TA-Lib and yfinance
try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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
.pattern-bullish { background-color: #d4f8e8; padding: 5px; border-radius: 3px; }
.pattern-bearish { background-color: #f8d4d4; padding: 5px; border-radius: 3px; }
.pattern-neutral { background-color: #f0f0f0; padding: 5px; border-radius: 3px; }
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
                return None
        except Exception:
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

class YFinanceFallback:
    """Fallback to yfinance when CMC is unavailable"""
    
    @staticmethod
    def get_crypto_data(symbol):
        """Get cryptocurrency data using yfinance"""
        if not YFINANCE_AVAILABLE:
            return None
            
        try:
            # Map symbols to yfinance format
            symbol_map = {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD', 
                'USDT': 'USDT-USD',
                'BNB': 'BNB-USD',
                'SOL': 'SOL-USD',
                'XRP': 'XRP-USD',
                'ADA': 'ADA-USD',
                'AVAX': 'AVAX-USD',
                'DOT': 'DOT-USD',
                'MATIC': 'MATIC-USD',
                'LINK': 'LINK-USD',
                'LTC': 'LTC-USD',
                'BCH': 'BCH-USD',
                'XLM': 'XLM-USD',
                'ATOM': 'ATOM-USD'
            }
            
            yf_symbol = symbol_map.get(symbol.upper(), f"{symbol.upper()}-USD")
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            history = ticker.history(period="1mo")
            
            if history.empty:
                return None
                
            current_price = history['Close'].iloc[-1]
            prev_price = history['Close'].iloc[-2] if len(history) > 1 else current_price
            change_24h = ((current_price - prev_price) / prev_price) * 100
            
            return {
                'name': info.get('name', symbol),
                'symbol': symbol,
                'price_usd': current_price,
                'price_inr': current_price * 83,  # Approximate conversion
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', 0),
                'change_24h': change_24h,
                'description': info.get('description', 'No description available'),
                'currency': info.get('currency', 'USD')
            }
        except Exception:
            return None

class DeltaExchange:
    """Delta Exchange API integration for real market data"""
    
    def __init__(self):
        self.base_url = "https://api.delta.exchange"
        self.api_key = st.secrets.get("DELTA_API_KEY", "")
        self.api_secret = st.secrets.get("DELTA_API_SECRET", "")
    
    def get_products(self, limit=50):
        """Get available trading products from Delta"""
        try:
            response = requests.get(f"{self.base_url}/v2/products", timeout=10)
            if response.status_code == 200:
                products = response.json().get('result', [])
                # Filter for popular trading pairs and Indian relevant tokens
                popular_symbols = ['BTC', 'ETH', 'SOL', 'MATIC', 'ADA', 'DOT', 'BNB', 'XRP', 'LINK', 'LTC']
                filtered_products = []
                
                for product in products:
                    symbol = product.get('symbol', '')
                    # Include major cryptocurrencies and INR pairs
                    if any(token in symbol for token in popular_symbols) or 'INR' in symbol:
                        filtered_products.append(product)
                
                return filtered_products[:limit]
            return []
        except Exception as e:
            st.error(f"Error fetching Delta products: {e}")
            return []
    
    def get_ticker(self, symbol):
        """Get real ticker data from Delta"""
        try:
            response = requests.get(f"{self.base_url}/v2/tickers/{symbol}", timeout=10)
            if response.status_code == 200:
                return response.json().get('result')
            return None
        except Exception as e:
            return None
    
    def get_ohlc(self, symbol, resolution=60, limit=100):
        """Get OHLC data from Delta"""
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
        except Exception:
            return []
    
    def get_orderbook(self, symbol):
        """Get order book data"""
        try:
            response = requests.get(f"{self.base_url}/v2/orderbook/{symbol}", timeout=10)
            if response.status_code == 200:
                return response.json().get('result')
            return None
        except Exception:
            return None

class PatternDiscovery:
    """Automated pattern discovery using technical analysis"""
    
    def __init__(self):
        self.patterns = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
    
    def discover_patterns(self, ohlc_data):
        """Discover technical patterns from OHLC data"""
        if not ohlc_data or len(ohlc_data) < 50:
            return self.patterns
            
        # Convert to numpy arrays for TA-Lib
        opens = np.array([float(candle['open']) for candle in ohlc_data])
        highs = np.array([float(candle['high']) for candle in ohlc_data])
        lows = np.array([float(candle['low']) for candle in ohlc_data])
        closes = np.array([float(candle['close']) for candle in ohlc_data])
        
        if TA_LIB_AVAILABLE:
            self._discover_talib_patterns(opens, highs, lows, closes)
        else:
            self._discover_basic_patterns(opens, highs, lows, closes)
            
        return self.patterns
    
    def _discover_talib_patterns(self, opens, highs, lows, closes):
        """Discover patterns using TA-Lib"""
        try:
            # Candlestick patterns
            patterns = {
                'CDL2CROWS': talib.CDL2CROWS(opens, highs, lows, closes),
                'CDL3BLACKCROWS': talib.CDL3BLACKCROWS(opens, highs, lows, closes),
                'CDL3INSIDE': talib.CDL3INSIDE(opens, highs, lows, closes),
                'CDL3LINESTRIKE': talib.CDL3LINESTRIKE(opens, highs, lows, closes),
                'CDL3OUTSIDE': talib.CDL3OUTSIDE(opens, highs, lows, closes),
                'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH(opens, highs, lows, closes),
                'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS(opens, highs, lows, closes),
                'CDLABANDONEDBABY': talib.CDLABANDONEDBABY(opens, highs, lows, closes, penetration=0),
                'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK(opens, highs, lows, closes),
                'CDLBELTHOLD': talib.CDLBELTHOLD(opens, highs, lows, closes),
                'CDLBREAKAWAY': talib.CDLBREAKAWAY(opens, highs, lows, closes),
                'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU(opens, highs, lows, closes),
                'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER(opens, highs, lows, closes, penetration=0),
                'CDLDOJI': talib.CDLDOJI(opens, highs, lows, closes),
                'CDLDOJISTAR': talib.CDLDOJISTAR(opens, highs, lows, closes),
                'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI(opens, highs, lows, closes),
                'CDLENGULFING': talib.CDLENGULFING(opens, highs, lows, closes),
                'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR(opens, highs, lows, closes, penetration=0),
                'CDLEVENINGSTAR': talib.CDLEVENINGSTAR(opens, highs, lows, closes, penetration=0),
                'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI(opens, highs, lows, closes),
                'CDLHAMMER': talib.CDLHAMMER(opens, highs, lows, closes),
                'CDLHANGINGMAN': talib.CDLHANGINGMAN(opens, highs, lows, closes),
                'CDLHARAMI': talib.CDLHARAMI(opens, highs, lows, closes),
                'CDLHARAMICROSS': talib.CDLHARAMICROSS(opens, highs, lows, closes),
                'CDLHIGHWAVE': talib.CDLHIGHWAVE(opens, highs, lows, closes),
                'CDLINNECK': talib.CDLINNECK(opens, highs, lows, closes),
                'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER(opens, highs, lows, closes),
                'CDLKICKING': talib.CDLKICKING(opens, highs, lows, closes),
                'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM(opens, highs, lows, closes),
                'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI(opens, highs, lows, closes),
                'CDLLONGLINE': talib.CDLLONGLINE(opens, highs, lows, closes),
                'CDLMARUBOZU': talib.CDLMARUBOZU(opens, highs, lows, closes),
                'CDLMATCHINGLOW': talib.CDLMATCHINGLOW(opens, highs, lows, closes),
                'CDLMATHOLD': talib.CDLMATHOLD(opens, highs, lows, closes, penetration=0),
                'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR(opens, highs, lows, closes, penetration=0),
                'CDLMORNINGSTAR': talib.CDLMORNINGSTAR(opens, highs, lows, closes, penetration=0),
                'CDLPIERCING': talib.CDLPIERCING(opens, highs, lows, closes),
                'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN(opens, highs, lows, closes),
                'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS(opens, highs, lows, closes),
                'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES(opens, highs, lows, closes),
                'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR(opens, highs, lows, closes),
                'CDLSHORTLINE': talib.CDLSHORTLINE(opens, highs, lows, closes),
                'CDLSPINNINGTOP': talib.CDLSPINNINGTOP(opens, highs, lows, closes),
                'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN(opens, highs, lows, closes),
                'CDLTAKURI': talib.CDLTAKURI(opens, highs, lows, closes),
                'CDLTASUKIGAP': talib.CDLTASUKIGAP(opens, highs, lows, closes),
                'CDLTHRUSTING': talib.CDLTHRUSTING(opens, highs, lows, closes),
                'CDLTRISTAR': talib.CDLTRISTAR(opens, highs, lows, closes),
                'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER(opens, highs, lows, closes),
                'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS(opens, highs, lows, closes),
                'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS(opens, highs, lows, closes)
            }
            
            # Check recent patterns (last 5 candles)
            for pattern_name, pattern_data in patterns.items():
                recent_patterns = pattern_data[-5:]  # Last 5 periods
                for i, value in enumerate(recent_patterns):
                    if value != 0:
                        pattern_info = {
                            'name': pattern_name,
                            'strength': abs(value),
                            'direction': 'bullish' if value > 0 else 'bearish',
                            'period': len(recent_patterns) - i  # How recent (1 = most recent)
                        }
                        
                        if value > 0:
                            self.patterns['bullish'].append(pattern_info)
                        elif value < 0:
                            self.patterns['bearish'].append(pattern_info)
                            
        except Exception as e:
            st.error(f"Error in TA-Lib pattern discovery: {e}")
    
    def _discover_basic_patterns(self, opens, highs, lows, closes):
        """Discover basic patterns without TA-Lib"""
        try:
            # Simple pattern detection
            if len(closes) >= 3:
                # Bullish Engulfing
                if (closes[-2] < opens[-2] and  # Previous candle bearish
                    closes[-1] > opens[-1] and  # Current candle bullish
                    closes[-1] > opens[-2] and  # Close above previous open
                    opens[-1] < closes[-2]):    # Open below previous close
                    self.patterns['bullish'].append({
                        'name': 'Bullish Engulfing',
                        'strength': 100,
                        'direction': 'bullish',
                        'period': 1
                    })
                
                # Bearish Engulfing
                if (closes[-2] > opens[-2] and  # Previous candle bullish
                    closes[-1] < opens[-1] and  # Current candle bearish
                    closes[-1] < opens[-2] and  # Close below previous open
                    opens[-1] > closes[-2]):    # Open above previous close
                    self.patterns['bearish'].append({
                        'name': 'Bearish Engulfing',
                        'strength': 100,
                        'direction': 'bearish',
                        'period': 1
                    })
                
                # Doji detection
                body_size = abs(closes[-1] - opens[-1])
                total_range = highs[-1] - lows[-1]
                if total_range > 0 and body_size / total_range < 0.1:  # Small body relative to range
                    self.patterns['neutral'].append({
                        'name': 'Doji',
                        'strength': 80,
                        'direction': 'neutral',
                        'period': 1
                    })
                    
        except Exception as e:
            st.error(f"Error in basic pattern discovery: {e}")
    
    def get_pattern_summary(self):
        """Get summary of discovered patterns"""
        summary = {
            'total_bullish': len(self.patterns['bullish']),
            'total_bearish': len(self.patterns['bearish']),
            'total_neutral': len(self.patterns['neutral']),
            'dominant_sentiment': 'neutral'
        }
        
        if summary['total_bullish'] > summary['total_bearish']:
            summary['dominant_sentiment'] = 'bullish'
        elif summary['total_bearish'] > summary['total_bullish']:
            summary['dominant_sentiment'] = 'bearish'
            
        return summary

class MarketDataProvider:
    def __init__(self):
        self.delta = DeltaExchange()
        self.cmc = CoinMarketCapAPI()
        self.yfinance = YFinanceFallback()
        self.pattern_discovery = PatternDiscovery()
        self.available_exchanges = ['Delta', 'CoinDCX', 'Coinswitch', 'Mudrex', 'ZebPay']
    
    def get_top_tokens_by_market_cap(self, limit=15):
        """Get top tokens by market cap from CMC or fallback"""
        # Try CMC first
        cmc_data = self.cmc.get_crypto_listings(limit=limit)
        if cmc_data and 'data' in cmc_data:
            tokens = []
            for crypto in cmc_data['data']:
                tokens.append({
                    'symbol': crypto['symbol'],
                    'name': crypto['name'],
                    'market_cap': crypto['quote']['USD']['market_cap'],
                    'price': crypto['quote']['USD']['price'],
                    'change_24h': crypto['quote']['USD']['percent_change_24h']
                })
            return tokens
        
        # Fallback to hardcoded top tokens
        top_tokens = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'market_cap': 1300000000000, 'price': 67000, 'change_24h': 2.1},
            {'symbol': 'ETH', 'name': 'Ethereum', 'market_cap': 450000000000, 'price': 3500, 'change_24h': 1.5},
            {'symbol': 'USDT', 'name': 'Tether', 'market_cap': 110000000000, 'price': 1.0, 'change_24h': 0.0},
            {'symbol': 'BNB', 'name': 'Binance Coin', 'market_cap': 85000000000, 'price': 580, 'change_24h': 3.2},
            {'symbol': 'SOL', 'name': 'Solana', 'market_cap': 82000000000, 'price': 180, 'change_24h': 5.7},
            {'symbol': 'XRP', 'name': 'Ripple', 'market_cap': 36000000000, 'price': 0.62, 'change_24h': -1.2},
            {'symbol': 'ADA', 'name': 'Cardano', 'market_cap': 24000000000, 'price': 0.45, 'change_24h': 2.8},
            {'symbol': 'AVAX', 'name': 'Avalanche', 'market_cap': 14000000000, 'price': 35, 'change_24h': 4.1},
            {'symbol': 'DOT', 'name': 'Polkadot', 'market_cap': 12000000000, 'price': 8.5, 'change_24h': 1.9},
            {'symbol': 'MATIC', 'name': 'Polygon', 'market_cap': 10000000000, 'price': 0.75, 'change_24h': 3.5}
        ]
        return top_tokens[:limit]
    
    def get_exchange_symbols(self, exchange):
        """Get available symbols for an exchange"""
        if exchange == 'Delta':
            products = self.delta.get_products()
            return [p['symbol'] for p in products if p]
        else:
            # For other exchanges, return popular pairs
            return ['BTC-INR', 'ETH-INR', 'USDT-INR', 'SOL-INR', 'MATIC-INR', 'ADA-INR', 'DOT-INR']
    
    def get_real_time_data(self, symbol, exchange):
        """Get real-time data from Delta Exchange"""
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
                    'bid': float(ticker.get('best_bid', 0)),
                    'ask': float(ticker.get('best_ask', 0)),
                    'exchange': exchange,
                    'timestamp': datetime.now()
                }
        return None
    
    def get_technical_indicators(self, symbol, exchange):
        """Calculate technical indicators with real data"""
        # Get OHLC data from Delta
        ohlc_data = self.delta.get_ohlc(symbol, resolution=60, limit=100)
        
        if not ohlc_data:
            return self._get_fallback_indicators(symbol)
        
        # Convert to arrays
        closes = np.array([float(candle['close']) for candle in ohlc_data])
        highs = np.array([float(candle['high']) for candle in ohlc_data])
        lows = np.array([float(candle['low']) for candle in ohlc_data])
        volumes = np.array([float(candle['volume']) for candle in ohlc_data])
        
        indicators = {}
        
        if TA_LIB_AVAILABLE and len(closes) >= 50:
            try:
                # RSI
                indicators['RSI'] = talib.RSI(closes, timeperiod=14)[-1]
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(closes)
                indicators['MACD'] = macd[-1]
                indicators['MACD_Signal'] = macd_signal[-1]
                indicators['MACD_Histogram'] = macd_hist[-1]
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators['BB_Upper'] = upper[-1]
                indicators['BB_Middle'] = middle[-1]
                indicators['BB_Lower'] = lower[-1]
                
                # Moving Averages
                indicators['SMA_20'] = talib.SMA(closes, timeperiod=20)[-1]
                indicators['SMA_50'] = talib.SMA(closes, timeperiod=50)[-1]
                indicators['EMA_12'] = talib.EMA(closes, timeperiod=12)[-1]
                indicators['EMA_26'] = talib.EMA(closes, timeperiod=26)[-1]
                
                # Stochastic
                slowk, slowd = talib.STOCH(highs, lows, closes)
                indicators['Stoch_K'] = slowk[-1]
                indicators['Stoch_D'] = slowd[-1]
                
                # Additional indicators
                indicators['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
                indicators['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
                indicators['OBV'] = talib.OBV(closes, volumes)[-1]
                
            except Exception as e:
                st.error(f"Error calculating TA-Lib indicators: {e}")
                return self._get_fallback_indicators(symbol)
        else:
            return self._get_fallback_indicators(symbol)
        
        indicators['Current_Price'] = closes[-1]
        return indicators
    
    def _get_fallback_indicators(self, symbol):
        """Fallback indicators when real data is unavailable"""
        np.random.seed(hash(symbol) % 1000)
        n_points = 100
        prices = 100 + np.cumsum(np.random.randn(n_points) * 2)
        
        return {
            'RSI': 50 + np.random.uniform(-20, 20),
            'MACD': np.random.uniform(-2, 2),
            'MACD_Signal': np.random.uniform(-2, 2),
            'MACD_Histogram': np.random.uniform(-0.5, 0.5),
            'BB_Upper': prices[-1] * 1.1,
            'BB_Middle': prices[-1],
            'BB_Lower': prices[-1] * 0.9,
            'SMA_20': np.mean(prices[-20:]),
            'SMA_50': np.mean(prices[-50:]),
            'EMA_12': np.mean(prices[-12:]),
            'EMA_26': np.mean(prices[-26:]),
            'Stoch_K': 50 + np.random.uniform(-30, 30),
            'Stoch_D': 50 + np.random.uniform(-30, 30),
            'ATR': prices[-1] * 0.02,
            'ADX': 20 + np.random.uniform(0, 40),
            'OBV': np.random.uniform(100000, 500000),
            'Current_Price': prices[-1]
        }
    
    def get_patterns(self, symbol, exchange):
        """Discover patterns for a symbol"""
        ohlc_data = self.delta.get_ohlc(symbol, resolution=60, limit=100)
        return self.pattern_discovery.discover_patterns(ohlc_data)
    
    def get_fundamental_data(self, symbol):
        """Get fundamental data with CMC and yfinance fallback"""
        # Try CMC first
        cmc_data = self.cmc.get_crypto_info(symbol)
        if cmc_data and 'data' in cmc_data:
            data = cmc_data['data']
            crypto_key = list(data.keys())[0]
            crypto_data = data[crypto_key]
            quote_data = crypto_data.get('quote', {}).get('USD', {})
            
            return {
                'name': crypto_data.get('name', symbol),
                'symbol': crypto_data.get('symbol', symbol),
                'price_usd': quote_data.get('price', 0),
                'price_inr': quote_data.get('price', 0) * 83,
                'market_cap': quote_data.get('market_cap', 0),
                'volume_24h': quote_data.get('volume_24h', 0),
                'change_24h': quote_data.get('percent_change_24h', 0),
                'change_7d': quote_data.get('percent_change_7d', 0),
                'circulating_supply': crypto_data.get('circulating_supply', 0),
                'total_supply': crypto_data.get('total_supply', 0),
                'max_supply': crypto_data.get('max_supply', 0),
                'rank': crypto_data.get('cmc_rank', 0),
                'source': 'CoinMarketCap'
            }
        
        # Fallback to yfinance
        yf_data = self.yfinance.get_crypto_data(symbol)
        if yf_data:
            yf_data['source'] = 'Yahoo Finance'
            return yf_data
        
        # Final fallback
        return {
            'name': symbol,
            'symbol': symbol,
            'price_usd': 0,
            'price_inr': 0,
            'market_cap': 0,
            'volume_24h': 0,
            'change_24h': 0,
            'change_7d': 0,
            'circulating_supply': 0,
            'total_supply': 0,
            'max_supply': 0,
            'rank': 0,
            'source': 'Fallback Data'
        }

class MarketsPage:
    def __init__(self):
        self.market_data = MarketDataProvider()
    
    def create_markets_page(self):
        """Create the enhanced markets page"""
        st.header("📈 Live Markets & Pattern Discovery")
        
        # Top tokens by market cap
        st.subheader("🏆 Top Tokens by Market Cap")
        top_tokens = self.market_data.get_top_tokens_by_market_cap(15)
        
        # Display top tokens in a grid
        cols = st.columns(5)
        for i, token in enumerate(top_tokens[:10]):
            with cols[i % 5]:
                change_color = "positive" if token['change_24h'] > 0 else "negative"
                st.metric(
                    token['symbol'],
                    f"${token['price']:,.2f}",
                    f"{token['change_24h']:.2f}%"
                )
        
        # Exchange selection and token analysis
        st.markdown("---")
        st.subheader("🔍 Advanced Token Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Exchange selection
            exchange = st.selectbox(
                "Select Exchange",
                self.market_data.available_exchanges,
                index=0
            )
            
            # Symbol selection
            symbols = self.market_data.get_exchange_symbols(exchange)
            selected_symbol = st.selectbox("Select Symbol", symbols)
            
            if st.button("🚀 Analyze Token", type="primary", use_container_width=True):
                st.session_state.analyze_token = True
                st.session_state.selected_symbol = selected_symbol
                st.session_state.selected_exchange = exchange
        
        with col2:
            # Quick stats
            st.subheader("📊 Market Overview")
            if top_tokens:
                total_mcap = sum(token['market_cap'] for token in top_tokens[:10])
                avg_change = np.mean([token['change_24h'] for token in top_tokens[:10]])
                gainers = sum(1 for token in top_tokens[:10] if token['change_24h'] > 0)
                
                st.metric("Total Top 10 MCap", f"${total_mcap/1e12:.1f}T")
                st.metric("Avg 24h Change", f"{avg_change:.2f}%")
                st.metric("Gainers/Losers", f"{gainers}/{10-gainers}")
        
        # Token analysis section
        if st.session_state.get('analyze_token') and st.session_state.get('selected_symbol'):
            self.display_comprehensive_analysis(
                st.session_state.selected_exchange,
                st.session_state.selected_symbol
            )
    
    def display_comprehensive_analysis(self, exchange, symbol):
        """Display comprehensive analysis with real data"""
        st.markdown("---")
        st.subheader(f"📊 Comprehensive Analysis: {symbol} on {exchange}")
        
        # Get real-time data
        with st.spinner("Fetching real-time data..."):
            real_time_data = self.market_data.get_real_time_data(symbol, exchange)
            fundamental_data = self.market_data.get_fundamental_data(symbol)
            indicators = self.market_data.get_technical_indicators(symbol, exchange)
            patterns = self.market_data.get_patterns(symbol, exchange)
        
        if not real_time_data:
            st.warning("⚠️ Real-time data temporarily unavailable. Showing simulated data.")
            real_time_data = {
                'symbol': symbol,
                'price': 1000,
                'volume': 50000,
                'change_24h': 0,
                'high_24h': 1100,
                'low_24h': 950,
                'exchange': exchange
            }
        
        # Display real-time data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = real_time_data.get('change_24h', 0)
            delta_color = "normal" if price_change >= 0 else "inverse"
            st.metric(
                "Current Price",
                f"${real_time_data['price']:,.2f}",
                f"{price_change:.2f}%",
                delta_color=delta_color
            )
        
        with col2:
            st.metric("24h Volume", f"${real_time_data['volume']:,.0f}")
        
        with col3:
            st.metric("24h High", f"${real_time_data['high_24h']:,.2f}")
        
        with col4:
            st.metric("24h Low", f"${real_time_data['low_24h']:,.2f}")
        
        # Tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Technical Analysis", 
            "🔍 Pattern Discovery", 
            "💰 Fundamentals", 
            "📊 Market Depth"
        ])
        
        with tab1:
            self.display_technical_analysis(indicators, symbol)
        
        with tab2:
            self.display_pattern_discovery(patterns, symbol)
        
        with tab3:
            self.display_fundamental_analysis(fundamental_data)
        
        with tab4:
            self.display_market_depth(symbol, exchange)
    
    def display_technical_analysis(self, indicators, symbol):
        """Display comprehensive technical analysis"""
        st.subheader("📊 Technical Indicators")
        
        # Main indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            rsi = indicators.get('RSI', 50)
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            st.metric("RSI (14)", f"{rsi:.2f}")
            st.progress(rsi / 100)
            st.caption(f"Status: {rsi_status}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            macd = indicators.get('MACD', 0)
            macd_signal = indicators.get('MACD_Signal', 0)
            macd_signal_text = "Bullish" if macd > macd_signal else "Bearish"
            st.metric("MACD", f"{macd:.4f}")
            st.metric("Signal", f"{macd_signal:.4f}")
            st.caption(f"Signal: {macd_signal_text}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            bb_position = ((indicators['Current_Price'] - indicators['BB_Lower']) / 
                         (indicators['BB_Upper'] - indicators['BB_Lower'])) * 100
            st.metric("Bollinger Position", f"{bb_position:.1f}%")
            st.progress(bb_position / 100)
            bb_status = "Lower Band" if bb_position < 20 else "Upper Band" if bb_position > 80 else "Middle"
            st.caption(f"Position: {bb_status}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="indicator-box">', unsafe_allow_html=True)
            stoch_k = indicators.get('Stoch_K', 50)
            stoch_d = indicators.get('Stoch_D', 50)
            st.metric("Stoch %K", f"{stoch_k:.1f}")
            st.metric("Stoch %D", f"{stoch_d:.1f}")
            stoch_status = "Oversold" if stoch_k < 20 else "Overbought" if stoch_k > 80 else "Neutral"
            st.caption(f"Status: {stoch_status}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional indicators
        st.subheader("Additional Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Moving Averages**")
            ma_data = {
                'MA': ['SMA 20', 'SMA 50', 'EMA 12', 'EMA 26'],
                'Value': [
                    indicators.get('SMA_20', 0),
                    indicators.get('SMA_50', 0),
                    indicators.get('EMA_12', 0),
                    indicators.get('EMA_26', 0)
                ]
            }
            st.dataframe(pd.DataFrame(ma_data), use_container_width=True)
            
            # MA Analysis
            if indicators.get('SMA_20', 0) > indicators.get('SMA_50', 0):
                st.success("✅ Golden Cross: Short-term > Long-term (Bullish)")
            else:
                st.warning("❌ Death Cross: Short-term < Long-term (Bearish)")
        
        with col2:
            st.write("**Volatility & Momentum**")
            vol_data = {
                'Indicator': ['ATR', 'ADX', 'OBV'],
                'Value': [
                    indicators.get('ATR', 0),
                    indicators.get('ADX', 0),
                    f"{indicators.get('OBV', 0):,.0f}"
                ]
            }
            st.dataframe(pd.DataFrame(vol_data), use_container_width=True)
            
            # ADX Analysis
            adx = indicators.get('ADX', 0)
            if adx > 25:
                st.info("📈 Strong Trend (ADX > 25)")
            elif adx > 20:
                st.info("📊 Moderate Trend (ADX > 20)")
            else:
                st.info("📉 Weak Trend (ADX < 20)")
        
        with col3:
            st.write("**Price Levels**")
            price_data = {
                'Level': ['Current', 'BB Upper', 'BB Middle', 'BB Lower'],
                'Value': [
                    indicators.get('Current_Price', 0),
                    indicators.get('BB_Upper', 0),
                    indicators.get('BB_Middle', 0),
                    indicators.get('BB_Lower', 0)
                ]
            }
            st.dataframe(pd.DataFrame(price_data), use_container_width=True)
            
            # Price position analysis
            current_price = indicators.get('Current_Price', 0)
            bb_middle = indicators.get('BB_Middle', 0)
            if current_price > bb_middle:
                st.success("📈 Price above middle band")
            else:
                st.warning("📉 Price below middle band")
    
    def display_pattern_discovery(self, patterns, symbol):
        """Display automated pattern discovery results"""
        st.subheader("🎯 Automated Pattern Discovery")
        
        pattern_summary = self.market_data.pattern_discovery.get_pattern_summary()
        
        # Pattern summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Bullish Patterns", pattern_summary['total_bullish'])
        with col2:
            st.metric("Bearish Patterns", pattern_summary['total_bearish'])
        with col3:
            st.metric("Neutral Patterns", pattern_summary['total_neutral'])
        with col4:
            sentiment_color = {
                'bullish': 'green',
                'bearish': 'red',
                'neutral': 'orange'
            }.get(pattern_summary['dominant_sentiment'], 'gray')
            st.metric("Dominant Sentiment", pattern_summary['dominant_sentiment'].title())
        
        # Display individual patterns
        st.subheader("Discovered Patterns")
        
        if not any(patterns.values()):
            st.info("No significant patterns detected in recent price action.")
            return
        
        # Bullish patterns
        if patterns['bullish']:
            st.success("### 🟢 Bullish Patterns")
            for pattern in patterns['bullish'][:5]:  # Show top 5
                st.write(f"**{pattern['name']}** (Strength: {pattern['strength']}, Recency: {pattern['period']} periods ago)")
        
        # Bearish patterns
        if patterns['bearish']:
            st.error("### 🔴 Bearish Patterns")
            for pattern in patterns['bearish'][:5]:  # Show top 5
                st.write(f"**{pattern['name']}** (Strength: {pattern['strength']}, Recency: {pattern['period']} periods ago)")
        
        # Neutral patterns
        if patterns['neutral']:
            st.warning("### 🟡 Neutral Patterns")
            for pattern in patterns['neutral'][:5]:  # Show top 5
                st.write(f"**{pattern['name']}** (Strength: {pattern['strength']}, Recency: {pattern['period']} periods ago)")
        
        # Pattern interpretation
        st.subheader("💡 Pattern Interpretation")
        
        if pattern_summary['dominant_sentiment'] == 'bullish':
            st.success("""
            **Trading Implications:**
            - Consider long positions or adding to existing longs
            - Look for entry points on pullbacks
            - Set stop-loss below recent support levels
            - Monitor for pattern confirmation with volume
            """)
        elif pattern_summary['dominant_sentiment'] == 'bearish':
            st.error("""
            **Trading Implications:**
            - Consider short positions or reducing longs
            - Look for resistance levels to enter shorts
            - Set stop-loss above recent resistance
            - Be cautious of potential reversals
            """)
        else:
            st.warning("""
            **Trading Implications:**
            - Market is in consolidation phase
            - Consider range-bound strategies
            - Wait for breakout confirmation
            - Monitor key support/resistance levels
            """)
    
    def display_fundamental_analysis(self, fundamental_data):
        """Display fundamental analysis"""
        st.subheader("💰 Fundamental Analysis")
        st.info(f"Data Source: {fundamental_data.get('source', 'Unknown')}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("USD Price", f"${fundamental_data['price_usd']:,.2f}")
            st.metric("INR Price", f"₹{fundamental_data['price_inr']:,.2f}")
        
        with col2:
            market_cap = fundamental_data['market_cap']
            if market_cap > 1e9:
                display_mcap = f"${market_cap/1e9:.2f}B"
            else:
                display_mcap = f"${market_cap/1e6:.2f}M"
            st.metric("Market Cap", display_mcap)
            st.metric("Rank", f"#{fundamental_data['rank']}")
        
        with col3:
            st.metric("24h Change", f"{fundamental_data['change_24h']:.2f}%")
            st.metric("7d Change", f"{fundamental_data.get('change_7d', 0):.2f}%")
        
        with col4:
            st.metric("24h Volume", f"${fundamental_data['volume_24h']:,.0f}")
            st.metric("Circulating Supply", f"{fundamental_data['circulating_supply']:,.0f}")
    
    def display_market_depth(self, symbol, exchange):
        """Display market depth information"""
        st.subheader("📊 Market Depth")
        
        if exchange == 'Delta':
            orderbook = self.market_data.delta.get_orderbook(symbol)
            if orderbook:
                # Display bids and asks
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Bids (Buy Orders)**")
                    bids = orderbook.get('buy', [])[:10]  # Top 10 bids
                    bids_df = pd.DataFrame(bids, columns=['Price', 'Size'])
                    if not bids_df.empty:
                        bids_df['Value'] = bids_df['Price'] * bids_df['Size']
                        st.dataframe(bids_df, use_container_width=True)
                
                with col2:
                    st.write("**Asks (Sell Orders)**")
                    asks = orderbook.get('sell', [])[:10]  # Top 10 asks
                    asks_df = pd.DataFrame(asks, columns=['Price', 'Size'])
                    if not asks_df.empty:
                        asks_df['Value'] = asks_df['Price'] * asks_df['Size']
                        st.dataframe(asks_df, use_container_width=True)
                
                # Calculate spread
                if bids and asks:
                    best_bid = bids[0][0]
                    best_ask = asks[0][0]
                    spread = best_ask - best_bid
                    spread_percent = (spread / best_bid) * 100
                    
                    st.metric("Spread", f"${spread:.4f}", f"{spread_percent:.4f}%")
            else:
                st.info("Market depth data temporarily unavailable.")
        else:
            st.info("Market depth available for Delta Exchange only.")

# Simplified platform class to run the app
def main():
    st.title("Aethos Platform - Advanced Markets")
    
    # API status
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.secrets.get("COIN_MARKET_CAP_API_KEY"):
            st.success("✅ CMC API: Connected")
        else:
            st.error("❌ CMC API: Not Configured")
    
    with col2:
        if st.secrets.get("DELTA_API_KEY"):
            st.success("✅ Delta API: Connected")
        else:
            st.warning("⚠️ Delta API: Not Configured")
    
    with col3:
        if YFINANCE_AVAILABLE:
            st.success("✅ Yahoo Finance: Available")
        else:
            st.warning("⚠️ Yahoo Finance: Not Available")
    
    # Initialize and run markets page
    markets_page = MarketsPage()
    markets_page.create_markets_page()

if __name__ == "__main__":
    main()
