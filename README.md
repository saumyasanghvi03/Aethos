# Aethos - BlockVista Terminal for Indian Crypto Traders

## Overview

Aethos is an advanced trading platform specifically designed for the Indian cryptocurrency market. As part of the BlockVista ecosystem, Aethos empowers Indian crypto traders with professional-grade tools, real-time analytics, and compliance features tailored for the Indian regulatory landscape.

Our vision is to democratize algorithmic trading and provide Indian traders with institutional-quality infrastructure for navigating the dynamic crypto markets with confidence and precision.

## Features

### Trading Capabilities
- **Semi-Automated Trading Bots**: Customizable algorithmic strategies with manual oversight
- **Fully Automated Trading**: Set-and-forget algorithmic execution with advanced risk management
- **Multi-Exchange Support**: Seamless integration with major cryptocurrency exchanges
- **Indian Trading Pairs**: Native support for INR pairs (BTC/INR, ETH/INR, USDT/INR, etc.)

### Analytics & Intelligence
- **Real-Time Market Analytics**: Live price tracking, volume analysis, and market depth visualization
- **Technical Indicators**: Comprehensive suite of indicators (RSI, MACD, Bollinger Bands, and more)
- **Portfolio Tracking**: Monitor your holdings across multiple exchanges in one unified dashboard
- **Historical Data Analysis**: Backtesting capabilities with extensive historical market data

### India-Specific Tools
- **INR Integration**: Native rupee support for deposits, withdrawals, and P&L calculations
- **Tax Compliance Tools**: Automated reporting features aligned with Indian cryptocurrency taxation
- **Regulatory Compliance**: Built-in safeguards to ensure adherence to Indian crypto regulations
- **UPI Integration**: Seamless fiat on/off-ramp via Unified Payments Interface

## Tech Stack

- **Python**: Core application framework and trading logic
- **Streamlit**: Interactive web-based user interface
- **Plotly**: Advanced charting and data visualization
- **Pandas**: Data manipulation and time-series analysis
- **NumPy**: High-performance numerical computing
- **Exchange APIs**: Direct integration with Binance, WazirX, CoinDCX, and other major platforms
- **SQLite/PostgreSQL**: Trade history and portfolio data storage
- **WebSocket**: Real-time market data streaming

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- API keys from your preferred cryptocurrency exchange(s)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saumyasanghvi03/Aethos.git
   cd Aethos
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   EXCHANGE_API_KEY=your_api_key_here
   EXCHANGE_API_SECRET=your_api_secret_here
   DATABASE_URL=sqlite:///aethos.db
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the platform**:
   Open your browser and navigate to `http://localhost:8501`

## Usage Guide

Aethos is organized into four main sections:

### 1. Dashboard
Your central hub for monitoring portfolio performance, recent trades, and market overview.
- View real-time P&L in INR
- Track open positions and order status
- Monitor key market indicators
- Quick access to favorite trading pairs

### 2. Algo Trading
Design, backtest, and deploy automated trading strategies.
- Strategy builder with visual workflow
- Backtesting engine with historical data
- Paper trading mode for risk-free testing
- Live deployment with customizable parameters
- Risk management controls (stop-loss, take-profit, position sizing)

### 3. Trader Tools
Advanced analysis and trading utilities.
- Technical analysis charts with 50+ indicators
- Market depth and order book visualization
- Trade journal and performance analytics
- Alert system for price movements and signals

### 4. Markets
Comprehensive market data and analysis.
- Real-time price feeds for all major cryptocurrencies
- Indian exchange rates (INR pairs)
- Market sentiment indicators
- News aggregation and analysis
- Exchange comparison tools

## Contact

**Developer**: Saumya Sanghvi  
**GitHub**: [@saumyasanghvi03](https://github.com/saumyasanghvi03)  
**Project Repository**: [Aethos - BlockVista Terminal](https://github.com/saumyasanghvi03/Aethos)

For bug reports, feature requests, or contributions, please open an issue on GitHub.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Saumya Sanghvi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgments

- Thanks to the Indian crypto trading community for valuable feedback and feature requests
- Built with support from open-source libraries and the Python ecosystem
- Special thanks to contributors who have helped improve the platform
- Inspired by the vision of accessible, professional trading tools for all

---

**Disclaimer**: Cryptocurrency trading involves substantial risk of loss. Aethos is a tool to assist with trading decisions but does not provide financial advice. Always conduct your own research and trade responsibly. Ensure compliance with all applicable Indian laws and regulations regarding cryptocurrency trading and taxation.
