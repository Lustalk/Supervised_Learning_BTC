# Bitcoin Trading Bot using Supervised Machine Learning

A robust, production-ready Bitcoin trading bot that uses supervised machine learning to generate BUY, SELL, and HOLD signals through rigorous backtesting and risk management.

## 🎯 Project Overview

This project implements a **classification-based trading strategy** that predicts Bitcoin's directional movement over a future time horizon. Unlike traditional regression approaches that predict exact prices, our model classifies market movements into actionable trading signals.

### Key Features
- **Supervised Learning Approach**: Uses historical price data to train a classification model
- **No Look-Ahead Bias**: Strictly forward-walking validation prevents data leakage
- **Comprehensive Backtesting**: Realistic simulation with transaction fees and risk metrics
- **Risk Management Ready**: Designed for integration with stop-loss and position sizing rules

## 🏗️ Architecture & Methodology

### Step 1: Target Labeling (✅ Complete)
Our core innovation lies in the target variable definition:

- **BUY (1)**: Price rises by >2% within 24 periods
- **SELL (-1)**: Price falls by >2% within 24 periods  
- **HOLD (0)**: Price movement stays within ±2% threshold

The 2% threshold covers transaction fees (~0.1%) and generates meaningful profit, while the 24-period horizon balances signal frequency with noise reduction.

### Step 2: Feature Engineering (✅ Complete)
Comprehensive technical indicators including:
- Moving averages (EMA, SMA)
- Momentum indicators (RSI, MACD)
- Volatility measures (Bollinger Bands, ATR)
- Time-based features (day of week, month)

*Optimized for performance with batch operations and zero warnings*

### Step 3: Model Training (⏳ Planned)
- **LightGBM Classifier**: Primary model for its speed and tabular data performance
- **TimeSeriesSplit**: Forward-walking validation to prevent data leakage
- **Feature Scaling**: Applied after train-test split to prevent leakage

### Step 4: Backtesting Engine (⏳ Planned)
- Portfolio simulation with transaction fees
- Risk metrics: Sharpe ratio, max drawdown, win rate
- Realistic trading constraints and slippage

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from bitcoin_trading_bot import create_labels, validate_labels

# Create target labels for your OHLCV data
targets = create_labels(df, T=24, X=0.02)

# Validate the generated labels
validate_labels(targets)
```

### Run Tests
```bash
pytest test_bitcoin_trading_bot.py -v
pytest test_feature_engineering.py -v
```

## 📊 Data Requirements

Your DataFrame must contain these columns:
- `open`: Opening price
- `high`: Highest price in period
- `low`: Lowest price in period  
- `close`: Closing price
- `volume`: Trading volume

## 🔬 Technical Implementation

### Target Labeling Logic
```python
def create_labels(df, T=24, X=0.02):
    """
    T: Prediction horizon (periods)
    X: Threshold percentage (0.02 = 2%)
    
    For each row t:
    1. Check if max(high[t+1:t+T]) > close[t] * (1 + X) → BUY
    2. Check if min(low[t+1:t+T]) < close[t] * (1 - X) → SELL  
    3. Otherwise → HOLD
    
    BUY takes precedence over SELL if both conditions are met.
    """
```

### Key Design Principles
1. **No Data Leakage**: Only future data used for labeling
2. **Transaction Cost Aware**: Threshold covers fees and generates profit
3. **Signal Precedence**: BUY signals take priority over SELL
4. **Configurable Parameters**: Adjustable horizon and threshold

## 📈 Performance Metrics

The backtesting engine will evaluate:
- **Total Return**: Portfolio P&L over time
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Trade Frequency**: Number of signals generated

## 🛡️ Risk Management

While the ML model generates signals, a complete trading strategy requires:

- **Stop-Loss**: Exit trades at predefined loss levels
- **Take-Profit**: Lock in profits at target levels
- **Position Sizing**: Risk 1-2% of capital per trade
- **Portfolio Limits**: Maximum exposure constraints

## 🔮 Future Enhancements

- Real-time data streaming with CCXT
- Ensemble methods (XGBoost, Random Forest)
- Advanced feature engineering (market microstructure)
- Cloud deployment with Docker
- Live trading integration

## 📚 Dependencies

- **Core ML**: scikit-learn, LightGBM, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest

## 🤝 Contributing

This project follows strict data science best practices:
- PEP 8 code style
- Comprehensive test coverage
- Conventional commit messages
- No look-ahead bias in validation

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk. Past performance does not guarantee future results. Always conduct thorough testing before live deployment.

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ by a Data Scientist specializing in quantitative finance**
