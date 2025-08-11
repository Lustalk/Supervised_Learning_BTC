"""
Bitcoin Trading Bot using Supervised Machine Learning
Author: Senior Data Scientist
Date: 2024

This project implements a classification-based trading strategy that predicts
BUY, SELL, or HOLD signals for Bitcoin trading.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def create_labels(df: pd.DataFrame, T: int = 24, X: float = 0.02) -> pd.Series:
    """
    Create target labels for supervised learning based on future price movements.
    
    This function implements the core labeling logic for our classification task:
    - BUY (1): Price will rise by more than threshold X% within T periods
    - SELL (-1): Price will fall by more than threshold X% within T periods  
    - HOLD (0): Price movement stays within +/-X% threshold
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        T (int): Prediction horizon in periods (default: 24)
        X (float): Threshold percentage for significant price movement (default: 0.02 = 2%)
    
    Returns:
        pd.Series: Target labels where 1=BUY, -1=SELL, 0=HOLD
        
    Note:
        - Avoids look-ahead bias by using only future data for labeling
        - BUY condition takes precedence if both conditions are met
        - Threshold X should cover transaction fees and generate profit
    """
    
    # Validate input DataFrame has required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Initialize target column with HOLD (0)
    targets = pd.Series(0, index=df.index, name='target')
    
    # Calculate price thresholds for each row
    buy_threshold = df['close'] * (1 + X)  # Price must rise above this for BUY
    sell_threshold = df['close'] * (1 - X)  # Price must fall below this for SELL
    
    # Iterate through each row to create labels
    for i in range(len(df) - T):  # Stop T periods before the end to avoid index errors
        current_close = df.iloc[i]['close']
        current_buy_threshold = buy_threshold.iloc[i]
        current_sell_threshold = sell_threshold.iloc[i]
        
        # Look ahead T periods to check future price movements
        future_window = df.iloc[i+1:i+1+T]
        
        # Check if BUY condition is met (price rises above threshold)
        max_future_price = future_window['high'].max()
        if max_future_price > current_buy_threshold:
            targets.iloc[i] = 1  # BUY signal
            continue  # BUY takes precedence, skip SELL check
        
        # Check if SELL condition is met (price falls below threshold)
        min_future_price = future_window['low'].min()
        if min_future_price < current_sell_threshold:
            targets.iloc[i] = -1  # SELL signal
    
    return targets


def validate_labels(targets: pd.Series) -> None:
    """
    Validate the generated target labels for data quality.
    
    Args:
        targets (pd.Series): Target labels to validate
    """
    print("=== Target Label Validation ===")
    print(f"Total samples: {len(targets)}")
    print(f"BUY signals (1): {sum(targets == 1)} ({sum(targets == 1)/len(targets)*100:.2f}%)")
    print(f"SELL signals (-1): {sum(targets == -1)} ({sum(targets == -1)/len(targets)*100:.2f}%)")
    print(f"HOLD signals (0): {sum(targets == 0)} ({sum(targets == 0)/len(targets)*100:.2f}%)")
    
    # Check for class imbalance
    class_counts = targets.value_counts()
    if len(class_counts) < 3:
        print("⚠️  Warning: Not all three classes are present in the data")
    
    # Check for reasonable distribution (avoid extreme imbalance)
    min_class_ratio = class_counts.min() / class_counts.max()
    if min_class_ratio < 0.1:
        print("⚠️  Warning: Severe class imbalance detected")


# ============================================================================
# STEP 2: DATA COLLECTION AND FEATURE ENGINEERING
# ============================================================================

def fetch_bitcoin_data(symbol: str = 'BTC/USDT', timeframe: str = '4h', limit: int = 1000) -> pd.DataFrame:
    """
    Fetch Bitcoin OHLCV data using yfinance (free alternative to ccxt).
    
    Args:
        symbol (str): Trading pair symbol (default: 'BTC/USDT')
        timeframe (str): Data granularity (default: '4h')
        limit (int): Number of data points to fetch (default: 1000)
    
    Returns:
        pd.DataFrame: OHLCV data with standardized column names
    """
    try:
        import yfinance as yf
        
        # Convert timeframe to yfinance format
        tf_mapping = {'4h': '4h', '1h': '1h', '1d': '1d'}
        yf_tf = tf_mapping.get(timeframe, '4h')
        
        # Use BTC-USD for yfinance
        ticker = yf.Ticker('BTC-USD')
        
        # Fetch data
        df = ticker.history(period=f"{limit*4}h", interval=yf_tf)
        
        if df.empty:
            raise ValueError("No data retrieved from yfinance")
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Available: {df.columns.tolist()}")
        
        print(f"✅ Successfully fetched {len(df)} {timeframe} data points for {symbol}")
        return df[required_cols]
        
    except ImportError:
        print("⚠️  yfinance not available, using sample data")
        return generate_sample_data(limit)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("⚠️  Falling back to sample data")
        return generate_sample_data(limit)

def generate_sample_data(limit: int = 1000) -> pd.DataFrame:
    """
    Generate realistic sample Bitcoin OHLCV data for testing.
    
    Args:
        limit (int): Number of data points to generate
    
    Returns:
        pd.DataFrame: Sample OHLCV data
    """
    np.random.seed(42)
    
    # Generate realistic Bitcoin price movements
    dates = pd.date_range('2024-01-01', periods=limit, freq='4h')
    base_price = 50000
    
    # Simulate realistic price movements with volatility clustering
    returns = np.random.normal(0, 0.01, limit)
    volatility = np.random.gamma(2, 0.005, limit)
    returns = returns * np.sqrt(volatility)
    
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Prevent negative prices
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, limit)
    }, index=dates)
    
    print(f"📊 Generated {limit} sample data points")
    return df

def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive technical indicators from OHLCV data.
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        pd.DataFrame: DataFrame with original data plus technical indicators
    """
    df = df.copy()
    
    # Moving Averages
    df['sma_12'] = df['close'].rolling(window=12).mean()
    df['sma_26'] = df['close'].rolling(window=26).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # Moving Average Crossovers
    df['sma_12_26_cross'] = df['sma_12'] - df['sma_26']
    df['ema_12_26_cross'] = df['ema_12'] - df['ema_26']
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Momentum Features
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(periods=period)
        df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
    
    # Volatility Features
    df['volatility_5'] = df['return_1'].rolling(window=5).std()
    df['volatility_10'] = df['return_1'].rolling(window=10).std()
    df['volatility_20'] = df['return_1'].rolling(window=20).std()
    
    # Volume Features
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Price Position Features
    df['price_position_20'] = (df['close'] - df['close'].rolling(window=20).min()) / \
                              (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
    
    # Time-based Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    print(f"✅ Created {len(df.columns) - 5} technical indicators")
    return df

def preprocess_features(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess features and prepare for machine learning.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target
        target_col (str): Name of target column
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series
    """
    # Separate features and target
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Drop rows with NaN values
    initial_rows = len(X)
    X = X.dropna()
    y = y.loc[X.index]
    
    dropped_rows = initial_rows - len(X)
    if dropped_rows > 0:
        print(f"⚠️  Dropped {dropped_rows} rows with NaN values")
    
    # Remove constant features
    constant_features = X.columns[X.nunique() == 1]
    if len(constant_features) > 0:
        X = X.drop(columns=constant_features)
        print(f"⚠️  Removed {len(constant_features)} constant features: {constant_features.tolist()}")
    
    # Remove highly correlated features
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(np.tri(correlation_matrix.shape[0], k=1, dtype=bool))
    high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    
    if len(high_corr_features) > 0:
        X = X.drop(columns=high_corr_features)
        print(f"⚠️  Removed {len(high_corr_features)} highly correlated features (>0.95)")
    
    print(f"✅ Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y

def step2_data_collection_and_features():
    """
    Execute Step 2: Data Collection and Feature Engineering.
    """
    print("\n" + "="*60)
    print("STEP 2: DATA COLLECTION AND FEATURE ENGINEERING")
    print("="*60)
    
    # 1. Data Collection
    print("\n📊 1. Collecting Bitcoin OHLCV data...")
    df = fetch_bitcoin_data(limit=1000)
    
    # 2. Create Target Labels (from Step 1)
    print("\n🎯 2. Creating target labels...")
    targets = create_labels(df, T=24, X=0.02)
    df['target'] = targets
    
    # 3. Feature Engineering
    print("\n🔧 3. Creating technical indicators...")
    df_with_features = create_technical_indicators(df)
    
    # 4. Preprocessing
    print("\n🧹 4. Preprocessing features...")
    X, y = preprocess_features(df_with_features)
    
    # 5. Display Results
    print("\n📈 Feature Engineering Results:")
    print(f"   • Original data shape: {df.shape}")
    print(f"   • Final features shape: {X.shape}")
    print(f"   • Target distribution:")
    print(f"     - BUY (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"     - SELL (-1): {sum(y == -1)} ({sum(y == -1)/len(y)*100:.1f}%)")
    print(f"     - HOLD (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    
    # 6. Feature Importance Preview
    print("\n🔍 5. Feature Overview (first 20 features):")
    feature_info = pd.DataFrame({
        'Feature': X.columns[:20],
        'Type': ['Technical' if 'sma' in col or 'ema' in col or 'rsi' in col or 'macd' in col or 'bb' in col or 'atr' in col 
                 else 'Momentum' if 'return' in col or 'volatility' in col 
                 else 'Volume' if 'volume' in col 
                 else 'Time' if col in ['hour', 'day_of_week', 'month', 'is_weekend']
                 else 'Other' for col in X.columns[:20]],
        'Non-Null': [X[col].count() for col in X.columns[:20]],
        'Unique': [X[col].nunique() for col in X.columns[:20]]
    })
    print(feature_info.to_string(index=False))
    
    return X, y, df_with_features


if __name__ == "__main__":
    # Example usage and testing
    print("Bitcoin Trading Bot - Step 1: Target Labeling")
    print("=" * 50)
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='4H')
    
    # Generate realistic OHLCV data
    base_price = 50000  # Starting Bitcoin price
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV DataFrame
    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    print("Sample OHLCV Data:")
    print(sample_data.head())
    print("\n" + "="*50)
    
    # Create target labels
    print("Creating target labels...")
    targets = create_labels(sample_data, T=24, X=0.02)
    
    # Validate labels
    validate_labels(targets)
    
    # Show sample results
    print("\nSample Results (first 10 rows):")
    results_df = sample_data.copy()
    results_df['target'] = targets
    print(results_df[['close', 'target']].head(10))
