"""
Comprehensive Feature Engineering Module for Bitcoin Trading Bot
Author: Data Scientist
Date: 2024

This module provides extensive technical indicators and features for machine learning:
- Moving Averages (EMA, SMA)
- Momentum Indicators (RSI, MACD, Stochastic, Williams %R)
- Volatility Measures (Bollinger Bands, ATR, Keltner Channels)
- Volume Indicators (OBV, VWAP, Volume Profile)
- Time-based Features (cyclical encoding, market sessions)
- Advanced ML Features (lagged features, rolling statistics)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering class for financial time series data.
    
    This class implements a wide range of technical indicators and features
    commonly used in quantitative trading and machine learning applications.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer with OHLCV data.
        
        Args:
            df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self.validate_input_data()
        
    def validate_input_data(self) -> None:
        """Validate that input DataFrame has required OHLCV columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check if DataFrame is empty
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")
            
        print(f"✅ Input data validated: {len(self.df)} rows × {len(self.df.columns)} columns")
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all available features in the correct order.
        
        Returns:
            pd.DataFrame: DataFrame with all features added
        """
        print("🚀 Starting comprehensive feature engineering...")
        
        # 1. Moving Averages (foundation for other indicators)
        self._add_moving_averages()
        
        # 2. Momentum Indicators
        self._add_momentum_indicators()
        
        # 3. Volatility Measures
        self._add_volatility_indicators()
        
        # 4. Volume Indicators
        self._add_volume_indicators()
        
        # 5. Time-based Features
        self._add_time_features()
        
        # 6. Advanced ML Features
        self._add_advanced_features()
        
        # 7. Feature Interactions and Crossovers
        self._add_feature_interactions()
        
        print(f"✅ Feature engineering complete: {len(self.df.columns)} total columns")
        return self.df
    
    def _add_moving_averages(self) -> None:
        """Add comprehensive moving average indicators."""
        print("📊 Adding moving averages...")
        
        # Simple Moving Averages (SMA)
        sma_periods = [5, 8, 10, 12, 20, 26, 50, 100, 200]
        for period in sma_periods:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages (EMA)
        ema_periods = [5, 8, 12, 20, 21, 26, 34, 55, 89, 200]
        for period in ema_periods:
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period).mean()
        
        # Weighted Moving Average (WMA)
        wma_periods = [10, 20, 50]
        for period in wma_periods:
            weights = np.arange(1, period + 1)
            self.df[f'wma_{period}'] = self.df['close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # Hull Moving Average (HMA) - reduces lag
        for period in [9, 16, 25]:
            wma_half = self.df['close'].rolling(window=period//2).apply(
                lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
            )
            wma_full = self.df['close'].rolling(window=period).apply(
                lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True
            )
            self.df[f'hma_{period}'] = wma_half * 2 - wma_full
        
        print(f"   • Added {len(sma_periods) + len(ema_periods) + len(wma_periods) + len([9, 16, 25])} moving averages")
    
    def _add_momentum_indicators(self) -> None:
        """Add comprehensive momentum indicators."""
        print("📈 Adding momentum indicators...")
        
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            delta = self.df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            # Handle division by zero and infinite values
            rs = rs.replace([np.inf, -np.inf], np.nan)
            self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            # Ensure RSI is within valid range [0, 100]
            self.df[f'rsi_{period}'] = self.df[f'rsi_{period}'].clip(0, 100)
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = self.df['close'].ewm(span=12).mean()
        ema_26 = self.df['close'].ewm(span=26).mean()
        self.df['macd'] = ema_12 - ema_26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        self.df['macd_histogram_ma'] = self.df['macd_histogram'].rolling(window=9).mean()
        
        # Stochastic Oscillator
        for period in [14, 21]:
            lowest_low = self.df['low'].rolling(window=period).min()
            highest_high = self.df['high'].rolling(window=period).max()
            self.df[f'stoch_k_{period}'] = 100 * (self.df['close'] - lowest_low) / (highest_high - lowest_low)
            self.df[f'stoch_d_{period}'] = self.df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        # Williams %R
        for period in [14, 21]:
            highest_high = self.df['high'].rolling(window=period).max()
            lowest_low = self.df['low'].rolling(window=period).min()
            self.df[f'williams_r_{period}'] = -100 * (highest_high - self.df['close']) / (highest_high - lowest_low)
        
        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            self.df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        # Rate of Change (ROC)
        for period in [10, 14, 20]:
            self.df[f'roc_{period}'] = ((self.df['close'] - self.df['close'].shift(period)) / 
                                       self.df['close'].shift(period)) * 100
        
        print(f"   • Added 6 types of momentum indicators")
    
    def _add_volatility_indicators(self) -> None:
        """Add comprehensive volatility measures."""
        print("📊 Adding volatility indicators...")
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_middle = self.df['close'].rolling(window=period).mean()
            bb_std = self.df['close'].rolling(window=period).std()
            self.df[f'bb_middle_{period}'] = bb_middle
            self.df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
            self.df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
            self.df[f'bb_width_{period}'] = (self.df[f'bb_upper_{period}'] - self.df[f'bb_lower_{period}']) / bb_middle
            self.df[f'bb_position_{period}'] = (self.df['close'] - self.df[f'bb_lower_{period}']) / \
                                              (self.df[f'bb_upper_{period}'] - self.df[f'bb_lower_{period}'])
            self.df[f'bb_squeeze_{period}'] = self.df[f'bb_width_{period}'] / self.df[f'bb_width_{period}'].rolling(window=20).mean()
        
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            high_low = self.df['high'] - self.df['low']
            high_close = np.abs(self.df['high'] - self.df['close'].shift())
            low_close = np.abs(self.df['low'] - self.df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            self.df[f'atr_{period}'] = true_range.rolling(window=period).mean()
            self.df[f'atr_ratio_{period}'] = self.df[f'atr_{period}'] / self.df['close']
        
        # Keltner Channels
        for period in [20, 50]:
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            atr = self.df[f'atr_14'] if f'atr_14' in self.df.columns else self.df['close'].rolling(window=period).std()
            self.df[f'kc_upper_{period}'] = typical_price + (atr * 2)
            self.df[f'kc_lower_{period}'] = typical_price - (atr * 2)
            self.df[f'kc_width_{period}'] = (self.df[f'kc_upper_{period}'] - self.df[f'kc_lower_{period}']) / typical_price
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = self.df['close'].pct_change(fill_method=None)
            self.df[f'hist_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(252 * 24)  # Annualized for hourly data
        
        print(f"   • Added 4 types of volatility indicators")
    
    def _add_volume_indicators(self) -> None:
        """Add comprehensive volume-based indicators."""
        print("📊 Adding volume indicators...")
        
        # Volume Moving Averages
        for period in [10, 20, 50]:
            self.df[f'volume_sma_{period}'] = self.df['volume'].rolling(window=period).mean()
            self.df[f'volume_ema_{period}'] = self.df['volume'].ewm(span=period).mean()
        
        # Volume Ratio and Relative Volume
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']
        self.df['relative_volume'] = self.df['volume'] / self.df['volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        self.df['obv'] = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        self.df['obv_sma'] = self.df['obv'].rolling(window=20).mean()
        self.df['obv_ratio'] = self.df['obv'] / self.df['obv_sma']
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['vwap'] = (typical_price * self.df['volume']).rolling(window=20).sum() / self.df['volume'].rolling(window=20).sum()
        self.df['vwap_distance'] = (self.df['close'] - self.df['vwap']) / self.df['vwap']
        
        # Chaikin Money Flow
        for period in [14, 21]:
            money_flow_multiplier = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / \
                                   (self.df['high'] - self.df['low'])
            money_flow_volume = money_flow_multiplier * self.df['volume']
            self.df[f'cmf_{period}'] = money_flow_volume.rolling(window=period).sum() / \
                                      self.df['volume'].rolling(window=period).sum()
        
        # Volume Price Trend (VPT)
        self.df['vpt'] = (self.df['volume'] * ((self.df['close'] - self.df['close'].shift()) / self.df['close'].shift())).fillna(0).cumsum()
        self.df['vpt_sma'] = self.df['vpt'].rolling(window=20).mean()
        
        print(f"   • Added 6 types of volume indicators")
    
    def _add_time_features(self) -> None:
        """Add comprehensive time-based features."""
        print("🕐 Adding time-based features...")
        
        # Check if index is datetime, if not create a dummy datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            # Create a dummy datetime index for testing purposes
            dummy_dates = pd.date_range('2024-01-01', periods=len(self.df), freq='4h')
            self.df.index = dummy_dates
            print("   ⚠️  Created dummy datetime index for time features")
        
        # Extract time components
        hour = self.df.index.hour
        day_of_week = self.df.index.dayofweek
        day_of_month = self.df.index.day
        week_of_year = self.df.index.isocalendar().week
        month = self.df.index.month
        quarter = self.df.index.quarter
        year = self.df.index.year
        
        # Prepare all new features in a dictionary
        new_features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'week_of_year': week_of_year,
            'month': month,
            'quarter': quarter,
            'year': year,
            
            # Cyclical encoding for periodic features
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            
            # Market session indicators
            'is_weekend': (day_of_week >= 5).astype(int),
            'is_month_start': (day_of_month <= 3).astype(int),
            'is_month_end': (day_of_month >= 28).astype(int),
            
            # Asian, European, and US market hours (approximate UTC)
            'is_asian_session': ((hour >= 0) & (hour < 8)).astype(int),
            'is_european_session': ((hour >= 7) & (hour < 16)).astype(int),
            'is_us_session': ((hour >= 13) & (hour < 21)).astype(int),
            
            # Time since market open/close
            'hours_since_market_open': hour,
            'hours_until_market_close': 24 - hour
        }
        
        # Add all features at once
        self.df = pd.concat([self.df, pd.DataFrame(new_features, index=self.df.index)], axis=1)
        
        print(f"   • Added 8 types of time-based features")
    
    def _add_advanced_features(self) -> None:
        """Add advanced ML-friendly features."""
        print("🧠 Adding advanced ML features...")
        
        new_features = {}
        
        # Lagged features (previous values)
        for lag in [1, 2, 3, 5, 10]:
            new_features[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            new_features[f'volume_lag_{lag}'] = self.df['volume'].shift(lag)
            if 'rsi_14' in self.df.columns:
                new_features[f'rsi_14_lag_{lag}'] = self.df['rsi_14'].shift(lag)
        
        # Rolling statistics
        for period in [5, 10, 20]:
            # Price statistics
            new_features[f'close_std_{period}'] = self.df['close'].rolling(window=period).std()
            new_features[f'close_skew_{period}'] = self.df['close'].rolling(window=period).skew()
            new_features[f'close_kurt_{period}'] = self.df['close'].rolling(window=period).kurt()
            
            # Volume statistics
            new_features[f'volume_std_{period}'] = self.df['volume'].rolling(window=period).std()
            new_features[f'volume_skew_{period}'] = self.df['volume'].rolling(window=period).skew()
        
        # Price momentum and acceleration
        for period in [1, 2, 3, 5, 10, 20]:
            new_features[f'return_{period}'] = self.df['close'].pct_change(periods=period)
            new_features[f'log_return_{period}'] = np.log(self.df['close'] / self.df['close'].shift(period))
            new_features[f'return_acceleration_{period}'] = new_features[f'return_{period}'].diff()
        
        # Volatility clustering features
        for period in [5, 10, 20]:
            returns = self.df['close'].pct_change(fill_method=None)
            new_features[f'volatility_{period}'] = returns.rolling(window=period).std()
            new_features[f'volatility_ma_{period}'] = new_features[f'volatility_{period}'].rolling(window=period).mean()
            new_features[f'volatility_ratio_{period}'] = new_features[f'volatility_{period}'] / new_features[f'volatility_ma_{period}']
        
        # Support and resistance levels
        for period in [20, 50]:
            new_features[f'support_{period}'] = self.df['low'].rolling(window=period).min()
            new_features[f'resistance_{period}'] = self.df['high'].rolling(window=period).max()
            new_features[f'price_to_support_{period}'] = (self.df['close'] - new_features[f'support_{period}']) / new_features[f'support_{period}']
            new_features[f'price_to_resistance_{period}'] = (new_features[f'resistance_{period}'] - self.df['close']) / self.df['close']
        
        # Feature ratios and differences
        if 'rsi_14' in self.df.columns:
            new_features['rsi_momentum'] = self.df['rsi_14'] - self.df['rsi_14'].shift(1)
            new_features['rsi_acceleration'] = new_features['rsi_momentum'].diff()
        
        if 'macd' in self.df.columns:
            new_features['macd_momentum'] = self.df['macd'] - self.df['macd'].shift(1)
            new_features['macd_acceleration'] = new_features['macd_momentum'].diff()
        
        # Add all features at once
        self.df = pd.concat([self.df, pd.DataFrame(new_features, index=self.df.index)], axis=1)
        
        print(f"   • Added 5 types of advanced ML features")
    
    def _add_feature_interactions(self) -> None:
        """Add feature interaction and crossover indicators."""
        print("🔗 Adding feature interactions...")
        
        new_features = {}
        
        # Moving average crossovers
        if 'sma_12' in self.df.columns and 'sma_26' in self.df.columns:
            new_features['sma_12_26_cross'] = self.df['sma_12'] - self.df['sma_26']
            new_features['sma_12_26_cross_signal'] = pd.Series(np.where(new_features['sma_12_26_cross'] > 0, 1, -1), index=self.df.index)
            new_features['sma_12_26_cross_change'] = new_features['sma_12_26_cross_signal'].diff()
        
        if 'ema_12' in self.df.columns and 'ema_26' in self.df.columns:
            new_features['ema_12_26_cross'] = self.df['ema_12'] - self.df['ema_26']
            new_features['ema_12_26_cross_signal'] = pd.Series(np.where(new_features['ema_12_26_cross'] > 0, 1, -1), index=self.df.index)
            new_features['ema_12_26_cross_change'] = new_features['ema_12_26_cross_signal'].diff()
        
        # Price vs moving average positions
        for ma_type in ['sma', 'ema']:
            for period in [20, 50, 200]:
                col_name = f'{ma_type}_{period}'
                if col_name in self.df.columns:
                    new_features[f'price_vs_{col_name}'] = (self.df['close'] - self.df[col_name]) / self.df[col_name]
                    new_features[f'price_above_{col_name}'] = (self.df['close'] > self.df[col_name]).astype(int)
        
        # RSI divergence (price vs RSI)
        if 'rsi_14' in self.df.columns:
            price_high = self.df['close'].rolling(window=14).max()
            rsi_high = self.df['rsi_14'].rolling(window=14).max()
            new_features['rsi_divergence'] = (price_high - self.df['close']) - (rsi_high - self.df['rsi_14'])
        
        # Volume-price divergence
        if 'volume_ratio' in self.df.columns:
            price_change = self.df['close'].pct_change(fill_method=None)
            volume_change = self.df['volume_ratio'].pct_change(fill_method=None)
            new_features['volume_price_divergence'] = price_change - volume_change
        
        # Bollinger Band squeeze and expansion
        if 'bb_width_20' in self.df.columns:
            new_features['bb_squeeze'] = self.df['bb_width_20'] / self.df['bb_width_20'].rolling(window=20).mean()
            new_features['bb_expansion'] = self.df['bb_width_20'] / self.df['bb_width_20'].shift(1)
        
        # Add all features at once
        self.df = pd.concat([self.df, pd.DataFrame(new_features, index=self.df.index)], axis=1)
        
        print(f"   • Added 4 types of feature interactions")
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Get a comprehensive summary of all created features.
        
        Returns:
            pd.DataFrame: Feature summary with types and statistics
        """
        feature_info = []
        
        for col in self.df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:  # Skip original columns
                # Determine feature type
                if any(ma in col for ma in ['sma_', 'ema_', 'wma_', 'hma_']):
                    feature_type = 'Moving Average'
                elif any(mom in col for mom in ['rsi_', 'macd', 'stoch', 'williams', 'cci', 'roc']):
                    feature_type = 'Momentum'
                elif any(vol in col for vol in ['bb_', 'atr_', 'kc_', 'hist_vol']):
                    feature_type = 'Volatility'
                elif any(vol_ind in col for vol_ind in ['volume_', 'obv', 'vwap', 'cmf', 'vpt']):
                    feature_type = 'Volume'
                elif any(time_feat in col for time_feat in ['hour', 'day', 'month', 'week', 'year']):
                    feature_type = 'Time'
                elif any(adv in col for adv in ['lag_', 'std_', 'skew_', 'kurt', 'support', 'resistance']):
                    feature_type = 'Advanced ML'
                elif any(inter in col for inter in ['cross', 'divergence', 'squeeze']):
                    feature_type = 'Interaction'
                else:
                    feature_type = 'Other'
                
                feature_info.append({
                    'Feature': col,
                    'Type': feature_type,
                    'Non_Null': self.df[col].count(),
                    'Unique_Values': self.df[col].nunique(),
                    'Mean': self.df[col].mean() if self.df[col].dtype in ['float64', 'int64'] else 'N/A',
                    'Std': self.df[col].std() if self.df[col].dtype in ['float64', 'int64'] else 'N/A'
                })
        
        return pd.DataFrame(feature_info)
    
    def clean_features(self, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Clean and prepare features for machine learning.
        
        Args:
            target_col (str): Name of target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned features and target
        """
        print("🧹 Cleaning features for ML...")
        
        # Separate features and target
        if target_col in self.df.columns:
            y = self.df[target_col]
            X = self.df.drop(columns=[target_col])
        else:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Handle NaN values more gracefully
        initial_rows = len(X)
        
        # Forward fill some features that can be reasonably filled
        fillable_features = [col for col in X.columns if any(fillable in col for fillable in ['sma_', 'ema_', 'wma_', 'hma_', 'bb_', 'volume_'])]
        if fillable_features:
            X[fillable_features] = X[fillable_features].ffill()
        
        # Drop remaining rows with NaN values
        X = X.dropna()
        y = y.loc[X.index]
        
        dropped_rows = initial_rows - len(X)
        if dropped_rows > 0:
            print(f"   ⚠️  Dropped {dropped_rows} rows with NaN values")
        
        # Remove constant features
        constant_features = X.columns[X.nunique() == 1]
        if len(constant_features) > 0:
            X = X.drop(columns=constant_features)
            print(f"   ⚠️  Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features (>0.99) - very lenient for small datasets
        if len(X.columns) > 10:  # Only check correlation if we have many features
            correlation_matrix = X.corr().abs()
            upper_triangle = correlation_matrix.where(np.tri(correlation_matrix.shape[0], k=1, dtype=bool))
            high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.99)]
            
            # Limit the number of features we remove to preserve diversity
            max_to_remove = min(len(high_corr_features), len(X.columns) - 10)
            if max_to_remove > 0:
                high_corr_features = high_corr_features[:max_to_remove]
                X = X.drop(columns=high_corr_features)
                print(f"   ⚠️  Removed {len(high_corr_features)} highly correlated features (>0.99)")
        
        # Ensure we have at least some features left
        if len(X.columns) == 0:
            # If all features were removed, keep at least the basic ones
            basic_features = ['sma_20', 'ema_20', 'rsi_14', 'bb_upper_20', 'bb_lower_20', 'volume_ma_20']
            available_basic = [f for f in basic_features if f in self.df.columns]
            if available_basic:
                X = self.df[available_basic].dropna()
                y = y.loc[X.index]
                print(f"   🔄 Restored {len(available_basic)} basic features after aggressive cleaning")
        
        print(f"   ✅ Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        return X, y


def create_comprehensive_features(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create all features in one call.
    
    Args:
        df (pd.DataFrame): Input OHLCV DataFrame
        target_col (str): Name of target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]: Features, target, full DataFrame with features, and feature summary
    """
    # Initialize feature engineer
    fe = FeatureEngineer(df)
    
    # Create all features
    df_with_features = fe.create_all_features()
    
    # Clean features for ML
    X, y = fe.clean_features(target_col)
    
    # Get feature summary
    feature_summary = fe.get_feature_summary()
    
    return X, y, df_with_features, feature_summary


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module - Example Usage")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='4h')
    
    base_price = 50000
    returns = np.random.normal(0, 0.01, 500)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 500)
    }, index=dates)
    
    # Add target column
    sample_data['target'] = np.random.choice([-1, 0, 1], size=500, p=[0.3, 0.4, 0.3])
    
    print("Sample data shape:", sample_data.shape)
    print("Creating comprehensive features...")
    
    # Create features
    X, y, df_with_features, feature_summary = create_comprehensive_features(sample_data)
    
    print(f"\nFeature engineering complete!")
    print(f"Original data: {sample_data.shape}")
    print(f"Features: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    print(f"\nFeature summary (first 20 features):")
    print(feature_summary.head(20).to_string(index=False))
