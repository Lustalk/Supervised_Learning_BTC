"""
Comprehensive test suite for the Feature Engineering module.
Tests all technical indicators, edge cases, and convenience functions.
"""

import pytest
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer, create_comprehensive_features


@pytest.fixture
def sample_data_with_target():
    """Create sample data with target column for all tests."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='4h')
    
    base_price = 50000
    returns = np.random.normal(0, 0.01, 300)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 300)
    }, index=dates)
    
    df['target'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.3, 0.4, 0.3])
    return df


class TestFeatureEngineer:
    """Test the main FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='4h')
        
        base_price = 50000
        returns = np.random.normal(0, 0.01, 300)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 300)
        }, index=dates)
    

    
    def test_initialization(self, sample_data):
        """Test FeatureEngineer initialization."""
        fe = FeatureEngineer(sample_data)
        assert fe.df.shape == sample_data.shape
        assert all(col in fe.df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_validation_missing_columns(self):
        """Test validation with missing columns."""
        invalid_df = pd.DataFrame({'open': [100], 'high': [101]})
        with pytest.raises(ValueError, match="Missing required columns"):
            FeatureEngineer(invalid_df)
    
    def test_validation_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        with pytest.raises(ValueError, match="DataFrame is empty"):
            FeatureEngineer(empty_df)
    
    def test_moving_averages(self, sample_data):
        """Test moving average calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_moving_averages()
        
        # Check that SMA columns were added
        assert 'sma_20' in fe.df.columns
        assert 'ema_20' in fe.df.columns
        assert 'wma_20' in fe.df.columns
        
        # Check calculations
        assert not fe.df['sma_20'].isna().all()
        assert not fe.df['ema_20'].isna().all()
    
    def test_momentum_indicators(self, sample_data):
        """Test momentum indicator calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_momentum_indicators()
        
        # Check RSI
        assert 'rsi_14' in fe.df.columns
        # RSI should be either NaN (for initial periods) or between 0 and 100
        rsi_valid = fe.df['rsi_14'].dropna()
        if len(rsi_valid) > 0:
            assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()
        
        # Check MACD
        assert 'macd' in fe.df.columns
        assert 'macd_signal' in fe.df.columns
        assert 'macd_histogram' in fe.df.columns
    
    def test_volatility_indicators(self, sample_data):
        """Test volatility indicator calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_volatility_indicators()
        
        # Check Bollinger Bands
        assert 'bb_middle_20' in fe.df.columns
        assert 'bb_upper_20' in fe.df.columns
        assert 'bb_lower_20' in fe.df.columns
        assert 'bb_width_20' in fe.df.columns
        
        # Check ATR
        assert 'atr_14' in fe.df.columns
        assert fe.df['atr_14'].min() >= 0 or fe.df['atr_14'].isna().all()
    
    def test_volume_indicators(self, sample_data):
        """Test volume indicator calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_volume_indicators()
        
        # Check volume moving averages
        assert 'volume_sma_20' in fe.df.columns
        assert 'volume_ema_20' in fe.df.columns
        
        # Check OBV
        assert 'obv' in fe.df.columns
        assert 'obv_sma' in fe.df.columns
    
    def test_time_features(self, sample_data):
        """Test time-based feature calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_time_features()
        
        # Check basic time features
        assert 'hour' in fe.df.columns
        assert 'day_of_week' in fe.df.columns
        assert 'month' in fe.df.columns
        
        # Check cyclical encoding
        assert 'hour_sin' in fe.df.columns
        assert 'hour_cos' in fe.df.columns
        
        # Check market session indicators
        assert 'is_asian_session' in fe.df.columns
        assert 'is_european_session' in fe.df.columns
        assert 'is_us_session' in fe.df.columns
    
    def test_advanced_features(self, sample_data):
        """Test advanced ML feature calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_advanced_features()
        
        # Check lagged features
        assert 'close_lag_1' in fe.df.columns
        assert 'volume_lag_1' in fe.df.columns
        
        # Check rolling statistics
        assert 'close_std_20' in fe.df.columns
        assert 'close_skew_20' in fe.df.columns
        
        # Check support and resistance
        assert 'support_20' in fe.df.columns
        assert 'resistance_20' in fe.df.columns
        assert 'price_to_support_20' in fe.df.columns
        assert 'price_to_resistance_20' in fe.df.columns
    
    def test_feature_interactions(self, sample_data):
        """Test feature interaction calculations."""
        fe = FeatureEngineer(sample_data)
        fe._add_moving_averages()
        fe._add_momentum_indicators()
        fe._add_volatility_indicators()
        fe._add_feature_interactions()
        
        # Check moving average crossovers
        assert 'sma_12_26_cross' in fe.df.columns
        assert 'ema_12_26_cross' in fe.df.columns
        
        # Check price vs moving average positions
        assert 'price_vs_sma_20' in fe.df.columns
        assert 'price_above_sma_20' in fe.df.columns
        
        # Check Bollinger Band features
        assert 'bb_squeeze' in fe.df.columns
        assert 'bb_expansion' in fe.df.columns
    
    def test_create_all_features(self, sample_data):
        """Test creating all features at once."""
        fe = FeatureEngineer(sample_data)
        df_with_features = fe.create_all_features()
        
        # Should have significantly more columns than original
        assert len(df_with_features.columns) > len(sample_data.columns)
        
        # Check that all feature types were added
        feature_columns = [col for col in df_with_features.columns if col not in sample_data.columns]
        assert len(feature_columns) >= 100  # Should have many features
    
    def test_feature_summary(self, sample_data):
        """Test feature summary generation."""
        fe = FeatureEngineer(sample_data)
        fe.create_all_features()
        summary = fe.get_feature_summary()
        
        # Check summary structure
        assert 'Feature' in summary.columns
        assert 'Type' in summary.columns
        assert 'Non_Null' in summary.columns
        
        # Should have summary for all features (excluding original OHLCV columns)
        original_columns = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in fe.df.columns if col not in original_columns]
        assert len(summary) >= len(feature_columns) - 5  # Allow for some flexibility
    
    def test_clean_features(self, sample_data_with_target):
        """Test feature cleaning for ML."""
        fe = FeatureEngineer(sample_data_with_target)
        fe.create_all_features()
        
        X, y = fe.clean_features('target')
        
        # Check shapes
        assert len(X) == len(y)
        assert 'target' not in X.columns
        
        # Check for no NaN values
        assert not X.isna().any().any()
        assert not y.isna().any()
    
    def test_clean_features_no_target(self, sample_data):
        """Test feature cleaning without target column."""
        fe = FeatureEngineer(sample_data)
        fe.create_all_features()
        
        # Add target column for testing
        fe.df['target'] = np.random.choice([-1, 0, 1], size=len(fe.df))
        
        X, y = fe.clean_features('target')
        
        # Check shapes
        assert len(X) == len(y)
        assert 'target' not in X.columns


class TestConvenienceFunctions:
    """Test convenience functions."""
    

    
    def test_create_comprehensive_features(self, sample_data_with_target):
        """Test the convenience function for creating all features."""
        X, y, df_with_features, feature_summary = create_comprehensive_features(sample_data_with_target)
        
        # Check return types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(df_with_features, pd.DataFrame)
        assert isinstance(feature_summary, pd.DataFrame)
        
        # Check shapes
        assert len(X) == len(y)
        assert len(X) <= len(sample_data_with_target)  # May drop some rows due to NaN
        
        # Check that features were created
        assert len(X.columns) > 5  # Should have many features


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            FeatureEngineer(empty_df)
    
    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        single_row_df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        })
        
        fe = FeatureEngineer(single_row_df)
        # Should handle gracefully, though most indicators will be NaN
        df_with_features = fe.create_all_features()
        
        # Should still have the original data
        assert len(df_with_features) == 1
        assert all(col in df_with_features.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_missing_values_in_data(self):
        """Test with DataFrame containing missing values."""
        df_with_nulls = pd.DataFrame({
            'open': [100, 101, np.nan, 103],
            'high': [101, 102, 102.5, 104],
            'low': [99, 100, 101.5, 102],
            'close': [100.5, 101.5, 102, 103.5],
            'volume': [1000, 1100, 1200, 1300]
        })
        
        fe = FeatureEngineer(df_with_nulls)
        # Should handle gracefully
        df_with_features = fe.create_all_features()
        
        # Should still have the original data
        assert len(df_with_features) == 4
        assert all(col in df_with_features.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_non_datetime_index(self):
        """Test with non-datetime index."""
        df_with_range_index = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        fe = FeatureEngineer(df_with_range_index)
        # Should handle gracefully by creating dummy datetime index
        df_with_features = fe.create_all_features()
        
        # Should still have the original data
        assert len(df_with_features) == 3
        assert all(col in df_with_features.columns for col in ['open', 'high', 'low', 'close', 'volume'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
