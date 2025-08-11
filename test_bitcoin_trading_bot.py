"""
Test suite for Bitcoin Trading Bot target labeling functionality.
Tests the core create_labels function and validation logic.
"""

import pytest
import pandas as pd
import numpy as np
from bitcoin_trading_bot import create_labels, validate_labels


class TestCreateLabels:
    """Test cases for the create_labels function."""
    
    def setup_method(self):
        """Set up test data before each test method."""
        # Create sample OHLCV data for testing
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='4H')
        
        # Generate realistic price data
        base_price = 50000
        returns = np.random.normal(0, 0.01, 50)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 50)
        }, index=dates)
    
    def test_create_labels_basic_functionality(self):
        """Test basic functionality of create_labels function."""
        targets = create_labels(self.test_data, T=5, X=0.02)
        
        # Check that targets are created
        assert len(targets) == len(self.test_data)
        assert targets.name == 'target'
        
        # Check that targets contain only valid values
        assert all(val in [-1, 0, 1] for val in targets)
        
        # Check that targets are properly aligned with DataFrame index
        assert targets.index.equals(self.test_data.index)
    
    def test_create_labels_required_columns(self):
        """Test that create_labels requires proper OHLCV columns."""
        # Test with missing columns
        incomplete_data = self.test_data.drop(columns=['high'])
        
        with pytest.raises(ValueError, match="DataFrame must contain columns"):
            create_labels(incomplete_data, T=5, X=0.02)
    
    def test_create_labels_buy_signal_logic(self):
        """Test that BUY signals are correctly generated."""
        # Create data where we can control future price movements
        controlled_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # With T=3 and X=0.02, a price of 105 should trigger a BUY signal
        # since 105 > 100 * (1 + 0.02) = 102
        targets = create_labels(controlled_data, T=3, X=0.02)
        
        # First row should have BUY signal (1) because future high reaches 105
        assert targets.iloc[0] == 1
    
    def test_create_labels_sell_signal_logic(self):
        """Test that SELL signals are correctly generated."""
        # Create data where we can control future price movements
        controlled_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # With T=3 and X=0.02, a price of 95 should trigger a SELL signal
        # since 95 < 100 * (1 - 0.02) = 98
        targets = create_labels(controlled_data, T=3, X=0.02)
        
        # First row should have SELL signal (-1) because future low reaches 95
        assert targets.iloc[0] == -1
    
    def test_create_labels_hold_signal_logic(self):
        """Test that HOLD signals are correctly generated."""
        # Create data where price stays within threshold
        controlled_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [100, 101, 101.5, 101.8, 101.9, 102, 102.1, 102.2, 102.3, 102.4],
            'low': [100, 99, 98.5, 98.2, 98.1, 98, 97.9, 97.8, 97.7, 97.6],
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # With T=3 and X=0.02, price movements should stay within ±2%
        # 100 * (1 + 0.02) = 102, 100 * (1 - 0.02) = 98
        targets = create_labels(controlled_data, T=3, X=0.02)
        
        # First row should have HOLD signal (0) because future prices stay within ±2%
        assert targets.iloc[0] == 0
    
    def test_create_labels_buy_precedence(self):
        """Test that BUY signals take precedence over SELL signals."""
        # Create data where both BUY and SELL conditions could be met
        controlled_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'low': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        # With T=3 and X=0.02:
        # BUY condition: 105 > 100 * (1 + 0.02) = 102 ✓
        # SELL condition: 95 < 100 * (1 - 0.02) = 98 ✓
        # BUY should take precedence
        targets = create_labels(controlled_data, T=3, X=0.02)
        
        # First row should have BUY signal (1) due to precedence rule
        assert targets.iloc[0] == 1
    
    def test_create_labels_horizon_parameter(self):
        """Test that the T (horizon) parameter works correctly."""
        # Test with different horizon values
        targets_T5 = create_labels(self.test_data, T=5, X=0.02)
        targets_T10 = create_labels(self.test_data, T=10, X=0.02)
        
        # Different horizons should produce different results
        assert not targets_T5.equals(targets_T10)
        
        # Longer horizon should generally produce more signals
        # (more opportunities for price to exceed threshold)
        assert abs(targets_T5).sum() <= abs(targets_T10).sum()
    
    def test_create_labels_threshold_parameter(self):
        """Test that the X (threshold) parameter works correctly."""
        # Test with different threshold values
        targets_X01 = create_labels(self.test_data, T=5, X=0.01)  # 1%
        targets_X02 = create_labels(self.test_data, T=5, X=0.02)  # 2%
        targets_X05 = create_labels(self.test_data, T=5, X=0.05)  # 5%
        
        # Higher threshold should produce fewer signals
        # (harder to exceed higher threshold)
        assert abs(targets_X01).sum() >= abs(targets_X02).sum()
        assert abs(targets_X02).sum() >= abs(targets_X05).sum()
    
    def test_create_labels_no_look_ahead_bias(self):
        """Test that the function doesn't use future data for current predictions."""
        # Create data with a clear pattern
        controlled_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        targets = create_labels(controlled_data, T=3, X=0.02)
        
        # The last few rows should not have signals because there's not enough future data
        # With T=3, the last 3 rows should have no signals (or be excluded)
        assert targets.iloc[-3:].sum() == 0 or targets.iloc[-3:].isna().all()


class TestValidateLabels:
    """Test cases for the validate_labels function."""
    
    def test_validate_labels_balanced_distribution(self):
        """Test validation with balanced class distribution."""
        # Create balanced targets
        balanced_targets = pd.Series([1, -1, 0, 1, -1, 0, 1, -1, 0])
        
        # Should not raise any warnings
        validate_labels(balanced_targets)
    
    def test_validate_labels_imbalanced_distribution(self):
        """Test validation with imbalanced class distribution."""
        # Create imbalanced targets (mostly HOLD)
        imbalanced_targets = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, -1])
        
        # Should detect class imbalance
        validate_labels(imbalanced_targets)
    
    def test_validate_labels_missing_classes(self):
        """Test validation when some classes are missing."""
        # Create targets with only two classes
        two_class_targets = pd.Series([1, 1, 1, -1, -1, -1])
        
        # Should warn about missing classes
        validate_labels(two_class_targets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
