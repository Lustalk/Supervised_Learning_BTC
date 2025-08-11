"""
Model Training Module for Bitcoin Trading Bot
Author: Senior Data Scientist
Date: 2024

This module implements Phase 3: Model Training with XGBoost classifier,
TimeSeriesSplit validation, and proper feature scaling to prevent data leakage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# XGBoost for classification
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Feature engineering integration
from feature_engineering import create_comprehensive_features


class BitcoinTradingModel:
    """
    XGBoost-based classification model for Bitcoin trading signals.
    
    This class implements the complete ML pipeline including:
    - Feature engineering and preprocessing
    - Time series validation with TimeSeriesSplit
    - Feature scaling after train-test split
    - XGBoost model training and evaluation
    - Hyperparameter optimization
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the trading model.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_weights = None
        
        # Training results
        self.training_history = {}
        self.validation_scores = {}
        self.feature_importance = None
        
        print("🚀 Bitcoin Trading Model initialized with XGBoost")
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by creating features and cleaning.
        
        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame
            target_col (str): Name of target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned features and target
        """
        print("📊 Preparing data for model training...")
        
        # Create comprehensive features
        X, y, df_with_features, feature_summary = create_comprehensive_features(df, target_col)
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Data prepared: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"📈 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_time_series_split(self, n_splits: int = 5) -> TimeSeriesSplit:
        """
        Create TimeSeriesSplit for forward-walking validation.
        
        Args:
            n_splits (int): Number of splits for cross-validation
            
        Returns:
            TimeSeriesSplit: Configured time series splitter
        """
        return TimeSeriesSplit(
            n_splits=n_splits,
            test_size=int(0.2 * 1000),  # 20% of data for testing
            gap=0  # No gap between train and test
        )
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler (fitted on training data only).
        
        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled training and validation features
        """
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled
    
    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Compute balanced class weights to handle class imbalance.
        
        Args:
            y (pd.Series): Target labels
            
        Returns:
            Dict[int, float]: Class weights for each class
        """
        classes = np.unique(y)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )
        
        self.class_weights = dict(zip(classes, weights))
        print(f"⚖️  Class weights: {self.class_weights}")
        
        return self.class_weights
    
    def create_xgboost_model(self, **kwargs) -> xgb.XGBClassifier:
        """
        Create XGBoost classifier with optimized parameters.
        
        Args:
            **kwargs: Additional XGBoost parameters
            
        Returns:
            xgb.XGBClassifier: Configured XGBoost model
        """
        # Default parameters optimized for financial time series
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
            
            # Tree parameters
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            
            # Learning parameters
            'learning_rate': 0.1,
            'n_estimators': 200,
            'early_stopping_rounds': 20,
            
            # Regularization
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1
        }
        
        # Update with any custom parameters
        default_params.update(kwargs)
        
        # Add class weights if available
        if self.class_weights:
            default_params['scale_pos_weight'] = self.class_weights.get(1, 1.0)
        
        return xgb.XGBClassifier(**default_params)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   n_splits: int = 5, **model_params) -> Dict[str, Any]:
        """
        Train the XGBoost model using time series cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target labels
            n_splits (int): Number of CV splits
            **model_params: Additional model parameters
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        print("🎯 Starting model training with TimeSeriesSplit...")
        
        # Create time series split
        tscv = self.create_time_series_split(n_splits)
        
        # Initialize results storage
        fold_scores = []
        fold_models = []
        fold_predictions = []
        
        # Compute class weights
        self.compute_class_weights(y)
        
        # Time series cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"🔄 Training fold {fold}/{n_splits}...")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features (fit on train, transform val)
            X_train_scaled, X_val_scaled = self.scale_features(X_train, X_val)
            
            # Create and train model
            model = self.create_xgboost_model(**model_params)
            
            # Train with early stopping
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)
            
            # Calculate metrics
            fold_score = {
                'fold': fold,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            }
            
            # Store results
            fold_scores.append(fold_score)
            fold_models.append(model)
            fold_predictions.append({
                'y_true': y_val,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
            
            print(f"   ✅ Fold {fold} - Accuracy: {fold_score['accuracy']:.4f}, F1: {fold_score['f1']:.4f}")
        
        # Store training results
        self.training_history = {
            'fold_scores': fold_scores,
            'fold_models': fold_models,
            'fold_predictions': fold_predictions
        }
        
        # Calculate average metrics
        avg_scores = self._calculate_average_scores(fold_scores)
        self.validation_scores = avg_scores
        
        print(f"\n🎉 Training complete! Average CV scores:")
        print(f"   📊 Accuracy: {avg_scores['accuracy']:.4f} ± {avg_scores['accuracy_std']:.4f}")
        print(f"   🎯 F1-Score: {avg_scores['f1']:.4f} ± {avg_scores['f1_std']:.4f}")
        print(f"   📈 Precision: {avg_scores['precision']:.4f} ± {avg_scores['precision_std']:.4f}")
        print(f"   📉 Recall: {avg_scores['recall']:.4f} ± {avg_scores['recall_std']:.4f}")
        
        return self.training_history
    
    def _calculate_average_scores(self, fold_scores: List[Dict]) -> Dict[str, float]:
        """Calculate average and standard deviation of cross-validation scores."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        avg_scores = {}
        
        for metric in metrics:
            values = [score[metric] for score in fold_scores]
            avg_scores[metric] = np.mean(values)
            avg_scores[f'{metric}_std'] = np.std(values)
        
        return avg_scores
    
    def select_best_model(self, metric: str = 'f1') -> xgb.XGBClassifier:
        """
        Select the best model based on validation performance.
        
        Args:
            metric (str): Metric to optimize for ('accuracy', 'precision', 'recall', 'f1')
            
        Returns:
            xgb.XGBClassifier: Best performing model
        """
        if not self.training_history:
            raise ValueError("No training history available. Train the model first.")
        
        # Find best fold
        fold_scores = self.training_history['fold_scores']
        best_fold_idx = np.argmax([score[metric] for score in fold_scores])
        
        # Get best model
        self.model = self.training_history['fold_models'][best_fold_idx]
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"🏆 Best model selected from fold {best_fold_idx + 1} based on {metric}")
        print(f"   📊 {metric.capitalize()}: {fold_scores[best_fold_idx][metric]:.4f}")
        
        return self.model
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model on new data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): True labels
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No trained model available. Train and select a model first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if binary classification
        if len(np.unique(y)) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        print("📊 Model Evaluation Results:")
        for metric, value in metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)):
        """Plot feature importance from the trained model."""
        if self.feature_importance is None:
            print("No feature importance available. Train and select a model first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot top N features
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances (XGBoost)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, fold_idx: int = 0, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix for a specific fold."""
        if not self.training_history:
            print("No training history available. Train the model first.")
            return
        
        if fold_idx >= len(self.training_history['fold_predictions']):
            print(f"Invalid fold index. Available folds: 0-{len(self.training_history['fold_predictions'])-1}")
            return
        
        # Get predictions for the specified fold
        fold_data = self.training_history['fold_predictions'][fold_idx]
        y_true = fold_data['y_true']
        y_pred = fold_data['y_pred']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['SELL', 'HOLD', 'BUY'],
                   yticklabels=['SELL', 'HOLD', 'BUY'])
        plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No trained model available.")
        
        import joblib
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights,
            'training_history': self.training_history,
            'validation_scores': self.validation_scores,
            'feature_importance': self.feature_importance
        }, filepath)
        
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model."""
        import joblib
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Restore model state
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.class_weights = model_data['class_weights']
        self.training_history = model_data['training_history']
        self.validation_scores = model_data['validation_scores']
        self.feature_importance = model_data['feature_importance']
        
        print(f"📂 Model loaded from: {filepath}")


def train_bitcoin_model(df: pd.DataFrame, target_col: str = 'target', 
                       n_splits: int = 5, random_state: int = 42, **model_params) -> BitcoinTradingModel:
    """
    Convenience function to train a Bitcoin trading model.
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame with target column
        target_col (str): Name of target column
        n_splits (int): Number of CV splits
        random_state (int): Random seed for reproducibility
        **model_params: Additional XGBoost parameters
        
    Returns:
        BitcoinTradingModel: Trained model instance
    """
    # Initialize model
    model = BitcoinTradingModel(random_state=random_state)
    
    # Prepare data
    X, y = model.prepare_data(df, target_col)
    
    # Train model
    model.train_model(X, y, n_splits, **model_params)
    
    # Select best model
    model.select_best_model()
    
    return model


if __name__ == "__main__":
    print("Model Training Module - Example Usage")
    print("=" * 50)
    
    # This would typically be used with real data
    print("To use this module:")
    print("1. Prepare your OHLCV data with target labels")
    print("2. Use train_bitcoin_model() function")
    print("3. Or create BitcoinTradingModel instance for more control")
    print("\nExample:")
    print("model = train_bitcoin_model(df, target_col='target')")
    print("model.plot_feature_importance()")
    print("model.plot_confusion_matrix()")
