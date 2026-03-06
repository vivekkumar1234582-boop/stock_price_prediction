"""
Data Preprocessing module for Stock Price Prediction
Handles normalization, feature engineering, and data splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import config


class DataPreprocessor:
    """
    A class to preprocess stock data for ANN model input
    """
    
    def __init__(self, lookback_window=None, feature_columns=None):
        """
        Initialize the DataPreprocessor
        
        Args:
            lookback_window: Number of past days to use for prediction
            feature_columns: List of columns to use as features
        """
        self.lookback_window = lookback_window or config.LOOKBACK_WINDOW
        self.feature_columns = feature_columns or config.FEATURES
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = None
        self.original_data = None
        
    def fit_transform(self, data):
        """
        Fit the scaler and transform the data
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Scaled data
        """
        # Extract features
        features = data[self.feature_columns].values
        
        # Scale features
        self.scaled_data = self.scaler.fit_transform(features)
        self.original_data = data
        
        return self.scaled_data
    
    def transform(self, data):
        """
        Transform data using fitted scaler
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Scaled data
        """
        features = data[self.feature_columns].values
        return self.scaler.transform(features)
    
    def create_sequences(self, data, target_col_index=None):
        """
        Create sequences for time series prediction using sliding window
        
        Args:
            data: Scaled data array
            target_col_index: Index of target column (usually Close price)
            
        Returns:
            X: Input sequences
            y: Target values (next day price)
        """
        if target_col_index is None:
            # Find the index of Close price in feature columns
            target_col_index = self.feature_columns.index(config.TARGET)
            
        X, y = [], []
        
        for i in range(self.lookback_window, len(data)):
            # Create sequence of past 'lookback_window' days
            X.append(data[i - self.lookback_window:i])
            # Target is the next day's closing price
            y.append(data[i, target_col_index])
            
        return np.array(X), np.array(y)
    
    def create_sequences_with_features(self, data):
        """
        Create sequences using multiple features
        
        Args:
            data: Scaled data array
            
        Returns:
            X: Input sequences (samples, lookback, features)
            y: Target values (next day Close price)
        """
        # Assume Close is the 3rd column (index 3) after Open, High, Low, Close, Volume
        # Adjust based on FEATURES order: ['Open', 'High', 'Low', 'Close', 'Volume']
        target_col_index = 3  # Close price index
        
        X, y = [], []
        
        for i in range(self.lookback_window, len(data)):
            X.append(data[i - self.lookback_window:i])
            y.append(data[i, target_col_index])
            
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, val_split=None, test_split=None):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Input sequences
            y: Target values
            val_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if test_split is None:
            test_split = config.TEST_SPLIT
        if val_split is None:
            val_split = config.VALIDATION_SPLIT
            
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=False  # Keep time order
        )
        
        # Second split: train vs val
        val_size = len(X_train_val) * val_split / (1 - test_split)
        val_size = int(val_size)
        
        X_train = X_train_val[:-val_size]
        X_val = X_train_val[-val_size:]
        y_train = y_train_val[:-val_size]
        y_val = y_train_val[-val_size:]
        
        print(f"Data split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform_target(self, scaled_values):
        """
        Inverse transform scaled target values back to original scale
        
        Args:
            scaled_values: Scaled target values
            
        Returns:
            Original scale values
        """
        # Create a dummy array with same shape as original data
        # We need to add columns to match the scaler's expected input
        dummy = np.zeros((len(scaled_values), len(self.feature_columns)))
        dummy[:, 3] = scaled_values  # Close price is at index 3
        
        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)
        return inversed[:, 3]
    
    def prepare_data(self, data):
        """
        Complete data preparation pipeline
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            X, y: Prepared sequences and targets
        """
        # Scale the data
        scaled_data = self.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences_with_features(scaled_data)
        
        print(f"Prepared data shape:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        
        return X, y
    
    def get_latest_sequence(self, data):
        """
        Get the latest sequence for prediction
        
        Args:
            data: DataFrame with recent stock data
            
        Returns:
            Numpy array ready for prediction
        """
        scaled_data = self.transform(data)
        
        # Get the last 'lookback_window' days
        sequence = scaled_data[-self.lookback_window:]
        
        return sequence.reshape(1, self.lookback_window, len(self.feature_columns))


class FeatureEngineer:
    """
    A class for feature engineering on stock data
    """
    
    @staticmethod
    def add_lag_features(df, lags=[1, 2, 3, 5, 7, 14, 21]):
        """
        Add lagged price features
        
        Args:
            df: DataFrame with stock data
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            
        return df
    
    @staticmethod
    def add_rolling_features(df, windows=[5, 10, 20, 50]):
        """
        Add rolling window features
        
        Args:
            df: DataFrame with stock data
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            # Rolling statistics for Close price
            df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Close_rolling_min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Close_rolling_max_{window}'] = df['Close'].rolling(window=window).max()
            
            # Rolling statistics for Volume
            df[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window=window).mean()
            
        return df
    
    @staticmethod
    def add_date_features(df):
        """
        Add date-based features
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with date features
        """
        df = df.copy()
        
        # Extract date features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfYear'] = df.index.dayofyear
        df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
        
        # One-hot encode cyclical features
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df


def prepare_stock_data(symbol, lookback_window=None, add_indicators=True):
    """
    Complete data preparation pipeline for a stock
    
    Args:
        symbol: Stock symbol
        lookback_window: Number of past days for prediction
        add_indicators: Whether to add technical indicators
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    from data_loader import StockDataLoader
    
    # Load data
    loader = StockDataLoader(symbol)
    data = loader.download_data()
    
    # Add technical indicators if requested
    if add_indicators:
        data = loader.add_technical_indicators()
    
    # Preprocess data
    preprocessor = DataPreprocessor(lookback_window=lookback_window)
    X, y = preprocessor.prepare_data(data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import StockDataLoader
    
    print("Testing DataPreprocessor...")
    
    # Load sample data
    loader = StockDataLoader('AAPL')
    data = loader.download_data()
    data = loader.add_technical_indicators()
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(data)
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    print("\nData preprocessing completed successfully!")
