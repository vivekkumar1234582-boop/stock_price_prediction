"""
Training module for Stock Price Prediction ANN Model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import config
from data_loader import StockDataLoader
from preprocess import DataPreprocessor
from model import StockPricePredictionModel


def train_stock_model(symbol, save_model=True, plot_results=True):
    """
    Train ANN model for a specific stock
    
    Args:
        symbol: Stock symbol to train on
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
        
    Returns:
        Trained model and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model for {symbol}")
    print(f"{'='*60}")
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    loader = StockDataLoader(symbol)
    data = loader.download_data()
    
    # Add technical indicators
    print("\n[2/5] Adding technical indicators...")
    data = loader.add_technical_indicators()
    
    # Save raw data
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    data.to_csv(f"{config.RESULTS_DIR}{symbol}_data.csv")
    print(f"Data saved to {config.RESULTS_DIR}{symbol}_data.csv")
    
    # Step 2: Preprocess data
    print("\n[3/5] Preprocessing data...")
    preprocessor = DataPreprocessor(lookback_window=config.LOOKBACK_WINDOW)
    X, y = preprocessor.prepare_data(data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Step 3: Build and train model
    print("\n[4/5] Building and training model...")
    input_shape = (config.LOOKBACK_WINDOW, len(config.FEATURES))
    model = StockPricePredictionModel(input_shape)
    
    # Train
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Step 4: Evaluate model
    print("\n[5/5] Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE METRICS")
    print("="*40)
    print(f"RMSE:           {metrics['RMSE']:.6f}")
    print(f"MAE:            {metrics['MAE']:.6f}")
    print(f"MAPE:           {metrics['MAPE']:.2f}%")
    print(f"R2 Score:       {metrics['R2_Score']:.4f}")
    print(f"Accuracy (5%):  {metrics['Accuracy_5%']:.2f}%")
    print("="*40)
    
    # Save model
    if save_model:
        model_path = f"models/{symbol}_model.h5"
        os.makedirs("models", exist_ok=True)
        model.save_model(model_path)
    
    # Plot results
    if plot_results:
        plot_training_results(model, X_test, y_test, history, symbol)
    
    return model, metrics, preprocessor


def plot_training_results(model, X_test, y_test, history, symbol):
    """
    Plot training and prediction results
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test targets
        history: Training history
        symbol: Stock symbol
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training loss
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: MAE
    ax2 = axes[0, 1]
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    if 'val_mae' in history.history:
        ax2.plot(history.history['val_mae'], label='Validation MAE', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Training and Validation MAE')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Actual vs Predicted
    ax3 = axes[1, 0]
    y_pred = model.predict(X_test)
    ax3.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Price')
    ax3.set_ylabel('Predicted Price')
    ax3.set_title('Actual vs Predicted Prices')
    ax3.grid(True)
    
    # Plot 4: Price prediction over time (sample)
    ax4 = axes[1, 1]
    sample_size = min(100, len(y_test))
    indices = range(sample_size)
    ax4.plot(indices, y_test[:sample_size], label='Actual', color='blue', alpha=0.7)
    ax4.plot(indices, y_pred[:sample_size], label='Predicted', color='red', alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Price (Normalized)')
    ax4.set_title('Actual vs Predicted Over Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{config.RESULTS_DIR}{symbol}_training_results.png", dpi=150, bbox_inches='tight')
    print(f"Training plots saved to {config.RESULTS_DIR}{symbol}_training_results.png")
    
    plt.close()


def train_multiple_stocks(symbols=None):
    """
    Train models for multiple stocks
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dictionary with results for each stock
    """
    if symbols is None:
        symbols = config.STOCK_SYMBOLS
        
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\n\n{'#'*60}")
            print(f"# Training model for {symbol}")
            print(f"{'#'*60}")
            
            model, metrics, preprocessor = train_stock_model(symbol)
            
            results[symbol] = {
                'model': model,
                'metrics': metrics,
                'preprocessor': preprocessor,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error training model for {symbol}: {e}")
            results[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
            continue
    
    # Print summary
    print("\n\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for symbol, result in results.items():
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"{symbol}: MAPE={metrics['MAPE']:.2f}%, R2={metrics['R2_Score']:.4f}")
        else:
            print(f"{symbol}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stock Price Prediction Model')
    parser.add_argument('--symbol', type=str, default=config.DEFAULT_STOCK, 
                        help='Stock symbol to train')
    parser.add_argument('--all', action='store_true',
                        help='Train models for all stocks')
    
    args = parser.parse_args()
    
    if args.all:
        train_multiple_stocks()
    else:
        train_stock_model(args.symbol)
