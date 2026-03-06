"""
Main entry point for Stock Price Prediction using ANN
This script runs the complete pipeline: data loading, training, and prediction
"""

import os
import sys
import argparse
from datetime import datetime

import config
from data_loader import StockDataLoader
from preprocess import DataPreprocessor
from model import StockPricePredictionModel
from train import train_stock_model, train_multiple_stocks
from predict import StockPredictor, predict_stock, predict_all_stocks
from live_prediction import LiveStockPredictor, run_live_prediction


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║      STOCK MARKET PRICE PREDICTION USING ARTIFICIAL NEURAL      ║
    ║                      NETWORK (ANN)                               ║
    ║                                                                  ║
    ║                    Version 1.0.0                                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def show_menu():
    """Show main menu"""
    menu = """
    Select an option:
    
    [1] Train model for a single stock
    [2] Train models for all stocks
    [3] Make prediction for a single stock
    [4] Make predictions for all stocks
    [5] Run live prediction (single stock)
    [6] Run live prediction (all stocks)
    [7] Download and view stock data
    [8] Exit
    
    Enter your choice: """
    return input(menu)


def download_and_view_data():
    """Download and display stock data"""
    print("\n" + "="*60)
    print("DOWNLOAD AND VIEW STOCK DATA")
    print("="*60)
    
    # Show available stocks
    print("\nAvailable stocks:")
    for i, symbol in enumerate(config.STOCK_SYMBOLS, 1):
        print(f"  {i}. {symbol}")
    
    # Get user choice
    choice = input("\nEnter stock number or symbol (or 'all'): ").strip().upper()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(config.STOCK_SYMBOLS):
            symbols = [config.STOCK_SYMBOLS[idx]]
        else:
            print("Invalid selection!")
            return
    elif choice == 'ALL':
        symbols = config.STOCK_SYMBOLS
    else:
        symbols = [choice]
    
    for symbol in symbols:
        try:
            print(f"\nDownloading data for {symbol}...")
            loader = StockDataLoader(symbol)
            data = loader.download_data()
            
            # Add technical indicators
            data = loader.add_technical_indicators()
            
            # Show stock info
            info = loader.get_stock_info()
            print(f"\n--- Stock Information ---")
            print(f"Name:        {info['name']}")
            print(f"Sector:      {info['sector']}")
            print(f"Industry:    {info['industry']}")
            print(f"Market Cap: {info['market_cap']}")
            print(f"Current Price: ${info['current_price']}")
            print(f"52W High:    ${info['52_week_high']}")
            print(f"52W Low:     ${info['52_week_low']}")
            
            # Show data sample
            print(f"\n--- Data Sample (Last 5 Days) ---")
            print(data.tail())
            
            # Show statistics
            print(f"\n--- Data Statistics ---")
            print(data.describe())
            
        except Exception as e:
            print(f"Error: {e}")


def train_single():
    """Train model for a single stock"""
    print("\n" + "="*60)
    print("TRAIN MODEL FOR SINGLE STOCK")
    print("="*60)
    
    # Show available stocks
    print("\nAvailable stocks:")
    for i, symbol in enumerate(config.STOCK_SYMBOLS, 1):
        print(f"  {i}. {symbol}")
    
    # Get user choice
    choice = input("\nEnter stock symbol (e.g., AAPL): ").strip().upper()
    
    if not choice:
        choice = config.DEFAULT_STOCK
        
    # Train the model
    try:
        train_stock_model(choice)
        print(f"\n✓ Model trained successfully for {choice}!")
    except Exception as e:
        print(f"Error: {e}")


def train_all():
    """Train models for all stocks"""
    print("\n" + "="*60)
    print("TRAIN MODELS FOR ALL STOCKS")
    print("="*60)
    print(f"This will train models for {len(config.STOCK_SYMBOLS)} stocks.")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm == 'y':
        try:
            results = train_multiple_stocks()
            print(f"\n✓ Training completed!")
        except Exception as e:
            print(f"Error: {e}")


def predict_single():
    """Make prediction for a single stock"""
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTION")
    print("="*60)
    
    # Get stock symbol
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    
    if not symbol:
        symbol = config.DEFAULT_STOCK
    
    try:
        # Get prediction
        predictor = StockPredictor(symbol)
        summary = predictor.get_prediction_summary()
        
        print(f"\n{'='*50}")
        print(f"PREDICTION SUMMARY FOR {summary['symbol']}")
        print(f"{'='*50}")
        print(f"Company:           {summary['company_name']}")
        print(f"Sector:            {summary['sector']}")
        print(f"Current Price:     ${summary['current_price']:.2f}")
        print(f"Predicted Price:   ${summary['predicted_price']:.2f}")
        print(f"Change:            ${summary['predicted_change']:.2f} ({summary['predicted_change_percent']:+.2f}%)")
        print(f"Prediction Date:   {summary['prediction_date'].date()}")
        print(f"52 Week High:       ${summary['52_week_high']:.2f}")
        print(f"52 Week Low:        ${summary['52_week_low']:.2f}")
        print(f"{'='*50}")
        
        # Trading recommendation
        change_pct = summary['predicted_change_percent']
        if change_pct > 2:
            recommendation = "STRONG BUY"
        elif change_pct > 0:
            recommendation = "BUY"
        elif change_pct > -2:
            recommendation = "HOLD"
        else:
            recommendation = "SELL"
            
        print(f"Recommendation:    {recommendation}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: You may need to train the model first using option [2]")


def predict_all():
    """Make predictions for all stocks"""
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTIONS FOR ALL STOCKS")
    print("="*60)
    
    try:
        predictions = predict_all_stocks()
        
        print(f"\n{'='*70}")
        print(f"{'SYMBOL':<10} {'CURRENT':<12} {'PREDICTED':<12} {'CHANGE':<12} {'RECOMMENDATION'}")
        print(f"{'='*70}")
        
        for symbol, pred in predictions.items():
            if 'error' not in pred:
                change = pred['predicted_change_percent']
                if change > 2:
                    rec = "STRONG BUY"
                elif change > 0:
                    rec = "BUY"
                elif change > -2:
                    rec = "HOLD"
                else:
                    rec = "SELL"
                    
                print(f"{symbol:<10} ${pred['current_price']:<11.2f} ${pred['predicted_price']:<11.2f} {change:>+10.2f}%  {rec}")
        
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: You may need to train the models first using option [2]")


def live_predict_single():
    """Run live prediction for a single stock"""
    print("\n" + "="*60)
    print("LIVE STOCK PRICE PREDICTION")
    print("="*60)
    
    # Get stock symbol
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    
    if not symbol:
        symbol = config.DEFAULT_STOCK
    
    # Get update interval
    interval = input("Enter update interval in seconds (default 300): ").strip()
    if not interval:
        interval = 300
    else:
        interval = int(interval)
    
    # Get duration
    duration = input("Enter duration in seconds (press Enter for indefinite): ").strip()
    if duration:
        duration = int(duration)
    else:
        duration = None
    
    print(f"\nStarting live prediction for {symbol}...")
    print("Press Ctrl+C to stop\n")
    
    try:
        run_live_prediction(symbol, duration=duration)
    except KeyboardInterrupt:
        print("\nLive prediction stopped.")


def live_predict_all():
    """Run live prediction for all stocks"""
    print("\n" + "="*60)
    print("LIVE STOCK PRICE PREDICTION (ALL STOCKS)")
    print("="*60)
    
    # Get update interval
    interval = input("Enter update interval in seconds (default 300): ").strip()
    if not interval:
        interval = 300
    else:
        interval = int(interval)
    
    # Get duration
    duration = input("Enter duration in seconds (press Enter for indefinite): ").strip()
    if duration:
        duration = int(duration)
    else:
        duration = None
    
    print(f"\nStarting live prediction for all stocks...")
    print("Press Ctrl+C to stop\n")
    
    try:
        from live_prediction import run_multi_stock_live
        run_multi_stock_live(duration=duration)
    except KeyboardInterrupt:
        print("\nLive prediction stopped.")


def main():
    """Main function"""
    # Print banner
    print_banner()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction using ANN')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'live', 'data'],
                        help='Operation mode')
    parser.add_argument('--symbol', type=str, help='Stock symbol')
    parser.add_argument('--all', action='store_true', help='Process all stocks')
    
    args = parser.parse_args()
    
    # If command line arguments provided, run in CLI mode
    if args.mode:
        if args.mode == 'data':
            if args.all:
                download_and_view_data()
            else:
                symbol = args.symbol or config.DEFAULT_STOCK
                loader = StockDataLoader(symbol)
                data = loader.download_data()
                data = loader.add_technical_indicators()
                print(data.tail())
                
        elif args.mode == 'train':
            if args.all:
                train_multiple_stocks()
            else:
                symbol = args.symbol or config.DEFAULT_STOCK
                train_stock_model(symbol)
                
        elif args.mode == 'predict':
            if args.all:
                predict_all_stocks()
            else:
                symbol = args.symbol or config.DEFAULT_STOCK
                predict_stock(symbol)
                
        elif args.mode == 'live':
            if args.all:
                run_multi_stock_live()
            else:
                symbol = args.symbol or config.DEFAULT_STOCK
                run_live_prediction(symbol)
    
    # Otherwise, run in interactive mode
    else:
        while True:
            choice = show_menu()
            
            if choice == '1':
                train_single()
            elif choice == '2':
                train_all()
            elif choice == '3':
                predict_single()
            elif choice == '4':
                predict_all()
            elif choice == '5':
                live_predict_single()
            elif choice == '6':
                live_predict_all()
            elif choice == '7':
                download_and_view_data()
            elif choice == '8':
                print("\nExiting... Thank you!")
                break
            else:
                print("\nInvalid choice! Please try again.")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
