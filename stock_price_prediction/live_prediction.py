"""
Live Prediction module for real-time stock price predictions
Continuously fetches live data and provides updated predictions
"""

import time
import threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import config
from data_loader import StockDataLoader, download_live_data
from preprocess import DataPreprocessor
from model import StockPricePredictionModel
from predict import StockPredictor


class LiveStockPredictor:
    """
    Real-time stock price prediction system
    Continuously updates predictions with live data
    """
    
    def __init__(self, symbol, update_interval=None):
        """
        Initialize live predictor
        
        Args:
            symbol: Stock symbol
            update_interval: Seconds between updates
        """
        self.symbol = symbol.upper()
        self.update_interval = update_interval or config.UPDATE_INTERVAL
        self.predictor = StockPredictor(symbol)
        self.is_running = False
        self.current_prediction = None
        self.prediction_history = []
        
    def start(self):
        """
        Start live prediction
        """
        self.is_running = True
        print(f"Starting live prediction for {self.symbol}")
        print(f"Update interval: {self.update_interval} seconds")
        
        # Make initial prediction
        self.update_prediction()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop(self):
        """
        Stop live prediction
        """
        self.is_running = False
        print(f"Stopped live prediction for {self.symbol}")
        
    def update_prediction(self):
        """
        Update the current prediction with latest data
        """
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating prediction...")
            
            # Get prediction
            prediction = self.predictor.get_prediction_summary()
            
            # Store current prediction
            self.current_prediction = prediction
            
            # Add to history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'current_price': prediction['current_price'],
                'predicted_price': prediction['predicted_price'],
                'predicted_change_percent': prediction['predicted_change_percent']
            })
            
            # Print prediction
            self._print_prediction(prediction)
            
            return prediction
            
        except Exception as e:
            print(f"Error updating prediction: {e}")
            return None
    
    def _print_prediction(self, prediction):
        """
        Print current prediction details
        """
        print("-" * 50)
        print(f"Stock: {prediction['symbol']} - {prediction['company_name']}")
        print(f"Current Price:  ${prediction['current_price']:.2f}")
        print(f"Predicted:     ${prediction['predicted_price']:.2f}")
        print(f"Change:        ${prediction['predicted_change']:.2f} ({prediction['predicted_change_percent']:+.2f}%)")
        print(f"52W High:      ${prediction['52_week_high']:.2f}")
        print(f"52W Low:       ${prediction['52_week_low']:.2f}")
        print("-" * 50)
        
    def _update_loop(self):
        """
        Background loop for periodic updates
        """
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self.update_prediction()
                
    def get_current_prediction(self):
        """
        Get the most recent prediction
        
        Returns:
            Current prediction dictionary
        """
        return self.current_prediction
    
    def get_prediction_history(self):
        """
        Get prediction history
        
        Returns:
            List of historical predictions
        """
        return self.prediction_history
    
    def get_live_price(self):
        """
        Get current live price
        
        Returns:
            Current stock price
        """
        try:
            live_data = download_live_data(self.symbol, period='1d', interval='1m')
            if not live_data.empty:
                return live_data['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching live price: {e}")
        return None


class MultiStockLivePredictor:
    """
    Live predictor for multiple stocks
    """
    
    def __init__(self, symbols=None, update_interval=None):
        """
        Initialize multi-stock predictor
        
        Args:
            symbols: List of stock symbols
            update_interval: Seconds between updates
        """
        self.symbols = symbols or config.STOCK_SYMBOLS
        self.update_interval = update_interval or config.UPDATE_INTERVAL
        self.predictors = {}
        self.is_running = False
        
        # Create predictors for each stock
        for symbol in self.symbols:
            self.predictors[symbol] = LiveStockPredictor(symbol, update_interval)
            
    def start_all(self):
        """
        Start live prediction for all stocks
        """
        self.is_running = True
        print(f"Starting live prediction for {len(self.symbols)} stocks")
        print(f"Stocks: {', '.join(self.symbols)}")
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop_all(self):
        """
        Stop all live predictions
        """
        self.is_running = False
        print("Stopped all live predictions")
        
    def update_all(self):
        """
        Update predictions for all stocks
        """
        results = {}
        
        for symbol in self.symbols:
            try:
                predictor = self.predictors[symbol]
                prediction = predictor.update_prediction()
                results[symbol] = prediction
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
                results[symbol] = {'error': str(e)}
                
        return results
    
    def _update_loop(self):
        """
        Background loop for periodic updates
        """
        while self.is_running:
            time.sleep(self.update_interval)
            if self.is_running:
                self.update_all()
                
    def get_all_predictions(self):
        """
        Get current predictions for all stocks
        
        Returns:
            Dictionary with predictions
        """
        return {symbol: predictor.get_current_prediction() 
                for symbol, predictor in self.predictors.items()}


def run_live_prediction(symbol, duration=None):
    """
    Run live prediction for a specified duration
    
    Args:
        symbol: Stock symbol
        duration: Duration in seconds (None for indefinite)
    """
    predictor = LiveStockPredictor(symbol)
    predictor.start()
    
    try:
        if duration:
            time.sleep(duration)
            predictor.stop()
        else:
            # Run indefinitely
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        predictor.stop()
        

def run_multi_stock_live(symbols=None, duration=None):
    """
    Run live prediction for multiple stocks
    
    Args:
        symbols: List of stock symbols
        duration: Duration in seconds (None for indefinite)
    """
    predictor = MultiStockLivePredictor(symbols)
    predictor.start_all()
    
    try:
        if duration:
            time.sleep(duration)
            predictor.stop_all()
        else:
            # Run indefinitely
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        predictor.stop_all()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Stock Price Prediction')
    parser.add_argument('--symbol', type=str, default=config.DEFAULT_STOCK,
                        help='Stock symbol for live prediction')
    parser.add_argument('--all', action='store_true',
                        help='Run live prediction for all stocks')
    parser.add_argument('--interval', type=int, default=config.UPDATE_INTERVAL,
                        help='Update interval in seconds')
    parser.add_argument('--duration', type=int, default=None,
                        help='Duration in seconds (default: indefinite)')
    
    args = parser.parse_args()
    
    if args.all:
        print(f"Starting live prediction for all stocks...")
        run_multi_stock_live(duration=args.duration)
    else:
        print(f"Starting live prediction for {args.symbol}...")
        run_live_prediction(args.symbol, duration=args.duration)
