"""
Prediction module for Stock Price Prediction
Handles making predictions with trained ANN models
"""

import numpy as np
import pandas as pd
import config
from data_loader import StockDataLoader
from preprocess import DataPreprocessor
from model import StockPricePredictionModel


class StockPredictor:
    """
    A class to make stock price predictions using trained ANN models
    """
    
    def __init__(self, symbol, model_path=None):
        """
        Initialize the StockPredictor
        
        Args:
            symbol: Stock symbol
            model_path: Path to trained model file
        """
        self.symbol = symbol.upper()
        self.model_path = model_path or f"models/{symbol}_model.h5"
        self.model = None
        self.preprocessor = None
        self.loader = None
        
    def load_model(self):
        """
        Load a trained model
        
        Returns:
            Loaded model
        """
        print(f"Loading model for {self.symbol}...")
        
        # Create model instance
        input_shape = (config.LOOKBACK_WINDOW, len(config.FEATURES))
        self.model = StockPricePredictionModel(input_shape)
        
        # Try to load saved model
        try:
            self.model.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using untrained model - predictions may not be accurate!")
            
        return self.model
    
    def prepare_recent_data(self):
        """
        Prepare recent data for prediction
        
        Returns:
            Prepared data for prediction
        """
        # Load recent data
        self.loader = StockDataLoader(self.symbol)
        
        # Get more data than needed for lookback window
        total_days = config.LOOKBACK_WINDOW + 30  # Extra days for indicators
        data = self.loader.get_recent_data(days=total_days)
        
        # Add technical indicators
        data = self.loader.add_technical_indicators()
        
        # Create preprocessor and transform data
        self.preprocessor = DataPreprocessor(lookback_window=config.LOOKBACK_WINDOW)
        
        # Get the latest sequence for prediction
        X = self.preprocessor.get_latest_sequence(data)
        
        return X, data
    
    def predict_next_day(self):
        """
        Predict the next day's stock price
        
        Returns:
            Predicted price for next day
        """
        if self.model is None:
            self.load_model()
            
        # Prepare data
        X, data = self.prepare_recent_data()
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Inverse transform to get actual price
        # We need to create a dummy array with the scaled value
        # The scaler expects all features, so we use the Close price position
        dummy = np.zeros((1, len(config.FEATURES)))
        dummy[0, 3] = prediction  # Close is at index 3
        
        # Inverse transform
        actual_price = self.preprocessor.scaler.inverse_transform(dummy)[0, 3]
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'predicted_price': actual_price,
            'predicted_change': actual_price - current_price,
            'predicted_change_percent': ((actual_price - current_price) / current_price) * 100,
            'prediction_date': pd.Timestamp.now() + pd.Timedelta(days=1)
        }
    
    def predict_multiple_days(self, days=5):
        """
        Predict stock prices for multiple future days
        
        Args:
            days: Number of days to predict ahead
            
        Returns:
            List of predictions
        """
        predictions = []
        current_data = self.loader.get_recent_data(days=config.LOOKBACK_WINDOW + 30)
        
        for i in range(days):
            # Add technical indicators
            current_data = self.loader.add_technical_indicators()
            
            # Prepare data
            self.preprocessor = DataPreprocessor(lookback_window=config.LOOKBACK_WINDOW)
            X = self.preprocessor.get_latest_sequence(current_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Inverse transform
            dummy = np.zeros((1, len(config.FEATURES)))
            dummy[0, 3] = prediction
            actual_price = self.preprocessor.scaler.inverse_transform(dummy)[0, 3]
            
            predictions.append({
                'day': i + 1,
                'date': pd.Timestamp.now() + pd.Timedelta(days=i+1),
                'predicted_price': actual_price
            })
            
            # Add prediction to data for next iteration (recursive prediction)
            new_row = current_data.iloc[-1].copy()
            new_row['Close'] = actual_price
            new_row['Open'] = actual_price
            new_row['High'] = actual_price * 1.02
            new_row['Low'] = actual_price * 0.98
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        return predictions
    
    def get_prediction_summary(self):
        """
        Get a comprehensive prediction summary
        
        Returns:
            Dictionary with prediction details
        """
        # Get next day prediction
        next_day = self.predict_next_day()
        
        # Get stock info
        stock_info = self.loader.get_stock_info()
        
        return {
            'symbol': self.symbol,
            'company_name': stock_info.get('name', 'N/A'),
            'sector': stock_info.get('sector', 'N/A'),
            'current_price': next_day['current_price'],
            'predicted_price': next_day['predicted_price'],
            'predicted_change': next_day['predicted_change'],
            'predicted_change_percent': next_day['predicted_change_percent'],
            'prediction_date': next_day['prediction_date'],
            '52_week_high': stock_info.get('52_week_high', 'N/A'),
            '52_week_low': stock_info.get('52_week_low', 'N/A'),
            'market_cap': stock_info.get('market_cap', 'N/A')
        }


def predict_stock(symbol, model_path=None):
    """
    Convenience function to get stock price prediction
    
    Args:
        symbol: Stock symbol
        model_path: Path to trained model
        
    Returns:
        Prediction dictionary
    """
    predictor = StockPredictor(symbol, model_path)
    return predictor.get_prediction_summary()


def predict_all_stocks():
    """
    Get predictions for all configured stocks
    
    Returns:
        Dictionary with predictions for all stocks
    """
    predictions = {}
    
    for symbol in config.STOCK_SYMBOLS:
        try:
            predictor = StockPredictor(symbol)
            summary = predictor.get_prediction_summary()
            predictions[symbol] = summary
            print(f"{symbol}: ${summary['current_price']:.2f} -> ${summary['predicted_price']:.2f} "
                  f"({summary['predicted_change_percent']:+.2f}%)")
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
            predictions[symbol] = {'error': str(e)}
    
    return predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--symbol', type=str, default=config.DEFAULT_STOCK,
                        help='Stock symbol to predict')
    parser.add_argument('--all', action='store_true',
                        help='Predict for all stocks')
    parser.add_argument('--days', type=int, default=1,
                        help='Number of days to predict')
    
    args = parser.parse_args()
    
    if args.all:
        print("Getting predictions for all stocks...")
        predictions = predict_all_stocks()
    else:
        predictor = StockPredictor(args.symbol)
        
        if args.days > 1:
            predictions = predictor.predict_multiple_days(args.days)
            print(f"\nMulti-day predictions for {args.symbol}:")
            for pred in predictions:
                print(f"  Day {pred['day']} ({pred['date'].date()}): ${pred['predicted_price']:.2f}")
        else:
            summary = predictor.get_prediction_summary()
            print(f"\nPrediction Summary for {args.symbol}:")
            print("="*50)
            print(f"Company:      {summary['company_name']}")
            print(f"Sector:       {summary['sector']}")
            print(f"Current Price: ${summary['current_price']:.2f}")
            print(f"Predicted:    ${summary['predicted_price']:.2f}")
            print(f"Change:       ${summary['predicted_change']:.2f} ({summary['predicted_change_percent']:+.2f}%)")
            print(f"52W High:     ${summary['52_week_high']:.2f}")
            print(f"52W Low:      ${summary['52_week_low']:.2f}")
            print("="*50)
