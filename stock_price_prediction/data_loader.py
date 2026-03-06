"""
Data Loader module for fetching stock data from Yahoo Finance using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config


class StockDataLoader:
    """
    A class to load and manage stock data from Yahoo Finance
    """
    
    def __init__(self, symbol, start_date=None, end_date=None):
        """
        Initialize the StockDataLoader
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        """
        self.symbol = symbol.upper()
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        self.data = None
        self.ticker = None
        
    def download_data(self):
        """
        Download stock data from Yahoo Finance
        
        Returns:
            DataFrame with stock data
        """
        print(f"Downloading data for {self.symbol}...")
        print(f"Date range: {self.start_date} to {self.end_date or 'Present'}")
        
        try:
            # Create ticker object
            self.ticker = yf.Ticker(self.symbol)
            
            # Download historical data
            self.data = self.ticker.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Clean the data
            self.data = self._clean_data(self.data)
            
            print(f"Successfully downloaded {len(self.data)} records for {self.symbol}")
            print(f"Data shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            return self.data
            
        except Exception as e:
            print(f"Error downloading data for {self.symbol}: {e}")
            raise
            
    def _clean_data(self, df):
        """
        Clean and preprocess the downloaded data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Drop rows with missing values
        df = df.dropna()

        # Fill any remaining NaN values with forward/backward fill (pandas 3 compatibility)
        df = df.ffill().bfill()
        
        # Remove duplicate rows
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def get_latest_price(self):
        """
        Get the latest stock price
        
        Returns:
            Latest closing price
        """
        if self.data is None:
            self.download_data()
            
        return self.data[config.TARGET].iloc[-1]
    
    def get_recent_data(self, days=60):
        """
        Get the most recent N days of data
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame with recent data
        """
        if self.data is None:
            self.download_data()
            
        return self.data.tail(days)
    
    def add_technical_indicators(self):
        """
        Add technical indicators to the dataset
        
        Returns:
            DataFrame with additional technical indicators
        """
        if self.data is None:
            self.download_data()
            
        df = self.data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        self.data = df
        return df
    
    def save_to_csv(self, filepath=None):
        """
        Save data to CSV file
        
        Args:
            filepath: Path to save the CSV file
        """
        if self.data is None:
            print("No data to save. Please download data first.")
            return
            
        if filepath is None:
            filepath = f"{self.symbol}_data.csv"
            
        self.data.to_csv(filepath)
        print(f"Data saved to {filepath}")
        
    def get_stock_info(self):
        """
        Get detailed stock information
        
        Returns:
            Dictionary with stock info
        """
        if self.ticker is None:
            self.ticker = yf.Ticker(self.symbol)
            
        info = self.ticker.info
        
        return {
            'symbol': self.symbol,
            'name': info.get('shortName', info.get('longName', 'N/A')),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'pe_ratio': info.get('peRatio', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A')
        }


def load_multiple_stocks(symbols, start_date=None, end_date=None):
    """
    Load data for multiple stock symbols
    
    Args:
        symbols: List of stock symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    stock_data = {}
    
    for symbol in symbols:
        try:
            loader = StockDataLoader(symbol, start_date, end_date)
            data = loader.download_data()
            stock_data[symbol] = data
            print(f"Loaded {symbol}: {len(data)} records")
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
            continue
            
    return stock_data


def download_live_data(symbol, period='1d', interval='1m'):
    """
    Download live/recent data for a symbol
    
    Args:
        symbol: Stock symbol
        period: Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 1wk, 1mo)
        
    Returns:
        DataFrame with live data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    return data


if __name__ == "__main__":
    # Test the data loader
    print("Testing StockDataLoader...")
    
    # Load single stock
    loader = StockDataLoader('AAPL')
    data = loader.download_data()
    
    print("\nStock Info:")
    info = loader.get_stock_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nData Sample:")
    print(data.head())
    
    print("\nData Statistics:")
    print(data.describe())
