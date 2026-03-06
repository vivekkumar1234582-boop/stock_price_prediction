"""
Configuration settings for Stock Price Prediction Project
"""

# Stock symbols to predict
STOCK_SYMBOLS = [
    'AAPL',   # Apple Inc.
    'GOOGL',  # Alphabet Inc.
    'MSFT',   # Microsoft Corporation
    'AMZN',   # Amazon.com Inc.
    'TSLA',   # Tesla Inc.
    'META',   # Meta Platforms Inc.
    'NVDA',   # NVIDIA Corporation
    'JPM',    # JPMorgan Chase & Co.
    'V',      # Visa Inc.
    'WMT'     # Walmart Inc.
]

# Default stock to use for single prediction
DEFAULT_STOCK = 'AAPL'

# Data parameters
START_DATE = '2015-01-01'
END_DATE = None  # None means current date (live data)
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Sliding window parameters
LOOKBACK_WINDOW = 60  # Number of past days to use for prediction

# Model parameters
HIDDEN_LAYERS = [128, 64, 32]  # Number of neurons in each hidden layer
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32

# Live prediction parameters
UPDATE_INTERVAL = 300  # Seconds between live updates (5 minutes)

# Data columns to use
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

# Target column for prediction
TARGET = 'Close'

# Model save path
MODEL_PATH = 'models/ann_model.h5'

# Results directory
RESULTS_DIR = 'results/'
