# Stock Market Price Prediction Using Artificial Neural Network (ANN)

## Project Overview
This project implements a multi-layer feedforward neural network with backpropagation to predict stock prices using live data from Yahoo Finance (yfinance).

## Features
- **Live Data Fetching**: Real-time stock data retrieval using yfinance
- **ANN Model**: Multi-layer feedforward neural network with backpropagation
- **Data Preprocessing**: Normalization/scaling, feature extraction from price and volume data
- **Performance Metrics**: MAPE, RMSE, and prediction accuracy
- **Live Prediction**: Continuous prediction updates with periodic data refresh

## Installation
```
bash
pip install -r requirements.txt
```

## Usage
1. Run main script for complete workflow:
```
bash
python main.py
```

2. Run live prediction for real-time updates:
```
bash
python live_prediction.py
```

3. Train model separately:
```
bash
python train.py
```

## Project Structure
- `config.py` - Configuration settings
- `data_loader.py` - Data fetching from yfinance
- `preprocess.py` - Data preprocessing and feature engineering
- `model.py` - ANN model definition
- `train.py` - Model training script
- `predict.py` - Prediction module
- `live_prediction.py` - Real-time prediction system
- `main.py` - Main execution script

## Supported Stocks
The project supports multiple popular stocks including:
- AAPL (Apple)
- GOOGL (Alphabet/Google)
- MSFT (Microsoft)
- AMZN (Amazon)
- TSLA (Tesla)
- META (Meta Platforms)
- NVDA (NVIDIA)
- JPM (JPMorgan Chase)
- V (Visa)
- WMT (Walmart)

## Model Architecture
- Input Layer: Historical prices (sliding window)
- Hidden Layers: Multiple dense layers with ReLU activation
- Output Layer: Single neuron for next day price prediction
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

## Performance Metrics
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Error (RMSE)
- R² Score (Prediction accuracy)

## Requirements
- Python 3.8+
- TensorFlow/Keras
- yfinance
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
