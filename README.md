Stock Price Prediction using Artificial Neural Networks (ANN)

This project implements a Stock Market Price Prediction system using Artificial Neural Networks (ANN) to forecast future stock prices based on historical market data. The model is designed to learn patterns from past stock performance and generate predictions for upcoming price movements.

The system automatically retrieves historical and live stock market data from Yahoo Finance using the yfinance API. The dataset includes key financial features such as Open, High, Low, Close, and Volume, which are used as inputs for the machine learning model. Before training, the data undergoes preprocessing steps including data cleaning, normalization, sliding window sequence generation, and feature engineering to ensure optimal model performance.

The prediction model is built using TensorFlow/Keras, implementing a multi-layer feedforward neural network with ReLU activation and dropout regularization. The model is trained using backpropagation with the Adam optimizer and Mean Squared Error (MSE) loss function. Training includes validation monitoring, early stopping, and learning rate adjustments to improve convergence and prevent overfitting.

The project supports predictions for multiple major stocks such as Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), Tesla (TSLA), Google (GOOGL), and others, defined in the configuration file. 

config

Additionally, the system includes a real-time prediction module that continuously fetches updated market data and generates live forecasts.
