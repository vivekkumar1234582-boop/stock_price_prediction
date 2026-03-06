"""
ANN Model module for Stock Price Prediction
Implements a multi-layer feedforward neural network with backpropagation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import config


class StockPricePredictionModel:
    """
    Artificial Neural Network for Stock Price Prediction
    Multi-layer feedforward neural network with backpropagation
    """
    
    def __init__(self, input_shape, hidden_layers=None, learning_rate=None):
        """
        Initialize the ANN model
        
        Args:
            input_shape: Shape of input data (lookback_window, num_features)
            hidden_layers: List of neurons in each hidden layer
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers or config.HIDDEN_LAYERS
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.model = None
        self.history = None
        self.build_model()
        
    def build_model(self):
        """
        Build the ANN model architecture
        
        Architecture:
        - Input layer: Historical prices (sliding window)
        - Hidden layers: Dense layers with ReLU activation
        - Output layer: Single neuron for next day price
        """
        model = keras.Sequential(name='StockPricePredictionANN')
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Flatten input if it's 3D (sequence data)
        model.add(layers.Flatten())
        
        # Hidden layers with ReLU activation
        for i, neurons in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                neurons,
                activation='relu',
                kernel_initializer='he_normal',
                name=f'hidden_layer_{i+1}'
            ))
            # Add dropout for regularization
            model.add(layers.Dropout(0.2))
        
        # Output layer (single neuron for price prediction)
        model.add(layers.Dense(1, activation='linear', name='output_layer'))
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        self.model.summary()
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        """
        Train the ANN model with backpropagation
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        print("\nStarting model training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet!")
            
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet!")
            
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate MAE
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate MAPE
        # Avoid division by zero
        non_zero_indices = y_test != 0
        mape = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / y_test[non_zero_indices])) * 100
        
        # Calculate R2 Score
        r2 = r2_score(y_test, y_pred)
        
        # Calculate accuracy (within 5% error)
        accuracy = np.mean(np.abs((y_test - y_pred) / y_test) < 0.05) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2_Score': r2,
            'Accuracy_5%': accuracy
        }
        
        return metrics
    
    def save_model(self, filepath=None):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if filepath is None:
            filepath = config.MODEL_PATH
            
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")
            
    def load_model(self, filepath=None):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        if filepath is None:
            filepath = config.MODEL_PATH
            
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            Model summary as string
        """
        if self.model is not None:
            return self.model.summary()
        return "No model loaded"


class AdvancedStockModel:
    """
    Advanced ANN model with additional features like batch normalization
    """
    
    def __init__(self, input_shape, hidden_layers=None, learning_rate=None):
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers or [256, 128, 64, 32]
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.model = None
        self.build_model()
        
    def build_model(self):
        """
        Build an advanced ANN model with batch normalization
        """
        model = keras.Sequential(name='AdvancedStockPricePrediction')
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Flatten input
        model.add(layers.Flatten())
        
        # First hidden layer
        model.add(layers.Dense(self.hidden_layers[0], kernel_initializer='he_normal'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.3))
        
        # Additional hidden layers
        for i in range(1, len(self.hidden_layers)):
            model.add(layers.Dense(self.hidden_layers[i], kernel_initializer='he_normal'))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
            model.add(layers.Dropout(0.2))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        self.model.summary()


def create_model(input_shape, model_type='standard'):
    """
    Factory function to create a stock prediction model
    
    Args:
        input_shape: Shape of input data
        model_type: Type of model ('standard' or 'advanced')
        
    Returns:
        StockPricePredictionModel or AdvancedStockModel instance
    """
    if model_type == 'advanced':
        return AdvancedStockModel(input_shape)
    else:
        return StockPricePredictionModel(input_shape)


if __name__ == "__main__":
    # Test the model
    print("Testing ANN Model...")
    
    # Create sample input shape
    input_shape = (config.LOOKBACK_WINDOW, len(config.FEATURES))
    print(f"Input shape: {input_shape}")
    
    # Create model
    model = StockPricePredictionModel(input_shape)
    
    print("\nModel created successfully!")
    
    # Test with random data
    import numpy as np
    X_test = np.random.rand(100, config.LOOKBACK_WINDOW, len(config.FEATURES))
    y_test = np.random.rand(100)
    
    predictions = model.predict(X_test)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
