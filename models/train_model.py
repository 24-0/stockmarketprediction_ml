import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import joblib
import os

def build_lstm_model(input_shape):
    """
    Build and compile LSTM model for stock prediction.
    
    Args:
        input_shape (tuple): Shape of input data
    
    Returns:
        Sequential: Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, epochs=50, batch_size=32):
    """
    Train the LSTM model.
    
    Args:
        X (np.array): Features
        y (np.array): Targets
        epochs (int): Number of epochs
        batch_size (int): Batch size
    
    Returns:
        Sequential: Trained model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_lstm_model((X_train.shape[1], 1))
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
    return model

def save_model(model, path):
    """
    Save the trained model.
    
    Args:
        model (Sequential): Trained model
        path (str): Save path
    """
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    model.save(path)

if __name__ == "__main__":
    # Load preprocessed data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    X = pd.read_csv(os.path.join(data_dir, 'aapl_X.csv'), header=None).values
    y = pd.read_csv(os.path.join(data_dir, 'aapl_y.csv'), header=None).values.flatten()
    
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    
    # Train model
    print("Training LSTM model...")
    model = train_model(X, y)
    
    # Save model
    save_model(model, 'models/lstm_model.h5')
    print("Model saved as models/lstm_model.h5")
