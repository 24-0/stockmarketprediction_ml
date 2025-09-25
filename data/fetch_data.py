import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def fetch_stock_data(symbol, period='5y'):
    """
    Fetch historical stock data for a given symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Period for data (e.g., '5y')
    
    Returns:
        pd.DataFrame: Historical data
    """
    data = yf.download(symbol, period=period)
    if data.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    return data

def preprocess_data(data, lookback=60):
    """
    Preprocess the data for LSTM model.
    
    Args:
        data (pd.DataFrame): Historical data
        lookback (int): Number of past days to use for prediction
    
    Returns:
        tuple: (X, y, scaler) where X is features, y is targets, scaler for inverse transform
    """
    close_prices = data['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def save_data(data, path):
    """
    Save data to CSV file.
    
    Args:
        data (pd.DataFrame or np.array): Data to save
        path (str): File path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, pd.DataFrame):
        data.to_csv(path)
    else:
        pd.DataFrame(data).to_csv(path, index=False)

if __name__ == "__main__":
    # Demo: Fetch and preprocess AAPL data
    symbol = 'AAPL'
    print(f"Fetching data for {symbol}...")
    raw_data = fetch_stock_data(symbol)
    save_data(raw_data, 'data/aapl_raw.csv')
    
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(raw_data)
    save_data(X.reshape(X.shape[0], -1), 'data/aapl_X.csv')
    save_data(y, 'data/aapl_y.csv')
    
    # Save scaler for later use (using joblib)
    import joblib
    joblib.dump(scaler, 'data/scaler.pkl')
    
    print(f"Data saved for {symbol}. Shape of X: {X.shape}, y: {y.shape}")
