import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Set global session for yfinance to avoid cloud impersonation issues
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})
yf._session = session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set page config for black theme
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if os.path.exists('static/styles.css'):
    load_css('static/styles.css')

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = 'models/lstm_model.h5'
    if not os.path.exists(model_path):
        model_path = 'lstm_model.h5'
        if not os.path.exists(model_path):
            st.error("No model file found. Please ensure lstm_model.h5 is in the project root or models/ directory.")
            st.stop()
    
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='mse')  # Compile to build metrics
    
    scaler_path = 'data/scaler.pkl'
    if not os.path.exists(scaler_path):
        st.error("Scaler file not found. Please ensure scaler.pkl is in the data/ directory.")
        st.stop()
    
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Failed to load model or scaler: {str(e)}")
    st.stop()

# Function to fetch data
def fetch_data(symbol, period='2y'):
    from datetime import datetime, timedelta
    import time
    max_retries = 3
    end = datetime.now()
    if period == '2y':
        start = end - timedelta(days=730)  # Approx 2 years
    else:
        start = end - timedelta(days=365 * int(period[:-1]))
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, start=start_str, end=end_str, interval='1d', prepost=False, progress=False)
            if data.empty:
                st.error(f"yfinance returned empty data for {symbol}. This could be due to invalid symbol, network issues, or Yahoo Finance API limits. Try a different symbol or check your connection.")
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                st.error(f"Error fetching data for {symbol} after {max_retries} attempts: {str(e)}")
                return pd.DataFrame()

# Function to preprocess for prediction
def preprocess_for_prediction(data, lookback=60):
    close_prices = data['Close'].values[-lookback:].reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    X = np.array([scaled_data])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X

# Function to predict future prices
def predict_future_prices(model, last_data, days=30):
    predictions = []
    current_input = last_data.copy()
    
    for _ in range(days):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0][0])
        # Update input with prediction
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1, 0] = pred[0][0]
    
    # Inverse transform predictions
    pred_array = np.array(predictions).reshape(-1, 1)
    predictions_inv = scaler.inverse_transform(pred_array)
    return predictions_inv.flatten()

# Main app
def main():
    st.title("📈 Stock Market Prediction & Analysis")
    st.markdown("Predict future stock prices using LSTM machine learning model.")
    
    # Sidebar
    st.sidebar.header("Input Parameters")
    popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"]
    selected_symbol = st.sidebar.selectbox("Select Popular Stock", popular_stocks, index=0)
    symbol = st.sidebar.text_input("Or Custom Symbol", value=selected_symbol, help="Enter stock ticker (e.g., AAPL, GOOGL)")
    prediction_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=90, value=30)
    
    if st.sidebar.button("Analyze & Predict"):
        try:
            with st.spinner("Fetching data and making predictions..."):
                # Fetch data
                data = fetch_data(symbol)
                if data.empty:
                    st.error(f"No data found for symbol {symbol}")
                    return
                
                # Preprocess for prediction
                last_data = preprocess_for_prediction(data)
                
                # Predict
                predictions = predict_future_prices(model, last_data, prediction_days)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Historical Data Chart
                st.subheader(f"Historical Data for {symbol}")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data['Close'], label='Historical Close Price', color='#00ff00')
                ax.set_title(f'{symbol} Historical Close Price', color='white')
                ax.set_xlabel('Date', color='white')
                ax.set_ylabel('Price (USD)', color='white')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.patch.set_facecolor('#000000')
                ax.set_facecolor('#000000')
                ax.tick_params(colors='white')
                st.pyplot(fig)
                
                # Prediction Chart
                st.subheader(f"Price Prediction for Next {prediction_days} Days")
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
                
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(data.index[-60:], data['Close'][-60:], label='Recent Historical', color='#00ff00')
                ax2.plot(future_dates, predictions, label='Predicted', color='#ff0000', linestyle='--')
                ax2.set_title(f'{symbol} Price Prediction', color='white')
                ax2.set_xlabel('Date', color='white')
                ax2.set_ylabel('Price (USD)', color='white')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                fig2.patch.set_facecolor('#000000')
                ax2.set_facecolor('#000000')
                ax2.tick_params(colors='white')
                st.pyplot(fig2)
                
                # Analysis
                st.subheader("Analysis Summary")
                current_price = data['Close'].iloc[-1].item()
                predicted_price = predictions[-1].item()
                change = ((predicted_price - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Predicted Price (End)", f"${predicted_price:.2f}")
                with col3:
                    st.metric("Predicted Change", f"{change:.2f}%", delta=f"{change:.2f}%")
                
                if change > 0:
                    st.success(f"📈 Predicted upward trend of {change:.2f}% over {prediction_days} days.")
                else:
                    st.error(f"📉 Predicted downward trend of {change:.2f}% over {prediction_days} days.")
                
                # Data Table
                st.subheader("Recent Data")
                st.dataframe(data.tail(10))
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Please check the symbol and try again.")

if __name__ == "__main__":
    main()
