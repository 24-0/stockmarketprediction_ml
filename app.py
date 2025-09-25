import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

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
    model = load_model('models/lstm_model.h5')
    model.compile(optimizer='adam', loss='mse')  # Compile to build metrics
    scaler = joblib.load('data/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Function to fetch data
def fetch_data(symbol, period='2y'):
    data = yf.download(symbol, period=period, auto_adjust=True)
    return data

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

if __name__ == "__main__":
    main()
