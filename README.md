# 📈 Stock Market Prediction App

A sophisticated web application for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. Built with Streamlit, this app provides real-time stock analysis and future price predictions with an elegant black-themed interface.

## 🚀 Features

- **Real-time Stock Data**: Fetch historical stock data using Yahoo Finance API
- **LSTM Model Predictions**: Advanced machine learning model for accurate price forecasting
- **Interactive Charts**: Visualize historical data and predictions with matplotlib
- **Customizable Predictions**: Choose prediction timeframe (1-90 days)
- **Black Theme UI**: Sleek, modern interface optimized for extended use
- **Comprehensive Analysis**: View current prices, predicted prices, and trend analysis
- **Data Tables**: Examine recent stock data in tabular format

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning framework for LSTM model
- **yfinance**: Yahoo Finance API for stock data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning utilities

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## 🔧 Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd StockPredictionApp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install streamlit yfinance pandas numpy tensorflow matplotlib scikit-learn
   ```

## 📊 Training the Model

Before running the app, you need to train the LSTM model on historical data:

1. **Fetch and preprocess data**:
   ```bash
   python data/fetch_data.py
   ```

2. **Train the model**:
   ```bash
   python models/train_model.py
   ```

This will create:
- `data/aapl_X.csv` and `data/aapl_y.csv`: Preprocessed training data
- `data/scaler.pkl`: MinMaxScaler for data normalization
- `models/lstm_model.h5`: Trained LSTM model

## 🎯 Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your web browser and go to `http://localhost:8501`

3. **Use the app**:
   - Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
   - Adjust the prediction days slider (1-90 days)
   - Click "Analyze & Predict" to generate forecasts
   - View historical charts, predictions, and analysis

## 📁 Project Structure

```
StockPredictionApp/
│
├── app.py                    # Main Streamlit application
├── data/
│   ├── fetch_data.py         # Data fetching and preprocessing script
│   ├── aapl_X.csv           # Preprocessed features (generated)
│   ├── aapl_y.csv           # Preprocessed targets (generated)
│   └── scaler.pkl           # Data scaler (generated)
├── models/
│   └── train_model.py       # LSTM model training script
│   └── lstm_model.h5        # Trained model (generated)
├── static/
│   └── styles.css           # Custom CSS for black theme
├── TODO.md                  # Project task list
└── README.md                # This file
```

## 🔍 How It Works

1. **Data Collection**: Historical stock prices are fetched using yfinance
2. **Preprocessing**: Data is normalized and sequenced for LSTM input
3. **Model Training**: LSTM network learns patterns from historical data
4. **Prediction**: Model forecasts future prices based on recent trends
5. **Visualization**: Results are displayed in interactive charts and metrics

## 🎨 Customization

### Changing the Stock Symbol
- Modify the default symbol in `app.py` or enter any valid ticker in the UI

### Adjusting Model Parameters
- Edit `models/train_model.py` to change:
  - LSTM layers and units
  - Training epochs and batch size
  - Lookback window for predictions

### Styling
- Customize the black theme in `static/styles.css`
- Modify chart colors and layouts in `app.py`

## 🚨 Disclaimer

This application is for educational and informational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consult with financial professionals before making investment choices.

## 📄 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🚀 Deployment

### Deployment without GitHub

Yes, you can deploy without connecting to GitHub. Here are GitHub-free options:

#### Heroku (Free Tier Available)

1. **Install Heroku CLI**:
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   # Login: heroku login
   ```

2. **Initialize Git Locally** (no GitHub needed):
   ```bash
   cd StockPredictionApp
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```

4. **Create Procfile** (in project root):
   ```
   web: streamlit run app.py --server.port $PORT --server.headless true --server.enableCORS false
   ```

5. **Deploy**:
   ```bash
   git push heroku main
   heroku open
   ```

#### PythonAnywhere (Free Beginner Account)

1. **Sign up** at [pythonanywhere.com](https://www.pythonanywhere.com)
2. **Upload Files**: Use the Files tab to upload your entire project folder
3. **Install Dependencies**: In a console, run `pip3.10 install --user -r requirements.txt`
4. **Create Web App**: Go to Web tab, create new web app, set source code path to your uploaded folder
5. **Configure**: Set WSGI file to run Streamlit (custom setup needed; see their docs for Streamlit)
6. **Reload**: Your app will be at `yourusername.pythonanywhere.com`

#### Quick Sharing with ngrok (Temporary Public URL)

1. **Run Locally**:
   ```bash
   cd StockPredictionApp
   streamlit run app.py --server.port 8501
   ```

2. **Install ngrok**: Download from [ngrok.com](https://ngrok.com)
3. **Expose Port**:
   ```bash
   ngrok http 8501
   ```
4. **Share URL**: Use the generated public URL (e.g., `https://abc123.ngrok.io`) - free but temporary

#### VPS/Server (e.g., DigitalOcean, AWS EC2)

1. **Set up Server**: Launch Ubuntu instance, SSH in
2. **Install Python/Docker**: Follow server setup guides
3. **Upload Project**: Use SCP or Git (local)
4. **Run with Screen/Tmux**: `screen -S app`, then `streamlit run app.py --server.address 0.0.0.0`
5. **Use PM2 or Supervisor**: For production process management
6. **Nginx Reverse Proxy**: For HTTPS and domain

### Streamlit Cloud (Requires GitHub)

1. **Create a GitHub Repository**:
   - Upload your project to GitHub (public repository required for free tier)

2. **Sign up for Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account

3. **Deploy the App**:
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy"

4. **Access Your App**:
   - Once deployed, you'll get a public URL (e.g., `https://your-app.streamlit.app`)

### Docker Deployment (Self-Hosted)

1. **Create Dockerfile** (in project root):
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
   ```

2. **Build and Run Locally**:
   ```bash
   docker build -t stock-app .
   docker run -p 8501:8501 stock-app
   ```

3. **Deploy to Server**: Push image to Docker Hub or run on VPS

### Netlify Deployment

**Note**: Netlify is primarily designed for static sites and JAMstack applications. Streamlit requires a Python backend server, which Netlify's free tier does not support. However, you can deploy a static export of your Streamlit app for demonstration purposes (no interactivity):

1. **Install streamlit-static**:
   ```bash
   pip install streamlit-static
   ```

2. **Export to Static HTML**:
   ```bash
   streamlit-static app.py --output static_export
   ```

3. **Deploy to Netlify**:
   - Upload the `static_export` folder to Netlify
   - This creates a static version with no real-time data fetching or predictions

For full functionality, use platforms that support Python backends like Heroku, Render, or VPS.

### Other Platforms

- **AWS Elastic Beanstalk**: Use EB CLI to deploy (no GitHub needed)
- **Google Cloud Run**: Build Docker image, deploy via gcloud CLI
- **Azure App Service**: Zip upload or CLI deployment
- **Render.com**: Direct GitHub or manual deploy (CLI option available)

## 📞 Support

If you encounter any issues or have questions:
1. Check the installation steps
2. Ensure all dependencies are installed
3. Verify internet connection for data fetching
4. Check console logs for error messages

---

**Happy Trading! 📈**
