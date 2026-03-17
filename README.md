# BiLSTM Stock Price Predictor (India) 🚀

A deep learning-based stock price prediction tool tailored for the Indian stock market (NSE). This project utilizes **Bidirectional Long Short-Term Memory (BiLSTM)** networks to forecast future stock prices based on historical trends.

## 🌟 Features
- **Custom Model Training**: Train models for any Indian stock ticker (e.g., INFY, RELIANCE, TCS).
- **7-Day Forecasting**: Predict stock prices for the next 7 days.
- **Interactive Dashboard**: A clean and modern Streamlit UI for easy data visualization.
- **Historical Trends**: View the last 60 days of historical data alongside predictions.
- **Automated Data Fetching**: Uses `yfinance` to get real-time stock data from Yahoo Finance.

## 🛠️ Tech Stack
- **Languages**: Python
- **Deep Learning**: PyTorch (BiLSTM)
- **Data Handling**: Pandas, NumPy, yfinance
- **Preprocessing**: Scikit-Learn (MinMaxScaler)
- **Visualization**: Matplotlib
- **Web Framework**: Streamlit

## 📁 Project Structure
- `app.py`: The main Streamlit application for making predictions and visualizing results.
- `Model_train.py`: Script to download historical data and train a specific model for a stock ticker.
- `Stock_price_Forecast.ipynb`: A research notebook for experimentation and model development.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. You can install the required dependencies using:

```bash
pip install torch streamlit yfinance scikit-learn matplotlib numpy
```

### 1. Training a Model
Before predicting, you must train a model for your chosen stock ticker. Run:

```bash
python Model_train.py
```
- Enter the ticker name when prompted (e.g., `RELIANCE`).
- The script will save `model_<TICKER>.pt` and `scaler_<TICKER>.pkl` in the root directory.

### 2. Running the Prediction App
Once the model is trained, launch the Streamlit dashboard:

```bash
streamlit run app.py
```
- Enter the ticker name in the UI.
- Click **Predict Next 7 Days** to view the forecast and chart.

## 🧠 Model Architecture
The project uses a **Bidirectional LSTM** architecture which processes sequences in both forward and backward directions, capturing complex patterns in stock price movements.
- **Input Layer**: Sequence of 60 days of normalized close prices.
- **BiLSTM Layers**: 2 layers with 64 hidden units each.
- **Output Layer**: Fully connected layer predicting the stock price.

---
*Disclaimer: This tool is for educational purposes only. Stock market investments are subject to market risks. Use predictions at your own risk.*
