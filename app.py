# app.py

import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# BiLSTM Model
class BiLSTMStockModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(BiLSTMStockModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Predict Next 7 Days 
def predict_next_7_days(model, recent_seq, scaler):
    model.eval()
    preds = []
    input_seq = recent_seq.copy()

    for _ in range(7):
        x = torch.tensor(input_seq[-60:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)
        input_seq = np.vstack([input_seq, [[pred]]])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds

# Streamlit UI
st.title("BiLSTM Stock Price Predictor (India)")

stock = st.text_input("Enter Indian stock ticker (e.g. INFY, RELIANCE):", "INFY").upper()

if st.button("Predict Next 7 Days"):
    try:
        full_ticker = stock + ".NS"
        end = datetime.today()
        start = end - timedelta(days=365)
        df = yf.download(full_ticker, start=start, end=end)

        if df.empty:
            st.error("No data found for this ticker.")
        else:
            close_prices = df['Close'].values.reshape(-1, 1)

            # Load scaler and model
            with open(f"scaler_{stock}.pkl", 'rb') as f:
                scaler = pickle.load(f)
            model = BiLSTMStockModel()
            model.load_state_dict(torch.load(f"model_{stock}.pt"))

            scaled = scaler.transform(close_prices)
            recent_seq = scaled[-60:]

            preds = predict_next_7_days(model, recent_seq, scaler)

            # Show results
            next_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
            pred_df = {
                "Date": [d.strftime("%Y-%m-%d") for d in next_dates],
                "Predicted Close Price (INR)": preds
            }

            st.subheader(f" Predicted Prices for {stock} (Next 7 Days)")
            st.table(pred_df)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(df.index[-60:], close_prices[-60:], label="Last 60 Days")
            ax.plot(next_dates, preds, label="Next 7 Days (Predicted)", color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
    except FileNotFoundError:
        st.error("Model not found. Please train it first using the training script.")
