# train_bilstm_stock.py

import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class BiLSTMStockModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(BiLSTMStockModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_len=60):
    xs, ys = [], []
    for i in range(len(data) - seq_len - 7):
        x = data[i:i + seq_len]
        y = data[i + seq_len:i + seq_len + 7]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(ticker, epochs=100, seq_len=80):
    full_ticker = ticker + ".NS"
    end = datetime.today()
    start = end - timedelta(days=1365)
    df = yf.download(full_ticker, start=start, end=end)

    if df.empty:
        print("No data found.")
        return

    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = create_sequences(scaled_data, seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = BiLSTMStockModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y[:, -1])  # Predict last of the 7 days
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), f"model_{ticker}.pt")
    with open(f"scaler_{ticker}.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model and scaler saved as model_{ticker}.pt and scaler_{ticker}.pkl")

if __name__ == "__main__":
    user_input = input("Enter Indian stock ticker (e.g. INFY, RELIANCE): ").upper()
    train_model(user_input)
    
    
    
    
    
