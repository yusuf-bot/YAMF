import ccxt
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import matplotlib.pyplot as plt

# === 1. Download 4H BTC/USDT data from Binance via ccxt ===
def download_data(symbol="BTC/USDT", timeframe="4h", limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.dropna(inplace=True)
    return df

# === 2. Add technical indicators ===
def add_indicators(df):
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.supertrend(append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    df.fillna(method="bfill", inplace=True)
    return df

# === 3. Dataset Preparation ===
class SequenceDataset(Dataset):
    def __init__(self, df, seq_len=24, threshold=0.003):
        features = [
            "Open", "High", "Low", "Close", "Volume",
            "EMA_20", "EMA_50", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9",
            "ATRr_14", "BBL_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "SUPERT_7_3.0",
            "STOCHk_14", "STOCHd_14", "MFI_14"
        ]
        data = df[features].values
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        self.sequences, self.labels = [], []
        for i in range(len(data) - seq_len - 1):
            seq = data[i:i + seq_len]
            future_return = (df["Close"].iloc[i + seq_len + 1] - df["Close"].iloc[i + seq_len]) / df["Close"].iloc[i + seq_len]
            label = 1 if future_return > threshold else 0
            self.sequences.append(torch.tensor(seq, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# === 4. Transformer Model ===
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.output(x).squeeze()

# === 5. Train Function ===
def train_model(model, dataloader, val_loader, epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(dataloader))

        model.eval()
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                loss = criterion(preds, y)
                total_val += loss.item()
        val_loss.append(total_val / len(val_loader))
        print(f"Epoch {epoch+1}: Train Loss={train_loss[-1]:.4f} | Val Loss={val_loss[-1]:.4f}")

    return model

# === 6. Backtest Function ===
def backtest(model, dataset, threshold=0.5):
    model.eval()
    prices = []
    preds_bin, true_bin = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            prob = torch.sigmoid(model(x.unsqueeze(0))).item()
            pred = 1 if prob > threshold else 0
            preds_bin.append(pred)
            true_bin.append(int(y.item()))
            prices.append(x[-1][3].item())  # closing price in last candle of sequence

    returns = []
    for i in range(1, len(prices)):
        if preds_bin[i - 1] == 1:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
        else:
            returns.append(0)

    equity = np.cumprod([1 + r for r in returns])
    plt.plot(equity)
    plt.title("Backtest Equity Curve")
    plt.xlabel("Trades")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()

    acc = np.mean(np.array(preds_bin) == np.array(true_bin))
    print(f"Prediction Accuracy: {acc * 100:.2f}%")

# === 7. Run the Full Pipeline ===
df = download_data()
df = add_indicators(df)
dataset = SequenceDataset(df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

input_dim = dataset[0][0].shape[1]
seq_len = dataset[0][0].shape[0]

model = TransformerClassifier(input_dim=input_dim, seq_len=seq_len)
model = train_model(model, train_loader, val_loader, epochs=15)

backtest(model, val_set)
