# ✅ Enhanced Smart Money Crypto Predictor with High Accuracy Improvements
# This version applies all optimization steps:
# - Trend/volatility filters
# - Expanded indicators
# - Confidence-based labels
# - Class-weighted BCE loss
# - More training samples

import time, datetime, numpy as np, pandas as pd, os
import requests, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from hmmlearn.hmm import GaussianHMM
import pandas_ta as ta
import joblib

SYMBOL = "BTCUSDT"
INTERVAL = "4h"
SEQ_LEN, PRED_LEN = 30, 1  # Longer sequence
HIDDEN_STATES, EPOCHS, BATCH_SIZE = 4, 100, 32
LR = 1e-4
FETCH_LIMIT = 3000  # More data
THRESHOLD = 0.01  # Higher filter
CONFIDENCE_THRESHOLD = 0.55

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1])
        return self.fc(out)

def compute_indicators(df):
    # === Trend Indicators ===
    df["ema8"] = ta.ema(df["close"], length=8)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    df["ema_trend"] = ((df["close"] > df["ema8"]).astype(int) + (df["ema8"] > df["ema21"]).astype(int) + (df["ema21"] > df["ema50"]).astype(int)) / 3

    # === Volatility and Momentum ===
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_pct"] = df["atr"] / df["close"]

    # === Volume Features ===
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["volume_sma"]

    # === VWAP ===
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]

    # === MACD ===
    macd = ta.macd(df["close"])
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # === ADX ===
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx["ADX_14"]
    df["dmp"] = adx["DMP_14"]
    df["dmn"] = adx["DMN_14"]
    df["dmi_diff"] = df["dmp"] - df["dmn"]

    # === Bollinger Bands ===
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # === Stochastic RSI ===
    stoch = ta.stochrsi(df["close"], length=14)
    df["stoch_k"] = stoch["STOCHRSIk_14_14_3_3"]
    df["stoch_d"] = stoch["STOCHRSId_14_14_3_3"]

    df["ema50_dist"] = (df["close"] - df["ema50"]) / df["close"]
    return df.dropna()

def compute_hmm(df):
    features = df[["close", "rsi14", "atr", "vol_ratio", "ema50_dist"]].dropna().values
    scaler = RobustScaler()
    scaled = scaler.fit_transform(features)
    hmm = GaussianHMM(n_components=HIDDEN_STATES, covariance_type="full", n_iter=200)
    hmm.fit(scaled)
    states = hmm.predict(scaled)
    return states

def create_dataset(df, states):
    df = df.copy()
    df["state"] = states
    feats = df[[
        "close", "rsi14", "atr", "atr_pct", "vol_ratio", "vwap_dist",
        "ema_trend", "macd_line", "macd_signal", "macd_hist",
        "adx", "dmi_diff", "bb_width", "bb_position",
        "stoch_k", "stoch_d", "state"
    ]].values
    X, y = [], []
    for i in range(len(feats) - SEQ_LEN - PRED_LEN):
        cur, fut = feats[i+SEQ_LEN-1][0], feats[i+SEQ_LEN+PRED_LEN-1][0]
        change = (fut - cur) / cur
        atr = feats[i+SEQ_LEN-1][2]
        vol = feats[i+SEQ_LEN-1][3]
        adx_val = feats[i+SEQ_LEN-1][6]

        if abs(change) < THRESHOLD: continue
        if atr < 0.005: continue
       
        if adx_val < 20: continue

        X.append(feats[i:i+SEQ_LEN])
        y.append(1 if change > 0 else 0)

    X, y = np.array(X), np.array(y)
    print(f"Label balance: {np.bincount(y.astype(int))}")

    from sklearn.utils import resample
    pos = resample(X[y==1], n_samples=min(np.bincount(y.astype(int))), replace=False)
    neg = resample(X[y==0], n_samples=min(np.bincount(y.astype(int))), replace=False)
    X = np.concatenate([pos, neg])
    y = np.array([1]*len(pos) + [0]*len(neg))
    shuff = np.random.permutation(len(X))
    return X[shuff], y[shuff]

def fetch_history(symbol="BTCUSDT", interval="4h", limit=2000):
    url, all_data = "https://api.binance.com/api/v3/klines", []
    end_time = int(time.time() * 1000)
    while len(all_data) < limit:
        fetch_count = min(1000, limit - len(all_data))
        params = {"symbol": symbol.upper(), "interval": interval, "limit": fetch_count, "endTime": end_time}
        r = requests.get(url, params=params)
        data = r.json()
        if not data: break
        all_data = data + all_data
        end_time = data[0][0] - 1
        time.sleep(0.5)  # Slower for 4h data
    
    df = pd.DataFrame(all_data, columns=["open_time", "o", "h", "l", "c", "v", "close_time", "qav", "n", "tbv", "tqv", "i"])
    df = df.astype({"o": float, "h": float, "l": float, "c": float, "v": float})
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["o", "h", "l", "c", "v"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    return df

def main():
    df = fetch_history(SYMBOL, INTERVAL, FETCH_LIMIT)
    df = compute_indicators(df)
    states = compute_hmm(df)
    X, y = create_dataset(df, states)
    X_scaled = RobustScaler().fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    split = int(len(X) * 0.8)
    trainX = torch.tensor(X_scaled[:split]).float()
    trainY = torch.tensor(y[:split]).float().unsqueeze(1)
    testX = torch.tensor(X_scaled[split:]).float()
    testY = torch.tensor(y[split:]).float().unsqueeze(1)

    model = SimpleLSTMClassifier(trainX.shape[-1])
    pos_weight = torch.tensor([len(trainY) / trainY.sum() - 1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=LR)

    best_loss, patience = float('inf'), 0
    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(trainX))
        for i in range(0, len(trainX), BATCH_SIZE):
            b = idx[i:i+BATCH_SIZE]
            pred = model(trainX[b])
            loss = loss_fn(pred, trainY[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(testX)
            val_loss = loss_fn(val_pred, testY)
            val_acc = ((torch.sigmoid(val_pred) > 0.5) == testY).float().mean()
            val_auc = roc_auc_score(testY.cpu(), torch.sigmoid(val_pred).cpu())
        print(f"Epoch {epoch+1} | Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | AUC: {val_auc:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss; patience = 0
            torch.save(model.state_dict(), "model.pt")
        else:
            patience += 1
        if patience > 15:
            print("Early stopping"); break

    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    val_df = df.iloc[-len(testX):]
    prev_price = None
    wins, total, profit = 0, 0, 0

    for i in range(len(testX)):
        prob = torch.sigmoid(model(testX[i].unsqueeze(0))).item()
        if 1 - CONFIDENCE_THRESHOLD < prob < CONFIDENCE_THRESHOLD:
            continue
        direction = 1 if prob > 0.5 else 0
        cur_price = val_df["close"].iloc[i]
        if prev_price is not None:
            actual = 1 if cur_price > prev_price else 0
            correct = (actual == direction)
            pct = abs(cur_price - prev_price) / prev_price
            profit += pct * (1 if correct else -1) * 100
            wins += int(correct)
            total += 1
            print(f"[{val_df.index[i]}] {'✅' if correct else '❌'} Pred: {['DOWN','UP'][direction]} | Price: {prev_price:.2f} → {cur_price:.2f} | Profit: {profit:.2f}%")
        prev_price = cur_price

    if total > 0:
        print("\n📈 FINAL RESULTS")
        print(f"Success Rate: {wins}/{total} = {wins/total:.2%}")
        print(f"Net PnL: {profit:.2f}% | Avg/trade: {profit/total:.3f}%")

if __name__ == "__main__":
    main()
