import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------- Data & Indicators -----------------------------
def fetch_4h(symbol='BTC/USDT', limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)  # ✅ required for VWAP
    return df


def add_indicators(df):
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.roc(length=10, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.stochrsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    vwap=df.ta.vwap(append=False)
    df['VWAP'] = vwap if isinstance(vwap, pd.Series) else vwap.iloc[:, 0]

    df['ema_trend'] = (df['EMA_50'] > df['EMA_200']).astype(int)
    df['price_slope'] = df['close'].diff(4)
    return df.dropna()

def add_hmm_states(df, n_states=4):
    features = df[['close', 'volume', 'RSI_14', 'MACD_12_26_9', 'price_slope']].dropna()
    hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=100)
    hmm.fit(features)
    states = hmm.predict(features)
    df = df.iloc[-len(states):].copy()
    df['hmm_state'] = states
    return df

def label_direction(df):
    df['future_close'] = df['close'].shift(-1)
    df['direction'] = (df['future_close'] > df['close']).astype(int)
    return df.dropna()

# ----------------------------- Feature Creation -----------------------------

def create_sequences(df, seq_len=16):
    cols = [
        'close', 'volume', 'RSI_14', 'MACD_12_26_9', 'EMA_50', 'EMA_200',
        'ROC_10', 'MFI_14', 'STOCHRSIk_14_14_3_3', 'ADX_14',
        'BBL_20_2.0', 'BBU_20_2.0', 'VWAP', 'ema_trend', 'price_slope', 'hmm_state'
    ]
    X, y = [], []
    for i in range(len(df) - seq_len - 1):
        X.append(df[cols].iloc[i:i+seq_len].values)
        y.append(df['direction'].iloc[i+seq_len])
    return np.array(X), np.array(y)

# ----------------------------- Models -----------------------------

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.norm(x[:, -1])
        x = self.drop(x)
        return torch.sigmoid(self.fc(x))

def train_lstm(X_train, y_train, input_size, epochs=10):
    model = BiLSTM(input_size)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        xb = torch.tensor(X_train, dtype=torch.float32)
        yb = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
    return model

def train_lgbm(X, y):
    X_flat = X.reshape(X.shape[0], -1)
    clf = LGBMClassifier(n_estimators=100)
    clf.fit(X_flat, y)
    return clf

def ensemble_predict(lstm_preds, lgbm_preds, threshold=0.5):
    combined = 0.6 * lstm_preds + 0.4 * lgbm_preds
    return (combined > threshold).astype(int)

# ----------------------------- Backtesting -----------------------------

def backtest(df, preds, initial=1000, fee=0.001):
    bal = initial
    eq_curve = []
    for i in range(len(preds)):
        price = df['close'].iloc[i]
        next_price = df['close'].iloc[i+1] if i+1 < len(df) else price
        if preds[i] == 1:
            bal -= bal * fee
            pnl = (next_price - price) / price
            bal += bal * (1 + pnl)
        eq_curve.append(bal)
    return pd.Series(eq_curve), bal

# ----------------------------- Run Full Pipeline -----------------------------

df = fetch_4h()
df = add_indicators(df)
df = add_hmm_states(df)
df = label_direction(df)
X, y = create_sequences(df)

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.2)

print("\n🔁 Training LSTM...")
lstm_model = train_lstm(X_train, y_train, X.shape[2])
lstm_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()

print("\n🌲 Training LightGBM...")
lgbm_model = train_lgbm(X_train, y_train)
lgbm_preds = lgbm_model.predict_proba(X_val.reshape(X_val.shape[0], -1))[:, 1]

print("\n🧠 Running Ensemble...")
final_preds = ensemble_predict(lstm_preds, lgbm_preds)

print("\n📊 Backtesting...")
equity, final_balance = backtest(df.iloc[-len(final_preds):], final_preds)
print(f"Final Balance: ${final_balance:.2f}")

equity.plot(title="Equity Curve", figsize=(10, 4))
plt.xlabel("Trades")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.show()
