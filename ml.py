import ccxt
import pandas as pd
import numpy as np
import talib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize Binance exchange
binance = ccxt.binance({'enableRateLimit': True})

# Fetch 10 years of 1-hour BTC/USDT data
symbol = 'BTC/USDT'
timeframe = '1h'
since = int(pd.Timestamp('2015-04-24').timestamp() * 1000)  # 10 years back from April 24, 2025
limit = 1000  # Binance API limit per request
ohlcv_data = []

# Fetch data in chunks
while True:
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv:
        break
    ohlcv_data.extend(ohlcv)
    since = ohlcv[-1][0] + 1  # Update since to the last timestamp
    if len(ohlcv) < limit:
        break

df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.drop_duplicates(subset=['timestamp'])  # Remove duplicates
print(f"Fetched {len(df)} candles for {symbol}")

# Squeeze Momentum Indicator Parameters
length = 20  # BB Length
mult = 2.0  # BB MultFactor
lengthKC = 20  # KC Length
multKC = 1.5  # KC MultFactor
useTrueRange = True

# Calculate Bollinger Bands (BB)
source = df['close']
basis = talib.SMA(source, timeperiod=length)
dev = mult * talib.STDDEV(source, timeperiod=length)
upperBB = basis + dev
lowerBB = basis - dev

# Calculate Keltner Channels (KC)
ma = talib.SMA(source, timeperiod=lengthKC)
if useTrueRange:
    tr = pd.concat([
        (df['high'] - df['low']),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
else:
    tr = df['high'] - df['low']
rangema = talib.SMA(tr, timeperiod=lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

# Squeeze conditions
sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
noSqz = (~sqzOn) & (~sqzOff)

# Calculate momentum (val)
highest_high = df['high'].rolling(window=lengthKC).max()
lowest_low = df['low'].rolling(window=lengthKC).min()
avg_hl = (highest_high + lowest_low) / 2
sma_close = talib.SMA(source, timeperiod=lengthKC)
avg_combined = (avg_hl + sma_close) / 2
diff = source - avg_combined
val = talib.LINEARREG(diff, timeperiod=lengthKC)
val = val * 10000  # Normalize as in Pine Script

# Additional Indicators
sma_20 = talib.SMA(source, timeperiod=20)  # 20-period SMA
tr_series = tr  # True Range (ATR)
rsi = talib.RSI(source, timeperiod=14)  # 14-period RSI
macd, macd_signal, macd_hist = talib.MACD(source, fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)  # 14-period ADX

# Prepare features
features = pd.DataFrame({
    'open': df['open'],
    'high': df['high'],
    'low': df['low'],
    'close': df['close'],
    'smi': val,
    'tr': tr_series,
    'sma_20': sma_20,
    'rsi': rsi,
    'macd': macd,
    'macd_signal': macd_signal,
    'macd_hist': macd_hist,
    'adx': adx,
    'sqzOn': sqzOn.astype(int),
    'sqzOff': sqzOff.astype(int),
    'noSqz': noSqz.astype(int)
}).dropna()

# Add previous 50 price and indicator points as features
lookback = 50
for i in range(1, lookback + 1):
    features[f'close_lag_{i}'] = features['close'].shift(i)
    features[f'smi_lag_{i}'] = features['smi'].shift(i)
    features[f'tr_lag_{i}'] = features['tr'].shift(i)
    features[f'sma_20_lag_{i}'] = features['sma_20'].shift(i)
    features[f'rsi_lag_{i}'] = features['rsi'].shift(i)
    features[f'macd_lag_{i}'] = features['macd'].shift(i)
    features[f'adx_lag_{i}'] = features['adx'].shift(i)

features = features.dropna()

# Define labels: Buy (1), Sell (-1), Sideways (0)
price_change = (features['close'].shift(-1) - features['close']) / features['close'] * 100
threshold = 0.3  # Reduced to 0.3%
labels = pd.Series(0, index=price_change.index)  # Default to sideways
labels[price_change > threshold] = 1  # Buy
labels[price_change < -threshold] = -1  # Sell

# Align features and labels
features = features.iloc[:-1]  # Remove last row (no label for it)
labels = labels.iloc[:-1]
df_aligned = df.loc[features.index]  # Align df with features for timestamps

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Train neural network
model = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=1000, random_state=42, learning_rate_init=0.001)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Generate signals on test data
predictions = model.predict(X_test)
test_indices = features.index[-len(X_test):]  # Indices for test data
signals = pd.DataFrame({
    'timestamp': df_aligned.loc[test_indices, 'timestamp'],
    'close': features.loc[test_indices, 'close'],
    'signal': predictions
})
signals['signal_text'] = signals['signal'].map({1: 'Buy', -1: 'Sell', 0: 'Sideways'})
print("\nRecent Signals:")
print(signals.tail(10)[['timestamp', 'close', 'signal_text']])