import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
import datetime

# Step 1: Fetch historical data from Binance
symbol = 'PEPEUSDT'
timeframe = '1m'  # 1-minute candles
total_limit = 8000  # Total number of candles to fetch
binance_limit = 1000  # Binance API limit per request

def get_binance_data(symbol='PEPEUSDT', timeframe='1m', days=1):
    exchange = ccxt.binance()
    
    # Calculate timestamps
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Convert to milliseconds timestamp
    since = int(start_date.timestamp() * 1000)
    
    # Fetch OHLCV data in batches
    all_candles = []
    while since < end_date.timestamp() * 1000:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since)
        if len(candles) == 0:
            break
        
        since = candles[-1][0] + 1  # Next timestamp after the last received
        all_candles.extend(candles)
    
    # Create DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    print(f"Data fetched. Total candles: {len(df)}")
    
    return df

data = get_binance_data(symbol=symbol, timeframe=timeframe, days=8)
print(data.shape)
print(data.head())

# Step 2: Calculate ATR and breakout levels
lookback = 5  # Lookback period for ATR and previous high/low
atr_factor = 0.75  # ATR multiplier for breakout threshold

# Calculate True Range (TR)
data['high_low'] = data['high'] - data['low']
data['high_close'] = np.abs(data['high'] - data['close'].shift(1))
data['low_close'] = np.abs(data['low'] - data['close'].shift(1))
data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)

# Calculate ATR
data['atr'] = data['tr'].rolling(window=lookback).mean()

# Calculate previous close
data['prev_close'] = data['close'].shift(1)

# Calculate breakout levels
data['long_breakout'] = data['prev_close'] + data['atr'] * atr_factor
data['short_breakout'] = data['prev_close'] - data['atr'] * atr_factor
data = data.dropna()
data = data[['open', 'high', 'low', 'close', 'volume', 'atr', 'long_breakout', 'short_breakout']].astype(float)
data_list = data.to_numpy().tolist()
print(data_list[0])

# Step 3: Calculate PnL
initial_capital = 100
capital = initial_capital
position = 0  # 1 for long, -1 for short, 0 for neutral
entry_price = 0
pnl = 0

for i in range(len(data_list)):
    open_price = data_list[i][0]
    high_price = data_list[i][1]
    low_price = data_list[i][2]
    close_price = data_list[i][3]
    long_breakout = data_list[i][6]
    short_breakout = data_list[i][7]

    # Detect breakout conditions
    long_condition = high_price > long_breakout and low_price < long_breakout
    short_condition = high_price > short_breakout and low_price < short_breakout

    # Resolve conflict if both conditions are true
    if long_condition and short_condition:
        if abs(open_price - long_breakout) < abs(open_price - short_breakout):
            short_condition = False
        else:
            long_condition = False

    # Close existing position before entering a new one
    if position == 1 and short_condition:  # Close long, enter short
        exit_price = short_breakout
        position_size = capital / entry_price  # How many units you bought
        pnl += (exit_price - entry_price) * position_size
        capital = initial_capital + pnl
        position = -1
        entry_price = short_breakout
    elif position == -1 and long_condition:  # Close short, enter long
        exit_price = long_breakout
        position_size = capital / entry_price  # How many units you bought
        pnl += (exit_price - entry_price) * position_size  # Short profit: (entry - exit)
        capital = initial_capital + pnl
        position = 1                                                       
        entry_price = long_breakout
    # Enter new position if none exists
    elif position == 0:
        if long_condition:                                          
            position = 1
            entry_price = long_breakout
        elif short_condition:
            position = -1
            entry_price = short_breakout

# Close final position at the last close price
if position != 0:
    exit_price = data_list[-1][3]  # Last close price
    if position == 1:  # Long position
        position_size = capital / entry_price  # How many units you bought
        pnl += (exit_price - entry_price) * position_size
    elif position == -1:  # Short position
        position_size = capital / entry_price  # How many units you bought
        pnl += (exit_price - entry_price) * position_size
    capital = initial_capital + pnl

print("Final PnL:", pnl)
print("Final Capital:", capital)