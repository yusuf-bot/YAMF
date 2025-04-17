import pandas as pd
import numpy as np
from datetime import datetime

def read_ohlcv_from_csv(file_path):
    df = pd.read_csv(file_path)
    df.set_index('timestamp', inplace=True)
    return df
    

def calculate_atr(df, length, num_atrs):
    df_result = df.copy()
    
    # Calculate the True Range components
    df_result['high_low'] = df_result['high'] - df_result['low']
    df_result['high_close'] = abs(df_result['high'] - df_result['close'].shift(1))
    df_result['low_close'] = abs(df_result['low'] - df_result['close'].shift(1))
    
    # Calculate True Range (maximum of the three components)
    df_result['tr'] = df_result[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # Calculate Simple Moving Average of True Range
    df_result['atr_sma'] = df_result['tr'].rolling(window=length).mean()
    
    # Multiply ATR SMA by the specified factor
    df_result['atrs'] = df_result['atr_sma'] * num_atrs
    
    # Drop temporary columns used for calculation
    df_result.drop(['high_low', 'high_close', 'low_close'], axis=1, inplace=True)
    
    return df_result

def generate_signals(df, length, initial_capital=1000):
    df_signals = df.copy()
    
    # lose[1]+atrs and close[1]-atrs
    df_signals['long_entry_price'] = df_signals['close'].shift(1) + df_signals['atrs']
    df_signals['short_entry_price'] = df_signals['close'].shift(1) - df_signals['atrs']

    # qty= 1000/close[1]
    df_signals['position_size'] = initial_capital / df_signals['close'].shift(1)
    
    # if (not na(close[length]))
    df_signals['valid_signal'] = ~df_signals['close'].shift(length).isna()
    
    return df_signals


def backtest_strategy(df_signals,length):
    bt = df_signals.copy()
    
    # Initialize position and equity columns
    bt['position'] = 0
    bt['equity'] = 0
    bt['cash'] = 1000  # Initial capital
    
    current_position = 0
    
    # Loop through each row to simulate trading
    for i in range(length, len(bt)):
        # Skip if no valid signal
        if not bt.loc[bt.index[i], 'valid_signal']:
            bt.loc[bt.index[i], 'position'] = current_position
            continue
            
        price = bt.loc[bt.index[i], 'close']
        long_entry = bt.loc[bt.index[i], 'long_entry_price']
        short_entry = bt.loc[bt.index[i], 'short_entry_price']
        position_size = bt.loc[bt.index[i], 'position_size']
        
        # Check for entry conditions
        if price >= long_entry and current_position <= 0:
            # Close short position if exists
            if current_position < 0:
                # Close short position
                bt.loc[bt.index[i], 'cash'] -= abs(current_position) * price
                
            # Enter long position
            current_position = position_size
            bt.loc[bt.index[i], 'cash'] -= position_size * price
            
        elif price <= short_entry and current_position >= 0:
            # Close long position if exists
            if current_position > 0:
                # Close long position
                bt.loc[bt.index[i], 'cash'] += current_position * price
                
            # Enter short position
            current_position = -position_size
            bt.loc[bt.index[i], 'cash'] += position_size * price
            
        # Update position and equity
        bt.loc[bt.index[i], 'position'] = current_position
        bt.loc[bt.index[i], 'equity'] = bt.loc[bt.index[i], 'cash'] + (current_position * price)
    
    return bt



df = read_ohlcv_from_csv('btcusdt.csv')
df_result=calculate_atr(df, 1, 1)
df_signals=generate_signals(df_result, 1)
result=backtest_strategy(df_signals,1)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(result['equity'])
plt.title('Equity Curve - Volatility Expansion Close Strategy')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.show()