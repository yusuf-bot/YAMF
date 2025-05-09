import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
from sklearn.linear_model import LinearRegression
import datetime

# Step 1: Fetch historical data from Binance
symbol = 'PEPEUSDT'
timeframe = '5m'  # 1-minute candles
total_limit = 8000  # Total number of candles to fetch
binance_limit = 1000  # Binance API limit per request
print('v 0.1.6')
def get_binance_data(symbol='PEPEUSDT', timeframe='5m', days=7):
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

data = get_binance_data(symbol=symbol, timeframe=timeframe, days=21)
print(data.shape)
print(data.head())


# Calculate True Range (TR)
data['hl2'] = (data['high'] + data['low']) / 2
data['hi'] = data['high'].rolling(7).max()

# Calculate lowest price over last 'period' candles
data['lo'] = data['low'].rolling(7).min()

# Calculate SMA of hl2 over last 'period' candles
data['av'] = data['hl2'].rolling(7).mean()
data['avg_hla'] = (data['hi'] + data['lo'] + data['av']) / 3

# Calculate the oscillator raw value
data['osc_raw'] = (data['close'] - data['avg_hla']) / (data['hi'] - data['lo']) * 100
data.dropna(inplace=True)
# Apply linear regression for each window of size linreg_period
data['sig_linreg'] = np.nan

for i in range(7, len(data) + 1):
    # Get window of data
    window = data['osc_raw'].iloc[i-7:i].values
    
    # Create X as indices (0 to linreg_period-1)
    X = np.arange(7).reshape(-1, 1)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, window)
    
    # Get the predicted value at the last position (equivalent to offset 0)
    data.loc[data.index[i-1], 'sig_linreg'] = model.predict(np.array([[6]]))[0]


# Apply EMA to the linreg result
data['sig'] = data['sig_linreg'].ewm(span=3, adjust=False).mean()
data['sgD'] = data['sig'].rolling(2).mean()
data['entry'] = np.minimum(data['sig'], data['sgD']) < -40
data['exit'] = np.maximum(data['sig'], data['sgD']) > 40


data['sig_prev'] = data['sig'].shift(1)
data['sgD_prev'] = data['sgD'].shift(1)
data['dot'] = ((data['sig'] > data['sgD']) & (data['sig_prev'] <= data['sgD_prev'])) 


final_data = data[['open', 'high', 'low', 'close', 'volume', 'sig', 'sgD','entry','exit','dot']].astype(float)
final_data.dropna(inplace=True)
data_list = data.to_numpy().tolist()
print(data_list[0])

# Step 2: Backtesting logic (modified for margin style)

initial_capital = 100
capital = initial_capital

order_fraction = 0.5  # 50% of total equity per trade
capital_growth = [capital]
total=0
position_size = 0
entry_prices = []
entry_amounts = []  # Track how much was invested per trade
max_pyramiding = 2
tp = 0.18  # 6% Take Profit
sl = 0.09  # 5% Stop Loss (corrected to 5%, you had 0.5 before)
commission = 0.0001  # 0.01% per trade

for idx, row in final_data.iterrows():
    open_price = row['open']
    high_price = row['high']
    low_price = row['low']
    close_price = row['close']
    long_condition = row['dot'] and row['entry']
    exit_condition = row['dot'] and row['exit']

    # Check for exits first
    if position_size > 0:
        new_entry_prices = []
        new_entry_amounts = []
        for entry_price, amount_invested in zip(entry_prices, entry_amounts):
            tp_price = entry_price * (1 + tp)
            sl_price = entry_price * (1 - sl)

            if high_price >= tp_price:
                # Take Profit hit
                profit = (tp_price - entry_price) / entry_price
                profit_cash = amount_invested * (1 + profit - commission)
                capital += (profit_cash - amount_invested)  # Add net profit
                position_size -= 1
                print(f"[{idx}] TAKE PROFIT at {tp_price} (Entry {entry_price}) -> Capital: {capital}")
            elif low_price <= sl_price:
                # Stop Loss hit
                loss = (sl_price - entry_price) / entry_price
                loss_cash = amount_invested * (1 + loss - commission)
                capital += (loss_cash - amount_invested)  # Add net loss
                position_size -= 1
                print(f"[{idx}] STOP LOSS at {sl_price} (Entry {entry_price}) -> Capital: {capital}")
            elif exit_condition==1:
                # Manual exit
                profit = (close_price - entry_price) / entry_price
                profit_cash = amount_invested * (1 + profit - commission)
                capital += (profit_cash - amount_invested)
                position_size -= 1
                print(f"[{idx}] MANUAL EXIT at {close_price} (Entry {entry_price}) -> Capital: {capital}")
            else:
                new_entry_prices.append(entry_price)
                new_entry_amounts.append(amount_invested)

        entry_prices = new_entry_prices
        entry_amounts = new_entry_amounts

    # New entry
    if long_condition:
        if position_size < max_pyramiding:
            trade_amount = capital * order_fraction

            if trade_amount > 0:
                entry_prices.append(open_price)
                entry_amounts.append(trade_amount)
                position_size += 1
                total+=1
                # No need to deduct available cash separately, it's margin trading
                print(f"[{idx}] ENTRY at {open_price} -> Trade Amount: {trade_amount} -> Capital: {capital:.2f} (Position size: {position_size})")

    capital_growth.append(capital)

print("Final Capital:", capital)
print("Total PnL (%):", ((capital - initial_capital) / initial_capital) * 100)
print(total)
# Plot Capital Growth
plt.plot(capital_growth)
plt.title('Capital Growth')
plt.xlabel('Trade')
plt.ylabel('Capital')
plt.grid(True)
plt.show()
