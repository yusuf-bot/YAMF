from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Initialize clients
api_key = 'PKYXKXZCBA155CHGFHJ2'
secret_key = "fZoyUsSQ6SdN7w4pipiGJidwmd9MT4EPUcjXjPk9"
crypto_client = StockHistoricalDataClient(api_key, secret_key)

# Strategy parameters
len1 = 10  # First SMA length
len2 = 20  # Second SMA length
tp_pct = 0.5 / 100  # Take profit percentage
sl_pct = 0.2 / 100  # Stop loss percentage
initial_capital = 1000
quantity = 1  # Quantity to trade

# Get historical data
end = datetime(2025, 4, 14)
start = end - timedelta(days=22)  # 1 year of data

request_params = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Minute,
    start=start,
    end=end
)

# Get data
bars = crypto_client.get_stock_bars(request_params)
df = bars.df.reset_index()
print("DataFrame columns:", df.columns.tolist())
print("DataFrame sample:", df.head())

# Calculate indicators
df['SMA1'] = df['close'].rolling(window=len1).mean()
df['SMA2'] = df['close'].rolling(window=len2).mean()

# Generate signals
df['buy_signal'] = False
df['sell_signal'] = False

# Crossover/Crossunder logic
for i in range(1, len(df)):
    # Crossover (SMA1 crosses above SMA2)
    if df['SMA1'].iloc[i-1] <= df['SMA2'].iloc[i-1] and df['SMA1'].iloc[i] > df['SMA2'].iloc[i]:
        df.loc[df.index[i], 'buy_signal'] = True
    
    # Crossunder (SMA1 crosses below SMA2)
    if df['SMA1'].iloc[i-1] >= df['SMA2'].iloc[i-1] and df['SMA1'].iloc[i] < df['SMA2'].iloc[i]:
        df.loc[df.index[i], 'sell_signal'] = True

# Backtest the strategy
df['position'] = 0  # 1 for long, -1 for short, 0 for no position
df['entry_price'] = np.nan
df['take_profit'] = np.nan
df['stop_loss'] = np.nan
df['trade_result'] = np.nan
df['equity'] = initial_capital

current_position = 0
open_trades = 0
entry_price = 0
take_profit_level = 0
stop_loss_level = 0

for i in range(len(df)):
    # Skip until we have both SMAs
    if pd.isna(df['SMA1'].iloc[i]) or pd.isna(df['SMA2'].iloc[i]):
        continue
    
    # Check for exit conditions if we have an open position
    if current_position != 0:
        current_price = df['close'].iloc[i]
        
        # Check for take profit or stop loss
        if current_position == 1:  # Long position
            if current_price >= take_profit_level or current_price <= stop_loss_level:
                # Close the position
                trade_result = (current_price - entry_price) / entry_price
                df.loc[df.index[i], 'trade_result'] = trade_result
                df.loc[df.index[i], 'equity'] = df['equity'].iloc[i-1] * (1 + trade_result * current_position)
                current_position = 0
                open_trades = 0
            else:
                df.loc[df.index[i], 'equity'] = df['equity'].iloc[i-1]
        
        elif current_position == -1:  # Short position
            if current_price <= take_profit_level or current_price >= stop_loss_level:
                # Close the position
                trade_result = (entry_price - current_price) / entry_price
                df.loc[df.index[i], 'trade_result'] = trade_result
                df.loc[df.index[i], 'equity'] = df['equity'].iloc[i-1] * (1 + trade_result * abs(current_position))
                current_position = 0
                open_trades = 0
            else:
                df.loc[df.index[i], 'equity'] = df['equity'].iloc[i-1]
    else:
        df.loc[df.index[i], 'equity'] = df['equity'].iloc[i-1] if i > 0 else initial_capital
    
    # Check for entry signals if we don't have an open position
    if open_trades == 0:
        if df['buy_signal'].iloc[i]:
            # Enter long position
            current_position = 1
            open_trades = 1
            entry_price = df['close'].iloc[i]
            take_profit_level = entry_price * (1 + tp_pct)
            stop_loss_level = entry_price * (1 - sl_pct)
            
            df.loc[df.index[i], 'position'] = current_position
            df.loc[df.index[i], 'entry_price'] = entry_price
            df.loc[df.index[i], 'take_profit'] = take_profit_level
            df.loc[df.index[i], 'stop_loss'] = stop_loss_level
        
        elif df['sell_signal'].iloc[i]:
            # Enter short position
            current_position = -1
            open_trades = 1
            entry_price = df['close'].iloc[i]
            take_profit_level = entry_price * (1 - tp_pct)
            stop_loss_level = entry_price * (1 + sl_pct)
            
            df.loc[df.index[i], 'position'] = current_position
            df.loc[df.index[i], 'entry_price'] = entry_price
            df.loc[df.index[i], 'take_profit'] = take_profit_level
            df.loc[df.index[i], 'stop_loss'] = stop_loss_level

# Calculate strategy metrics
total_trades = df['trade_result'].count()
winning_trades = df[df['trade_result'] > 0]['trade_result'].count()
losing_trades = df[df['trade_result'] < 0]['trade_result'].count()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

final_equity = df['equity'].iloc[-1]
total_return = (final_equity - initial_capital) / initial_capital * 100
max_drawdown = (df['equity'].cummax() - df['equity']).max() / df['equity'].cummax().max() * 100

# Plot results
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

# Price and SMAs
ax1.plot(df['timestamp'], df['close'], label='BTC/USD', color='white', alpha=0.8)
ax1.plot(df['timestamp'], df['SMA1'], label=f'SMA {len1}', color='red')
ax1.plot(df['timestamp'], df['SMA2'], label=f'SMA {len2}', color='blue')

# Plot buy and sell signals
buy_signals = df[df['buy_signal']]
sell_signals = df[df['sell_signal']]
ax1.scatter(buy_signals['timestamp'], buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
ax1.scatter(sell_signals['timestamp'], sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')

ax1.set_title(f'BTC/USD with SMA Crossover Strategy (SMA {len1} & SMA {len2})', fontsize=14)
ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# Equity curve
ax2.plot(df['timestamp'], df['equity'], color='green', linewidth=1.5)
ax2.set_title('Strategy Equity Curve', fontsize=14)
ax2.set_ylabel('Equity ($)', fontsize=12)
ax2.grid(alpha=0.3)

# Format x-axis dates
fig.autofmt_xdate()
date_format = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(date_format)

# Add strategy metrics as text
metrics_text = (
    f"Initial Capital: ${initial_capital}\n"
    f"Final Equity: ${final_equity:.2f}\n"
    f"Total Return: {total_return:.2f}%\n"
    f"Total Trades: {total_trades}\n"
    f"Win Rate: {win_rate:.2f}%\n"
    f"Max Drawdown: {max_drawdown:.2f}%"
)

# Add text box with metrics
props = dict(boxstyle='round', facecolor='black', alpha=0.5)
ax2.text(0.02, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('c:\\Users\\ASAA\\Desktop\\projects\\vibe\\sma_strategy_results.png')
plt.show()

# Save results to CSV
df.to_csv('c:\\Users\\ASAA\\Desktop\\projects\\vibe\\sma_strategy_data.csv')

# Print summary
print("\nStrategy Results:")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Equity: ${final_equity:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")

print(f"\nChart saved to c:\\Users\\ASAA\\Desktop\\projects\\vibe\\sma_strategy_results.png")
print(f"Data exported to c:\\Users\\ASAA\\Desktop\\projects\\vibe\\sma_strategy_data.csv")