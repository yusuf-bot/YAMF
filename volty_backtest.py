from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
# Apply nest_asyncio to allow running the event loop
load_dotenv()

# Get API keys from environment variables
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

data_client = StockHistoricalDataClient(api_key, secret_key)

# Strategy parameters
length = 1  # SMA length for TR
num_atrs = 1  # ATR multiplier
initial_capital = 100
symbol = "AMD"  # Stock symbol format

# Get historical data
end = datetime(2025, 4, 15)
start = end - timedelta(days=31)  # 1 year of data

request_params = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Minute,
    start=start,
    end=end
)

# Get data
bars = data_client.get_stock_bars(request_params)
df = bars.df.reset_index()

# Calculate True Range and ATR
df['high_low'] = df['high'] - df['low']
df['high_close'] = abs(df['high'] - df['close'].shift(1))
df['low_close'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_sma'] = df['tr'].rolling(window=length).mean()
df['atrs'] = df['atr_sma'] * num_atrs

# Generate signals
df['long_entry_price'] = df['close'].shift(1) + df['atrs'].shift(1)
df['short_entry_price'] = df['close'].shift(1) - df['atrs'].shift(1)

# Initialize columns for backtesting
df['position'] = 0
df['shares'] = 0
df['long_triggered'] = False
df['short_triggered'] = False
df['equity'] = float(initial_capital)
df['trade_type'] = ''
df['entry_price'] = 0.0
df['exit_price'] = 0.0
df['profit_loss'] = 0.0

# Backtest the strategy
position = 0
shares = 0
equity = float(initial_capital)
entry_price = 0
trade_entry_time = None

for i in range(length + 1, len(df)):
    current_price = df['close'].iloc[i]
    prev_price = df['close'].iloc[i-1]
    
    # Check if we have a valid setup
    if not pd.isna(df['close'].iloc[i-length]):
        long_entry = df['long_entry_price'].iloc[i-1]
        short_entry = df['short_entry_price'].iloc[i-1]
        
        # Check for long entry
        if position <= 0 and current_price >= long_entry:
            # Close any existing short position
            if position < 0:
                profit = (entry_price - prev_price) * abs(shares)
                equity += profit
                # Record exit for short position
                df.loc[df.index[i], 'exit_price'] = prev_price
                df.loc[df.index[i], 'profit_loss'] = profit
            
            # Enter long position
            shares = 1
            position = 1
            entry_price = long_entry
            trade_entry_time = df['timestamp'].iloc[i]
            df.loc[df.index[i], 'long_triggered'] = True
            df.loc[df.index[i], 'trade_type'] = 'LONG'
            df.loc[df.index[i], 'entry_price'] = entry_price
        
        # Check for short entry
        elif position >= 0 and current_price <= short_entry:
            # Close any existing long position
            if position > 0:
                profit = (prev_price - entry_price) * shares
                equity += profit
                # Record exit for long position
                df.loc[df.index[i], 'exit_price'] = prev_price
                df.loc[df.index[i], 'profit_loss'] = profit
            
            # Enter short position
            shares = 1  # Fixed: use short_entry instead of long_entry
            position = -1
            entry_price = short_entry
            trade_entry_time = df['timestamp'].iloc[i]
            df.loc[df.index[i], 'short_triggered'] = True
            df.loc[df.index[i], 'trade_type'] = 'SHORT'
            df.loc[df.index[i], 'entry_price'] = entry_price
    
    # Update position and equity
    df.loc[df.index[i], 'position'] = position
    df.loc[df.index[i], 'shares'] = shares
    
    # Calculate equity (mark-to-market)
    if position > 0:
        unrealized_profit = (current_price - entry_price) * shares
    elif position < 0:
        unrealized_profit = (entry_price - current_price) * shares
    else:
        unrealized_profit = 0
    
    df.loc[df.index[i], 'equity'] = float(equity + unrealized_profit)

# Create a trades dataframe
trades_df = df[(df['long_triggered'] == True) | (df['short_triggered'] == True) | 
               ((df['exit_price'] != 0) & ~pd.isna(df['exit_price']))].copy()

trades_df = trades_df[['timestamp', 'trade_type', 'entry_price', 'exit_price', 
                       'shares', 'profit_loss']].copy()

# Calculate strategy metrics
total_trades = df['long_triggered'].sum() + df['short_triggered'].sum()
long_trades = df['long_triggered'].sum()
short_trades = df['short_triggered'].sum()

final_equity = df['equity'].iloc[-1]
total_return = (final_equity - initial_capital) / initial_capital * 100
max_drawdown = (df['equity'].cummax() - df['equity']).max() / df['equity'].cummax().max() * 100

# Plot results
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

# Price and entry levels
ax1.plot(df['timestamp'], df['close'], label=f'{symbol}', color='white', alpha=0.8)
ax1.plot(df['timestamp'], df['long_entry_price'], label='Long Entry', color='green', alpha=0.5, linestyle='--')
ax1.plot(df['timestamp'], df['short_entry_price'], label='Short Entry', color='red', alpha=0.5, linestyle='--')

# Plot entry points
long_entries = df[df['long_triggered']]
short_entries = df[df['short_triggered']]
ax1.scatter(long_entries['timestamp'], long_entries['long_entry_price'], marker='^', color='green', s=100, label='Long Entry')
ax1.scatter(short_entries['timestamp'], short_entries['short_entry_price'], marker='v', color='red', s=100, label='Short Entry')

ax1.set_title(f'{symbol} with Volatility Expansion Close Strategy (Length: {length}, ATR Mult: {num_atrs})', fontsize=14)
ax1.set_ylabel('Price ($)', fontsize=12)
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
    f"Long Trades: {long_trades}\n"
    f"Short Trades: {short_trades}\n"
    f"Max Drawdown: {max_drawdown:.2f}%"
)

# Add text box with metrics
props = dict(boxstyle='round', facecolor='black', alpha=0.5)
ax2.text(0.02, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Create results directory if it doesn't exist
results_dir = os.path.join('c:\\Users\\ASAA\\Desktop\\projects\\vibe', 'backtest_results')
os.makedirs(results_dir, exist_ok=True)

# Save results as JPG
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'volty_expan_{symbol}_results.jpg'), format='jpg', dpi=300)

# Save trades to CSV
trades_csv_path = os.path.join(results_dir, f'volty_expan_{symbol}_trades.csv')
trades_df.to_csv(trades_csv_path, index=False)

# Save detailed results to CSV
df.to_csv(os.path.join(results_dir, f'volty_expan_{symbol}_data.csv'))

# Save summary results
summary = {
    'symbol': symbol,
    'length': length,
    'num_atrs': num_atrs,
    'initial_capital': initial_capital,
    'final_equity': final_equity,
    'total_return': total_return,
    'total_trades': total_trades,
    'long_trades': long_trades,
    'short_trades': short_trades,
    'max_drawdown': max_drawdown,
    'start_date': start.strftime('%Y-%m-%d'),
    'end_date': end.strftime('%Y-%m-%d')
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(results_dir, f'volty_expan_{symbol}_summary.csv'), index=False)

# Print summary
print("\nVolatility Expansion Close Strategy Results:")
print(f"Symbol: {symbol}")
print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
print(f"Initial Capital: ${initial_capital}")
print(f"Final Equity: ${final_equity:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Total Trades: {total_trades}")
print(f"Long Trades: {long_trades}")
print(f"Short Trades: {short_trades}")
print(f"Max Drawdown: {max_drawdown:.2f}%")

print(f"\nResults saved to {results_dir}")
print(f"Trades CSV saved to: {trades_csv_path}")