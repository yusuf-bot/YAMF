import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Binance client with API keys from environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

# Strategy parameters
SYMBOL = 'BTCUSDT'  # Trading pair
TIMEFRAME = Client.KLINE_INTERVAL_1MINUTE  # Timeframe (1h)
TREND_PERIOD = 200  # Trend period
ATR_PERIOD = 14  # ATR period
INITIAL_CAPITAL = 100  # Initial capital
POSITION_SIZE = 0.15  # Fraction of capital to use per trade

# Function to calculate ATR
def calculate_atr(high, low, close, period):
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# Function to fetch historical data from Binance
def fetch_historical_data(symbol, interval, lookback_days):
    end_time = datetime.now()
    # Convert lookback to days
    start_time = end_time - timedelta(days=lookback_days)
    
    # Convert to timestamp in milliseconds
    end_time_ms = int(end_time.timestamp() * 1000)
    start_time_ms = int(start_time.timestamp() * 1000)
    
    print(f"Fetching data from {start_time} to {end_time}")
    
    # Fetch historical klines
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=str(start_time_ms),
        end_str=str(end_time_ms)
    )
    
    # Create DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert relevant columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"Fetched {len(df)} bars of historical data")
    return df

# Function to calculate HLC3
def calculate_hlc3(df):
    return (df['high'] + df['low'] + df['close']) / 3

# Function to implement the Q-Trend Strategy
def q_trend_strategy(df, trend_period, atr_period):
    # Calculate HLC3
    df['hlc3'] = calculate_hlc3(df)
    
    # Calculate highest and lowest over the trend period (from the previous bar)
    df['highest'] = df['hlc3'].shift(1).rolling(window=trend_period).max()
    df['lowest'] = df['hlc3'].shift(1).rolling(window=trend_period).min()
    
    # Calculate middle point
    df['middle'] = (df['highest'] + df['lowest']) / 2
    
    # Calculate ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    df['prev_middle'] = df['middle'].shift(1)
    df['prev_atr'] = df['atr'].shift(1)
    # Calculate entry conditions
    df['long_signal'] = df['hlc3'].shift(1) > (df['prev_middle'] + df['prev_atr'])
    df['short_signal'] = df['hlc3'].shift(1) < (df['prev_middle'] - df['prev_atr'])
    df=df.dropna()
    # Initialize position and equity columns
    df['position'] = 0
    df['equity'] = INITIAL_CAPITAL
    df['trail_stop'] = 0.0
    df['entry_price'] = 0.0
    df['trade_id'] = 0
    
    # Track open positions and PnL
    current_position = 0
    entry_price = 0
    trail_stop = 0
    trade_id = 0
    
    # Implement trading logic
    for i in range(trend_period + atr_period, len(df)):
        current_price = df['close'].iloc[i]
        
        # Update trailing stop if in a position
        if current_position == 1:  # Long position
            new_trail_stop = current_price - df['atr'].iloc[i] * 0.5
            trail_stop = max(trail_stop, new_trail_stop)
            
            # Check if trail stop hit
            if current_price < trail_stop:
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_id'] = 0
                current_position = 0
                
        elif current_position == -1:  # Short position
            new_trail_stop = current_price + df['atr'].iloc[i] * 0.5
            trail_stop = min(trail_stop, new_trail_stop) if trail_stop != 0 else new_trail_stop
            
            # Check if trail stop hit
            if current_price > trail_stop:
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_id'] = 0
                current_position = 0
        
        # Check for entry signals if not in a position
        if current_position == 0:
            if df['long_signal'].iloc[i] and not (current_position > 0):
                df.loc[df.index[i], 'position'] = 1
                current_position = 1
                entry_price = current_price
                trail_stop = current_price - df['atr'].iloc[i]
                trade_id += 1
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'trade_id'] = trade_id
                
            elif df['short_signal'].iloc[i] and not (current_position < 0):
                df.loc[df.index[i], 'position'] = -1
                current_position = -1
                entry_price = current_price
                trail_stop = current_price + df['atr'].iloc[i]
                trade_id += 1
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'trade_id'] = trade_id
        
        # Record trail stop level
        df.loc[df.index[i], 'trail_stop'] = trail_stop
    
    return df

# Function to calculate equity and performance metrics
def calculate_performance(df):
    # Calculate returns based on position
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    
    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    df['strategy_cumulative_returns'] = (1 + df['strategy_returns']).cumprod().fillna(1)
    
    # Calculate equity
    df['equity'] = INITIAL_CAPITAL * df['strategy_cumulative_returns']
    
    # Calculate drawdown
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    return df

# Function to analyze trade statistics
def analyze_trades(df):
    # Find trade entries and exits - this is where the issue is
    df['trade_entry'] = df['trade_id'] != df['trade_id'].shift(1)
    df['trade_exit'] = (df['trade_id'] != df['trade_id'].shift(-1)) & (df['trade_id'] != 0)
    
    # Create lists to store trade data
    trade_list = []
    
    # Loop through the dataframe to extract trade details
    current_trade_id = 0
    entry_price = 0
    entry_time = None
    position_type = 0
    
    for i in range(len(df)):
        # New trade entry
        if df['trade_entry'].iloc[i] and df['trade_id'].iloc[i] != 0:
            current_trade_id = df['trade_id'].iloc[i]
            entry_price = df['close'].iloc[i]
            entry_time = df.index[i]
            position_type = df['position'].iloc[i]
        
        # Trade exit
        elif df['trade_exit'].iloc[i] and current_trade_id != 0:
            exit_price = df['close'].iloc[i]
            exit_time = df.index[i]
            
            # Calculate trade metrics
            pnl = 0
            if position_type == 1:  # Long
                pnl = (exit_price - entry_price) / entry_price
            elif position_type == -1:  # Short
                pnl = (entry_price - exit_price) / entry_price
            
            duration = (exit_time - entry_time).total_seconds() / 3600  # hours
            
            # Add trade to list
            trade_list.append({
                'trade_id': current_trade_id,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position_type': 'LONG' if position_type == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'duration': duration
            })
            
            # Reset trade tracking
            current_trade_id = 0
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trade_list)
    
    # Debug information to help identify the issue
    print(f"Total position changes: {(df['position'] != df['position'].shift(1)).sum()}")
    print(f"Total trade entries identified: {df['trade_entry'].sum()}")
    print(f"Total trade exits identified: {df['trade_exit'].sum()}")
    print(f"Total trades recorded: {len(trade_list)}")
    
    if len(trades_df) > 0:
        # Calculate trade statistics
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] <= 0]
        
        stats = {
            'total_trades': len(trades_df),
            'win_rate': len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': win_trades['pnl'].mean() if len(win_trades) > 0 else 0,
            'avg_loss': loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0,
            'profit_factor': abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0 else float('inf'),
            'avg_trade_duration': trades_df['duration'].mean(),
            'long_trades': len(trades_df[trades_df['position_type'] == 'LONG']),
            'short_trades': len(trades_df[trades_df['position_type'] == 'SHORT']),
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min()
        }
        
        # Save trades to CSV
        csv_filename = f"qtrend_trades_{SYMBOL}_{TIMEFRAME}.csv"
        
        # Check if file exists to append or create new
        if os.path.exists(csv_filename):
            # Read existing trades to avoid duplicates
            existing_trades = pd.read_csv(csv_filename)
            
            # Convert string dates back to datetime for comparison
            if 'entry_time' in existing_trades.columns:
                existing_trades['entry_time'] = pd.to_datetime(existing_trades['entry_time'])
                existing_trades['exit_time'] = pd.to_datetime(existing_trades['exit_time'])
            
            # Find new trades not in existing file
            if len(existing_trades) > 0:
                # Create a unique identifier for each trade
                trades_df['trade_key'] = trades_df['entry_time'].astype(str) + "_" + trades_df['exit_time'].astype(str)
                existing_trades['trade_key'] = existing_trades['entry_time'].astype(str) + "_" + existing_trades['exit_time'].astype(str)
                
                # Filter for new trades
                new_trades = trades_df[~trades_df['trade_key'].isin(existing_trades['trade_key'])]
                
                # Drop the temporary key column
                new_trades = new_trades.drop('trade_key', axis=1)
                
                # Append only if there are new trades
                if len(new_trades) > 0:
                    new_trades.to_csv(csv_filename, mode='a', header=False, index=False)
                    print(f"Added {len(new_trades)} new trades to {csv_filename}")
            else:
                # If existing file is empty, just write the new trades
                trades_df.to_csv(csv_filename, index=False)
                print(f"Saved {len(trades_df)} trades to {csv_filename}")
        else:
            # Create new file
            trades_df.to_csv(csv_filename, index=False)
            print(f"Saved {len(trades_df)} trades to {csv_filename}")
        
        return trades_df, stats
    else:
        # Alternative method to identify trades if the first method fails
        # This looks directly at position changes
        alt_trades = []
        in_position = False
        entry_price = 0
        entry_time = None
        position_type = 0
        trade_count = 0
        
        for i in range(1, len(df)):
            # Entry
            if df['position'].iloc[i] != 0 and df['position'].iloc[i-1] == 0:
                in_position = True
                entry_price = df['close'].iloc[i]
                entry_time = df.index[i]
                position_type = df['position'].iloc[i]
                trade_count += 1
            
            # Exit
            elif df['position'].iloc[i] == 0 and df['position'].iloc[i-1] != 0:
                if in_position:
                    exit_price = df['close'].iloc[i]
                    exit_time = df.index[i]
                    
                    # Calculate PnL
                    pnl = 0
                    if position_type == 1:  # Long
                        pnl = (exit_price - entry_price) / entry_price
                    elif position_type == -1:  # Short
                        pnl = (entry_price - exit_price) / entry_price
                    
                    duration = (exit_time - entry_time).total_seconds() / 3600
                    
                    alt_trades.append({
                        'trade_id': trade_count,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'position_type': 'LONG' if position_type == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'duration': duration
                    })
                    
                    in_position = False
        
        print(f"Alternative method found {len(alt_trades)} trades")
        
        if len(alt_trades) > 0:
            alt_trades_df = pd.DataFrame(alt_trades)
            win_trades = alt_trades_df[alt_trades_df['pnl'] > 0]
            loss_trades = alt_trades_df[alt_trades_df['pnl'] <= 0]
            
            stats = {
                'total_trades': len(alt_trades_df),
                'win_rate': len(win_trades) / len(alt_trades_df) if len(alt_trades_df) > 0 else 0,
                'avg_win': win_trades['pnl'].mean() if len(win_trades) > 0 else 0,
                'avg_loss': loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0,
                'profit_factor': abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0 else float('inf'),
                'avg_trade_duration': alt_trades_df['duration'].mean(),
                'long_trades': len(alt_trades_df[alt_trades_df['position_type'] == 'LONG']),
                'short_trades': len(alt_trades_df[alt_trades_df['position_type'] == 'SHORT']),
                'best_trade': alt_trades_df['pnl'].max(),
                'worst_trade': alt_trades_df['pnl'].min()
            }
            
            # Save alternative trades to CSV
            csv_filename = f"qtrend_trades_{SYMBOL}_{TIMEFRAME}.csv"
            
            # Check if file exists to append or create new
            if os.path.exists(csv_filename):
                # Read existing trades to avoid duplicates
                existing_trades = pd.read_csv(csv_filename)
                
                # Convert string dates back to datetime for comparison
                if 'entry_time' in existing_trades.columns:
                    existing_trades['entry_time'] = pd.to_datetime(existing_trades['entry_time'])
                    existing_trades['exit_time'] = pd.to_datetime(existing_trades['exit_time'])
                
                # Find new trades not in existing file
                if len(existing_trades) > 0:
                    # Create a unique identifier for each trade
                    alt_trades_df['trade_key'] = alt_trades_df['entry_time'].astype(str) + "_" + alt_trades_df['exit_time'].astype(str)
                    existing_trades['trade_key'] = existing_trades['entry_time'].astype(str) + "_" + existing_trades['exit_time'].astype(str)
                    
                    # Filter for new trades
                    new_trades = alt_trades_df[~alt_trades_df['trade_key'].isin(existing_trades['trade_key'])]
                    
                    # Drop the temporary key column
                    new_trades = new_trades.drop('trade_key', axis=1)
                    
                    # Append only if there are new trades
                    if len(new_trades) > 0:
                        new_trades.to_csv(csv_filename, mode='a', header=False, index=False)
                        print(f"Added {len(new_trades)} new trades to {csv_filename}")
                else:
                    # If existing file is empty, just write the new trades
                    alt_trades_df.to_csv(csv_filename, index=False)
                    print(f"Saved {len(alt_trades_df)} trades to {csv_filename}")
            else:
                # Create new file
                alt_trades_df.to_csv(csv_filename, index=False)
                print(f"Saved {len(alt_trades_df)} trades to {csv_filename}")
            
            return alt_trades_df, stats
        
        return pd.DataFrame(), {'total_trades': 0}

# Function to visualize backtest results
def plot_backtest_results(df, stats):
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Equity curve
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['equity'])
    plt.title('Equity Curve')
    plt.grid(True)
    
    # Plot 2: Drawdown
    plt.subplot(3, 1, 2)
    plt.fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.3)
    plt.title('Drawdown (%)')
    plt.grid(True)
    
    # Plot 3: Price with buys and sells
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['close'], color='black', alpha=0.5)
    
    # Plot middle line and ATR bands
    plt.plot(df.index, df['middle'], color='blue', alpha=0.5)
    plt.plot(df.index, df['middle'] + df['atr'], color='green', alpha=0.5, linestyle='--')
    plt.plot(df.index, df['middle'] - df['atr'], color='red', alpha=0.5, linestyle='--')
    
    # Plot buy and sell points
    buy_signals = df[df['position'].diff() == 1]
    sell_signals = df[df['position'].diff() == -2]  # From 1 to -1
    short_signals = df[df['position'].diff() == -1]
    cover_signals = df[df['position'].diff() == 2]  # From -1 to 1
    exit_signals = df[(df['position'] == 0) & (df['position'].shift(1) != 0)]
    
    plt.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', alpha=1, s=100)
    plt.scatter(short_signals.index, short_signals['close'], color='red', marker='v', alpha=1, s=100)
    plt.scatter(exit_signals.index, exit_signals['close'], color='black', marker='x', alpha=1, s=100)
    
    plt.title('Price with Entries and Exits')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Display statistics
    stats_text = f"""
    Total Trades: {stats.get('total_trades', 0)}
    Win Rate: {stats.get('win_rate', 0)*100:.2f}%
    Profit Factor: {stats.get('profit_factor', 0):.2f}
    Avg Win: {stats.get('avg_win', 0)*100:.2f}%
    Avg Loss: {stats.get('avg_loss', 0)*100:.2f}%
    Best Trade: {stats.get('best_trade', 0)*100:.2f}%
    Worst Trade: {stats.get('worst_trade', 0)*100:.2f}%
    Avg Duration: {stats.get('avg_trade_duration', 0):.2f} hours
    Final Equity: ${df['equity'].iloc[-1]:,.2f}
    Max Drawdown: {df['drawdown'].min()*100:.2f}%
    Total Return: {(df['equity'].iloc[-1]/INITIAL_CAPITAL-1)*100:.2f}%
    Sharpe Ratio: {np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std() if df['strategy_returns'].std() != 0 else 0:.2f}
    """
    
    plt.figtext(0.01, 0.01, stats_text, fontsize=10)
    plt.savefig('backtest_results.png')
    plt.show()
    
    return

# Function to run a complete backtest for a given period
def run_backtest(symbol, timeframe, lookback_days):
    print(f"Starting backtest for {symbol} on {timeframe} timeframe for past {lookback_days} days")
    
    # Fetch historical data
    df = fetch_historical_data(symbol, timeframe, lookback_days)
    
    # Apply strategy
    df = q_trend_strategy(df, TREND_PERIOD, ATR_PERIOD)
    df = calculate_performance(df)
    
    # Analyze trades
    trades_df, stats = analyze_trades(df)
    
    # Print backtest summary
    print("\n--- Backtest Summary ---")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}")
    print(f"Period: {df.index[0]} to {df.index[-1]} ({lookback_days} days)")
    print(f"Starting capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final equity: ${df['equity'].iloc[-1]:,.2f}")
    print(f"Total return: {(df['equity'].iloc[-1]/INITIAL_CAPITAL-1)*100:.2f}%")
    print(f"Maximum drawdown: {df['drawdown'].min()*100:.2f}%")
    print(f"Total trades: {stats['total_trades']}")
    
    if stats['total_trades'] > 0:
        print(f"Win rate: {stats['win_rate']*100:.2f}%")
        print(f"Profit factor: {stats['profit_factor']:.2f}")
        print(f"Average winning trade: {stats['avg_win']*100:.2f}%")
        print(f"Average losing trade: {stats['avg_loss']*100:.2f}%")
        print(f"Best trade: {stats['best_trade']*100:.2f}%")
        print(f"Worst trade: {stats['worst_trade']*100:.2f}%")
        
        print("\n--- Most Recent Trades ---")
        if len(trades_df) > 0:
            print(trades_df.tail(5).to_string())
    
    # Plot results
    plot_backtest_results(df, stats)
    
    return df, trades_df, stats

# Main function for paper trading
def run_paper_trading(backtest_df=None):
    print(f"Starting Q-Trend Strategy Paper Trading on {SYMBOL}")
    
    # Use backtest data if provided, otherwise fetch new data
    if backtest_df is None:
        # Fetch sufficient historical data to initialize the strategy
        lookback_days = (TREND_PERIOD + ATR_PERIOD + 10) * 0.3  # Estimate days needed based on timeframe
        df = fetch_historical_data(SYMBOL, TIMEFRAME, lookback_days)
        
        # Initialize the strategy with historical data
        df = q_trend_strategy(df, TREND_PERIOD, ATR_PERIOD)
        df = calculate_performance(df)
    else:
        df = backtest_df
    
    print("\nStrategy initialized with historical data")
    print(f"Latest close price: {df['close'].iloc[-1]}")
    print(f"Current position: {df['position'].iloc[-1]}")
    print(f"Current equity: {df['equity'].iloc[-1]:.2f}")
    
    # Main trading loop
    try:
        while True:
            current_time = datetime.now()
            print(f"\n--- {current_time} ---")
            
            # Fetch updated data
            lookback_days = (TREND_PERIOD + ATR_PERIOD + 10) * 0.3
            df = fetch_historical_data(SYMBOL, TIMEFRAME, lookback_days)
            
            # Apply strategy
            df = q_trend_strategy(df, TREND_PERIOD, ATR_PERIOD)
            df = calculate_performance(df)
            
            # Get current status
            current_price = df['close'].iloc[-1]
            current_position = df['position'].iloc[-1]
            current_equity = df['equity'].iloc[-1]
            
            print(f"Current price: {current_price}")
            print(f"Current position: {current_position}")
            print(f"Current equity: {current_equity:.2f}")
            
            if current_position == 1:
                print(f"Trailing stop for LONG: {df['trail_stop'].iloc[-1]}")
            elif current_position == -1:
                print(f"Trailing stop for SHORT: {df['trail_stop'].iloc[-1]}")
            
            # Wait for next update
            if TIMEFRAME == Client.KLINE_INTERVAL_1HOUR:
                wait_time = 300  # 5 minutes for hourly timeframe
            else:
                wait_time = 60  # Default 1 minute
                
            print(f"Waiting {wait_time}s for next update...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\nPaper trading stopped by user.")
        
        # Print final performance
        analyze_trades(df)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Ensure API keys are set
    if not api_key or not api_secret:
        print("Error: Binance API keys not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        print("Create a .env file with your Binance API key and secret:")
        print("BINANCE_API_KEY=your_api_key")
        print("BINANCE_API_SECRET=your_api_secret")
    else:
        # Ask user what they want to do
        print("Q-Trend Strategy - Options:")
        print("1. Run 1-year backtest")
        print("2. Run paper trading")
        print("3. Run backtest and then paper trading")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Run 1-year backtest
            backtest_df, trades_df, stats = run_backtest(SYMBOL, TIMEFRAME, int(5000/1440))
        elif choice == '2':
            # Run paper trading only
            run_paper_trading()
        elif choice == '3':
            # Run backtest first, then paper trading
            backtest_df, trades_df, stats = run_backtest(SYMBOL, TIMEFRAME, 365)
            run_paper_trading(backtest_df)
        else:
            print("Invalid choice. Exiting.")