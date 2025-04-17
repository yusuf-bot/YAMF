import pandas as pd
import ta
import matplotlib.pyplot as plt

def volty_expan_strategy(csv_file, initial_capital=1000, length=5, num_atrs=0.75):
    # Load data
    df = pd.read_csv(csv_file, parse_dates=True)
    df.columns = [col.lower() for col in df.columns]  # Standardize column names
    df.dropna(inplace=True)

    # Calculate ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=length)

    # Calculate stop levels based on close[2]
    df['stop_long'] = df['close'].shift(1) + df['atr'] * num_atrs
    df['stop_short'] = df['close'].shift(1) - df['atr'] * num_atrs
    df['qty'] = initial_capital / df['close'].shift(1)

    # Initialize
    df['position'] = 0  # 1 = long, -1 = short, 0 = flat
    df['cash'] = initial_capital
    df['equity'] = initial_capital
    df['entry_price'] = 0

    in_trade = False
    trade_type = None

    for i in range(2 + length, len(df)):
        price = df.iloc[i]['close']
        stop_long = df.iloc[i]['stop_long']
        stop_short = df.iloc[i]['stop_short']
        qty = df.iloc[i]['qty']

        # Simulate entries — only one position at a time
        if not in_trade:
            if price >= stop_long:
                df.at[df.index[i], 'position'] = 1
                df.at[df.index[i], 'entry_price'] = stop_long
                in_trade = True
                trade_type = 'long'
            elif price <= stop_short:
                df.at[df.index[i], 'position'] = -1
                df.at[df.index[i], 'entry_price'] = stop_short
                in_trade = True
                trade_type = 'short'
        else:
            # For simplicity, exit after 1 bar
            df.at[df.index[i], 'position'] = 0
            in_trade = False
            trade_type = None

        # Update equity
        if trade_type == 'long':
            df.at[df.index[i], 'equity'] = qty * price
        elif trade_type == 'short':
            df.at[df.index[i], 'equity'] = initial_capital + (df.iloc[i]['entry_price'] - price) * qty
        else:
            df.at[df.index[i], 'equity'] = df.at[df.index[i - 1], 'equity']

    # Plot equity
    df['equity'].plot(title='Equity Curve', figsize=(12, 6))
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.show()

    return df

# Example usage:
df_result = volty_expan_strategy("btcusdt.csv")
