import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import ccxt
import math


# Custom Pandas Data Feed
class PandasDataFeed(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
    )

# Enhanced Strategy with capital tracking
class EnhancedLuxAlgoStrategy(LuxAlgoStrategy):
    def __init__(self):
        super().__init__()
        self.capital_growth = []
        
    def next(self):
        # Record capital before trade execution
        self.capital_growth.append(self.broker.getvalue())
        
        # Execute the original strategy logic
        super().next()


def get_binance_data(symbol='PEPEUSDT', timeframe='5m', days=365):
    # First check if the specific file exists
    specific_filename = f"{symbol}.csv"
    
    if os.path.exists(specific_filename):
        print(f"Loading data from {specific_filename}")
        df = pd.read_csv(specific_filename, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded from CSV. Total candles: {len(df)}")
        return df
    
    # If specific file doesn't exist, check for the detailed filename
    detailed_filename = f"{symbol}_{timeframe}_{days}d.csv"
    
    if os.path.exists(detailed_filename):
        print(f"Loading data from {detailed_filename}")
        df = pd.read_csv(detailed_filename, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded from CSV. Total candles: {len(df)}")
        return df
    
    # If neither CSV exists, fetch data from Binance
    exchange = ccxt.binance()
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    since = int(start_date.timestamp() * 1000)
    
    all_candles = []
    while since < end_date.timestamp() * 1000:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if len(candles) == 0:
                break
            
            since = candles[-1][0] + 1
            all_candles.extend(candles)
            print(f"Fetched {len(candles)} candles. Total: {len(all_candles)}")
            
            # Add delay to avoid rate limits
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)  # Longer delay on error
            continue
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    print(f"Data fetched. Total candles: {len(df)}")
    
    # Save to both filenames for future use
    df.to_csv(specific_filename)
    print(f"Data saved to {specific_filename}")
    df.to_csv(detailed_filename)
    print(f"Data also saved to {detailed_filename}")
    
    return df


class LuxAlgoSignals(bt.Indicator):
    """
    Lux Algo Signals & Overlays indicator converted from PineScript to Backtrader
    """
    lines = ('smarttrail', 'trendcatcher', 'signal')
    
    params = (
        ('sensitivity', 5),        # Signal sensitivity (1-26)
        ('atr_length', 10),        # Signal tuner (1-25)
        ('show_signals', True),    # Enable/disable signals
        ('signal_mode', 'confirmation'),  # 'confirmation', 'contrarian', or 'none'
    )
    
    def __init__(self):
        # Calculate basic moving averages
        self.sma4 = bt.indicators.SMA(self.data.close, period=4)
        self.sma5 = bt.indicators.SMA(self.data.close, period=5)
        self.sma9 = bt.indicators.SMA(self.data.close, period=9)
        self.ema50 = bt.indicators.EMA(self.data.close, period=50)
        self.ema200 = bt.indicators.EMA(self.data.close, period=200)
        
        # ATR for signal filtering and stop loss
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_length)
        
        # Smart Trail calculation
        self.smart_trail_period = 10
        self.smart_trail_factor1 = 4
        self.smart_trail_factor2 = 8
        
        # Trend Catcher
        self.trend_catcher_period = 14
        
        # Store signal information
        self.last_signal_price = 0
        self.last_signal_bar = 0
        self.last_signal_type = None
        self.last_signal_atr = 0
        
        # Initialize lines
        self.lines.smarttrail = self.data.close * 0.0
        self.lines.trendcatcher = self.data.close * 0.0
        self.lines.signal = self.data.close * 0.0  # 1 for long, -1 for short, 0 for no signal
    
    def next(self):
        # Calculate Smart Trail
        self.lines.smarttrail[0] = self.calculate_smart_trail()
        
        # Calculate Trend Catcher
        self.lines.trendcatcher[0] = self.calculate_trend_catcher()
        
        # Calculate Signals
        self.calculate_signals()
    
    def calculate_smart_trail(self):
        # Simplified Smart Trail calculation
        high_max = max(self.data.high.get(size=self.smart_trail_period))
        low_min = min(self.data.low.get(size=self.smart_trail_period))
        
        # Calculate average true range for the period
        atr_val = self.atr[0]
        
        # Calculate smart trail line based on trend direction
        if self.data.close[0] > self.ema50[0]:
            # Bullish trend - support line
            return self.data.low[0] - atr_val * self.smart_trail_factor1 / self.smart_trail_factor2
        else:
            # Bearish trend - resistance line
            return self.data.high[0] + atr_val * self.smart_trail_factor1 / self.smart_trail_factor2
    
    def calculate_trend_catcher(self):
        # Simplified Trend Catcher calculation
        close_prices = np.array(self.data.close.get(size=self.trend_catcher_period))
        if len(close_prices) < self.trend_catcher_period:
            return self.data.close[0]
        
        # Calculate linear regression
        x = np.arange(len(close_prices))
        slope, _ = np.polyfit(x, close_prices, 1)
        
        # Return a value based on the slope
        if slope > 0:
            return self.data.close[0] * 1.01  # Slightly above price for bullish trend
        else:
            return self.data.close[0] * 0.99  # Slightly below price for bearish trend
    
    def calculate_signals(self):
        if not self.p.show_signals:
            self.lines.signal[0] = 0
            return
        
        # Default - no signal
        self.lines.signal[0] = 0
        
        # Calculate bar index
        current_bar = len(self.data)
        
        # Calculate conditions for signals
        long_confirmation = (
            self.data.close[0] > self.data.open[0] and
            self.data.close[0] > self.data.close[-1] and
            self.data.close[0] > self.sma5[0] and
            self.data.close[0] > self.sma9[0] and
            self.data.close[-1] < self.sma9[-1] and
            self.data.close[0] > self.ema50[0]
        )
        
        short_confirmation = (
            self.data.close[0] < self.data.open[0] and
            self.data.close[0] < self.data.close[-1] and
            self.data.close[0] < self.sma5[0] and
            self.data.close[0] < self.sma9[0] and
            self.data.close[-1] > self.sma9[-1] and
            self.data.close[0] < self.ema50[0]
        )
        
        long_contrarian = (
            self.data.close[0] < self.data.open[0] and
            self.data.close[0] < self.data.close[-1] and
            self.data.close[0] < self.sma5[0] and
            self.data.close[0] < self.sma9[0] and
            self.data.close[-1] > self.sma9[-1] and
            self.data.close[0] < self.ema50[0] and
            self.data.close[0] > self.ema200[0]
        )
        
        short_contrarian = (
            self.data.close[0] > self.data.open[0] and
            self.data.close[0] > self.data.close[-1] and
            self.data.close[0] > self.sma5[0] and
            self.data.close[0] > self.sma9[0] and
            self.data.close[-1] < self.sma9[-1] and
            self.data.close[0] > self.ema50[0] and
            self.data.close[0] < self.ema200[0]
        )
        
        # Long exit and short exit conditions
        long_exit = (
            self.data.close[0] < self.data.open[0] and
            self.data.close[0] < self.data.close[-1] and
            self.data.close[0] < self.sma5[0] and
            self.data.close[0] < self.sma9[0] and
            self.data.close[-1] > self.sma9[-1]
        )
        
        short_exit = (
            self.data.close[0] > self.data.open[0] and
            self.data.close[0] > self.data.close[-1] and
            self.data.close[0] > self.sma5[0] and
            self.data.close[0] > self.sma9[0] and
            self.data.close[-1] < self.sma9[-1]
        )
        
        # Determine signal conditions based on mode
        if self.p.signal_mode == 'confirmation':
            long_signal_condition = long_confirmation
            short_signal_condition = short_confirmation
        elif self.p.signal_mode == 'contrarian':
            long_signal_condition = long_contrarian
            short_signal_condition = short_contrarian
        else:
            long_signal_condition = False
            short_signal_condition = False
        
        # Exit conditions
        long_exit_condition = long_exit and self.last_signal_type == 'long'
        short_exit_condition = short_exit and self.last_signal_type == 'short'
        
        # Signal filtering
        bars_since_last_signal = current_bar - self.last_signal_bar
        min_bars_for_new_signal = 3
        
        if self.last_signal_price > 0:
            price_movement_from_last_signal = abs(self.data.close[0] - self.last_signal_price)
            min_price_movement = self.last_signal_atr * self.p.sensitivity
        else:
            price_movement_from_last_signal = float('inf')
            min_price_movement = 0
        
        # Valid signal conditions
        valid_long_signal = (
            long_signal_condition and 
            (self.last_signal_bar == 0 or bars_since_last_signal > min_bars_for_new_signal) and
            (self.last_signal_price == 0 or price_movement_from_last_signal > min_price_movement or self.last_signal_type == 'short')
        )
        
        valid_short_signal = (
            short_signal_condition and 
            (self.last_signal_bar == 0 or bars_since_last_signal > min_bars_for_new_signal) and
            (self.last_signal_price == 0 or price_movement_from_last_signal > min_price_movement or self.last_signal_type == 'long')
        )
        
        valid_long_exit = long_exit_condition and (self.last_signal_bar == 0 or bars_since_last_signal > min_bars_for_new_signal)
        valid_short_exit = short_exit_condition and (self.last_signal_bar == 0 or bars_since_last_signal > min_bars_for_new_signal)
        
        # Generate signals
        if valid_long_signal:
            self.lines.signal[0] = 1
            self.last_signal_bar = current_bar
            self.last_signal_price = self.data.close[0]
            self.last_signal_atr = self.atr[0]
            self.last_signal_type = 'long'
        elif valid_short_signal:
            self.lines.signal[0] = -1
            self.last_signal_bar = current_bar
            self.last_signal_price = self.data.close[0]
            self.last_signal_atr = self.atr[0]
            self.last_signal_type = 'short'
        elif valid_long_exit:
            self.lines.signal[0] = 2  # 2 for long exit
            self.last_signal_bar = current_bar
            self.last_signal_price = self.data.close[0]
            self.last_signal_atr = self.atr[0]
            self.last_signal_type = 'longExit'
        elif valid_short_exit:
            self.lines.signal[0] = -2  # -2 for short exit
            self.last_signal_bar = current_bar
            self.last_signal_price = self.data.close[0]
            self.last_signal_atr = self.atr[0]
            self.last_signal_type = 'shortExit'

# Example strategy using the LuxAlgoSignals indicator
class LuxAlgoStrategy(bt.Strategy):
    params = (
        ('tp', 0.18),              # Take profit percentage
        ('sl', 0.09),              # Stop loss percentage
        ('order_fraction', 0.5),   # Order size as fraction of portfolio
    )
    
    def __init__(self):
        # Initialize the Lux Algo Signals indicator
        self.lux_signals = LuxAlgoSignals(
            sensitivity=5,
            atr_length=10,
            show_signals=True,
            signal_mode='confirmation'
        )
        
        # Track positions and trades
        self.positions_count = 0
        self.entry_prices = []
        self.entry_amounts = []
    
    def next(self):
        # Check for signals
        if self.lux_signals.lines.signal[0] == 1:  # Long signal
            # Calculate trade amount
            trade_amount = self.broker.getvalue() * self.p.order_fraction
            
            if trade_amount > 0:
                # Add new position
                self.entry_prices.append(self.data.open[0])
                self.entry_amounts.append(trade_amount)
                self.positions_count += 1
                self.log(f"LONG ENTRY at {self.data.open[0]:.6f} -> Amount: {trade_amount:.2f}")
        
        elif self.lux_signals.lines.signal[0] == -1:  # Short signal
            # Calculate trade amount
            trade_amount = self.broker.getvalue() * self.p.order_fraction
            
            if trade_amount > 0:
                # Add new position (for simplicity, treating shorts like longs)
                self.entry_prices.append(self.data.open[0])
                self.entry_amounts.append(-trade_amount)  # Negative for short
                self.positions_count += 1
                self.log(f"SHORT ENTRY at {self.data.open[0]:.6f} -> Amount: {trade_amount:.2f}")
        
        # Process exits
        if self.positions_count > 0:
            new_entry_prices = []
            new_entry_amounts = []
            
            for entry_price, amount_invested in zip(self.entry_prices, self.entry_amounts):
                is_long = amount_invested > 0
                
                if is_long:
                    tp_price = entry_price * (1 + self.p.tp)
                    sl_price = entry_price * (1 - self.p.sl)
                    
                    if self.data.high[0] >= tp_price:
                        # Take Profit hit
                        self.log(f"LONG TP at {tp_price:.6f} (Entry {entry_price:.6f})")
                        self.positions_count -= 1
                    elif self.data.low[0] <= sl_price:
                        # Stop Loss hit
                        self.log(f"LONG SL at {sl_price:.6f} (Entry {entry_price:.6f})")
                        self.positions_count -= 1
                    elif self.lux_signals.lines.signal[0] == 2:  # Long exit signal
                        # Manual exit
                        self.log(f"LONG EXIT at {self.data.close[0]:.6f} (Entry {entry_price:.6f})")
                        self.positions_count -= 1
                    else:
                        new_entry_prices.append(entry_price)
                        new_entry_amounts.append(amount_invested)
                else:  # Short position
                    tp_price = entry_price * (1 - self.p.tp)
                    sl_price = entry_price * (1 + self.p.sl)
                    
                    if self.data.low[0] <= tp_price:
                        # Take Profit hit
                        self.log(f"SHORT TP at {tp_price:.6f} (Entry {entry_price:.6f})")
                        self.positions_count -= 1
                    elif self.data.high[0] >= sl_price:
                        # Stop Loss hit
                        self.log(f"SHORT SL at {sl_price:.6f} (Entry {entry_price:.6f})")
                        self.positions_count -= 1
                    elif self.lux_signals.lines.signal[0] == -2:  # Short exit signal
                        # Manual exit
                        self.log(f"SHORT EXIT at {self.data.close[0]:.6f} (Entry {entry_price:.6f})")
                        self.positions_count -= 1
                    else:
                        new_entry_prices.append(entry_price)
                        new_entry_amounts.append(amount_invested)
            
            self.entry_prices = new_entry_prices
            self.entry_amounts = new_entry_amounts
    
    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"[{dt}] {txt}")



if __name__ == '__main__':
    print('v 1.0.1 - LuxAlgo Signals Backtest')
    
    # Fetch data
    symbol = 'BTCUSDT'
    timeframe = '5m'
    days = 365  # One year of data
    data = get_binance_data(symbol=symbol, timeframe=timeframe, days=days)
    print(f"Data shape: {data.shape}")
    print(data.head())

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add our enhanced strategy
    cerebro.addstrategy(EnhancedLuxAlgoStrategy, 
                        tp=0.15,              # Take profit at 15%
                        sl=0.07,              # Stop loss at 7%
                        order_fraction=0.5)   # Use 50% of portfolio per trade

    # Add data feed to Cerebro
    data_feed = PandasDataFeed(dataname=data)
    cerebro.adddata(data_feed)

    # Set initial capital and commission
    initial_capital = 1000  # Start with $1000
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Print starting conditions
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    
    # Run the backtest
    results = cerebro.run()
    
    # Print final results
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: ${final_value:.2f}')
    print(f'Profit/Loss: ${final_value - initial_capital:.2f}')
    print(f'Return: {((final_value / initial_capital) - 1) * 100:.2f}%')

    # Extract strategy instance and capital growth
    strategy = results[0]
    capital_growth = strategy.capital_growth

    # Plot Capital Growth
    plt.figure(figsize=(12, 6))
    plt.plot(capital_growth)
    plt.title(f'LuxAlgo Strategy - Capital Growth ({symbol}, {timeframe})')
    plt.xlabel('Trading Periods')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    output_file = f'luxalgo_{symbol}_{timeframe}_results.png'
    plt.savefig(output_file)
    print(f"Results chart saved to {output_file}")
    
    # Optional: Show the plot
    plt.show()
    
    # Generate performance metrics
    returns = np.diff(capital_growth) / capital_growth[:-1]
    annual_return = ((final_value / initial_capital) ** (365 / len(capital_growth)) - 1) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 60 / int(timeframe[:-1]))
    max_drawdown = np.max(np.maximum.accumulate(capital_growth) - capital_growth) / np.max(capital_growth) * 100
    
    print("\nPerformance Metrics:")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")