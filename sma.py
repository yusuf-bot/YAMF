from datetime import datetime
import backtrader as bt
import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def get_binance_data(symbol='PEPEUSDT', timeframe='4h', days=7):
    # First check if the specific file exists
    specific_filename = f"{symbol}_{timeframe}_{days}.csv"
    
    if os.path.exists(specific_filename):
        print(f"Loading data from {specific_filename}")
        df = pd.read_csv(specific_filename, parse_dates=['datetime'], index_col='datetime')
        
        # Ensure we have valid data by replacing zeros and NaNs
        for col in ['open', 'high', 'low', 'close']:
            # Replace zeros with NaN
            df[col] = df[col].replace(0, np.nan)
            # Forward fill NaN values
            df[col] = df[col].fillna(method='ffill')
            # If there are still NaNs at the beginning, backfill
            df[col] = df[col].fillna(method='bfill')
            
        print(f"Data loaded from CSV. Total candles: {len(df)}")
        return df
    
    # If specific file doesn't exist, check for the detailed filename
    detailed_filename = f"{symbol}_{timeframe}_{days}d.csv"
    
    if os.path.exists(detailed_filename):
        print(f"Loading data from {detailed_filename}")
        df = pd.read_csv(detailed_filename, parse_dates=['datetime'], index_col='datetime')
        
        # Ensure we have valid data by replacing zeros and NaNs
        for col in ['open', 'high', 'low', 'close']:
            # Replace zeros with NaN
            df[col] = df[col].replace(0, np.nan)
            # Forward fill NaN values
            df[col] = df[col].fillna(method='ffill')
            # If there are still NaNs at the beginning, backfill
            df[col] = df[col].fillna(method='bfill')
            
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
    
    # Ensure we have valid data by replacing zeros and NaNs
    for col in ['open', 'high', 'low', 'close']:
        # Replace zeros with NaN
        df[col] = df[col].replace(0, np.nan)
        # Forward fill NaN values
        df[col] = df[col].fillna(method='ffill')
        # If there are still NaNs at the beginning, backfill
        df[col] = df[col].fillna(method='bfill')
    
    print(f"Data fetched. Total candles: {len(df)}")
    
    # Save to both filenames for future use
    df.to_csv(specific_filename)
    print(f"Data saved to {specific_filename}")
    df.to_csv(detailed_filename)
    print(f"Data also saved to {detailed_filename}")
    
    return df

# Define the base LuxAlgo Strategy
class LuxAlgoStrategy(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=14,         # period for the fast moving average
        pslow=28,         # period for the slow moving average
        tp=0.15,          # take profit percentage (15% by default)
        sl=0.07,          # stop loss percentage (7% by default)
        order_fraction=0.5  # portion of portfolio to use per trade (50% by default)
    )

    def __init__(self):
        # Initialize moving averages
        self.sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        self.sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(self.sma1, self.sma2)  # crossover signal
        
        # Keep track of take profit and stop loss prices
        self.take_profit_price = None
        self.stop_loss_price = None
        
        # Keep track of open orders
        self.order = None
        self.buy_price = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return
            
        # Check if order was completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                # Calculate take profit and stop loss prices
                self.take_profit_price = self.buy_price * (1.0 + self.p.tp)
                self.stop_loss_price = self.buy_price * (1.0 - self.p.sl)
                print(f'BUY EXECUTED at {order.executed.price}')
                print(f'Take Profit set at: {self.take_profit_price}')
                print(f'Stop Loss set at: {self.stop_loss_price}')
            elif order.issell():
                print(f'SELL EXECUTED at {order.executed.price}')
                
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f'Order Canceled/Margin/Rejected: {order.status}')
            
        # Reset order
        self.order = None

    def next(self):
        # Check if an order is pending - if yes, we cannot send a 2nd one
        if self.order:
            return
            
        # Check if we are in the market (have a position)
        if not self.position:
            # We are not in the market, look for entry signal
            if self.crossover > 0:  # if fast crosses slow to the upside
                # Calculate the stake based on the order_fraction parameter
                cash = self.broker.getcash()
                stake = cash * self.p.order_fraction
                price = self.data.close[0]
                
                # Ensure price is valid and greater than zero
                if price <= 0:
                    print(f'WARNING: Invalid price {price} detected, skipping buy signal')
                    return
                    
                size = stake / price
                
                # Ensure we're not trying to buy with more cash than available
                if size * price > cash:
                    size = cash / price * 0.99  # Use 99% of available cash to account for fees
                
                print(f'BUY SIGNAL: Price={price:.8f}, Cash={cash}, Stake={stake}, Size={size}')
                self.order = self.buy(size=size)
        else:
            # We are in the market, look for exit signals
            current_price = self.data.close[0]
            
            # Check for take profit condition
            if current_price >= self.take_profit_price:
                print(f'TAKE PROFIT TRIGGERED: Current Price={current_price}, TP Price={self.take_profit_price}')
                self.order = self.close()
                
            # Check for stop loss condition
            elif current_price <= self.stop_loss_price:
                print(f'STOP LOSS TRIGGERED: Current Price={current_price}, SL Price={self.stop_loss_price}')
                self.order = self.close()
                
            # Check for crossover exit condition
            elif self.crossover < 0:  # if fast crosses slow to the downside
                print(f'CROSSOVER EXIT SIGNAL: Current Price={current_price}')
                self.order = self.close()


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


if __name__ == '__main__':
    print('v 1.0.2 - SMA')
    
    # Fetch data
    symbol = 'PEPEUSDT'
    timeframe = '4h'
    days = 31+28  # One year of data
    data = get_binance_data(symbol=symbol, timeframe=timeframe, days=days)
    print(f"Data shape: {data.shape}")
    print(data.head())

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add our enhanced strategy with parameters
    cerebro.addstrategy(EnhancedLuxAlgoStrategy, 
                        pfast=10,               # Fast SMA period
                        pslow=30,               # Slow SMA period
                        tp=0.15,                # Take profit at 15%
                        sl=0.07,                # Stop loss at 7%
                        order_fraction=0.5)     # Use 50% of portfolio per trade

    # Add data feed to Cerebro
    data_feed = PandasDataFeed(dataname=data)
    cerebro.adddata(data_feed)

    # Set initial capital and commission
    initial_capital = 100  # Start with $1000
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Print starting conditions
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue()}')
    
    # Run the backtest
    results = cerebro.run()
    
    # Print final results
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: ${final_value}')
    print(f'Profit/Loss: ${final_value - initial_capital}')
    print(f'Return: {((final_value / initial_capital) - 1) * 100}%')

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
    if len(capital_growth) > 1:  # Make sure we have enough data points
        returns = np.diff(capital_growth) / capital_growth[:-1]
        annual_return = ((final_value / initial_capital) ** (365 / len(capital_growth)) - 1) * 100
        
        # Avoid division by zero for Sharpe ratio
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 60 / int(timeframe[:-1]))
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        peak = np.maximum.accumulate(capital_growth)
        drawdown = (peak - capital_growth) / peak
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
        
        print("\nPerformance Metrics:")
        print(f"Annual Return: {annual_return}%")
        print(f"Sharpe Ratio: {sharpe_ratio}")
        print(f"Max Drawdown: {max_drawdown}%")
        print('v 1.0.2 - SMA')
    
    else:
        print("\nNot enough data points to calculate performance metrics")