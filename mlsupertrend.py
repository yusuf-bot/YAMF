import backtrader as bt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import datetime
import ccxt
import time

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


class Supertrend:
    def __init__(self, factor, atr_period):
        self.factor = factor
        self.atr_period = atr_period
        self.upper = None
        self.lower = None
        self.output = None
        self.perf = 0
        self.trend = 0
        
    def update(self, hl2, atr, close, prev_close, prev_output):
        up = hl2 + atr * self.factor
        dn = hl2 - atr * self.factor
        
        if self.upper is None:
            self.upper = hl2
            self.lower = hl2
            self.output = hl2
            return
        
        # Update upper and lower bands
        if prev_close < self.upper:
            self.upper = min(up, self.upper)
        else:
            self.upper = up
            
        if prev_close > self.lower:
            self.lower = max(dn, self.lower)
        else:
            self.lower = dn
        
        # Update trend
        if close > self.upper:
            self.trend = 1
        elif close < self.lower:
            self.trend = 0
        
        # Update output
        self.output = self.lower if self.trend == 1 else self.upper
        
        # Update performance
        diff = np.sign(prev_close - prev_output) if prev_output is not None else 0
        self.perf += 2/(10+1) * ((close - prev_close) * diff - self.perf)
        
        return self.output


class MLAIStrategy(bt.Strategy):
    params = (
        ('atr_length', 10),            # ATR length
        ('min_mult', 1),            # Minimum multiplier
        ('max_mult', 5),            # Maximum multiplier
        ('step', 0.5),                # Step size for factors
        ('perf_alpha', 10),            # Performance memory
        ('from_cluster', 'Best'),     # Which cluster to use ('Best', 'Average', 'Worst')
        ('max_iter', 1000),           # Maximum K-means iterations
        ('max_data', 1000),          # Maximum historical data points for calculations
    )
    
    def __init__(self):
        # Create a list for all supertrend instances
        self.supertrends = []
        self.factors = []
        
        # Create factors
        for i in range(int((self.p.max_mult - self.p.min_mult) / self.p.step) + 1):
            factor = self.p.min_mult + i * self.p.step
            self.factors.append(factor)
            self.supertrends.append(Supertrend(factor, self.p.atr_length))
        
        # Initialize indicators
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_length)
        self.hl2 = (self.data.high + self.data.low) / 2
        
        # Initialize variables for signals
        self.target_factor = None
        self.perf_idx = 0
        self.perf_ama = None
        self.upper = self.hl2
        self.lower = self.hl2
        self.os = 0
        self.prev_os = 0
        self.ts = self.hl2
        
        # Keep track of performance for clustering
        self.performance_window = []
        self.factor_window = []
        
        # Determine which cluster to use
        if self.p.from_cluster == 'Best':
            self.cluster_idx = 2
        elif self.p.from_cluster == 'Average':
            self.cluster_idx = 1
        else:  # Worst
            self.cluster_idx = 0
        
        # For logging
        self.log_enabled = True
        
        # For position tracking
        self.current_position = None  # None, 'long', or 'short'
        self.entry_price = 0
        self.trade_count = 0
        self.total_pnl = 0
        
    def log(self, txt, dt=None):
        if not self.log_enabled:
            return
            
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def update_supertrends(self):
        """Update all supertrend instances"""
        for i, st in enumerate(self.supertrends):
            prev_close = self.data.close[-1] if len(self) > 1 else None
            prev_output = st.output
            st.update(self.hl2[0], self.atr[0], self.data.close[0], prev_close, prev_output)
            
            # Add performance data for clustering
            if len(self.performance_window) <= self.p.max_data:
                self.performance_window.append(st.perf)
                self.factor_window.append(self.factors[i])
    
    def perform_clustering(self):
        """Perform K-means clustering on performance data"""
        if len(self.performance_window) < 3:
            return
            
        # Prepare data for clustering
        X = np.array(self.performance_window).reshape(-1, 1)
        
        # Initialize kmeans using quartiles as initial centroids
        q25 = np.percentile(X, 25)
        q50 = np.percentile(X, 50)
        q75 = np.percentile(X, 75)
        initial_centroids = np.array([[q25], [q50], [q75]])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, init=initial_centroids, n_init=1, max_iter=self.p.max_iter)
        kmeans.fit(X)
        
        # Get clusters
        clusters = kmeans.predict(X)
        
        # Group factors by cluster
        cluster_factors = [[] for _ in range(3)]
        cluster_perfs = [[] for _ in range(3)]
        
        for i, cluster in enumerate(clusters):
            cluster_factors[cluster].append(self.factor_window[i])
            cluster_perfs[cluster].append(self.performance_window[i])
        
        # Sort clusters by performance (ascending)
        mean_perfs = [np.mean(perfs) if perfs else 0 for perfs in cluster_perfs]
        sorted_indices = np.argsort(mean_perfs)
        
        # Set target factor based on chosen cluster
        target_cluster = sorted_indices[self.cluster_idx]
        if cluster_factors[target_cluster]:
            self.target_factor = np.mean(cluster_factors[target_cluster])
        
        # Calculate performance index
        if cluster_perfs[target_cluster]:
            perf_mean = max(np.mean(cluster_perfs[target_cluster]), 0)
            denominator = np.mean(np.abs(np.diff(self.data.close.get(size=int(self.p.perf_alpha)))))
            if denominator != 0:
                self.perf_idx = perf_mean / denominator
    
    def update_trailing_stop(self):
        """Update trailing stop based on target factor"""
        if self.target_factor is None:
            return
            
        # Calculate new bounds
        up = self.hl2[0] + self.atr[0] * self.target_factor
        dn = self.hl2[0] - self.atr[0] * self.target_factor
        
        # Update upper and lower bands
        if self.data.close[-1] < self.upper:
            self.upper = min(up, self.upper)
        else:
            self.upper = up
            
        if self.data.close[-1] > self.lower:
            self.lower = max(dn, self.lower)
        else:
            self.lower = dn
        
        # Update trend
        self.prev_os = self.os
        if self.data.close[0] > self.upper:
            self.os = 1
        elif self.data.close[0] < self.lower:
            self.os = 0
        
        # Update trailing stop
        self.ts = self.lower if self.os else self.upper
        
        # Update trailing stop AMA
        if self.perf_ama is None:
            self.perf_ama = self.ts
        else:
            self.perf_ama += self.perf_idx * (self.ts - self.perf_ama)
    
    def close_position(self):
        """Close any existing position and report PnL"""
        if self.current_position is None:
            return
            
        current_price = self.data.close[0]
        
        if self.current_position == 'long':
            # Calculate PnL for long position
            position_size = self.getposition(self.data).size
            if position_size > 0:
                trade_pnl = (current_price - self.entry_price) * position_size
                self.total_pnl += trade_pnl
                
                self.log(f'CLOSING LONG - Entry: {self.entry_price:.6f}, Exit: {current_price:.6f}')
                self.log(f'TRADE PNL: {trade_pnl:.2f}, TOTAL PNL: {self.total_pnl:.2f}')
                
                # Close the position
                self.close()
        
        elif self.current_position == 'short':
            # Calculate PnL for short position
            position_size = self.getposition(self.data).size
            if position_size < 0:
                trade_pnl = (self.entry_price - current_price) * abs(position_size)
                self.total_pnl += trade_pnl
                
                self.log(f'CLOSING SHORT - Entry: {self.entry_price:.6f}, Exit: {current_price:.6f}')
                self.log(f'TRADE PNL: {trade_pnl:.2f}, TOTAL PNL: {self.total_pnl:.2f}')
                
                # Close the position
                self.close()
        
        # Update position tracking
        self.current_position = None
        self.entry_price = 0
    
    def next(self):
        # Update all supertrends
        self.update_supertrends()
        
        # Perform clustering every N bars
        if len(self) % 10 == 0 or self.target_factor is None:
            self.perform_clustering()
        
        # Update trailing stop based on target factor
        self.update_trailing_stop()
        
        # Check for signal changes
        if self.os != self.prev_os:
            # Calculate portfolio stats
            total_equity = self.broker.getvalue()
            cash = self.broker.getcash()
            position_value = total_equity - cash
            
            # Buy signal
            if self.os > self.prev_os:
                # Close any existing short position first
                self.close_position()
                
                self.log(f'BUY SIGNAL - Factor: {self.target_factor:.2f}, Perf: {int(self.perf_idx * 10)}')
                self.log(f'Total Equity: {total_equity:.2f}, Cash: {cash:.2f}, Position Value: {position_value:.2f}')
                
                # Execute long order
                self.buy()
                
                # Update position tracking
                self.current_position = 'long'
                self.entry_price = self.data.close[0]
                self.trade_count += 1
                
                # Show updated portfolio stats after trade
                new_total_equity = self.broker.getvalue()
                new_cash = self.broker.getcash()
                new_position_value = new_total_equity - new_cash
                self.log(f'AFTER BUY - Total Equity: {new_total_equity:.2f}, Cash: {new_cash:.2f}, Position Value: {new_position_value:.2f}')
            
            # Sell signal
            elif self.os < self.prev_os:
                # Close any existing long position first
                self.close_position()
                
                self.log(f'SELL SIGNAL - Factor: {self.target_factor:.2f}, Perf: {int(self.perf_idx * 10)}')
                self.log(f'Total Equity: {total_equity:.2f}, Cash: {cash:.2f}, Position Value: {position_value:.2f}')
                
                # Execute short order
                self.sell()
                
                # Update position tracking
                self.current_position = 'short'
                self.entry_price = self.data.close[0]
                self.trade_count += 1
                
                # Show updated portfolio stats after trade
                new_total_equity = self.broker.getvalue()
                new_cash = self.broker.getcash()
                new_position_value = new_total_equity - new_cash
                self.log(f'AFTER SELL - Total Equity: {new_total_equity:.2f}, Cash: {new_cash:.2f}, Position Value: {new_position_value:.2f}')


def run_strategy(symbol='BTCUSDT', timeframe='5m', days=18):
    # Get data
    df = get_binance_data(symbol=symbol, timeframe=timeframe, days=days)
    
    # Create backtrader instance
    cerebro = bt.Cerebro()
    
    # Add data feed
    data = PandasDataFeed(dataname=df)
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(MLAIStrategy,
                      atr_length=10,         # Optimized for crypto
                      min_mult=1,         # Minimum multiplier
                      max_mult=5,         # Maximum multiplier
                      step=0.5,             # Step size
                      perf_alpha=10,         # Performance memory
                      from_cluster='Best')  # Use best-performing cluster
    
    # Set broker parameters
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% fee
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run strategy
    print(f'Starting portfolio value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strategy = results[0]
    print(f'Final portfolio value: {cerebro.broker.getvalue():.2f}')
    
    # Print analysis
    print(f'Sharpe Ratio: {strategy.analyzers.sharpe.get_analysis()["sharperatio"]:.3f}')
    print(f'Return: {strategy.analyzers.returns.get_analysis()["rtot"]:.2%}')
    print(f'Max Drawdown: {strategy.analyzers.drawdown.get_analysis()["max"]["drawdown"]:.2%}')
    
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    print(f'Total Trades: {trade_analysis.total.closed}')
    if trade_analysis.total.closed > 0:
        print(f'Win Rate: {trade_analysis.won.total/trade_analysis.total.closed:.2%}')
        print(f'Average P/L: {trade_analysis.pnl.net/trade_analysis.total.closed:.4f}')
    
    # Plot results
    cerebro.plot(style='candlestick', volume=False)


if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run strategy
    run_strategy(symbol='BTCUSDT', timeframe='5m', days=30)