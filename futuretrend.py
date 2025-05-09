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


# Create a custom observer for trade tracking
class TradeObserver(bt.observer.Observer):
    lines = ('entry_price', 'exit_price', 'entry_size', 'exit_size', 'pnl')
    
    plotinfo = dict(plot=True, subplot=True, plotname='Trade PnL')
    plotlines = dict(
        entry_price=dict(marker='^', markersize=8, color='green', fillstyle='full'),
        exit_price=dict(marker='v', markersize=8, color='red', fillstyle='full'),
        entry_size=dict(_plotskip=True),
        exit_size=dict(_plotskip=True),
        pnl=dict(_plotskip=True),
    )
    
    def next(self):
        # Default to NaN
        self.lines.entry_price[0] = np.nan
        self.lines.exit_price[0] = np.nan
        self.lines.entry_size[0] = np.nan
        self.lines.exit_size[0] = np.nan
        self.lines.pnl[0] = np.nan
        
        # Look for trades in the system
        for trade in self._owner._tradespending:
            if trade.isclosed:
                self.lines.exit_price[0] = trade.price
                self.lines.exit_size[0] = trade.size
                self.lines.pnl[0] = trade.pnl
            elif trade.isopen:
                self.lines.entry_price[0] = trade.price
                self.lines.entry_size[0] = trade.size


class MLAIStrategy(bt.Strategy):
    params = (
        ('atr_length', 10),            # ATR length
        ('min_mult', 1),            # Minimum multiplier
        ('max_mult', 4),            # Maximum multiplier
        ('step', 0.5),                # Step size for factors
        ('perf_alpha', 10),            # Performance memory
        ('from_cluster', 'Best'),     # Which cluster to use ('Best', 'Average', 'Worst')
        ('max_iter', 1000),           # Maximum K-means iterations
        ('max_data', 10000),          # Maximum historical data points for calculations
        ('percent_equity', 50),       # Percentage of equity to use per trade (50% = 0.5)
        ('plot_signals', True),       # Plot buy/sell signals
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
        
        # For plotting buy/sell signals
        if self.p.plot_signals:
            self.buy_signals = bt.indicators.NumPyNDArrayIndicator(self.data, 
                                                                  name='Buy Signals',
                                                                  plottable=True, 
                                                                  plotskip=False,
                                                                  subplot=False)
            self.sell_signals = bt.indicators.NumPyNDArrayIndicator(self.data, 
                                                                   name='Sell Signals',
                                                                   plottable=True, 
                                                                   plotskip=False,
                                                                   subplot=False)
            # Plot the upper and lower bands
            self.upper_band = bt.indicators.NumPyNDArrayIndicator(self.data, 
                                                                 name='Upper Band',
                                                                 plottable=True, 
                                                                 plotskip=False,
                                                                 subplot=False)
            self.lower_band = bt.indicators.NumPyNDArrayIndicator(self.data, 
                                                                 name='Lower Band',
                                                                 plottable=True, 
                                                                 plotskip=False,
                                                                 subplot=False)
            # Plot target prices
            self.target_levels = bt.indicators.NumPyNDArrayIndicator(self.data, 
                                                                    name='Target Levels',
                                                                    plottable=True, 
                                                                    plotskip=False,
                                                                    subplot=False)
            self.stop_levels = bt.indicators.NumPyNDArrayIndicator(self.data, 
                                                                  name='Stop Levels',
                                                                  plottable=True, 
                                                                  plotskip=False,
                                                                  subplot=False)
        
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
            
        # Update plotting indicators
        if self.p.plot_signals:
            self.upper_band[0] = self.upper
            self.lower_band[0] = self.lower
    
    def next(self):
        # Update all supertrends
        self.update_supertrends()
        
        # Perform clustering every N bars
        if len(self) % 10 == 0 or self.target_factor is None:
            self.perform_clustering()
        
        # Update trailing stop based on target factor
        self.update_trailing_stop()
        
        # Reset plot values
        if self.p.plot_signals:
            self.buy_signals[0] = np.nan
            self.sell_signals[0] = np.nan
            self.target_levels[0] = np.nan
            self.stop_levels[0] = np.nan
        
        # Check for signal changes
        if self.os != self.prev_os:
            # Calculate position size (50% of equity)
            current_equity = self.broker.get_cash() + self.broker.get_value()
            equity_to_use = current_equity * (self.p.percent_equity / 100)
            entry_price = self.data.close[0]
            size = equity_to_use / entry_price
            
            # Buy signal
            if self.os > self.prev_os:
                self.log(f'BUY SIGNAL - Factor: {self.target_factor:.2f}, Perf: {int(self.perf_idx * 10)}')
                
                # Calculate target and stop prices
                target_pct = (int(self.perf_idx * 10)) / 100
                target_price = entry_price * (1 + target_pct)
                stop_price = entry_price * (1 - target_pct)
                
                self.log(f'Entry: {entry_price:.6f}, Target: {target_price:.6f}, Stop: {stop_price:.6f}')
                self.log(f'Using 50% equity: {equity_to_use:.2f}, Size: {size:.6f}')
                
                # Plot signals
                if self.p.plot_signals:
                    self.buy_signals[0] = entry_price
                    self.target_levels[0] = target_price
                    self.stop_levels[0] = stop_price
                
                # Execute orders with specified size
                self.buy(size=size)
                self.sell(size=size, exectype=bt.Order.Limit, price=target_price, 
                         oco=self.sell(size=size, exectype=bt.Order.Stop, price=stop_price))
            
            # Sell signal
            elif self.os < self.prev_os:
                self.log(f'SELL SIGNAL - Factor: {self.target_factor:.2f}, Perf: {int(self.perf_idx * 10)}')
                
                # Calculate target and stop prices
                target_pct = (int(self.perf_idx * 10)) / 100
                target_price = entry_price * (1 - target_pct)
                stop_price = entry_price * (1 + target_pct)
                
                self.log(f'Entry: {entry_price:.6f}, Target: {target_price:.6f}, Stop: {stop_price:.6f}')
                self.log(f'Using 50% equity: {equity_to_use:.2f}, Size: {size:.6f}')
                
                # Plot signals
                if self.p.plot_signals:
                    self.sell_signals[0] = entry_price
                    self.target_levels[0] = target_price
                    self.stop_levels[0] = stop_price
                
                # Execute orders with specified size
                self.sell(size=size)
                self.buy(size=size, exectype=bt.Order.Limit, price=target_price, 
                        oco=self.buy(size=size, exectype=bt.Order.Stop, price=stop_price))
                        
    def notify_trade(self, trade):
        """Notification for trade events"""
        if trade.isclosed:
            self.log(f'TRADE CLOSED - Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
            
            # Mark the exit on the chart
            if self.p.plot_signals and trade.pnl != 0:
                if trade.pnl > 0:
                    # Profitable trade
                    self.target_levels[0] = self.data.close[0]
                else:
                    # Losing trade
                    self.stop_levels[0] = self.data.close[0]


 days=days)
    
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
                      from_cluster='Best',  # Use best-performing cluster
                      percent_equity=50,    # Use 50% of equity per trade
                      plot_signals=True)    # Plot entry/exit signals
    
    # Set broker parameters
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% fee
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run strategy
    print(f'Starting portfolio value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strategy = results[0]
    print(f'Final portfolio value: {cerebro.broker.getvalue():.2f}')
    
    # Print analysis
    print(f'Return: {strategy.analyzers.returns.get_analysis()["rtot"]:.2%}')
    print(f'Max Drawdown: {strategy.analyzers.drawdown.get_analysis()["max"]["drawdown"]:.2%}')
    
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    print(f'Total Trades: {trade_analysis.total.closed}')
    if trade_analysis.total.closed > 0:
        print(f'Win Rate: {trade_analysis.won.total/trade_analysis.total.closed:.2%}')
        print(f'Average P/L: {trade_analysis.pnl.net/trade_analysis.total.closed:.4f}')
    
    # Configure plot
    plt_style = dict(
        style='candlestick',
        barup='green',
        bardown='red',
        barwidth=0.5,
        bartrans=0.0,
        volume=False,
        barup_wick='green',
        bardown_wick='red',
        valuetags=True,
    )
    
    # Add indicator plotting styling
    plt_schemes = {
        'Buy Signals': dict(
            marker='^',
            markersize=8,
            color='green',
            fillstyle='full',
            ls='',
            plotskip=False,
        ),
        'Sell Signals': dict(
            marker='v',
            markersize=8,
            color='red',
            fillstyle='full',
            ls='',
            plotskip=False,
        ),
        'Upper Band': dict(
            color='blue',
            ls='--',
            linewidth=1,
            plotskip=False,
        ),
        'Lower Band': dict(
            color='blue',
            ls='--',
            linewidth=1,
            plotskip=False,
        ),
        'Target Levels': dict(
            marker='o',
            markersize=6,
            color='darkgreen',
            fillstyle='full',
            ls='',
            plotskip=False,
        ),
        'Stop Levels': dict(
            marker='o',
            markersize=6,
            color='darkred',
            fillstyle='full',
            ls='',
            plotskip=False,
        ),
    }
    
    # Plot results with custom formatting
    figs = cerebro.plot(iplot=False, style=plt_style, scheme=plt_schemes)
    
    # Enhance figure after plotting
    if figs and len(figs) > 0:
        fig = figs[0][0]  # Get the main figure
        
        # Add title with strategy info
        fig.suptitle(f'{symbol} - {timeframe} - 50% Equity Per Trade', fontsize=14)
        
        # Add legend to the first subplot
        ax = fig.axes[0]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best')
        
        # Save plot
        save_filename = f"{symbol}_{timeframe}_trades.png"
        fig.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as {save_filename}")
        
        # Show plot
        plt.show()


if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run strategy
    run_strategy(symbol='DOGEUSDT', timeframe='5m', days=16)