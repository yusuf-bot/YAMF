import backtrader as bt
import yfinance as yf
import pandas as pd

# Download data from yfinance - using ^GSPC instead of SPX for S&P 500
data_df = yf.download('AMD', start='2022-01-01', end='2022-12-31', group_by='ticker')

# Check if we have data
if data_df.empty:
    print("No data downloaded. Please check the ticker symbol.")
    exit()

# Flatten multi-level columns
data_df.columns = data_df.columns.get_level_values(1)

# Set index name for Backtrader compatibility
data_df.index.name = 'datetime'

# Define strategy
class SMACrossover(bt.Strategy):
    params = (
        ('short_period', 10),
        ('long_period', 30),
    )
    
    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)

    def next(self):
        # Make sure we have enough data
        if len(self) <= self.params.long_period:
            return
            
        if not self.position:
            if self.sma_short[0] > self.sma_long[0]:
                self.buy()
        else:
            if self.sma_short[0] < self.sma_long[0]:
                self.close()

# Feed into Backtrader
data = bt.feeds.PandasData(dataname=data_df)

# Setup Cerebro
cerebro = bt.Cerebro()
cerebro.addstrategy(SMACrossover)
cerebro.adddata(data)
cerebro.broker.setcash(10000)

# Run Backtest
print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")

# Plot result
cerebro.plot()