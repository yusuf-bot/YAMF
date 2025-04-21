import backtrader as bt
from binance.client import Client
import pandas as pd
from datetime import datetime
import pytz
import time

# Initialize Binance client (no keys needed for public data)
client = Client()

# Fetch historical data for DOGEUSDT, 1m interval

def fetch_binance_data(symbol='DOGEUSDT', interval='1m', start_str='1 Jan 2024', end_str='28 Jan 2024'):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=[
        'datetime', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
    ])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    return df

class VoltyExpanCloseStrategy(bt.Strategy):
    params = (
        ('atr_period', 5),
        ('atr_mult', 0.75),
    )

    def __init__(self):
        self.tr = bt.indicators.TrueRange()
        self.atr = bt.indicators.SimpleMovingAverage(self.tr, period=self.p.atr_period)
        self.order = None

    def next(self):
        if len(self) <= self.p.atr_period:
            return

        if self.order:
            self.cancel(self.order)

        stop_dist = self.atr[0] * self.p.atr_mult
        cash = self.broker.getcash()
        size = int(cash / self.data.close[-1])

        long_stop = self.data.close[-1] + stop_dist
        short_stop = self.data.close[-1] - stop_dist

        if not self.position:
            self.buy(exectype=bt.Order.Stop, price=long_stop, size=size)
            self.sell(exectype=bt.Order.Stop, price=short_stop, size=size)

# Get Binance data
data_df = fetch_binance_data()

# Feed to Backtrader
data = bt.feeds.PandasData(dataname=data_df)

# Backtest engine
cerebro = bt.Cerebro()
cerebro.addstrategy(VoltyExpanCloseStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(1000)

print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
cerebro.run()
print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
cerebro.plot()
