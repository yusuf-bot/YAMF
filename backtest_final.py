import backtrader as bt
import pandas as pd
from binance.client import Client
import datetime

# -------------------------------
# Step 1: Fetch 1-min ETHUSDT Data from Binance
# -------------------------------
client = Client()

symbol = 'ETHUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE
start_str = '17 Jan, 2025'
end_str = '24 May, 2025'

klines = client.get_historical_klines(symbol, interval, start_str, end_str)

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# -------------------------------
# Step 2: Smart Grid Strategy (Backtrader)
# -------------------------------
class SmartGridStrategy(bt.Strategy):
    params = dict(
        upper_bound=3000,
        lower_bound=2000,
        grid_qty=20,
        leverage=3,
        stake_pct=10,
        commission=0.25 / 100,
    )

    def __init__(self):
        self.grid_lines = []
        self.order_flags = []
        self.total_comm = 0.0
        self.orders = []

        step = (self.p.upper_bound - self.p.lower_bound) / (self.p.grid_qty - 1)
        for i in range(self.p.grid_qty):
            price = self.p.lower_bound + i * step
            self.grid_lines.append(price)
            self.order_flags.append(False)

        self.t3 = bt.ind.EMA(self.data.close, period=70) * (1 + 0.7) - bt.ind.EMA(bt.ind.EMA(self.data.close, period=70), period=70) * 0.7

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f'[{dt}] {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dt = self.datas[0].datetime.datetime(0)
        if order.status == order.Completed:
            self.total_comm += order.executed.comm
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size:.4f}, Cost={order.executed.value:.2f}, Comm={order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size:.4f}, Cost={order.executed.value:.2f}, Comm={order.executed.comm:.2f}')
        elif order.status == order.Canceled:
            self.log(f'ORDER CANCELLED')
        elif order.status == order.Margin:
            self.log(f'ORDER MARGIN ISSUE')
        elif order.status == order.Rejected:
            self.log(f'ORDER REJECTED')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE CLOSED: Gross PnL={trade.pnl:.2f}, Net PnL={trade.pnlcomm:.2f}')

    def next(self):
        contracts = self.p.stake_pct * (self.broker.getvalue() / 100) / self.data.close[0] * self.p.leverage

        # Entry Logic
        if self.t3[0] > self.t3[-1] and self.data.close[0] > self.t3[0]:
            for i in range(len(self.grid_lines)):
                if self.data.close[0] < self.grid_lines[i] and not self.order_flags[i]:
                    self.buy(size=contracts)
                    self.order_flags[i] = True
                    break

        # Exit Logic
        for i in range(len(self.grid_lines) - 1):
            if self.order_flags[i]:
                next_price = self.grid_lines[i + 1]
                if self.data.close[0] > next_price:
                    self.sell(size=contracts)
                    self.order_flags[i] = False
                    break

    def stop(self):
        self.log(f'Final Portfolio Value: {self.broker.getvalue()}')
        self.log(f'Total Commission Paid: {self.total_comm:.2f}')
        if self.position:
            self.log(f'OPEN TRADE: Size={self.position.size:.4f}, Entry Price={self.position.price:.2f}, Current Price={self.data.close[0]:.2f}')


# -------------------------------
# Step 3: Run Backtest
# -------------------------------
data_feed = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()
cerebro.addstrategy(SmartGridStrategy)
cerebro.adddata(data_feed)
cerebro.broker.setcash(100.0)
cerebro.broker.setcommission(commission=0.00125)

print('Starting Portfolio Value:', cerebro.broker.getvalue())
cerebro.run()
print('Final Portfolio Value:', cerebro.broker.getvalue(),cerebro.broker.getcash())

cerebro.plot(style='candlestick')
