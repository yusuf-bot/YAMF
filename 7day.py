import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import ccxt
import time

class SevenDayHighLowStrategy(bt.Strategy):
    params = (
        ('lookback', 7),
        ('sma_period', 200),
        ('order_percentage', 0.5),  # 50% of capital for each order
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        self.sma200 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.p.sma_period)

        # Track high/low with 1-bar delay
        self.highest = bt.ind.Highest(self.datas[0].high(-1), period=self.p.lookback, subplot=False)
        self.lowest = bt.ind.Lowest(self.datas[0].low(-1), period=self.p.lookback, subplot=False)

        # Track capital growth (initialize with starting capital)
        self.capital_growth = [self.broker.getvalue()]
        
        # Add a trade counter
        self.trade_count = 0
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'[{dt}] {txt}')

    def next(self):
        # Record current capital
        self.capital_growth.append(self.broker.getvalue())

        if self.position:
            if self.dataclose[0] > self.highest[0]:
                self.log(f'SELL EXECUTED, Price: {self.dataclose[0]:.2f}, Value: {self.broker.getvalue():.2f}')
                self.close()
                self.trade_count += 1
                pnl = self.broker.getvalue() - self.initial_cash
            self.log(f'PnL after sell: {pnl:.2f} ({(pnl / self.initial_cash) * 100:.2f}%)')
        else:
            if self.dataclose[0] < self.lowest[0] and self.dataclose[0] > self.sma200[0]:
                # Calculate the size based on 50% of current portfolio value
                cash = self.broker.getcash()
                price = self.dataclose[0]
                size = int((cash * self.p.order_percentage) / price)
                self.log(f'BUY EXECUTED, Price: {price:.2f}, Size: {size}, Value: {self.broker.getvalue():.2f}')
                self.buy(size=size)
                self.trade_count += 1

    def stop(self):
        # Print final stats when the backtest ends
        self.log(f'End of backtest. Total trades: {self.trade_count}')
        self.log(f'Final portfolio value: {self.broker.getvalue():.2f}')
        pnl = self.broker.getvalue() - 100  # Assuming 100 is initial capital
        self.log(f'Profit/Loss: {pnl:.2f} ({pnl}%)')


# ========= CSV Loader or Binance Fetch Function =========
def get_binance_data(symbol='PEPEUSDT', timeframe='5m', days=21):
    specific_filename = f"{symbol}.csv"
    detailed_filename = f"{symbol}_{timeframe}_{days}d.csv"

    if os.path.exists(specific_filename):
        print(f"Loading data from {specific_filename}")
        df = pd.read_csv(specific_filename, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded. Total candles: {len(df)}")
        return df

    if os.path.exists(detailed_filename):
        print(f"Loading data from {detailed_filename}")
        df = pd.read_csv(detailed_filename, parse_dates=['datetime'], index_col='datetime')
        print(f"Data loaded. Total candles: {len(df)}")
        return df

    print(f"Fetching data from Binance for {symbol}, timeframe {timeframe}, {days} days")
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
            time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)

    df.to_csv(specific_filename)
    print(f"Data saved to {specific_filename}")
    df.to_csv(detailed_filename)
    print(f"Data also saved to {detailed_filename}")
    
    return df


# ========= Backtesting =========
if __name__ == '__main__':
    print("Starting backtest with the SevenDayHighLowStrategy v 1.3.0")
    df = get_binance_data('PEPEUSDT', '5m', days=365)  # Increased to 30 days for better testing

    class MyDataFeed(bt.feeds.PandasData):
        params = (('datetime', None),)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SevenDayHighLowStrategy)

    data = MyDataFeed(dataname=df)
    cerebro.adddata(data)

    initial_capital = 100
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.0010)  # 0.1% commission, adjusted to be more realistic

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # ========= Plot Capital Growth =========
    strategy = results[0]
    
    # Create a DataFrame for better plotting
    capital_growth_df = pd.DataFrame(strategy.capital_growth, columns=['Portfolio Value'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(capital_growth_df.index, capital_growth_df['Portfolio Value'])
    plt.title('Capital Growth Over Time')
    plt.xlabel('Bar Number')
    plt.ylabel('Portfolio Value ($)')
    
    # Set y-axis to start from a lower value to better show changes
    y_min = max(min(strategy.capital_growth) * 0.95, 0)  # 5% below minimum but not below 0
    y_max = max(strategy.capital_growth) * 1.05  # 5% above maximum
    plt.ylim(y_min, y_max)
    
    plt.grid(True)
    plt.tight_layout()
    
    # Add horizontal line for initial capital
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.legend()
    
    plt.savefig('capital_growth.png')
    print("Capital growth chart saved as 'capital_growth.png'")
    plt.show()