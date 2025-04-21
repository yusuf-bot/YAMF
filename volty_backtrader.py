from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
import backtrader as bt
import backtrader.analyzers as btanalyzers

# Load environment variables
load_dotenv()

# Get API keys from environment variables
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

# Strategy parameters
length = 5  # SMA length for TR
num_atrs = 0.75  # ATR multiplier
initial_capital = 100
symbol = "DOGE/USD"  # Crypto symbol format

# Define the Volatility Expansion Strategy
class VoltyExpansionStrategy(bt.Strategy):
    params = (
        ('length', 1),
        ('num_atrs', 1),
        ('debug', True),
    )
    
    def __init__(self):
        # Calculate True Range and ATR
        self.high_low = self.data.high - self.data.low
        self.high_close = abs(self.data.high - self.data.close(-1))
        self.low_close = abs(self.data.low - self.data.close(-1))
        self.tr = bt.Max(self.high_low, self.high_close, self.low_close)
        self.atr = bt.indicators.SMA(self.tr, period=self.params.length)
        self.atrs = self.atr * self.params.num_atrs
        
        # Generate signals
        self.long_entry = self.data.close(-1) + self.atrs(-1)
        self.short_entry = self.data.close(-1) - self.atrs(-1)
        
        # Keep track of trades
        self.trades = []
        self.trade_entry_price = 0
        self.trade_entry_time = None
        self.trade_type = None
        
        # Track equity for comparison with original implementation
        self.equity = initial_capital
        self.unrealized_profit = 0
        
        # Debug variables
        self.debug_info = []
        
    def next(self):
        # Similar logic to original implementation
        current_price = self.data.close[0]
        
        # Debug info
        if self.params.debug and len(self.data) > self.params.length + 1:
            debug_entry = {
                'date': self.data.datetime.datetime(),
                'close': current_price,
                'atr': self.atr[0],
                'long_entry': self.long_entry[0],
                'short_entry': self.short_entry[0]
            }
            self.debug_info.append(debug_entry)
        
        # Check for long entry
        if not self.position:  # Only enter if we have no position
            if current_price > self.long_entry[0]:
                # Enter long position with fixed position sizing
                cash = self.broker.getcash()
                size = 100  # Use current price instead of entry price
                self.trade_entry_price = current_price
                self.trade_entry_time = self.data.datetime.datetime()
                self.trade_type = 'LONG'
                self.buy(size=size)
                print(f"BUY SIGNAL at {self.trade_entry_time}: Price={current_price}, Entry={self.long_entry[0]}")
            
            # Check for short entry
            elif current_price < self.short_entry[0]:
                # Enter short position with fixed position sizing
                cash = self.broker.getcash()
                size = 100  # Use current price instead of entry price
                self.trade_entry_price = current_price
                self.trade_entry_time = self.data.datetime.datetime()
                self.trade_type = 'SHORT'
                self.sell(size=size)
                print(f"SELL SIGNAL at {self.trade_entry_time}: Price={current_price}, Entry={self.short_entry[0]}")
        
        # Check for exit conditions if we have a position
        elif self.position.size > 0:  # Long position
            if current_price < self.short_entry[0]:  # Exit long at short entry
                self.close()
                print(f"CLOSE LONG at {self.data.datetime.datetime()}: Price={current_price}")
        
        elif self.position.size < 0:  # Short position
            if current_price > self.long_entry[0]:  # Exit short at long entry
                self.close()
                print(f"CLOSE SHORT at {self.data.datetime.datetime()}: Price={current_price}")
    
    def notify_trade(self, trade):
        if trade.isclosed:
            # Record trade details
            profit_loss_pct = 0
            if trade.price != 0 and trade.size != 0:
                profit_loss_pct = trade.pnlcomm / trade.price / abs(trade.size) * 100
                
            self.trades.append({
                'entry_time': self.trade_entry_time,
                'exit_time': self.data.datetime.datetime(),
                'type': self.trade_type,
                'entry_price': self.trade_entry_price,
                'exit_price': trade.price,
                'size': trade.size,
                'profit_loss': trade.pnl,
                'profit_loss_pct': profit_loss_pct
            })
            
            print(f"TRADE CLOSED: {self.trade_type}, Entry=${self.trade_entry_price:.2f}, Exit=${trade.price:.2f}, P/L=${trade.pnl:.2f} ({profit_loss_pct:.2f}%)")

# Function to get data from Alpaca and prepare for Backtrader
def get_alpaca_data(symbol, start, end, timeframe):
    data_client = CryptoHistoricalDataClient(api_key, secret_key)
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end
    )
    
    # Get data
    bars = data_client.get_crypto_bars(request_params)
    df = bars.df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Format for Backtrader
    df = df.set_index('timestamp')
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    
    return df

# Main function to run the backtest
def run_backtest():
    # Get historical data
    end = datetime(2025, 4, 15)
    start = datetime(2025,1,1)  # Increase backtest period to 60 days
    
    # Get data from Alpaca
    df = get_alpaca_data(symbol, start, end, TimeFrame.Minute)
    
    # Print data summary
    print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy
    cerebro.addstrategy(VoltyExpansionStrategy, length=length, num_atrs=num_atrs)
    
    # Create a data feed from the dataframe
    data = bt.feeds.PandasData(
        dataname=df,
        # Use proper column indices or names
        datetime=None,  # Use index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1  # No open interest data
    )
    
    # Add the data feed to cerebro
    cerebro.adddata(data)
    
    # Set our starting cash
    cerebro.broker.setcash(initial_capital)
    
    # Set commission - 0.1% per trade
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    
    # Run the backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Get analyzer results
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analysis.get('sharperatio', 0.0)  # Default to 0.0 if not available
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trade_analysis = strategy.analyzers.trades.get_analysis()

    # Calculate metrics
    final_equity = cerebro.broker.getvalue()
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Safely extract drawdown information
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0.0)

    # Safely get trade statistics
    total_trades = 0
    long_trades = 0
    short_trades = 0

    # Check if there were any trades
    if trade_analysis:
        total_trades = trade_analysis.get('total', {}).get('closed', 0)
        long_trades = trade_analysis.get('long', {}).get('total', 0)
        short_trades = trade_analysis.get('short', {}).get('total', 0)
    
    # Create trades dataframe
    trades_df = pd.DataFrame(strategy.trades)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join('c:\\Users\\ASAA\\Desktop\\projects\\vibe', 'backtest_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Sanitize symbol name for file paths by replacing / with _
    safe_symbol = symbol.replace('/', '_')
    
    # Save trades to CSV if we have any
    if not trades_df.empty:
        trades_csv_path = os.path.join(results_dir, f'volty_expan_{safe_symbol}_trades.csv')
        trades_df.to_csv(trades_csv_path, index=False)
    
    # Plot results
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price and equity data for plotting
    price_data = df['close']
    dates = df.index
    
    # Price chart
    ax1.plot(dates, price_data, label=f'{symbol}', color='white', alpha=0.8)
    
    # Plot entry points if we have trades
    if not trades_df.empty:
        long_entries = trades_df[trades_df['type'] == 'LONG']
        short_entries = trades_df[trades_df['type'] == 'SHORT']
        
        if not long_entries.empty:
            ax1.scatter(long_entries['entry_time'], long_entries['entry_price'], 
                       marker='^', color='green', s=100, label='Long Entry')
        
        if not short_entries.empty:
            ax1.scatter(short_entries['entry_time'], short_entries['entry_price'], 
                       marker='v', color='red', s=100, label='Short Entry')
    
    ax1.set_title(f'{symbol} with Volatility Expansion Close Strategy (Length: {length}, ATR Mult: {num_atrs})', fontsize=14)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add strategy metrics as text
    metrics_text = (
        f"Initial Capital: ${initial_capital}\n"
        f"Final Equity: ${final_equity:.2f}\n"
        f"Total Return: {total_return:.2f}%\n"
        f"Total Trades: {total_trades}\n"
        f"Long Trades: {long_trades}\n"
        f"Short Trades: {short_trades}\n"
        f"Max Drawdown: {max_drawdown:.2f}%\n"
        f"Sharpe Ratio: {sharpe_ratio}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='black', alpha=0.5)
    ax2.text(0.02, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save results as JPG
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'volty_expan_{safe_symbol}_results.jpg'), format='jpg', dpi=300)
    
    # Save summary results
    summary = {
        'symbol': symbol,
        'length': length,
        'num_atrs': num_atrs,
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date': end.strftime('%Y-%m-%d')
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, f'volty_expan_{safe_symbol}_summary.csv'), index=False)
    
    # Print summary
    print("\nVolatility Expansion Close Strategy Results:")
    print(f"Symbol: {symbol}")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${initial_capital}")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Long Trades: {long_trades}")
    print(f"Short Trades: {short_trades}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    
    print(f"\nResults saved to {results_dir}")
    if not trades_df.empty:
        print(f"Trades CSV saved to: {trades_csv_path}")

if __name__ == "__main__":
    run_backtest()