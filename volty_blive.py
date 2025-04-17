import alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import logging
import os

# Setup logging
log_dir = os.path.join('c:\\Users\\ASAA\\Desktop\\projects\\vibe', 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'volty_expan_live.log')),
        logging.StreamHandler()
    ]
)

from dotenv import load_dotenv

# Apply nest_asyncio to allow running the event loop
load_dotenv()

# Get API keys from environment variables
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

paper = True  # Set to True for paper trading

# Initialize clients
trading_client = TradingClient(api_key, secret_key, paper=paper)
data_client = CryptoHistoricalDataClient(api_key, secret_key)

# Strategy parameters
symbol = "BTC/USD"
length = 1  # SMA length for TR
num_atrs = 1  # ATR multiplier
position_size = 1  # Number of shares to trade

def get_historical_data():
    """Get historical data for the symbol"""
    end = datetime.now()
    start = end - timedelta(days=30)  # Get 30 days of data
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end
    )
    
    try:
        bars = data_client.get_crypto_bars(request_params)
        df = bars.df.reset_index()
        return df
    except Exception as e:
        logging.error(f"Error getting historical data: {e}")
        return None

def calculate_signals(df):
    """Calculate signals based on the Volatility Expansion Close strategy"""
    # Calculate True Range and ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_sma'] = df['tr'].rolling(window=length).mean()
    df['atrs'] = df['atr_sma'] * num_atrs
    
    # Generate signals
    df['long_entry_price'] = df['close'] + df['atrs']
    df['short_entry_price'] = df['close'] - df['atrs']
    
    return df

def get_current_position():
    """Get current position for the symbol"""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == symbol:
                return float(position.qty)
        return 0
    except Exception as e:
        logging.error(f"Error getting positions: {e}")
        return 0

def place_order(side, qty, order_type="market"):
    """Place an order"""
    try:
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        order = trading_client.submit_order(order_data=order_data)
        logging.info(f"Order placed: {side} {qty} shares of {symbol}")
        return order
    except Exception as e:
        logging.error(f"Error placing order: {e}")
        return None

def run_strategy():
    """Run the Volatility Expansion Close strategy"""
    logging.info("Starting Volatility Expansion Close strategy for 24/7 crypto market")
    
    while True:
        try:
            # Get historical data
            df = get_historical_data()
            if df is None or len(df) < length + 1:
                logging.warning("Not enough data to calculate signals")
                time.sleep(60)
                continue
            
            # Calculate signals
            df = calculate_signals(df)
            
            # Get the latest signal
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Get current position
            current_position = get_current_position()
            
            # Check for entry conditions
            if not pd.isna(latest['close']) and not pd.isna(previous['close']):
                current_price = latest['close']
                long_entry = previous['long_entry_price']
                short_entry = previous['short_entry_price']
                
                logging.info(f"Current price: {current_price}, Long entry: {long_entry}, Short entry: {short_entry}")
                logging.info(f"Current position: {current_position}")
                
                # Long entry
                if current_position <= 0 and current_price >= long_entry:
                    # Close any existing short position
                    if current_position < 0:
                        place_order("buy", abs(current_position))
                    
                    # Enter long position
                    place_order("buy", position_size)
                    logging.info(f"Long entry triggered at {current_price}")
                
                # Short entry
                elif current_position >= 0 and current_price <= short_entry:
                    # Close any existing long position
                    if current_position > 0:
                        place_order("sell", current_position)
                    
                    # Enter short position
                    place_order("sell", position_size)
                    logging.info(f"Short entry triggered at {current_price}")
            
            # Wait for next check
            logging.info("Waiting for next check...")
            time.sleep(60)  # Check every hour
            
        except Exception as e:
            logging.error(f"Error in strategy execution: {e}")
            time.sleep(60)

if __name__ == "__main__":
    # For crypto markets, we don't need to check if the market is open
    logging.info("Starting strategy for 24/7 crypto market...")
    run_strategy()