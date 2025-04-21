from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import asyncio
import time
from datetime import datetime, timedelta
import logging

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Get API keys from environment variables
API_KEY =  os.environ.get('ALPACA_API_KEY') # Fallback to hardcoded values if env var not found
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
PAPER = True  # Set to False for live trading

# Strategy parameters - matching PineScript exactly
SYMBOL = "MSFT"  # Microsoft stock
TIMEFRAME = TimeFrame.Minute  # 1-minute bars
LENGTH = 5  # Length for TR calculation
ATR_MULT = 0.75  # ATR multiplier
BASE_CASH = 1000.0  # Base position size in USD

class VoltyExpanCloseStrategy:
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        # Use the enum value instead of a string
        self.stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)
        
        self.initial_equity = None
        self.current_position = 0
        self.df = None
        self.last_update_time = None
        
        # Track active orders
        self.active_long_order_id = None
        self.active_short_order_id = None
        self.last_bar_timestamp = None
        
        # Initialize equity
        self.get_account_equity()
        
        # Get initial data
        self.update_historical_data()

    def get_account_equity(self):
        """Get the current account equity"""
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            
            if not self.initial_equity:
                self.initial_equity = equity
                logger.info(f"Initial equity set to: ${self.initial_equity}")
                
            return equity
        except Exception as e:
            logger.error(f"Error getting account equity: {e}")
            return None

    def update_historical_data(self):
        """Fetch historical price data for the strategy"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=3)  # Get 3 days of data
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[SYMBOL],
                timeframe=TIMEFRAME,
                start=start_time,
                end=end_time,
                feed=DataFeed.IEX  # Use enum instead of string
            )
            
            bars_data = self.data_client.get_stock_bars(request_params)
            
            # Convert to dataframe
            df = bars_data.df.reset_index()
            
            # If there's no data or not enough data, exit
            if len(df) < LENGTH + 2:
                logger.error(f"Not enough data points. Got {len(df)}, need at least {LENGTH + 2}")
                return False
                
            # Calculate indicators
            self.df = self.calculate_indicators(df)
            self.last_update_time = datetime.now()
            logger.info(f"Updated historical data with {len(self.df)} bars")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return False
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators exactly as in PineScript"""
        # Make sure we have required columns
        if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            logger.error("Required price columns are missing")
            return df
            
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        
        # True Range calculation (exactly as in ta.tr)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # ATR calculation using SMA (exactly as in ta.sma(ta.tr, length))
        df['tr_sma'] = df['tr'].rolling(window=LENGTH).mean()
        df['atrs'] = df['tr_sma'] * ATR_MULT
        
        # Entry price calculations (matching PineScript)
        df['long_entry_price'] = df['close'] + df['atrs']
        df['short_entry_price'] = df['close'] - df['atrs']
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
        
        return df


    def get_position_size(self, price):
        """Calculate position size based on account equity growth"""
        current_equity = self.get_account_equity()
        
        if not current_equity:
            logger.error("Failed to get current equity")
            return 0
        
        equity_growth = current_equity / self.initial_equity
        
        # Limit position size to a percentage of available buying power
        account = self.trading_client.get_account()
        available_cash = float(account.buying_power) * 0.95  # Use 95% of available buying power
        
        position_cash = min(BASE_CASH * equity_growth, available_cash)
        position_qty = int(position_cash / price)  # Whole shares for stocks
        
        logger.info(f"Position cash: ${position_cash:.2f}, Position quantity: {position_qty} shares")
        
        return position_qty

    def get_current_position(self):
        """Get current position for the symbol"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                if position.symbol == SYMBOL:
                    qty = float(position.qty)
                    side = 'long' if qty > 0 else 'short'
                    return qty, side
            return 0, None
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return 0, None

    # check_and_cancel_orders method remains the same

    def place_stop_order(self, side, qty, stop_price):
        """Place a stop order through Alpaca - matching PineScript behavior"""
        try:
            # For stocks, we need whole shares
            qty = int(qty)
            stop_price = round(stop_price, 2)  # Round to 2 decimal places
            
            if qty <= 0:
                logger.error(f"Invalid quantity: {qty}")
                return None
            
            # Check if we have enough balance
            account = self.trading_client.get_account()
            
            # For sell orders, check if we have enough shares
            if side == "sell":
                positions = self.trading_client.get_all_positions()
                stock_position = 0
                for position in positions:
                    if position.symbol == SYMBOL:
                        stock_position = float(position.qty)
                        break
                
                if stock_position < qty:
                    logger.warning(f"Insufficient {SYMBOL} shares. Requested: {qty}, Available: {stock_position}")
                    qty = int(stock_position)  # Adjust quantity to available amount
                    if qty <= 0:
                        return None
            
            # For buy orders, check if we have enough USD
            else:  # side == "buy"
                required_usd = qty * stop_price * 1.01  # Add 1% buffer for price movement
                available_usd = float(account.buying_power)
                
                if available_usd < required_usd:
                    logger.warning(f"Insufficient USD balance. Required: ${required_usd}, Available: ${available_usd}")
                    qty = int((available_usd / stop_price) * 0.99)  # Adjust quantity and add buffer
                    if qty <= 0:
                        return None
                    
            logger.info(f"Placing {side} stop order for {qty} shares of {SYMBOL} at ${stop_price}")
            
            # Create a stop order - this matches PineScript strategy.entry with stop parameter
            order_data = StopOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,  # Use DAY for stocks
                stop_price=stop_price
            )
            
            order = self.trading_client.submit_order(order_data=order_data)
            logger.info(f"Stop order placed: {order.id} at ${stop_price}")
            return order.id
        
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            return None

    # check_for_signals method remains the same

    async def handle_bar(self, data):
        """Handle incoming bar data"""
        try:
            # Extract price data from the bar
            symbol = data.symbol
            
            if symbol != SYMBOL:
                return
                
            # Update the latest bar in our dataframe
            new_bar = pd.DataFrame({
                'timestamp': [data.timestamp],
                'open': [data.open],
                'high': [data.high],
                'low': [data.low],
                'close': [data.close],
                'volume': [data.volume]
            })
            
            # Check if we need to update historical data (every 15 minutes)
            current_time = datetime.now()
            if (self.last_update_time is None or 
                (current_time - self.last_update_time).total_seconds() > 900):  # 15 minutes
                self.update_historical_data()
            else:
                # Append the new bar to our dataframe and recalculate indicators
                if self.df is not None:
                    self.df = pd.concat([self.df, new_bar], ignore_index=True)
                    self.df = self.calculate_indicators(self.df)
                
            # Check for signals on bar close
            self.check_for_signals()
            
        except Exception as e:
            logger.error(f"Error handling bar data: {e}")

    async def start_streaming(self):
        """Start streaming market data"""
        try:
            # Subscribe to stock bars (OHLCV)
            # Note: The feed is already specified in the stream initialization
            self.stream.subscribe_bars(self.handle_bar, SYMBOL)
            
            # Start the stream
            logger.info(f"Starting data stream for {SYMBOL} using IEX feed")
            await self.stream._run_forever()
        except Exception as e:
            logger.error(f"Error starting data stream: {e}")

    def run(self):
        """Run the strategy"""
        logger.info("Starting Volatility Expansion Close Strategy")
        
        # Debug account information
        account = self.trading_client.get_account()
        logger.info(f"Account cash: ${account.cash}")
        logger.info(f"Account buying power: ${account.buying_power}")
        
        # Make sure we have initial data
        if not self.update_historical_data():
            logger.error("Failed to get initial data. Exiting.")
            return
        
        # Start the streaming data in an async event loop
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.start_streaming())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            # Cancel any open orders on exit
            self.check_and_cancel_orders()
        finally:
            loop.close()

if __name__ == "__main__":
    strategy = VoltyExpanCloseStrategy()
    strategy.run()