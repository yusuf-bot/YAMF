from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import asyncio
import time
from datetime import datetime, timedelta
import logging
from collections import deque

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
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
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')
PAPER = True  # Set to False for live trading

# Strategy parameters
SYMBOL = "AMD"  # Alpaca Stock format
TIMEFRAME = TimeFrame.Minute  # 1-minute bars
LENGTH = 5  # Length for ATR calculation
ATR_MULT = 0.75  # ATR multiplier
BASE_CASH = 10000.0  # Base position size in USD

class VoltyExpanCloseStrategy:
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        self.stream = StockDataStream(API_KEY, SECRET_KEY)
        
        self.initial_equity = None
        self.current_position = 0
        self.last_entry_price = None
        self.entry_type = None  # 'long' or 'short'
        self.latest_price = None
        
        # Store candles in a deque for easy management
        self.candles = deque(maxlen=100)  # Store up to 100 candles
        self.ready_to_trade = False
        self.active_orders = {}  # Track active stop orders
        
        # Initialize equity
        self.get_account_equity()

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

    def calculate_indicators(self):
        """Calculate all technical indicators needed for the strategy"""
        if len(self.candles) < LENGTH + 2:
            logger.warning(f"Not enough candles to calculate indicators. Have {len(self.candles)}, need {LENGTH + 2}")
            return None
            
        # Convert candles to dataframe
        df = pd.DataFrame(self.candles)
        
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        
        # True Range calculation
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # ATR calculation
        df['atr'] = df['tr'].rolling(window=LENGTH).mean()
        df['atrs'] = df['atr'] * ATR_MULT
        
        # Entry price calculations
        df['long_entry_price'] = df['prev_close'] + df['atrs']
        df['short_entry_price'] = df['prev_close'] - df['atrs']
        
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
        
        # For stocks, we need whole shares
        position_qty = int(position_cash / price)
        
        # Ensure at least 1 share
        position_qty = max(position_qty, 1)
        
        logger.info(f"Current equity: ${current_equity}, Growth multiplier: {equity_growth:.2f}")
        logger.info(f"Available cash: ${available_cash:.2f}")
        logger.info(f"Position cash: ${position_cash:.2f}, Position quantity: {position_qty} shares")
        
        return position_qty

    def get_current_position(self):
        """Get current position for the symbol"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                if position.symbol == SYMBOL:  # No need to replace "/" for stock symbols
                    qty = float(position.qty)
                    side = 'long' if qty > 0 else 'short'
                    return qty, side
            return 0, None
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return 0, None

    def cancel_existing_orders(self):
        """Cancel all existing orders for the symbol"""
        try:
            orders = self.trading_client.get_orders()
            for order in orders:
                if order.symbol == SYMBOL:  # No need to replace "/" for stock symbols
                    self.trading_client.cancel_order_by_id(order.id)
                    logger.info(f"Cancelled order: {order.id}")
                    
                    # Remove from tracking
                    if order.id in self.active_orders:
                        del self.active_orders[order.id]
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    def place_stop_order(self, side, qty, stop_price):
        """Place a stop order through Alpaca"""
        try:
            # First cancel any existing orders
            self.cancel_existing_orders()
            
            if qty <= 0:
                logger.error(f"Invalid quantity: {qty}")
                return None
                
            logger.info(f"Placing {side} stop order for {qty} shares of {SYMBOL} at ${stop_price:.2f}")
            
            order_data = StopOrderRequest(
                symbol=SYMBOL,  # No need to replace "/" for stock symbols
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_price
            )
            
            order = self.trading_client.submit_order(order_data=order_data)
            logger.info(f"Stop order placed: {order.id}")
            
            # Track the order
            self.active_orders[order.id] = {
                "side": side,
                "qty": qty,
                "stop_price": stop_price
            }
            
            return order
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            return None


    def close_position(self):
        """Close any existing position for the symbol"""
        try:
            logger.info(f"Closing position for {SYMBOL}")
            self.trading_client.close_position(symbol_or_asset_id=SYMBOL)  # No need to replace "/" for stock symbols
            logger.info("Position closed successfully")
        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def check_for_signals(self):
        """Check the latest data for entry signals"""
        if not self.ready_to_trade:
            if len(self.candles) >= LENGTH + 2:
                logger.info(f"Collected {len(self.candles)} candles, ready to start trading")
                self.ready_to_trade = True
            else:
                logger.info(f"Collecting candles: {len(self.candles)}/{LENGTH + 2}")
                return
                
        # Calculate indicators
        df = self.calculate_indicators()
        if df is None:
            return
            
        # Get the latest complete bar (not the current forming one)
        latest_bar = df.iloc[-2]
        
        # Get current price and position
        current_price = self.latest_price
        position_qty, position_side = self.get_current_position()
        
        if current_price is None:
            logger.error("No current price available")
            return
            
        logger.info(f"Current price: ${current_price}, Position: {position_qty} ({position_side})")
        
        # If we have no position, set up stop orders
        if position_qty == 0:
            long_entry_price = latest_bar['long_entry_price']
            short_entry_price = latest_bar['short_entry_price']
            
            logger.info(f"Long entry: ${long_entry_price}, Short entry: ${short_entry_price}")
            
            # Calculate position size based on current price
            position_qty = self.get_position_size(current_price)
            
            # Place stop orders for both long and short entries
            self.place_stop_order("buy", position_qty, long_entry_price)
            self.place_stop_order("sell", position_qty, short_entry_price)
            
            logger.info(f"Stop orders placed - Long: ${long_entry_price}, Short: ${short_entry_price}")

    def check_order_status(self):
        """Check the status of active orders"""
        if not self.ready_to_trade:
            return
            
        try:
            orders = self.trading_client.get_orders(status=OrderStatus.OPEN)
            
            # Update our tracking
            current_orders = {}
            for order in orders:
                if order.symbol == SYMBOL:  # No need to replace "/" for stock symbols
                    current_orders[order.id] = order
            
            # Check if any orders have been filled
            for order_id in list(self.active_orders.keys()):
                if order_id not in current_orders:
                    logger.info(f"Order {order_id} has been filled or cancelled")
                    
                    # Cancel other orders as we now have a position
                    self.cancel_existing_orders()
                    
                    # Remove from tracking
                    del self.active_orders[order_id]
                    
        except Exception as e:
            logger.error(f"Error checking order status: {e}")

    async def handle_bar(self, data):
        """Handle incoming bar data"""
        try:
            # Extract price data from the bar
            symbol = data.symbol
            
            if symbol != SYMBOL:
                return
                
            # Store the complete candle
            candle = {
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            }
            
            self.candles.append(candle)
            self.latest_price = data.close
            
            logger.info(f"Received new candle: O:{data.open} H:{data.high} L:{data.low} C:{data.close}")
            
            # Check for signals
            self.check_for_signals()
            
            # Check order status
            self.check_order_status()
            
        except Exception as e:
            logger.error(f"Error handling bar data: {e}")

    async def handle_trade(self, data):
        """Handle incoming trade data to update latest price"""
        try:
            if data.symbol == SYMBOL:
                self.latest_price = data.price
        except Exception as e:
            logger.error(f"Error handling trade data: {e}")

    async def start_streaming(self):
        """Start streaming market data with retry mechanism"""
        max_retries = 5
        retry_count = 0
        base_delay = 2  # Base delay in seconds
        
        while retry_count < max_retries:
            try:
                # Subscribe to Stock trades
                self.stream.subscribe_trades(self.handle_trade, SYMBOL)
                # Subscribe to Stock bars (OHLCV)
                self.stream.subscribe_bars(self.handle_bar, SYMBOL)
                
                # Start the stream
                logger.info(f"Starting data stream for {SYMBOL}")
                await self.stream._run_forever()
                
                # If we get here without an exception, break the loop
                break
                
            except ValueError as e:
                if "connection limit exceeded" in str(e).lower():
                    retry_count += 1
                    delay = base_delay * (2 ** retry_count)  # Exponential backoff
                    
                    logger.warning(f"Connection limit exceeded. Retry {retry_count}/{max_retries} in {delay} seconds...")
                    
                    # Close the current connection if possible
                    try:
                        await self.stream._close_connection()
                    except:
                        pass
                        
                    # Create a new stream instance
                    self.stream = StockDataStream(API_KEY, SECRET_KEY)
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Error starting data stream: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Error starting data stream: {e}")
                raise
                
        if retry_count >= max_retries:
            logger.error(f"Failed to connect after {max_retries} retries. Exiting.")
            raise ConnectionError("Maximum connection retries exceeded")

    def run(self):
        """Run the strategy"""
        logger.info("Starting Volatility Expansion Close Strategy for Stock")
        
        # Debug account information
        account = self.trading_client.get_account()
        logger.info(f"Account cash: ${account.cash}")
        logger.info(f"Account buying power: ${account.buying_power}")
        
        # Start the streaming data in an async event loop
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.start_streaming())
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            logger.info("Consider using historical data instead or waiting before retrying")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Clean up
            try:
                loop.run_until_complete(self.stream._close_connection())
            except:
                pass
            loop.close()

if __name__ == "__main__":
    strategy = VoltyExpanCloseStrategy()
    strategy.run()