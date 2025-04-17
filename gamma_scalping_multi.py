# Import necessary packets

from datetime import datetime, timedelta, time as dt_time
import time
import asyncio
import pandas as pd
import pytz
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import nest_asyncio
from alpaca.data.historical.option import OptionHistoricalDataClient, OptionLatestQuoteRequest
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.trading.models import TradeUpdate
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest, GetCalendarRequest
from alpaca.trading.enums import AssetStatus, ContractType, AssetClass
from dotnev import load_dotenv
import os
# Apply nest_asyncio to allow running the event loop
load_dotenv()

# Get API keys from environment variables
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

nest_asyncio.apply()
paper = True
# Initialize Alpaca clients

trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
trade_update_stream = TradingStream(api_key=api_key, secret_key=secret_key, paper=paper)
stock_data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
option_data_client = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def is_market_open():
    # Get current time in Eastern timezone (US market)
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if current time is between 9:30 AM and 4:00 PM Eastern
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    current_time = now.time()
    
    if market_open <= current_time <= market_close:
        # Check if today is a market holiday
        today = now.date()
        calendar_request = GetCalendarRequest(
            start=today.isoformat(),
            end=today.isoformat()
        )
        calendar = trading_client.get_calendar(calendar_request)
        
        # If calendar is empty, it's a holiday
        return len(calendar) > 0
    
    return False

# Fix the get_next_market_open function
def get_next_market_open():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Get calendar for the next week
    start_date = now.date()
    end_date = (now + timedelta(days=7)).date()
    
    calendar_request = GetCalendarRequest(
        start=start_date.isoformat(),
        end=end_date.isoformat()
    )
    calendar = trading_client.get_calendar(calendar_request)
    
    for day in calendar:
        try:
            # Check if day.open is already a datetime object
            if isinstance(day.open, datetime):
                open_date = day.open.astimezone(eastern)
            else:
                # Parse the date string if it's a string
                open_time_str = day.open
                if isinstance(open_time_str, str):
                    # Remove the 'Z' and add timezone info if needed
                    if open_time_str.endswith('Z'):
                        open_time_str = open_time_str[:-1] + '+00:00'
                    
                    # Parse the ISO format string
                    open_date = datetime.fromisoformat(open_time_str).astimezone(eastern)
                else:
                    # If it's neither a datetime nor a string, use the date with default market open time
                    open_date = datetime.combine(
                        day.date if hasattr(day, 'date') else start_date,
                        dt_time(9, 30)
                    ).replace(tzinfo=eastern)
            
            if open_date > now:
                return open_date
        except Exception as e:
            print(f"Error parsing market open time: {e}")
            # Continue to the next day
            continue
    
    # If no open days found in the next week, return next Monday 9:30 AM
    next_monday = now + timedelta(days=(7 - now.weekday()))
    next_monday = next_monday.replace(hour=9, minute=30, second=0, microsecond=0)
    return next_monday


def get_account_equity():
    account = trading_client.get_account()
    return float(account.equity)

# Store initial equity for scaling calculations
initial_equity = None 
# Configuration

underlying_symbols = ["F", "PLTR", "NIO"] 
max_abs_notional_delta = 500
risk_free_rate = 0.045
positions = {}

#liquidate existing positions

all_positions = trading_client.get_all_positions()


for p in all_positions:
    try:
        if p.asset_class == AssetClass.US_OPTION:
            option_contract = trading_client.get_option_contract(p.symbol)
            if option_contract.underlying_symbol in underlying_symbols:
                print(f"Attempting to liquidate {p.qty} of {p.symbol}")
                trading_client.close_position(p.symbol)
        elif p.asset_class == AssetClass.US_EQUITY:
            if p.symbol in underlying_symbols:
                print(f"Attempting to liquidate {p.qty} of {p.symbol}")
                trading_client.close_position(p.symbol)
    except Exception as e:
        print(f"Error closing position for {p.symbol}: {e}")
        # Continue with other positions even if one fails
        continue

for symbol in underlying_symbols:
    print(f"Adding {symbol} to position list")
    positions[symbol] = {'asset_class': 'us_equity', 'position': 0, 'initial_position': 0}

# Set expiration range for options

today = datetime.now().date()
min_expiration = today + timedelta(days=14)
max_expiration = today + timedelta(days=60)

# Get the latest price of the underlying stock

def get_underlying_price(symbol):

    underlying_trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    underlying_trade_response = stock_data_client.get_stock_latest_trade(underlying_trade_request)
    return underlying_trade_response[symbol].price

for underlying_symbol in underlying_symbols:
    # Get the latest price of the underlying stock
    underlying_price = get_underlying_price(underlying_symbol)
    min_strike = round(underlying_price * 1.01, 2)
    
    print(f"{underlying_symbol} price: {underlying_price}")
    print(f"Min Expiration: {min_expiration}, Max Expiration: {max_expiration}, Min Strike: {min_strike}")
    
    # Search for option contracts to add to the portfolio
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying_symbol],
        status=AssetStatus.ACTIVE,
        expiration_date_gte=min_expiration,
        expiration_date_lte=max_expiration,
        root_symbol=underlying_symbol,
        type=ContractType.CALL,
        strike_price_gte=str(min_strike),
        limit=5,
    )
    
    option_chain_list = trading_client.get_option_contracts(req).option_contracts
    
    # Add the first 3 options to the position list
    for option in option_chain_list[:3]:
        symbol = option.symbol
        print(f"Adding {symbol} to position list")
        positions[symbol] = {
            'asset_class': 'us_option',
            'underlying_symbol': option.underlying_symbol,
            'expiration_date': pd.Timestamp(option.expiration_date),
            'strike_price': float(option.strike_price),
            'type': option.type,
            'size': float(option.size),
            'position': 1.0,
            'initial_position': 1.0
        }

# Calculate implied volatility

def calculate_implied_volatility(option_price, S, K, T, r, option_type):
    def option_price_diff(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price - option_price

    return brentq(option_price_diff, 1e-6, 1)


# Calculate option Greeks (Delta and Gamma)

def calculate_greeks(option_price, strike_price, expiry, underlying_price, risk_free_rate, option_type):
    T = (expiry - pd.Timestamp.now()).days / 365
    implied_volatility = calculate_implied_volatility(option_price, underlying_price, strike_price, T, risk_free_rate, option_type)
    d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * implied_volatility ** 2) * T) / (implied_volatility * np.sqrt(T))
    d2 = d1 - implied_volatility * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(T))
    return delta, gamma


# Handle trade updates

async def on_trade_updates(data: TradeUpdate):
    symbol = data.order.symbol
    if symbol in positions:
        if data.event in {'fill', 'partial_fill'}:
            side = data.order.side
            qty = data.order.qty
            filled_avg_price = data.order.filled_avg_price
            position_qty = data.position_qty
            print(f"{data.event} event: {side} {qty} {symbol} @ {filled_avg_price}")
            print(f"updating position from {positions[symbol]['position']} to {position_qty}")
            positions[symbol]['position'] = float(position_qty)

trade_update_stream.subscribe_trade_updates(on_trade_updates)

# Execute initial trades

# Execute initial trades
# Modify the initial_trades function to use limit orders when market is closed
async def initial_trades():
    await asyncio.sleep(5)
    print("executing initial option trades")
    
    market_open = is_market_open()
    
    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_option' and pos['initial_position'] != 0:
            side = 'buy' if pos['initial_position'] > 0 else 'sell'
            
            try:
                # For options, we need to use limit orders when market is closed
                if not market_open and pos['asset_class'] == 'us_option':
                    # Get current option quote to set a reasonable limit price
                    try:
                        option_quote_request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
                        option_quote = option_data_client.get_option_latest_quote(option_quote_request)[symbol]
                        
                        # Set limit price slightly above ask for buy, slightly below bid for sell
                        if side == 'buy':
                            limit_price = option_quote.ask_price * 1.05  # 5% above ask
                        else:
                            limit_price = option_quote.bid_price * 0.95  # 5% below bid
                            
                        # Ensure limit price is valid (not zero)
                        if limit_price <= 0:
                            limit_price = 0.05  # Minimum price
                            
                        order_request = {
                            "symbol": symbol,
                            "qty": abs(pos['initial_position']),
                            "side": side,
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": round(limit_price, 2)
                        }
                        print(f"Submitting {side} limit order for {abs(pos['initial_position'])} contracts of {symbol} at ${limit_price:.2f}")
                    except Exception as e:
                        print(f"Error getting option quote for {symbol}: {e}")
                        print(f"Using default limit price of $0.05")
                        order_request = {
                            "symbol": symbol,
                            "qty": abs(pos['initial_position']),
                            "side": side,
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": 0.05
                        }
                else:
                    # Use market orders when market is open
                    order_request = {
                        "symbol": symbol,
                        "qty": abs(pos['initial_position']),
                        "side": side,
                        "type": "market",
                        "time_in_force": "day"
                    }
                    print(f"Submitting {side} market order for {abs(pos['initial_position'])} contracts of {symbol}")
                
                # Submit the order
                trading_client.submit_order(**order_request)
                
            except Exception as e:
                print(f"Error submitting order for {symbol}: {e}")
            
            # Add a small delay between orders to avoid rate limiting
            await asyncio.sleep(1)

# Maintain delta-neutral strategy for multiple symbols
def maintain_delta_neutral():
    global initial_equity
    if initial_equity is None:
        initial_equity = get_account_equity()
        print(f"Initial account equity: ${initial_equity}")

    current_equity = get_account_equity()
    equity_ratio = current_equity / initial_equity
    print(f"Current equity: ${current_equity}, Scaling factor: {equity_ratio:.2f}x")


    # Group positions by underlying symbol
    symbol_positions = {}
    for underlying in underlying_symbols:
        symbol_positions[underlying] = []
    
    # Organize positions by underlying
    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_equity' and symbol in underlying_symbols:
            symbol_positions[symbol].append((symbol, pos))
        elif pos['asset_class'] == 'us_option':
            underlying = pos['underlying_symbol']
            if underlying in underlying_symbols:
                symbol_positions[underlying].append((symbol, pos))
    
    # Process each underlying symbol separately
    for underlying_symbol, pos_list in symbol_positions.items():
        if not pos_list:
            continue
            
        current_delta = 0.0
        underlying_price = get_underlying_price(underlying_symbol)
        
        print(f"Current price of {underlying_symbol} is {underlying_price}")
        
        # Calculate total delta for this underlying
        for symbol, pos in pos_list:
            if pos['asset_class'] == 'us_equity':
                current_delta += pos['position']
            elif pos['asset_class'] == 'us_option':
                try:
                    option_quote_request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
                    option_quote = option_data_client.get_option_latest_quote(option_quote_request)[symbol]
                    option_quote_mid = (option_quote.bid_price + option_quote.ask_price) / 2
                    
                    delta, gamma = calculate_greeks(
                        option_price=option_quote_mid,
                        strike_price=pos['strike_price'],
                        expiry=pos['expiration_date'],
                        underlying_price=underlying_price,
                        risk_free_rate=risk_free_rate,
                        option_type=pos['type']
                    )
                    
                    current_delta += delta * pos['position'] * pos['size']
                except Exception as e:
                    print(f"Error calculating Greeks for {symbol}: {e}")
        
        # Adjust delta for this underlying
        scaled_max_delta = max_abs_notional_delta * equity_ratio
        adjust_delta(underlying_symbol, current_delta, underlying_price, scaled_max_delta)

def adjust_delta(underlying_symbol, current_delta, underlying_price, scaled_max_delta):
    if current_delta * underlying_price > scaled_max_delta:
        side = 'sell'
    elif current_delta * underlying_price < -scaled_max_delta:
        side = 'buy'
    else:
        return
    
    qty = abs(round(current_delta, 0))
    
    # Check if market is open
    market_open = is_market_open()
    
    if market_open:
        # Use market order when market is open
        order_request = MarketOrderRequest(
            symbol=underlying_symbol, 
            qty=qty, 
            side=side, 
            type='market', 
            time_in_force='day'
        )
        print(f"Submitting {side} market order for {qty} shares of {underlying_symbol}")
    else:
        # Use limit order when market is closed
        try:
            # Get current stock quote to set a reasonable limit price
            underlying_trade_request = StockLatestTradeRequest(symbol_or_symbols=underlying_symbol)
            underlying_trade = stock_data_client.get_stock_latest_trade(underlying_trade_request)[underlying_symbol]
            current_price = underlying_trade.price
            
            # Set limit price slightly above market for buy, slightly below for sell
            if side == 'buy':
                limit_price = current_price * 1.01  # 1% above current
            else:
                limit_price = current_price * 0.99  # 1% below current
                
            order_request = {
                "symbol": underlying_symbol,
                "qty": qty,
                "side": side,
                "type": "limit",
                "time_in_force": "day",
                "limit_price": round(limit_price, 2)
            }
            print(f"Submitting {side} limit order for {qty} shares of {underlying_symbol} at ${limit_price:.2f}")
        except Exception as e:
            print(f"Error getting stock price for {underlying_symbol}: {e}")
            # Skip order if we can't get a price
            return
    
    try:
        if isinstance(order_request, dict):
            trading_client.submit_order(**order_request)
        else:
            trading_client.submit_order(order_request)
    except Exception as e:
        print(f"Error submitting order for {underlying_symbol}: {e}")
# Add rebalancing function to reinvest profits
async def rebalance_positions(interval=3600):  # Default: rebalance every hour
    global initial_equity
    
    while True:
        await asyncio.sleep(interval)
        
        if initial_equity is None:
            initial_equity = get_account_equity()
            continue
            
        current_equity = get_account_equity()
        equity_ratio = current_equity / initial_equity
        
        # Only rebalance if equity has changed significantly (e.g., 5% or more)
        if abs(equity_ratio - 1.0) >= 0.05:
            print(f"Rebalancing positions due to equity change. Current ratio: {equity_ratio:.2f}x")
            
            # Close all current option positions
            for symbol, pos in list(positions.items()):
                if pos['asset_class'] == 'us_option' and pos['position'] != 0:
                    try:
                        print(f"Closing position for {symbol} to rebalance")
                        trading_client.close_position(symbol)
                    except Exception as e:
                        print(f"Error closing position for {symbol}: {e}")
            
            # Update option positions with new sizing for each underlying
            for underlying_symbol in underlying_symbols:
                # Get the latest price of the underlying stock
                underlying_price = get_underlying_price(underlying_symbol)
                min_strike = round(underlying_price * 1.01, 2)
                
                # Search for option contracts to add to the portfolio
                req = GetOptionContractsRequest(
                    underlying_symbols=[underlying_symbol],
                    status=AssetStatus.ACTIVE,
                    expiration_date_gte=min_expiration,
                    expiration_date_lte=max_expiration,
                    root_symbol=underlying_symbol,
                    type=ContractType.CALL,
                    strike_price_gte=str(min_strike),
                    limit=5,
                )
                
                option_chain_list = trading_client.get_option_contracts(req).option_contracts
                
                # Remove old options for this underlying
                for symbol in list(positions.keys()):
                    if positions[symbol]['asset_class'] == 'us_option' and positions[symbol]['underlying_symbol'] == underlying_symbol:
                        del positions[symbol]
                
                # Add new options with scaled position sizes
                for option in option_chain_list[:3]:
                    symbol = option.symbol
                    scaled_position = 1.0 * equity_ratio  # Scale the position size
                    print(f"Adding {symbol} to position list with scaled size {scaled_position:.2f}")
                    positions[symbol] = {
                        'asset_class': 'us_option',
                        'underlying_symbol': option.underlying_symbol,
                        'expiration_date': pd.Timestamp(option.expiration_date),
                        'strike_price': float(option.strike_price),
                        'type': option.type,
                        'size': float(option.size),
                        'position': 0.0,  # Will be updated after order fills
                        'initial_position': scaled_position
                    }
            
            # Execute trades for new positions
            await initial_trades()
            
            # Update initial equity for future scaling
            initial_equity = current_equity

# Update gamma_scalp to include rebalancing
async def gamma_scalp(initial_interval=30, interval=120, rebalance_interval=3600):
    while True:
        # Check if market is open
        if not is_market_open():
            next_open = get_next_market_open()
            now = datetime.now(pytz.timezone('US/Eastern'))
            wait_seconds = (next_open - now).total_seconds()
            
            print(f"Market is closed. Waiting until next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Sleeping for {wait_seconds:.0f} seconds ({wait_seconds/3600:.1f} hours)")
            
            # Sleep until market opens
            await asyncio.sleep(wait_seconds)
            continue
        
        # Market is open, proceed with strategy
        print("Market is open. Running gamma scalping strategy.")
        
        # Initial delta neutral adjustment
        maintain_delta_neutral()
        
        # Create rebalance task
        rebalance_task = asyncio.create_task(rebalance_positions(rebalance_interval))
        
        # Run strategy until market closes
        while is_market_open():
            await asyncio.sleep(interval)
            maintain_delta_neutral()
        
        # Market closed, cancel rebalance task
        rebalance_task.cancel()
        try:
            await rebalance_task
        except asyncio.CancelledError:
            pass
        
        print("Market closed. Pausing strategy.")
# Main event loop

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(
    trade_update_stream._run_forever(),
    initial_trades(),
    gamma_scalp()
))
loop.close()