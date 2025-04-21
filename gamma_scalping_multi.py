from datetime import datetime, timedelta
import time
import asyncio
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import nest_asyncio
from alpaca.data.historical.option import OptionHistoricalDataClient, OptionLatestQuoteRequest
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.trading.models import TradeUpdate
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.enums import AssetStatus, ContractType, AssetClass
from dotenv import load_dotenv
import os
load_dotenv()

# Get API keys from environment variables
api_key = os.environ.get('ALPACA_API_KEY')
secret_key = os.environ.get('ALPACA_SECRET_KEY')

nest_asyncio.apply()
paper = True
# Initialize Alpaca clients
# Initialize Alpaca clients

trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
trade_update_stream = TradingStream(api_key=api_key, secret_key=secret_key, paper=paper)
stock_data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
option_data_client = OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)

underlying_symbols = ["AAPL", "SPY", "QQQ"] 
max_abs_notional_delta = 500
risk_free_rate = 0.045
positions = {}

print(f"Liquidating pre-existing positions for all tracked symbols")
all_positions = trading_client.get_all_positions()

for p in all_positions:
    if p.asset_class == AssetClass.US_OPTION:
        option_contract = trading_client.get_option_contract(p.symbol)
        if option_contract.underlying_symbol in underlying_symbols:
            print(f"Liquidating {p.qty} of {p.symbol}")
            trading_client.close_position(p.symbol)
    elif p.asset_class == AssetClass.US_EQUITY:
        if p.symbol in underlying_symbols:
            print(f"Liquidating {p.qty} of {p.symbol}")
            trading_client.close_position(p.symbol)

# Add underlying symbols to positions list
for symbol in underlying_symbols:
    print(f"Adding {symbol} to position list")
    positions[symbol] = {'asset_class': 'us_equity', 'position': 0, 'initial_position': 0}


today = datetime.now().date()
min_expiration = today + timedelta(days=14)
max_expiration = today + timedelta(days=60)

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

def calculate_greeks(option_price, strike_price, expiry, underlying_price, risk_free_rate, option_type):
    T = (expiry - pd.Timestamp.now()).days / 365
    implied_volatility = calculate_implied_volatility(option_price, underlying_price, strike_price, T, risk_free_rate, option_type)
    d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * implied_volatility ** 2) * T) / (implied_volatility * np.sqrt(T))
    d2 = d1 - implied_volatility * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(T))
    return delta, gamma

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
async def initial_trades():
    await asyncio.sleep(5)
    print("executing initial option trades")
    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_option' and pos['initial_position'] != 0:
            side = 'buy' if pos['initial_position'] > 0 else 'sell'
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(pos['initial_position']),
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"Submitting order to {side} {abs(pos['initial_position'])} contracts of {symbol} at market")
            trading_client.submit_order(order_request)
            # Add a small delay between orders to avoid rate limiting
            await asyncio.sleep(1)

def maintain_delta_neutral():
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
        adjust_delta(underlying_symbol, current_delta, underlying_price)

def adjust_delta(underlying_symbol, current_delta, underlying_price):
    if current_delta * underlying_price > max_abs_notional_delta:
        side = 'sell'
    elif current_delta * underlying_price < -max_abs_notional_delta:
        side = 'buy'
    else:
        return
    
    qty = abs(round(current_delta, 0))
    order_request = MarketOrderRequest(symbol=underlying_symbol, qty=qty, side=side, type='market', time_in_force='day')
    print(f"Submitting {side} order for {qty} shares of {underlying_symbol} at market")
    trading_client.submit_order(order_request)


async def gamma_scalp(initial_interval=30, interval=120):
    await asyncio.sleep(initial_interval)
    maintain_delta_neutral()
    while True:
        await asyncio.sleep(interval)
        maintain_delta_neutral()



loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(
    trade_update_stream._run_forever(),
    #initial_trades(),
    gamma_scalp()
))
loop.close()