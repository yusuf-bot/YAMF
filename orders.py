api_key='PKYXKXZCBA155CHGFHJ2'
secret_key = "fZoyUsSQ6SdN7w4pipiGJidwmd9MT4EPUcjXjPk9"
paper = True

from datetime import datetime, timedelta
import alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
from alpaca.trading.stream import TradingStream
from alpaca.data import StockHistoricalDataClient
from alpaca.data import StockTradesRequest
from alpaca.data.live import StockDataStream, CryptoDataStream
# Initialize both trading and historical data clients
trading_client = TradingClient(api_key, secret_key, paper=paper)
"""
market order
market_order_data=MarketOrderRequest(
    symbol="BTCUSD",
    qty=0.01,
    side=OrderSide.BUY, 
    time_in_force=TimeInForce.GTC,
)

market_order=trading_client.submit_order(
    order_data=market_order_data
)
print(market_order)"""

"""
limit order
limit_order_data=LimitOrderRequest(
    symbol="SPY",
    qty=1,
    side=OrderSide.BUY, 
    time_in_force=TimeInForce.GTC,
    limit_price=538
)

market_order=trading_client.submit_order(
    order_data=limit_order_data
)
print(market_order)
"""



"""
HOW TO GET DATA
data_client = StockHistoricalDataClient(api_key, secret_key)

# Get historical trades
request_params = StockTradesRequest(
    symbol_or_symbols="AAPL",
    start=datetime(2024, 1, 30, 14, 30),
    end=datetime(2024, 1, 30, 14, 45)
)

trades_history = data_client.get_stock_trades(request_params)
print("Historical Trades:")
print(trades_history)"""


"""
cancel order

request_params = GetOrdersRequest(
    status=QueryOrderStatus.OPEN,
    side=OrderSide.BUY,

)


orders=trading_client.get_orders(request_params)
for order in orders:
    print(order.id)
    trading_client.cancel_order_by_id(order.id)"""

"""
get all posotions

positions=trading_client.get_all_positions()
for position in positions:
    print(position.symbol,position.current_price)
"""


"""
close all positions

trading_client.close_all_positions()

"""


"""
stream data real-time

stream = CryptoDataStream(api_key, secret_key)
async def handle_trade(data):
    print(data)

stream.subscribe_trades(handle_trade, "BTC/USDT")
stream.run()

"""