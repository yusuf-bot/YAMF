import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


from alpaca.trading.client import TradingClient

trading_client = TradingClient(API_KEY, API_SECRET,paper=True)

account = trading_client.get_account()

print(account)
print(account.cash)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# preparing orders
market_order_data = MarketOrderRequest(
                    symbol="ETH/USD",
                    qty=0.009975,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                    )

# Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
               )