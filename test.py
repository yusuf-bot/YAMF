from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
# Load Alpaca API credentials from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
try:
    account = trading_client.get_account()
    print(account)
except Exception as e:
    print(f"Error: {e}")