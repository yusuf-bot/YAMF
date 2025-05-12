import ccxt
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ['BINANCE_API_KEY']
API_SECRET = os.environ['BINANCE_API_SECRET']

# Initialize Binance Futures
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'defaultType': 'future'}
})
exchange.set_sandbox_mode(True) # Enable testnet

# Load markets
exchange.load_markets()

open_orders = exchange.fetch_open_orders(symbol='ETH/USDT:USDT')
for o in open_orders:
    print(o)


# Print position siz