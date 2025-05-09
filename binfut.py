from dotenv import load_dotenv
import os
import ccxt

# Load environment variables
load_dotenv()

API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

# Debug: Check if API credentials are loaded
print(f"API Key loaded: {'Yes' if API_KEY else 'No'}")
print(f"API Secret loaded: {'Yes' if API_SECRET else 'No'}")

# Initialize the exchange
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {  # Note: 'options' not 'option'
        'defaultType': 'future',  # Note: 'future' not 'futures'
    }
})

exchange.set_sandbox_mode(True)  # Set to True for testing in the sandbox environment

try:
    # Test connection and get account info
    balance = exchange.fetch_balance()
    print("Connection successful!")
    print(f"Available balances: {balance['free']}")
    
    orderbook = exchange.fetch_order_book('TAO/USDT')
    best_ask = orderbook['asks'][0][0]  # Get lowest ask price

    # Buy near the ask
    target_price = best_ask # 1% above ask
    print(f"Target price for buy order: {target_price}")
    order = exchange.create_limit_buy_order(
        symbol='TAO/USDT',
        amount=2,
        price=target_price
    )
    # Replace with your desired quantity
        
except Exception as e:
    print(f"Error: {e}")