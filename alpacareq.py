import requests
from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')
BASE_URL = 'https://api.alpaca.markets'

HEADERS = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': API_SECRET,
    "accept": "application/json",
    "content-type": "application/json",
    "paper":'true'
}

def place_crypto_order(symbol, qty, side):
    url = f"{BASE_URL}/v2/orders"
    order_data = {
        "symbol": symbol,             # e.g., "BTC/USD"
        "qty": qty,                   # amount in units (not dollars)
        "side": side,                 # "buy" or "sell"
        "type": "market",
        "time_in_force": "gtc"        # Good Till Canceled
    }
    response = requests.post(url, headers=HEADERS, json=order_data)
    if response.status_code == 200:
        print("Order placed:", response.json())
    else:
        print("Error placing order:", response.status_code, response.text)

# Example: Buy 0.001 BTC
place_crypto_order("BTC/USD", 0.001, "buy")
