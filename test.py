import ccxt
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = os.environ['BYBIT_API_KEY']
API_SECRET = os.environ['BYBIT_SECRET_KEY']

# Initialize Bybit exchange in testnet mode
bybit = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'testnet': True  # Enable testnet mode
})

# Verify connection
print(bybit.fetch_balance())  # Check testnet funds