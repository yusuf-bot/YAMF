"""import ccxt
import pandas as pd

coinbase = ccxt.coinbase()

# Fetch OHLCV data for ETH/USD with 5-minute timeframe
bars = coinbase.fetch_ohlcv('ETH/USD', timeframe='5m', limit=71)

# Create DataFrame and convert timestamp
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(df.tail())

print(len(df), "rows fetched")
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_coinbase_ohlcv(symbol='ETH-USD', granularity=300, limit=71):
    # granularity: 60 = 1m, 300 = 5m, 900 = 15m, etc.
    end = datetime.utcnow()
    start = end - timedelta(seconds=granularity * limit)

    params = {
        'start': start.isoformat(),
        'end': end.isoformat(),
        'granularity': granularity  # in seconds
    }

    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Failed to fetch:", response.status_code, response.text)
        return None

    data = response.json()

    # Coinbase returns: [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    df = df.sort_values("timestamp")

        
    print(df.tail())

    print(len(df), "rows fetched")

fetch_coinbase_ohlcv()