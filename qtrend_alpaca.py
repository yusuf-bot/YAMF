import requests
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse

# Generate dynamic start and end time (last 71 x 5-minute candles)
end_time = datetime.utcnow()
start_time = end_time - timedelta(minutes=72 * 5)

# Format for URL (ISO8601, then URL-encoded)
start_str = urllib.parse.quote(start_time.isoformat() + "Z")
end_str = urllib.parse.quote(end_time.isoformat() + "Z")

# Construct URL
url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=ETH%2FUSD&timeframe=5Min&start={start_str}&end={end_str}&limit=1000&sort=asc"

headers = {"accept": "application/json"}

# Make the request
response = requests.get(url, headers=headers)

# Parse and save
if response.status_code == 200:
    data = response.json()
    bars = data['bars']['ETH/USD']

    df = pd.DataFrame(bars)
    df['t'] = pd.to_datetime(df['t'])
    df = df.rename(columns={
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    })

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    print(df.head())
    print(len(df), "rows fetched")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    print(response.text)
