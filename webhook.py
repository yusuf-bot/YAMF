import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np
import urllib.parse
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# === CONFIG ===
SYMBOL = "ETH/USD"
EMA_LENGTH = 70
UPPER_BOUND = 3000
LOWER_BOUND = 2000
GRID_QTY = 30
MAX_PYRAMIDING = 4
LEVERAGE = 1
STAKE_PCT = 20  # Percent of equity per trade

# === INIT ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = True

trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

# Track active grid orders
order_tracker = [ False for i in range(GRID_QTY)]
qty_tracker = [0 for i in range(GRID_QTY)]


def fetch_bars(symbol, start_time, end_time):
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
        print(len(df), "rows fetched")
        return df
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print(response.text)


def compute_t3(close):
    return close.ewm(span=EMA_LENGTH).mean()

def build_grid():
    width = (UPPER_BOUND - LOWER_BOUND) / (GRID_QTY - 1)
    return [LOWER_BOUND + i * width for i in range(GRID_QTY)]

def should_buy(t3, close):
    return t3.iloc[-1] > t3.iloc[-2] and close.iloc[-1] > t3.iloc[-1]

async def run_strategy():
    grid_lines = build_grid()

    while True:
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=72*5)
            bars = fetch_bars(SYMBOL, start_time, end_time)
            print(f"Fetched {len(bars)} bars from {start_time} to {end_time}")
            if bars is not None:
                if len(bars) < EMA_LENGTH + 1:
                    logging.warning("Not enough bars yet.")
                    await asyncio.sleep(60)
                    continue

                t3 = compute_t3(bars['close'])

                if should_buy(t3, bars['close']):
                    active_count = order_tracker.count(True)
                    logging.info(f"Active grid orders: {active_count}")
                    if active_count < MAX_PYRAMIDING:
                        for i, grid_price in enumerate(grid_lines):
                            if not order_tracker[i] and bars['close'].iloc[-1] < grid_price:
                                account = trading_client.get_account()
                                equity = float(account.equity)
                                stake_usd = equity * (STAKE_PCT / 100)
                                quantity = round(stake_usd * LEVERAGE / bars['close'].iloc[-1], 4)
                                qty_tracker[i] = quantity*0.9975
                                trading_client.submit_order(MarketOrderRequest(
                                    symbol=SYMBOL,
                                    qty=quantity,
                                    side=OrderSide.BUY,
                                    type=OrderType.MARKET,
                                    time_in_force=TimeInForce.GTC
                                ))

                                order_tracker[i] = True
                                logging.info(f"BUY Grid {i} at {grid_price:.2f} | Qty: {quantity}")
                                break
                    else:
                        logging.info("Max pyramiding reached.")

                # Check for exits
                for i in range(GRID_QTY - 1):
                    if order_tracker[i]:
                        next_price = grid_lines[i + 1]
                        if bars['close'].iloc[-1] > next_price:
                            order_tracker[i] = False
                            try:
                                trading_client.submit_order(MarketOrderRequest(
                                    symbol=SYMBOL,
                                    qty=qty_tracker[i],
                                    side=OrderSide.SELL,
                                    type=OrderType.MARKET,
                                    time_in_force=TimeInForce.GTC
                                ))
                                qty_tracker[i] = 0
                                logging.info(f"CLOSE Grid {i} at {bars['close'].iloc[-1]}")
                            except Exception as e:
                                try:
                                    trading_client.close_position(SYMBOL.replace("/", ""))
                                    qty_tracker[i] = 0
                                except Exception as e:
                                    logging.warning(f"Tried to close but error: {e}")

        except Exception as e:
            logging.error(f"Strategy loop error: {e}")

        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(run_strategy())
