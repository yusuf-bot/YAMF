import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import requests
from dotenv import load_dotenv
import time
import logging
from threading import Thread

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
app = Flask(__name__)
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME", "YAMF")

headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

info_logger = logging.getLogger('info_logger')
info_logger.setLevel(logging.INFO)

warning_logger = logging.getLogger('warning_logger')
warning_logger.setLevel(logging.WARNING)

error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)

def load_supabase_data():
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?select=id,order_tracker,qty_tracker",
        headers=headers
    )

    if response.status_code == 200:
        rows = response.json()
        if not rows:
            warning_logger.warning("No data found.")
            return {"order_tracker": [], "qty_tracker": []}

        # Determine max id to initialize the arrays correctly
        max_id = max(row["id"] for row in rows)
        order_tracker = [False] * (max_id + 1)
        qty_tracker = [0.0] * (max_id + 1)

        for row in rows:
            idx = row["id"]
            order_tracker[idx] = row["order_tracker"]
            qty_tracker[idx] = row["qty_tracker"]

        result = {
            "order_tracker": order_tracker,
            "qty_tracker": qty_tracker
        }
        return result
    else:
        warning_logger.warning("Failed to fetch data")
        warning_logger.warning(f"Status Code: {response.status_code}")
        warning_logger.warning(f"Response: {response.text}")
        return {"order_tracker": [], "qty_tracker": []}



def update_rows(i, order_tracker, qty_tracker):
        data = {
            "order_tracker": order_tracker,
            "qty_tracker": qty_tracker
        }
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}?id=eq.{i}",
            headers=headers,
            json=data
        )
        info_logger.info(f"Update row {i}: {response.status_code}  {data}")


def fetch_bars(symbol, start_time, end_time):
    try:
        # Coinbase uses '-' instead of '/' in symbols
        product_id = symbol.replace('/', '-')
        granularity = 300  # 5-minute candles
        limit = 71

        params = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'granularity': granularity
        }

        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        response = requests.get(url, params=params)

        if response.status_code != 200:
            error_logger.error(f"Failed to fetch data. HTTP {response.status_code}: {response.text}")
            return None

        data = response.json()
        if not data:
            warning_logger.warning("No data returned from Coinbase.")
            return None

        # Columns: time, low, high, open, close, volume
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)

        info_logger.info(df.tail())
        info_logger.info(f"{len(df)} rows fetched")
        return df

    except Exception as e:
        error_logger.error(f"Failed to fetch data: {e}")
        return None


def compute_t3(close):
    """Compute T3 using EMA."""
    return close.ewm(span=EMA_LENGTH).mean()

def build_grid():
    """Build grid levels."""
    width = (UPPER_BOUND - LOWER_BOUND) / (GRID_QTY - 1)
    return [LOWER_BOUND + i * width for i in range(GRID_QTY)]

def should_buy(t3, close):
    """Check buy condition."""
    return t3.iloc[-1] > t3.iloc[-2] and close.iloc[-1] > t3.iloc[-1]

def run_trading_strategy():
    """Execute trading strategy."""
    time.sleep(20)
    try:
        # Load state
        state = load_supabase_data()
        info_logger.info(f"State loaded: {state}")


        # Fetch bars
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=(EMA_LENGTH + 1) * 5)
        bars = fetch_bars(SYMBOL, start_time, end_time)
        if bars is None or len(bars) < EMA_LENGTH + 1:
            warning_logger.warning("Insufficient bars")
            return "fail"


        # Compute T3
        t3 = compute_t3(bars['close'])

        # Grid setup
        grid_lines = build_grid()
        trades_bought=0
        trades_sold=0
        # Buy logic
        if should_buy(t3, bars['close']):
            info_logger.info("Buy condition met")
            active_count = state['order_tracker'].count(True)
            
            for i, grid_price in enumerate(grid_lines[:-1]):
                if active_count < MAX_PYRAMIDING:
                    if not state['order_tracker'][i] and bars['close'].iloc[-1] < grid_price:
                        account = trading_client.get_account()
                        equity = float(account.equity)
                        stake_usd = equity * (STAKE_PCT / 100)
                        quantity = round(stake_usd * LEVERAGE / bars['close'].iloc[-1], 4)
                        state['qty_tracker'][i] = quantity * 0.9975  # Adjust for fees
                        order = trading_client.submit_order(MarketOrderRequest(
                            symbol=SYMBOL,
                            qty=quantity,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            time_in_force=TimeInForce.GTC
                        ))
                        trades_bought+=1
                        state['order_tracker'][i] = True
                        update_rows(i, state['order_tracker'][i], state['qty_tracker'][i])
                        info_logger.info(f"BUY Grid {i} at {grid_price:.2f} | Qty: {quantity} | Predicted Close: {grid_lines[i + 1]:.2f}")


        # Exit logic
        for i in range(GRID_QTY - 1):
            if state['order_tracker'][i]:
                next_price = grid_lines[i + 1]
                if bars['close'].iloc[-1] > next_price:
                    try:
                        trading_client.submit_order(MarketOrderRequest(
                            symbol=SYMBOL,
                            qty=state['qty_tracker'][i],
                            side=OrderSide.SELL,
                            type=OrderType.MARKET,
                            time_in_force=TimeInForce.GTC
                        ))
                        state['order_tracker'][i] = False
                        state['qty_tracker'][i] = 0.0
                        update_rows(i, state['order_tracker'][i], state['qty_tracker'][i])
                        info_logger.info(f"CLOSE Grid {i} at {bars['close'].iloc[-1]:.2f}")
                        trades_sold+=1

                    except Exception as e:
                        warning_logger.warning(f"Sell failed: {e}, attempting close_position")
                        try:
                            trading_client.close_position(SYMBOL.replace("/", ""))
                            state['order_tracker'][i] = False
                            state['qty_tracker'][i] = 0.0
                            update_rows(i, state['order_tracker'][i], state['qty_tracker'][i])
                            info_logger.info(f"CLOSE Grid {i} via close_position")
                            trades_sold+=1
                        except Exception as e:
                            error_logger.error(f"Close position failed: {e}")


        info_logger.info( f"{trades_bought} opened {trades_sold} closed")
        return "pass"
    except Exception as e:
        error_logger.error(f"Strategy error: {e}")


@app.route('/')
def home():
    return jsonify({"message": "Flask is working!", "status": "success"})

@app.route('/test')
def test():
    return jsonify({"message": "Test endpoint working", "status": "success"})


@app.route('/run-strategy', methods=['GET'])
def run_strategy():
    def task():
        try:
            stat = run_trading_strategy()  # move all logic into here
        except Exception as e:
            error_logger.error(f"Background strategy error: {e}")

    Thread(target=task).start()
    return jsonify({"status": "started", "message": "Strategy running in background"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

