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
STATE_FILE = "trade_state.json"


# === INIT ===
app = Flask(__name__)
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")

API_SECRET = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)



def load_trade_state():
    """Load trade state from JSON file."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Validate state
                if (isinstance(state.get('order_tracker'), list) and len(state['order_tracker']) == GRID_QTY and
                    isinstance(state.get('qty_tracker'), list) and len(state['qty_tracker']) == GRID_QTY):
                    print("Loaded state:", state)
                    return state
        # Default state
        print("State file not found or invalid. Returning default state.")
         # Default state
        return {
            'order_tracker': [False] * GRID_QTY,
            'qty_tracker': [0.0] * GRID_QTY,
  
        }
    except Exception as e:
        print(f"Error loading state: {e}")
        print("Returning default state")
        return {
            'order_tracker': [False] * GRID_QTY,
            'qty_tracker': [0.0] * GRID_QTY,

        }

def save_trade_state(state):
    """Save trade state to JSON file with minimal storage."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, separators=(',', ':'))
    except Exception as e:
        print('error', e)


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
            print(f"Failed to fetch data. HTTP {response.status_code}: {response.text}")
            return None

        data = response.json()
        if not data:
            print("No data returned from Coinbase.")
            return None

        # Columns: time, low, high, open, close, volume
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(df.tail())
        print(len(df), "rows fetched")
        return df

    except Exception as e:
        print(f"Failed to fetch data: {e}")
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
        state = load_trade_state()
        

        # Fetch bars
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=(EMA_LENGTH + 1) * 5)
        bars = fetch_bars(SYMBOL, start_time, end_time)
        if bars is None or len(bars) < EMA_LENGTH + 1:
            print("Insufficient bars")
            return "fail"


        # Compute T3
        t3 = compute_t3(bars['close'])

        # Grid setup
        grid_lines = build_grid()
        trades_bought=0
        trades_sold=0
        # Buy logic
        if should_buy(t3, bars['close']):
            print("Buy condition met")
            active_count = sum(state['order_tracker'])
            
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
                        save_trade_state(state)
                        print(f"BUY Grid {i} at {grid_price:.2f} | Qty: {quantity}")


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
                        save_trade_state(state)
                        print(f"CLOSE Grid {i} at {bars['close'].iloc[-1]:.2f}")
                        trades_sold+=1

                    except Exception as e:
                        print(f"Sell failed: {e}, attempting close_position")
                        try:
                            trading_client.close_position(SYMBOL.replace("/", ""))
                            state['order_tracker'][i] = False
                            state['qty_tracker'][i] = 0.0
                            save_trade_state(state)
                            print(f"CLOSE Grid {i} via close_position")
                            trades_sold+=1
                        except Exception as e:
                            print(f"Close position failed: {e}")


        print({"status": "called", "message": f"{trades_bought} opened {trades_sold} closed"})
        return "pass"
    except Exception as e:
        import traceback
        print(f"Strategy error: {e}")
        traceback.print_exec()

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
            print(f"Background strategy error: {e}")

    Thread(target=task).start()
    return jsonify({"status": "started", "message": "Strategy running in background"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

