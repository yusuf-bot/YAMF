import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
from flask import Flask, jsonify
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
import requests
from dotenv import load_dotenv
import time, torch,joblib
from supervise import load_model, add_indicators
import logging
from threading import Thread

# === CONFIG ===
SYMBOL = "ETH/USD"
EMA_LENGTH = 70
UPPER_BOUND = 3000
LOWER_BOUND = 2000
GRID_QTY = 60
MAX_PYRAMIDING = 35
LEVERAGE = 1
STAKE_PCT = 2.7  # Percent of equity per trade

COMMISSION = 0.001  # Commission per trade (0.1% for Alpaca)
SYMBOL_AI = "BTC/USD"  # Symbol for AI strategy
TABLE_NAME_AI = "ai_strat"  # Supabase table for AI strategy

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



def fetch_last_supabase_row():
    url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME_AI}?order=id.desc&limit=1"
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.json():
        return response.json()[0]
    return {"pred": None, "equity": 100.0, "profit": 0.0}  # Default values if table is empty


def insert_supabase_row(pred, equity, profit):
    data = {"pred": pred, "equity": equity, "profit": profit}
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/{TABLE_NAME_AI}",
        json=data,
        headers={**headers, "Content-Type": "application/json"}
    )
    if response.status_code not in [200, 201]:
        print(f"Failed to insert row: {response.status_code} | {response.text}")


def run_ai_strat(seq_len=24, threshold=0.5):
    scaler = joblib.load("scaler.pkl")
    model = load_model(input_dim=18, seq_len=seq_len)

    # Step 1: Load past equity + prediction from Supabase
    last_row = fetch_last_supabase_row()
    prev_pred = last_row.get("pred")
    prev_equity = last_row.get("equity", 100.0)
    prev_price = None
    prev_profit = last_row.get("profit", 0.0)

    # Step 2: Fetch and process data
    df = fetch_bars(SYMBOL_AI, datetime.now(UTC) - timedelta(days=500), datetime.now(UTC))
    df = add_indicators(df)

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "EMA_20", "EMA_50", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9",
        "ATRr_14", "BBL_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "SUPERT_7_3.0",
        "STOCHk_14_3_3", "STOCHd_14_3_3", "MFI_14"
    ]

    data = df[features].values
    data = scaler.transform(data)
    seq = torch.tensor(data[-seq_len:], dtype=torch.float32).unsqueeze(0)

    # Step 3: Get prediction
    with torch.no_grad():
        prob = torch.sigmoid(model(seq)).item()
        curr_pred = True if prob > threshold else False

    curr_price = df["Close"].iloc[-1]

    # Step 4: Calculate new profit and equity
    if prev_pred is None or prev_pred != curr_pred:
        # New trade or switch direction → close previous position if any
        if prev_pred is not None:
            prev_price = df["Close"].iloc[-2]  # Simulate entry price as previous close
            raw_return = (curr_price - prev_price) / prev_price
            trade_return = raw_return if prev_pred else -raw_return
            trade_return -= COMMISSION  # Deduct commission
            equity = prev_equity * (1 + trade_return)
            profit = equity - 100.0
        else:
            equity = prev_equity
            profit = prev_profit
    else:
        equity = prev_equity
        profit = prev_profit

    insert_supabase_row(pred=curr_pred, equity=equity, profit=profit)

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}] Prediction: {'UP' if curr_pred else 'DOWN'} | Prob: {prob:.3f} | Equity: {equity:.4f} | Profit: {profit:.4f}")




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
    ema1 = close.ewm(span=70).mean()
    ema2 = ema1.ewm(span=70).mean()
    ema3 = ema2.ewm(span=70).mean()
    return ema3

def build_grid():
    """Build grid levels."""
    width = (UPPER_BOUND - LOWER_BOUND) / (GRID_QTY - 1)
    return [LOWER_BOUND + i * width for i in range(GRID_QTY)]

def should_buy(t3, close):
    """Check buy condition."""
    info_logger.info(f"Checking buy condition: Previous_T3={t3.iloc[-2]} T3={t3.iloc[-1]}, Close={close.iloc[-1]}")
    return t3.iloc[-1] > t3.iloc[-2] and close.iloc[-1] > t3.iloc[-1]

def run_trading_strategy():
    """Execute trading strategy."""
    time.sleep(20)
    try:
        # Load state
        state = load_supabase_data()
        info_logger.info(f"State loaded: {state}")


        # Fetch bars
        end_time = datetime.now(UTC)
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
                        active_count = state['order_tracker'].count(True)
                        update_rows(i, state['order_tracker'][i], state['qty_tracker'][i])
                        info_logger.info(f"BUY Grid {i} at {grid_price:.2f} | Qty: {quantity} | Predicted Close: {grid_lines[i + 1]:.2f}")


        # Exit logic
        for i in range(GRID_QTY - 1):
            if state['order_tracker'][i]:
                info_logger.info(f"Checking exit for Grid {i} | Price: {grid_lines[i]} | Close: {bars['close'].iloc[-1]} | Next Price: {grid_lines[i + 1]}")
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
            now = datetime.now(UTC)

            if now.hour % 4 == 0 and now.minute == 0:
                info_logger.info("Running AI strategy")
                run_ai_strat(seq_len=24, threshold=0.5)
            stat = run_trading_strategy()  # move all logic into here
        except Exception as e:
            error_logger.error(f"Background strategy error: {e}")

    Thread(target=task).start()
    return jsonify({"status": "started", "message": "Strategy running in background"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

