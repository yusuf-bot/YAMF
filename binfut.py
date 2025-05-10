from dotenv import load_dotenv
import os
import ccxt
import time
from datetime import datetime, timedelta

# === Load environment variables ===
load_dotenv()
API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

print(f"API Key loaded: {'Yes' if API_KEY else 'No'}")
print(f"API Secret loaded: {'Yes' if API_SECRET else 'No'}")

# === CONFIG ===
symbol = 'ETH/USDT'
base = 'ETH'
quote = 'USDT'
timeframe = '1m'
trend_timeframe = '1h'
trend_period = 200
atr_period = 14
trade_amount = 5

# === Initialize Exchange ===
exchange = ccxt.binanceusdm({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
exchange.set_sandbox_mode(True)

# === Strategy State ===
candles = []
position = None
entry_price = None
trail_price = None

# === Historical Candle Fetch ===
def preload_45m_candles():
    lookback_minutes = trend_period * 60
    since = int((datetime.utcnow() - timedelta(minutes=lookback_minutes)).timestamp() * 1000)
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=trend_timeframe, since=since, limit=trend_period + 1)
        return [{
            'timestamp': o[0],
            'open': o[1],
            'high': o[2],
            'low': o[3],
            'close': o[4]
        } for o in ohlcv]
    except Exception as e:
        print(f"Error fetching historical candles: {e}")
        return []

# === Live Candle Fetch (1m) ===
def fetch_latest_candle():
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
        if ohlcv:
            return {
                'timestamp': ohlcv[-1][0],
                'open': ohlcv[-1][1],
                'high': ohlcv[-1][2],
                'low': ohlcv[-1][3],
                'close': ohlcv[-1][4]
            }
    except Exception as e:
        print(f"Error fetching 1m candle: {e}")
    return None

# === Helpers ===
def calculate_atr(candles, period):
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i - 1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs[-period:]) / period if len(trs) >= period else None

def calculate_middle(candles, period):
    highs = [c['high'] for c in candles[-period-1:-1]]
    lows = [c['low'] for c in candles[-period-1:-1]]
    return (max(highs) + min(lows)) / 2 if highs and lows else None

def place_market_order(side, amount):
    try:
        print(f"Placing market {side} order for {amount} {symbol}")
        return exchange.create_market_order(symbol, side, amount)
    except Exception as e:
        print(f"Order Error: {e}")
        return None

def close_position():
    try:
        position_data = exchange.fapiPrivate_get_positionrisk()
        for p in position_data:
            if p['symbol'] == symbol.replace('/', '') and float(p['positionAmt']) != 0:
                amt = abs(float(p['positionAmt']))
                side = 'sell' if float(p['positionAmt']) > 0 else 'buy'
                place_market_order(side, amt)
    except Exception as e:
        print(f"Close position error: {e}")

# === Initialize candle history ===
candles = preload_45m_candles()
print(f"Preloaded {len(candles)} 45m candles.")

# === Main Loop ===
while True:
    candle = fetch_latest_candle()
    if candle:
        if not candles or candle['timestamp'] > candles[-1]['timestamp']:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} New candle: {candle}")
            candles.append(candle)
            if len(candles) > trend_period + 1:
                middle = calculate_middle(candles, trend_period)
                atr = calculate_atr(candles, atr_period)
                if middle is None or atr is None:
                    continue

                epsilon = atr
                src_prev = (candles[-2]['high'] + candles[-2]['low'] + candles[-2]['close']) / 3
                is_60min = (len(candles) - 1) % 60 == 0  # every 45th 1m candle = new 45m bar

                print(f"Middle: {middle}, ATR: {atr}, src_prev: {src_prev}, long: {src_prev > middle + epsilon}, short: {src_prev < middle - epsilon}")

                # === Entry logic on 45m candle ===
                if is_60min and position is None:
                    if src_prev > middle + epsilon:
                        res = place_market_order('buy', trade_amount)
                        if res:
                            position = 'long'
                            entry_price = candle['close']
                            trail_price = entry_price - atr
                            print("Long Entry", entry_price, trail_price)
                    elif src_prev < middle - epsilon:
                        res = place_market_order('sell', trade_amount)
                        if res:
                            position = 'short'
                            entry_price = candle['close']
                            trail_price = entry_price + atr
                            print("Short Entry", entry_price, trail_price)

                # === Exit logic every 1 minute ===
                if position == 'long':
                    trail_price = max(trail_price, candle['close'] - atr)
                    if candle['close'] < trail_price:
                        res = place_market_order('sell', trade_amount)
                        print("Long Exit", candle['close'], trail_price)
                        position = None
                        entry_price = None
                        trail_price = None

                elif position == 'short':
                    trail_price = min(trail_price, candle['close'] + atr)
                    if candle['close'] > trail_price:
                        res = place_market_order('buy', trade_amount)
                        print("Short Exit", candle['close'], trail_price)
                        position = None
                        entry_price = None
                        trail_price = None

    time.sleep(60)  # Run every 1 minute
