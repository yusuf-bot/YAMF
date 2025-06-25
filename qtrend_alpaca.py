from flask import Flask, jsonify
import pandas as pd, numpy as np, requests, joblib, logging
from datetime import datetime
import pandas_ta as ta
import os, time
from dotenv import load_dotenv

# === Setup ===
load_dotenv()
app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# === Constants ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE")
BINANCE_SYMBOL = "ETHUSDT"
TIMEFRAME = "4h"
MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"
CONFIDENCE_THRESHOLD = 0.55

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

# === Helpers ===
def fetch_ohlcv_binance(symbol, interval="4h", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return None

def add_technical_features(df):
    macd = ta.macd(df['close'])
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'], df['plus_di'], df['minus_di'] = adx['ADX_14'], adx['DMP_14'], adx['DMN_14']
    st = ta.supertrend(df['high'], df['low'], df['close'])
    df['supertrend'], df['supertrend_dir'] = st['SUPERT_7_3.0'], st['SUPERTd_7_3.0']
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    df['ema9_above_21'] = df['ema9'] > df['ema21']
    df['ema21_above_50'] = df['ema21'] > df['ema50']
    df['above_ema200'] = df['close'] > df['ema200']
    df['roc'] = ta.roc(df['close'], length=10)
    df['roc_5'] = ta.roc(df['close'], length=5)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_ma'] = ta.sma(df['rsi'], length=5)
    stochrsi = ta.stochrsi(df['close'], length=14)
    df['stochrsi_k'] = stochrsi['STOCHRSIk_14_14_3_3']
    df['stochrsi_d'] = stochrsi['STOCHRSId_14_14_3_3']
    bb = ta.bbands(df['close'], length=20)
    df['bb_upper'], df['bb_lower'] = bb['BBU_20_2.0'], bb['BBL_20_2.0']
    df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['close']
    df['bb_position'] = (df['close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_ma'] = ta.sma(df['atr'], length=5)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['momentum'] = ta.mom(df['close'], length=10)
    df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    df['atr_percentile'] = df['atr'].rolling(100).apply(lambda x: x.rank(pct=True).iloc[-1] if len(x.dropna()) else np.nan)
    df['trend_strength'] = abs(df['adx']) * np.where(df['plus_di'] > df['minus_di'], 1, -1)
    df['high_volatility'] = (df['atr_percentile'] > 0.7).astype(int)
    df['low_volatility'] = (df['atr_percentile'] < 0.3).astype(int)
    df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(3).sum()
    df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(3).sum()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def get_last_state():
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?order=timestamp.desc&limit=1"
    r = requests.get(url, headers=headers)
    if r.status_code == 200 and r.json():
        return r.json()[0]
    return {"capital": 10000, "position": None}

def save_trade(prediction, probability, capital, position, action, price, pnl=None):
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": int(prediction),
        "probability": float(probability),
        "capital": capital,
        "position": position,
        "action": action,
        "price": price,
        "pnl": pnl
    }
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
        headers={**headers, "Content-Type": "application/json"},
        json=payload
    )
    if r.status_code not in [200, 201]:
        logger.error(f"Supabase insert error: {r.text}")

# === Prediction Route ===
@app.route("/predict", methods=["GET"])
def predict():
    df = fetch_ohlcv_binance(BINANCE_SYMBOL, interval=TIMEFRAME)
    if df is None or len(df) < 50:
        return jsonify({"status": "error", "message": "Not enough data"}), 400

    df = add_technical_features(df)

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
    except Exception as e:
        return jsonify({"error": "Model or scaler not found"}), 500

    # Load last signal
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?order=timestamp.desc&limit=1",
        headers=headers
    )
    last_state = r.json()[0] if r.status_code == 200 and r.json() else {}

    capital = float(last_state.get("capital", 10000.0))
    prev_position = last_state.get("position", "none")
    entry_price = float(last_state.get("price", 0.0)) if prev_position in ["long", "short"] else None

    row = df[feature_cols].iloc[-1:].dropna()
    if row.empty:
        return jsonify({"error": "Invalid feature row"}), 500

    X_scaled = scaler.transform(row)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    curr_price = float(df["close"].iloc[-1])

    # Determine action and new position
    action = "none"
    new_position = prev_position
    pnl = None
    commission_rate = 0.0005  # 0.05% × 2 (entry + exit)
    confidence_threshold = 0.55

    # Check if we should open a new position
    if prediction == 1 and probability > confidence_threshold:
        new_position = "long"
        action = "open" if prev_position != "long" else "none"
    elif prediction == 0 and probability < (1 - confidence_threshold):
        new_position = "short"
        action = "open" if prev_position != "short" else "none"
    else:
        new_position = prev_position
        action = "none"

    # If position changes → calculate and apply PnL
    if prev_position in ["long", "short"] and prev_position != new_position and entry_price:
        price_change = curr_price - entry_price
        pnl_pct = price_change / entry_price
        if prev_position == "short":
            pnl_pct *= -1

        pnl_pct -= 2 * commission_rate
        pnl = capital * pnl_pct
        capital += pnl
        action = "close" if new_position == "none" else "close+open"

    # Insert new row to Supabase
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction,
        "probability": probability,
        "action": action,
        "position": new_position,
        "price": curr_price,
        "pnl": pnl,
        "capital": capital
    }

    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
        headers={**headers, "Content-Type": "application/json"},
        json=payload
    )

    if r.status_code not in [200, 201]:
        return jsonify({"error": "Failed to insert row", "details": r.text}), 500

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "price": curr_price,
        "capital": capital,
        "pnl": pnl,
        "position": new_position,
        "action": action,
        "insert_status": "success"
    })

# === Test Endpoint ===
@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "API is live!"})

# === Run App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
