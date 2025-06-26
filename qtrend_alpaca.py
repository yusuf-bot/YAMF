from flask import Flask, jsonify
import pandas as pd, numpy as np, requests, joblib, logging
from datetime import datetime,timedelta,UTC
import pandas_ta as ta
import os, time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

# === Setup ===
load_dotenv()
app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Environment Variable Validation ===
required_env_vars = ["SUPABASE_URL", "SUPABASE_KEY", "TABLE_NAME", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {missing_vars}")

# === Constants ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("TABLE_NAME")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
SYMBOL = "ETH/USDT"  # Alpaca crypto symbol
ALPACA_SYMBOL = "ETHUSD"   # 4 hour timeframe
MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"
CONFIDENCE_THRESHOLD = 0.55
MAX_LOSS_THRESHOLD = 0.20  # 20% maximum loss protection
MIN_DATA_POINTS = 250  # Minimum data points needed for indicators
EQUITY_USAGE = 0.95  # Use 95% of available equity (keep 5% as buffer)

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

# === Account Management ===
def get_account_info():
    """Get current account information from Alpaca"""
    try:
        account = trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked
        }
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return None

def get_current_positions():
    """Get current positions from Alpaca"""
    try:
        positions = trading_client.get_all_positions()
        current_position = None
        
        for position in positions:
            if position.symbol == ALPACA_SYMBOL:
                current_position = {
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "side": "long" if float(position.qty) > 0 else "short",
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "avg_entry_price": float(position.avg_entry_price)
                }
                break
        
        return current_position
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return None

def close_all_positions():
    """Close all positions for the symbol"""
    try:
        response = trading_client.close_position(ALPACA_SYMBOL)
        logger.info(f"Position closed: {response}")
        return True
    except APIError as e:
        if "position does not exist" in str(e).lower():
            logger.info("No position to close")
            return True
        logger.error(f"Error closing position: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error closing position: {e}")
        return False

def place_market_order(side, quantity):
    """Place a market order"""
    try:
        order_request = MarketOrderRequest(
            symbol=ALPACA_SYMBOL,
            qty=quantity,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        
        order = trading_client.submit_order(order_request)
        logger.info(f"Order submitted: {side} {quantity} {ALPACA_SYMBOL} - Order ID: {order.id}")
        
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "status": order.status.value,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None
        }
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None

def calculate_position_size(account_equity, current_price):
    """Calculate position size based on available equity"""
    try:
        # Use 95% of equity to leave buffer for fees
        usable_equity = account_equity * EQUITY_USAGE
        quantity = usable_equity / current_price
        
        # Round to appropriate decimal places (crypto typically 6 decimals)
        quantity = round(quantity, 6)
        
        logger.info(f"Position size calculation: Equity=${account_equity:.2f}, Price=${current_price:.2f}, Qty={quantity}")
        return quantity
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0

# === Data Validation ===
def validate_ohlcv_data(df):
    """Validate OHLCV data quality"""
    if df is None or df.empty:
        return False, "Empty dataframe"
    
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for null values
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    if df[numeric_cols].isnull().any().any():
        return False, "Contains null values in OHLCV data"
    
    # Check for non-positive prices
    price_cols = ['open', 'high', 'low', 'close']
    if (df[price_cols] <= 0).any().any():
        return False, "Contains non-positive prices"
    
    # Basic OHLC validation
    invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                   (df['high'] < df['close']) | (df['low'] > df['open']) | \
                   (df['low'] > df['close'])
    if invalid_ohlc.any():
        return False, "Invalid OHLC relationships (high < low, etc.)"
    
    return True, "Valid data"

# === Data Fetching ===
def fetch_ohlcv(symbol, start_time, end_time, granularity=14400):
    """Fetch OHLCV data from Binance for the specified symbol and timeframe"""
    try:
        granularity_map = {
            60: '1m',
            300: '5m',
            900: '15m',
            3600: '1h',
            14400: '4h',
            21600: '6h',
            86400: '1d'
        }
        if granularity not in granularity_map:
            logger.error(f"Invalid granularity: {granularity}. Supported: {list(granularity_map.keys())}")
            return None
        interval = granularity_map[granularity]

        # Map symbol to Binance format
        symbol_map = {
            'ETH/USD': 'ETHUSDT',
            'BTC/USD': 'BTCUSDT',
            # Add other mappings as needed
        }
        product_id = symbol_map.get(symbol, symbol.replace('/', '').upper())

        url = "https://api.binance.com/api/v3/klines"
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        max_candles = 1000
        delta_ms = granularity * 1000 * max_candles

        all_data = []
        current_start_ts = start_ts

        while current_start_ts < end_ts:
            current_end_ts = min(current_start_ts + delta_ms, end_ts)

            params = {
                'symbol': product_id,
                'interval': interval,
                'startTime': current_start_ts,
                'endTime': current_end_ts,
                'limit': max_candles
            }

            logger.info(f"Fetching Binance data with params: {params}")
            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Failed to fetch data. HTTP {response.status_code}: {response.text}")
                return None

            data = response.json()
            if not data:
                logger.warning(f"No data returned from {current_start_ts} to {current_end_ts}.")
                break

            all_data.extend(data)
            current_start_ts = current_end_ts + 1
            time.sleep(0.1)

        if not all_data:
            logger.warning("No data collected across all intervals.")
            return None

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.astype({
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float
        })
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Fetched {len(df)} rows from {start_time} to {end_time}")
        logger.info(df.tail())
        return df

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return None

def safe_indicator_calculation(func, *args, **kwargs):
    """Safely calculate technical indicators with error handling"""
    try:
        result = func(*args, **kwargs)
        if result is None:
            logger.warning(f"Indicator {func.__name__} returned None")
            return None
        return result
    except Exception as e:
        logger.warning(f"Error calculating {func.__name__}: {e}")
        return None

def add_technical_features(df):
    """Add technical indicators with comprehensive error handling"""
    if len(df) < MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: need at least {MIN_DATA_POINTS} points, got {len(df)}")
    
    df = df.copy()  # Avoid modifying original dataframe
    
    try:
        #close
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            logger.info(f"Added close_lag_{lag}")
        # MACD
        macd = safe_indicator_calculation(ta.macd, df['close'])
        if macd is not None:
            df['macd_hist'] = macd.get('MACDh_12_26_9', np.nan)
            df['macd_line'] = macd.get('MACD_12_26_9', np.nan)
            df['macd_signal'] = macd.get('MACDs_12_26_9', np.nan)
        else:
            df['macd_hist'] = df['macd_line'] = df['macd_signal'] = np.nan

        for lag in [1, 2, 3]:
            df[f'macd_hist_lag_{lag}'] = df['macd_hist'].shift(lag)
            logger.info(f"Added macd_hist_lag_{lag}")
        
        # ADX
        adx = safe_indicator_calculation(ta.adx, df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx.get('ADX_14', np.nan)
            df['plus_di'] = adx.get('DMP_14', np.nan)
            df['minus_di'] = adx.get('DMN_14', np.nan)
        else:
            df['adx'] = df['plus_di'] = df['minus_di'] = np.nan
        
        # SuperTrend
        st = safe_indicator_calculation(ta.supertrend, df['high'], df['low'], df['close'])
        if st is not None:
            df['supertrend'] = st.get('SUPERT_7_3.0', np.nan)
            df['supertrend_dir'] = st.get('SUPERTd_7_3.0', np.nan)
        else:
            df['supertrend'] = df['supertrend_dir'] = np.nan
        
        # EMAs
        df['ema9'] = safe_indicator_calculation(ta.ema, df['close'], length=9)
        df['ema21'] = safe_indicator_calculation(ta.ema, df['close'], length=21)
        df['ema50'] = safe_indicator_calculation(ta.ema, df['close'], length=50)
        df['ema200'] = safe_indicator_calculation(ta.ema, df['close'], length=200)
        
        # EMA relationships (handle NaN values)
        df['ema9_above_21'] = (df['ema9'] > df['ema21']).fillna(False)
        df['ema21_above_50'] = (df['ema21'] > df['ema50']).fillna(False)
        df['above_ema200'] = (df['close'] > df['ema200']).fillna(False)
        
        # ROC
        df['roc'] = safe_indicator_calculation(ta.roc, df['close'], length=10)
        df['roc_5'] = safe_indicator_calculation(ta.roc, df['close'], length=5)
        
        # RSI
        df['rsi'] = safe_indicator_calculation(ta.rsi, df['close'], length=14)
        df['rsi_ma'] = safe_indicator_calculation(ta.sma, df['rsi'], length=5) if 'rsi' in df and df['rsi'].notna().any() else np.nan
        
        for lag in [1, 2, 3]:
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            logger.info(f"Added rsi_lag_{lag}")

        # Stochastic RSI
        stochrsi = safe_indicator_calculation(ta.stochrsi, df['close'], length=14)
        if stochrsi is not None:
            df['stochrsi_k'] = stochrsi.get('STOCHRSIk_14_14_3_3', np.nan)
            df['stochrsi_d'] = stochrsi.get('STOCHRSId_14_14_3_3', np.nan)
        else:
            df['stochrsi_k'] = df['stochrsi_d'] = np.nan
        
        # Bollinger Bands
        bb = safe_indicator_calculation(ta.bbands, df['close'], length=20)
        if bb is not None:
            df['bb_upper'] = bb.get('BBU_20_2.0', np.nan)
            df['bb_lower'] = bb.get('BBL_20_2.0', np.nan)
            # Safe division for BB width and position
            bb_range = bb.get('BBU_20_2.0', np.nan) - bb.get('BBL_20_2.0', np.nan)
            df['bb_width'] = np.where(df['close'] > 0, bb_range / df['close'], np.nan)
            df['bb_position'] = np.where(bb_range > 0, 
                                       (df['close'] - bb.get('BBL_20_2.0', np.nan)) / bb_range, 
                                       np.nan)
        else:
            df['bb_upper'] = df['bb_lower'] = df['bb_width'] = df['bb_position'] = np.nan
        
        # ATR
        df['atr'] = safe_indicator_calculation(ta.atr, df['high'], df['low'], df['close'], length=14)
        df['atr_ma'] = safe_indicator_calculation(ta.sma, df['atr'], length=5) if 'atr' in df and df['atr'].notna().any() else np.nan
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Volume indicators
        df['volume_sma'] = safe_indicator_calculation(ta.sma, df['volume'], length=20)
        df['volume_ratio'] = np.where(df['volume_sma'] > 0, df['volume'] / df['volume_sma'], np.nan)
        
        for lag in [1, 2, 3]:
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            logger.info(f"Added volume_lag_{lag}")
        
        # Momentum
        df['momentum'] = safe_indicator_calculation(ta.mom, df['close'], length=10)
        df['williams_r'] = safe_indicator_calculation(ta.willr, df['high'], df['low'], df['close'], length=14)
        
        # Advanced features
        df['atr_percentile'] = df['atr'].rolling(100).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x.dropna()) >= 10 else np.nan, 
            raw=False
        )
        
        # Trend strength (handle NaN values)
        df['trend_strength'] = np.where(
            (df['adx'].notna()) & (df['plus_di'].notna()) & (df['minus_di'].notna()),
            abs(df['adx']) * np.where(df['plus_di'] > df['minus_di'], 1, -1),
            np.nan
        )
        
        # Volatility regime
        df['high_volatility'] = (df['atr_percentile'] > 0.7).astype(int)
        df['low_volatility'] = (df['atr_percentile'] < 0.3).astype(int)
        
        # Price patterns
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(3).sum()
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(3).sum()
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Fill NaN values for NUMERIC columns only (exclude timestamp and other non-numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove timestamp-related columns that might be numeric but shouldn't be filled
        numeric_columns = [col for col in numeric_columns if col not in ['timestamp']]
        
        # Use forward fill then backward fill for numeric columns only
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN (edge case), fill with 0
                df[col] = df[col].fillna(0)
        
        logger.info(f"Successfully calculated {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} technical features")
        return df
        
    except Exception as e:
        logger.error(f"Error in technical feature calculation: {e}")
        raise
    
# === Supabase Integration ===
def get_last_trade_state():
    """Get the last trading state from Supabase"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?order=timestamp.desc&limit=1"
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                return data[0]
        
        logger.warning("No previous trade state found in Supabase")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching last trade state: {e}")
        return None

def save_trade_to_supabase(payload):
    """Save trade data to Supabase with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            
            if r.status_code in [200, 201]:
                logger.info("Trade data saved to Supabase successfully")
                return True
            else:
                logger.warning(f"Supabase insert failed with status {r.status_code}: {r.text}")
                
        except Exception as e:
            logger.warning(f"Error saving trade data (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            time.sleep(1)
    
    logger.error("Failed to save trade data after all retries")
    return False

def calculate_position_action(prediction, probability, current_position, confidence_threshold):
    """Determine trading action based on prediction and current position"""
    action = "none"
    target_position = "none"
    
    # Strong bullish signal
    if prediction == 1 and probability > confidence_threshold:
        target_position = "long"
        if current_position != "long":
            action = "close+open" if current_position == "short" else "open"
    
    # Strong bearish signal
    elif prediction == 0 and probability < (1 - confidence_threshold):
        target_position = "short"
        if current_position != "short":
            action = "close+open" if current_position == "long" else "open"
    
    # Weak signal - close position if we have one
    else:
        target_position = "none"
        if current_position in ["long", "short"]:
            action = "close"
    
    return action, target_position

def execute_trading_action(action, target_position, account_info, current_price):
    """Execute the determined trading action"""
    orders_executed = []
    
    try:
        if action == "close":
            # Close existing position
            if close_all_positions():
                orders_executed.append({"action": "close", "success": True})
            else:
                orders_executed.append({"action": "close", "success": False})
        
        elif action == "open":
            # Open new position
            quantity = calculate_position_size(account_info["equity"], current_price)
            if quantity > 0:
                side = OrderSide.BUY if target_position == "long" else OrderSide.SELL
                order = place_market_order(side, quantity)
                if order:
                    orders_executed.append({"action": "open", "success": True})

             
        
        elif action == "close+open":
            # Close existing position first
            if close_all_positions():
                orders_executed.append({"action": "close", "success": True})
                
                # Wait a moment for the close to process
                time.sleep(2)
                
                # Refresh account info for new position
                updated_account = get_account_info()
                if updated_account:
                    account_info = updated_account
                
                # Open new position
                quantity = calculate_position_size(account_info["equity"], current_price)
                if quantity > 0:
                    side = OrderSide.BUY if target_position == "long" else OrderSide.SELL
                    order = place_market_order(side, quantity)
                    if order:
                        orders_executed.append({"action": "open", "order": order, "success": True})
                    else:
                        orders_executed.append({"action": "open", "success": False})
            else:
                orders_executed.append({"action": "close", "success": False})
        
        return orders_executed
        
    except Exception as e:
        logger.error(f"Error executing trading action: {e}")
        return [{"action": action, "success": False, "error": str(e)}]

# === Main Prediction Route ===
@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Get account information
        account_info = get_account_info()
        if not account_info:
            return jsonify({"error": "Failed to get account information"}), 500
        
        # Check account status
        if account_info["trading_blocked"] or account_info["account_blocked"]:
            return jsonify({"error": "Account is blocked from trading"}), 400
        
        # Get current positions
        current_alpaca_position = get_current_positions()
        current_position = "none"
        if current_alpaca_position:
            current_position = current_alpaca_position["side"]
        
        # Circuit breaker - check for maximum loss
        last_state = get_last_trade_state()
        try:
            initial_equity = float(last_state["initial_equity"]) if last_state and "initial_equity" in last_state else float(account_info["equity"])
            logger.info(f"Initial equity set to ${initial_equity:.2f} from {'Supabase' if last_state and 'initial_equity' in last_state else 'current account'}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid initial_equity value: {e}. Using current account equity.")
            initial_equity = float(account_info["equity"])
        
        if account_info["equity"] < initial_equity * (1 - MAX_LOSS_THRESHOLD):
            return jsonify({
                "status": "error",
                "message": f"Maximum loss threshold ({MAX_LOSS_THRESHOLD:.1%}) reached",
                "current_equity": account_info["equity"],
                "initial_equity": initial_equity
            }), 400

        # Fetch market data
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=84)  # Fetch 30 days of data
        df = fetch_ohlcv(SYMBOL, start_time, end_time)
        if df is None or len(df) < MIN_DATA_POINTS:
            return jsonify({
                "status": "error", 
                "message": f"Insufficient data: need at least {MIN_DATA_POINTS} points"
            }), 400

        # Add technical features
        df = add_technical_features(df)
        
        # Load ML models
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_cols = joblib.load(FEATURE_COLS_PATH)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return jsonify({"error": "Model loading failed", "details": str(e)}), 500

        # Validate features exist in dataframe
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            logger.error(f"Missing features in dataframe: {missing_features}")
            return jsonify({
                "error": "Feature validation failed", 
                "missing_features": list(missing_features)
            }), 500

        # Prepare data for prediction
        row = df[feature_cols].iloc[-1:].dropna(axis=1)  # Drop columns with NaN
        if row.empty:
            return jsonify({"error": "No valid feature data for prediction"}), 500

        # Ensure all feature columns are present (fill missing with 0)
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0.0

        # Reorder columns to match training
        row = row[feature_cols]
        
        # Scale features and predict
        X_scaled = scaler.transform(row)
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        probability = float(probabilities[1])  # Probability of class 1 (bullish)
        
        curr_price = float(df["close"].iloc[-1])
        
        # Determine trading action
        action, target_position = calculate_position_action(
            prediction, probability, current_position, CONFIDENCE_THRESHOLD
        )
        logger.info(f"Action determined: {action} with target position {target_position} (probability: {probability:.4f})")
        # Execute trading action
        orders_executed = []
        if action != "none":
            orders_executed = execute_trading_action(action, target_position, account_info, curr_price)
        print(orders_executed)
        # Get updated account and position info
        updated_account = get_account_info()
        updated_position = get_current_positions()
        print(f"Updated account info: {updated_position}")
        final_position = "none"
        if updated_position:
            final_position = updated_position["side"]

        # Calculate PnL if we have previous state
        pnl = 0.0
        if last_state and current_alpaca_position and action in ["close", "close+open"]:
            entry_price = float(last_state.get("entry_price", 0))
            if entry_price > 0:
                if current_position == "long":
                    pnl = (curr_price - entry_price) * float(current_alpaca_position["qty"])
                elif current_position == "short":
                    pnl = (entry_price - curr_price) * abs(float(current_alpaca_position["qty"]))

        # Prepare Supabase payload
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "probability": round(probability, 4),
            "position": final_position,
            "price": round(curr_price, 2),
            "entry_price": round(curr_price, 2) if action in ["open", "close+open"] else None,
            "pnl": round(pnl, 2) if pnl != 0 else None,
      
        }

        # Save to Supabase
        save_success = save_trade_to_supabase(payload)
        
        # Prepare response
        response = {
            "prediction": prediction,
            "probability": round(probability, 4),
            "price": round(curr_price, 2),
            "position": final_position,
            "pnl": round(pnl, 2) if pnl != 0 else None,
            "account": updated_account if updated_account else account_info,
            "orders_executed": orders_executed,
            "save_status": "success" if save_success else "failed",
            "timestamp": payload["timestamp"]
        }
        
        logger.info(f"Prediction completed: {action} -> {final_position} at ${curr_price:.2f}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Prediction failed", 
            "details": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

# === Health Check Endpoint ===
# Add this simple test endpoint

@app.route("/test-alpaca", methods=["GET"])
def test_alpaca():
    """Simple Alpaca connection test"""
    try:
        # Test the trading client initialization
        logger.info("Testing Alpaca trading client initialization...")
        
        # Check if credentials are set
        if not API_KEY or not API_SECRET:
            return jsonify({
                "error": "Missing Alpaca credentials",
                "api_key_set": bool(API_KEY),
                "api_secret_set": bool(API_SECRET)
            }), 400
        
        # Try to get account info
        logger.info("Attempting to get account information...")
        account = trading_client.get_account()
        
        return jsonify({
            "status": "SUCCESS",
            "account_status": account.status.value if hasattr(account, 'status') else 'unknown',
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
            "paper_trading": True,
            "timestamp": datetime.now(UTC).isoformat()
        })
        
    except APIError as e:
        logger.error(f"Alpaca API Error: {e}")
        return jsonify({
            "status": "API_ERROR",
            "error": str(e),
            "error_code": getattr(e, 'code', 'unknown'),
            "error_message": getattr(e, 'message', 'unknown'),
            "timestamp": datetime.utcnow().isoformat()
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        return jsonify({
            "status": "ERROR",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Comprehensive health check"""
    try:
        # Check database connection
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?limit=1"
        r = requests.get(url, headers=headers, timeout=5)
        db_status = "ok" if r.status_code == 200 else "error"
        
        # Check Binance API
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=1)  # Fetch 30 days of data
        test_data = fetch_ohlcv(SYMBOL, start_time, end_time)
        api_status = "ok" if test_data is not None else "error"
        
        # Check model files
        model_files = [MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH]
        models_exist = all(os.path.exists(f) for f in model_files)
        model_status = "ok" if models_exist else "error"
        
        overall_status = "ok" if all(s == "ok" for s in [db_status, api_status, model_status]) else "error"
        
        return jsonify({
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "database": db_status,
                "binance_api": api_status,
                "models": model_status
            },
            "version": "2.0"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500



    
# === Test Endpoint ===
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "ok", 
        "message": "Trading bot API is live!",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "2.0"
    })

# === Portfolio Status Endpoint ===
@app.route("/status", methods=["GET"])
def status():
    """Get current portfolio status"""
    try:
        last_state = get_last_trade_state()
        return jsonify({
            "capital": last_state.get("capital", 10000.0),
            "position": last_state.get("position", "none"),
            "last_price": last_state.get("price", 0.0),
            "timestamp": last_state.get("timestamp", ""),
            "status": "ok"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500



# === Run App ===
if __name__ == "__main__":
    logger.info("Starting Trading Bot API v2.0")
    logger.info(f"Target symbol: {ALPACA_SYMBOL}")
    logger.info(f"Timeframe: 4H")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    app.run(host="0.0.0.0", port=5000, debug=False)