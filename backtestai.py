import time, datetime, numpy as np, pandas as pd, os
import requests, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from hmmlearn.hmm import GaussianHMM
import pandas_ta as ta
import joblib
from scipy import stats

SYMBOL = "BTCUSDT"
INTERVAL = "4h"  # Changed to 4h for better accuracy
SEQ_LEN, PRED_LEN = 20, 1
HIDDEN_STATES, EPOCHS, BATCH_SIZE = 4, 150, 32  # Increased states and epochs for 4h
LR = 5e-4  # Lower LR for stability
FETCH_LIMIT = 1000 * 10  # 2000 4h candles = ~333 days
THRESHOLD = 0.008  # Higher threshold for 4h moves
CONFIDENCE_THRESHOLD = 0.52 # Higher confidence for 4h trades

# Simplified LSTM (reduced overfitting)
class EnhancedLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=128, layers=1):  # Simplified architecture
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden * 2)
        
        # Simplified classification head
        self.fc1 = nn.Linear(hidden * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm1(lstm_out)
        
        # Use last timestep
        x = lstm_out[:, -1]
        
        # Classification layers
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Fetch historical Binance data
def fetch_history(symbol="BTCUSDT", interval="4h", limit=2000):
    url, all_data = "https://api.binance.com/api/v3/klines", []
    end_time = int(time.time() * 1000)
    while len(all_data) < limit:
        fetch_count = min(1000, limit - len(all_data))
        params = {"symbol": symbol.upper(), "interval": interval, "limit": fetch_count, "endTime": end_time}
        r = requests.get(url, params=params)
        data = r.json()
        if not data: break
        all_data = data + all_data
        end_time = data[0][0] - 1
        time.sleep(0.5)  # Slower for 4h data
    
    df = pd.DataFrame(all_data, columns=["open_time", "o", "h", "l", "c", "v", "close_time", "qav", "n", "tbv", "tqv", "i"])
    df = df.astype({"o": float, "h": float, "l": float, "c": float, "v": float})
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["o", "h", "l", "c", "v"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    return df

# Smart Money & Advanced Indicators
def add_smart_money_indicators(df):
    # EMA Ribbon for trend
    df["ema8"] = ta.ema(df["close"], length=8)
    df["ema13"] = ta.ema(df["close"], length=13)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema34"] = ta.ema(df["close"], length=34)
    df["ema55"] = ta.ema(df["close"], length=55)
    df["ema200"] = ta.ema(df["close"], length=200)
    
    # EMA trend strength
    df["ema_trend"] = (
        (df["close"] > df["ema8"]).astype(int) +
        (df["ema8"] > df["ema13"]).astype(int) +
        (df["ema13"] > df["ema21"]).astype(int) +
        (df["ema21"] > df["ema34"]).astype(int) +
        (df["ema34"] > df["ema55"]).astype(int)
    ) / 5  # Normalized 0-1
    
    # VWAP (Volume Weighted Average Price)
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]
    
    # Ichimoku Cloud
    try:
        ich = ta.ichimoku(df["high"], df["low"], df["close"])
        if ich is not None and len(ich) > 0:
            ich_df = ich[0] if isinstance(ich, tuple) else ich
            
            # Get available columns
            available_cols = ich_df.columns.tolist()
            
            # Map to expected columns with fallbacks
            col_mapping = {
                'tenkan': ['ITS_9', 'tenkan_sen', 'conversion_line'],
                'kijun': ['IKS_26', 'kijun_sen', 'base_line'],
                'senkou_a': ['ISA_9', 'senkou_span_a', 'leading_span_a'],
                'senkou_b': ['ISB_26', 'senkou_span_b', 'leading_span_b'],
                'chikou': ['ICS_26', 'chikou_span', 'lagging_span']
            }
            
            for target_col, possible_names in col_mapping.items():
                found_col = None
                for name in possible_names:
                    if name in available_cols:
                        found_col = name
                        break
                
                if found_col:
                    df[target_col] = ich_df[found_col]
                else:
                    # Set to None if not found
                    df[target_col] = np.nan
        else:
            # Set all to NaN if ichimoku fails
            for col in ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']:
                df[col] = np.nan
    except:
        # Fallback if ichimoku completely fails
        df["tenkan"] = ta.sma(df["close"], length=9)
        df["kijun"] = ta.sma(df["close"], length=26)
        df["senkou_a"] = (df["tenkan"] + df["kijun"]) / 2
        df["senkou_b"] = ta.sma(df["close"], length=52)
        df["chikou"] = df["close"].shift(-26)
    
    # Cloud analysis
    df["cloud_top"] = np.maximum(df["senkou_a"], df["senkou_b"])
    df["cloud_bottom"] = np.minimum(df["senkou_a"], df["senkou_b"])
    df["above_cloud"] = (df["close"] > df["cloud_top"]).astype(int)
    df["cloud_thickness"] = (df["cloud_top"] - df["cloud_bottom"]) / df["close"]
    
    # Supertrend with multiple periods
    try:
        st1 = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=2)
        st2 = ta.supertrend(df["high"], df["low"], df["close"], length=14, multiplier=3)
        st3 = ta.supertrend(df["high"], df["low"], df["close"], length=21, multiplier=4)
        
        # Handle different possible column names
        def get_supertrend_col(st_result, length, multiplier):
            if st_result is not None and len(st_result.columns) > 0:
                possible_names = [
                    f"SUPERT_{length}_{multiplier}",
                    f"SUPERT_{length}_{multiplier:.1f}",
                    f"SUPERTd_{length}_{multiplier}",
                    f"SUPERTd_{length}_{multiplier:.1f}",
                    "SUPERT"
                ]
                for name in possible_names:
                    if name in st_result.columns:
                        return st_result[name]
                # If no match, return first column
                return st_result.iloc[:, 0]
            return np.nan
        
        df["supertr1"] = get_supertrend_col(st1, 10, 2.0)
        df["supertr2"] = get_supertrend_col(st2, 14, 3.0)
        df["supertr3"] = get_supertrend_col(st3, 21, 4.0)
        
    except:
        # Fallback supertrend calculation
        for i, (length, mult) in enumerate([(10, 2), (14, 3), (21, 4)], 1):
            hl2 = (df["high"] + df["low"]) / 2
            atr = ta.atr(df["high"], df["low"], df["close"], length=length)
            upper_band = hl2 + (mult * atr)
            lower_band = hl2 - (mult * atr)
            
            # Simple supertrend logic
            supertrend = np.where(df["close"] > upper_band.shift(1), lower_band, upper_band)
            df[f"supertr{i}"] = supertrend
    
    # Supertrend confluence
    df["st_bullish"] = (
        (df["close"] > df["supertr1"]).astype(int) +
        (df["close"] > df["supertr2"]).astype(int) +
        (df["close"] > df["supertr3"]).astype(int)
    ) / 3
    
    # ADX for trend strength
    try:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is not None and len(adx.columns) > 0:
            # Try different column name patterns
            adx_cols = adx.columns.tolist()
            adx_col = [col for col in adx_cols if 'ADX' in col and not ('DM' in col)]
            dmp_col = [col for col in adx_cols if 'DMP' in col or ('+DM' in col)]
            dmn_col = [col for col in adx_cols if 'DMN' in col or ('-DM' in col)]
            
            df["adx"] = adx[adx_col[0]] if adx_col else adx.iloc[:, 0]
            df["dmi_plus"] = adx[dmp_col[0]] if dmp_col else np.nan
            df["dmi_minus"] = adx[dmn_col[0]] if dmn_col else np.nan
        else:
            df["adx"] = np.nan
            df["dmi_plus"] = np.nan
            df["dmi_minus"] = np.nan
    except:
        # Simple ADX fallback
        df["adx"] = ta.rsi(df["close"], length=14)  # Use RSI as proxy
        df["dmi_plus"] = np.nan
        df["dmi_minus"] = np.nan
    
    df["dmi_diff"] = df["dmi_plus"] - df["dmi_minus"]
    
    # Enhanced momentum indicators
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["rsi21"] = ta.rsi(df["close"], length=21)
    
    # RSI Divergence detection (simplified)
    df["rsi_slope"] = df["rsi14"].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df["price_slope"] = df["close"].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df["rsi_divergence"] = np.where(
        (df["rsi_slope"] * df["price_slope"]) < 0, 1, 0
    )
    
    # MACD with histogram analysis
    try:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and len(macd.columns) > 0:
            macd_cols = macd.columns.tolist()
            
            macd_line_col = [col for col in macd_cols if 'MACD_' in col and 'MACDs' not in col and 'MACDh' not in col]
            macd_signal_col = [col for col in macd_cols if 'MACDs' in col]
            macd_hist_col = [col for col in macd_cols if 'MACDh' in col]
            
            df["macd_line"] = macd[macd_line_col[0]] if macd_line_col else macd.iloc[:, 0]
            df["macd_signal"] = macd[macd_signal_col[0]] if macd_signal_col else macd.iloc[:, 1] if len(macd.columns) > 1 else np.nan
            df["macd_hist"] = macd[macd_hist_col[0]] if macd_hist_col else macd.iloc[:, 2] if len(macd.columns) > 2 else np.nan
        else:
            df["macd_line"] = np.nan
            df["macd_signal"] = np.nan
            df["macd_hist"] = np.nan
    except:
        # Simple MACD fallback
        ema12 = ta.ema(df["close"], length=12)
        ema26 = ta.ema(df["close"], length=26)
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = ta.ema(df["macd_line"], length=9)
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    
    df["macd_hist_slope"] = df["macd_hist"].rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else 0, raw=False)
    
    # Stochastic RSI
    stoch = ta.stochrsi(df["close"], length=14)
    df["stochrsi_k"] = stoch["STOCHRSIk_14_14_3_3"]
    df["stochrsi_d"] = stoch["STOCHRSId_14_14_3_3"]
    
    # Williams %R
    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    
    # CCI (Commodity Channel Index)
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    
    # ATR and volatility
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_pct"] = df["atr14"] / df["close"]
    
    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_middle"] = bb["BBM_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # Keltner Channels
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2)
    if kc is not None and len(kc.columns) > 0:
        # Try different possible column names
        kc_cols = kc.columns.tolist()
        upper_col = [col for col in kc_cols if 'upper' in col.lower() or 'u' in col.lower()]
        lower_col = [col for col in kc_cols if 'lower' in col.lower() or 'l' in col.lower()]
        
        if upper_col and lower_col:
            df["kc_upper"] = kc[upper_col[0]]
            df["kc_lower"] = kc[lower_col[0]]
            df["kc_squeeze"] = ((df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])).astype(int)
        else:
            # Fallback: manual calculation
            df["kc_middle"] = ta.ema(df["close"], length=20)
            df["kc_upper"] = df["kc_middle"] + (2 * df["atr14"])
            df["kc_lower"] = df["kc_middle"] - (2 * df["atr14"])
            df["kc_squeeze"] = ((df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])).astype(int)
    else:
        # Manual Keltner Channel calculation
        df["kc_middle"] = ta.ema(df["close"], length=20)
        df["kc_upper"] = df["kc_middle"] + (2 * df["atr14"])
        df["kc_lower"] = df["kc_middle"] - (2 * df["atr14"])
        df["kc_squeeze"] = ((df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])).astype(int)
    
    # Donchian Channels
    df["dc_upper"] = df["high"].rolling(20).max()
    df["dc_lower"] = df["low"].rolling(20).min()
    df["dc_middle"] = (df["dc_upper"] + df["dc_lower"]) / 2
    df["dc_position"] = (df["close"] - df["dc_lower"]) / (df["dc_upper"] - df["dc_lower"])
    
    # Volume analysis
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    
    # Volume Profile approximation (price-volume relationship)
    df["pv_ratio"] = (df["close"].pct_change().abs() + 1e-8) / (df["volume_ratio"] + 1e-8)
    
    # Market Structure (simplified)
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["higher_low"] = (df["low"] > df["low"].shift(1)).astype(int)
    df["market_structure"] = df["higher_high"] + df["higher_low"] - 1  # -1 to 1 scale
    
    # Price action patterns
    df["doji"] = (abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8) < 0.1).astype(int)
    df["hammer"] = ((df["close"] > df["open"]) & 
                   ((df["open"] - df["low"]) > 2 * (df["close"] - df["open"])) &
                   ((df["high"] - df["close"]) < (df["close"] - df["open"]))).astype(int)
    
    # Multi-timeframe considerations (using rolling windows to simulate HTF)
    df["htf_trend"] = df["close"].rolling(96).mean()  # ~16 days for 4h = weekly trend
    df["htf_trend_slope"] = df["htf_trend"].rolling(12).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    return df.dropna()

# Enhanced HMM with more features
def compute_enhanced_hmm_states(df):
    # Select key features for HMM
    hmm_features = df[[
        "close", "volume_ratio", "rsi14", "atr_pct", "vwap_dist",
        "ema_trend", "st_bullish", "adx", "macd_hist", "bb_position"
    ]].values
    
    # Robust scaling for HMM
    scaler = RobustScaler()
    hmm_features_scaled = scaler.fit_transform(hmm_features)
    
    model = GaussianHMM(n_components=HIDDEN_STATES, covariance_type="full", n_iter=300)
    model.fit(hmm_features_scaled)
    states = model.predict(hmm_features_scaled)
    
    return states, model, scaler

# Enhanced dataset creation with smart money filters and class balancing
def create_enhanced_dataset(df, hmm_states):
    # Select comprehensive feature set
    feature_cols = [
        "close", "volume_ratio", "ema_trend", "vwap_dist", "rsi14", "rsi21",
        "atr_pct", "st_bullish", "above_cloud", "cloud_thickness",
        "adx", "dmi_diff", "rsi_divergence", "macd_line", "macd_signal", 
        "macd_hist", "macd_hist_slope", "stochrsi_k", "stochrsi_d",
        "williams_r", "cci", "bb_width", "bb_position", "kc_squeeze",
        "dc_position", "pv_ratio", "market_structure", "doji", "hammer",
        "htf_trend_slope"
    ]
    
    features = df[feature_cols].values
    features = np.hstack([features, hmm_states.reshape(-1, 1)])
    
    X, y = [], []
    
    for i in range(len(features) - SEQ_LEN - PRED_LEN):
        cur_idx = i + SEQ_LEN - 1
        fut_idx = i + SEQ_LEN + PRED_LEN - 1
        
        cur_price = features[cur_idx][0]  # close price
        fut_price = features[fut_idx][0]
        change = (fut_price - cur_price) / cur_price
        
        # Relaxed smart money filters
        atr_pct = features[cur_idx][6]  # atr_pct
        volume_ratio = features[cur_idx][1]  # volume_ratio
        
        # More lenient filters to get more data
        if atr_pct < 0.001 or volume_ratio < 0.3: 
            continue
        
        X.append(features[i:i+SEQ_LEN])
        y.append(1 if change > 0 else 0)
    
    X, y = np.array(X), np.array(y)
    
    # DIAGNOSTIC: Check class distribution
    print(f"🔍 Label distribution before balancing: {np.mean(y):.2f} → {np.bincount(y)}")
    
    # BALANCE THE DATASET
    if len(X) > 0:
        # Combine features and labels
        Xy = list(zip(X, y))
        class_0 = [xy for xy in Xy if xy[1] == 0]
        class_1 = [xy for xy in Xy if xy[1] == 1]
        
        # Match class sizes
        min_len = min(len(class_0), len(class_1))
        if min_len > 0:
            class_0 = resample(class_0, replace=False, n_samples=min_len, random_state=42)
            class_1 = resample(class_1, replace=False, n_samples=min_len, random_state=42)
            
            # Shuffle and unzip
            Xy_balanced = class_0 + class_1
            np.random.shuffle(Xy_balanced)
            X, y = zip(*Xy_balanced)
            X, y = np.array(X), np.array(y)
            
            print(f"✅ Label distribution after balancing: {np.mean(y):.2f} → {np.bincount(y)}")
    
    return X, y

def main():
    print("🚀 Enhanced Smart Money Crypto Predictor - 4H Timeframe")
    print("=" * 60)
    
    # Fetch and prepare data
    print("📊 Fetching historical data...")
    df = fetch_history(symbol=SYMBOL, interval=INTERVAL, limit=FETCH_LIMIT)
    print(f"✅ Fetched {len(df)} {INTERVAL} candles")
    
    # Add indicators
    print("🔧 Computing smart money indicators...")
    df = add_smart_money_indicators(df)
    print(f"✅ Added {len(df.columns)} features")
    
    # HMM states
    print("🧠 Computing HMM market regimes...")
    states, hmm_model, hmm_scaler = compute_enhanced_hmm_states(df)
    print(f"✅ Identified {HIDDEN_STATES} market regimes")
    
    # Create dataset
    print("📈 Creating training dataset...")
    X, y = create_enhanced_dataset(df, states)
    print(f"✅ Generated {len(X)} training samples")
    
    if len(X) == 0:
        print("❌ No valid samples generated. Adjust filters.")
        return
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Save scalers
    joblib.dump(scaler, f"scaler_{SYMBOL}_{INTERVAL}.pkl")
    joblib.dump(hmm_scaler, f"hmm_scaler_{SYMBOL}_{INTERVAL}.pkl")
    
    # Train/test split
    split = int(len(X_scaled) * 0.8)
    trainX = torch.tensor(X_scaled[:split]).float()
    trainY = torch.tensor(y[:split]).float().unsqueeze(1)
    testX = torch.tensor(X_scaled[split:]).float()
    testY = torch.tensor(y[split:]).float().unsqueeze(1)
    
    print(f"📚 Training: {len(trainX)}, Testing: {len(testX)}")
    
    # Calculate class weights for loss function
    pos_count = np.sum(y[:split])
    neg_count = len(y[:split]) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]) if pos_count > 0 else torch.tensor([1.0])
    print(f"📊 Class weight (pos): {pos_weight.item():.2f}")
    
    # Model setup with weighted loss
    model = EnhancedLSTMClassifier(trainX.shape[-1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    
    # Training loop
    print("🎯 Training enhanced model...")
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Shuffle training data
        idx = torch.randperm(len(trainX))
        for i in range(0, len(trainX), BATCH_SIZE):
            batch = idx[i:i+BATCH_SIZE]
            pred = model(trainX[batch])
            loss = loss_fn(pred, trainY[batch])
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
        
        # Validation with AUC
        model.eval()
        with torch.no_grad():
            val_pred = model(testX)
            val_loss = loss_fn(val_pred, testY)
            val_acc = ((torch.sigmoid(val_pred) > 0.5) == testY).float().mean()
            
            # Calculate AUC
            val_prob = torch.sigmoid(val_pred).detach().cpu().numpy()
            val_true = testY.cpu().numpy()
            try:
                val_auc = roc_auc_score(val_true, val_prob)
            except:
                val_auc = 0.5  # If AUC fails, use neutral value
        
        scheduler.step(val_loss)
        
        # Use AUC for model selection instead of loss
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"enhanced_predictor_{SYMBOL}_{INTERVAL}.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss/len(trainX)*BATCH_SIZE:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        if patience_counter >= 25:  # Increased patience
            print("Early stopping triggered")
            break
    
    # Load best model and backtest
    model.load_state_dict(torch.load(f"enhanced_predictor_{SYMBOL}_{INTERVAL}.pt"))
    model.eval()
    
    print(f"\n🔁 Backtesting on {INTERVAL} validation data...\n")
    print("=" * 80)
    
    wins, total, net_profit = 0, 0, 0.0
    last_pred_dir = None
    prev_price = None
    val_df = df.iloc[split + SEQ_LEN + PRED_LEN:]
    
    for i in range(len(testX)):
        seq = testX[i].unsqueeze(0)
        with torch.no_grad():
            logits = model(seq)
            pred_prob = torch.sigmoid(logits).item()
        
        # Higher confidence threshold for 4h trades
        if 1 - CONFIDENCE_THRESHOLD < pred_prob < CONFIDENCE_THRESHOLD:
            continue
        
        pred_dir = 1 if pred_prob > 0.5 else 0
        confidence = pred_prob if pred_dir == 1 else 1 - pred_prob
        cur_price = val_df["close"].iloc[i]
        
        if prev_price is not None and last_pred_dir is not None:
            actual_dir = 1 if cur_price > prev_price else 0
            correct = (actual_dir == last_pred_dir)
            pct_change = abs(cur_price - prev_price) / prev_price
            profit = pct_change * (1 if correct else -1) * 100  # Percentage profit
            
            wins += int(correct)
            total += 1
            net_profit += profit
            
            print(f"[{val_df.index[i].strftime('%Y-%m-%d %H:%M')}] {'✅' if correct else '❌'} "
                  f"Pred: {'UP' if last_pred_dir else 'DOWN'} | "
                  f"Price: {prev_price:.2f} → {cur_price:.2f} ({pct_change:.2%}) | "
                  f"Profit: {profit:+.2f}%")
            
            if total % 10 == 0:
                success_rate = wins / total
                print(f"📊 Success Rate: {wins}/{total} = {success_rate:.2%} | "
                      f"Net PnL: {net_profit:+.2f}% | "
                      f"Avg per trade: {net_profit/total:+.3f}%")
                print("-" * 80)
        
        last_pred_dir = pred_dir
        prev_price = cur_price
        
        direction = "📈 UP" if pred_dir else "📉 DOWN"
        print(f"🔮 Next {INTERVAL} prediction: {direction} (Confidence: {confidence:.1%})")
    
    if total > 0:
        final_success_rate = wins / total
        avg_profit_per_trade = net_profit / total
        print("\n" + "=" * 80)
        print("📈 FINAL BACKTEST RESULTS")
        print("=" * 80)
        print(f"🎯 Success Rate: {wins}/{total} = {final_success_rate:.2%}")
        print(f"💰 Net PnL: {net_profit:+.2f}%")
        print(f"📊 Avg per trade: {avg_profit_per_trade:+.3f}%")
        print(f"⏰ Timeframe: {INTERVAL}")
        print(f"💎 Confidence Threshold: {CONFIDENCE_THRESHOLD:.1%}")
        print("=" * 80)
    
    print("\n✅ Enhanced smart money predictor complete!")

if __name__ == "__main__":
    main()