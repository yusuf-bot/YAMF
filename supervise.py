import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from hmmlearn.hmm import GaussianHMM
import warnings
import joblib
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, symbol="ETH/USDT:USDT", timeframe="4h", lookback_periods=5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        self.scaler = StandardScaler()
        self.model = None
        self.hmm_model = None
        self.feature_cols = []
        
    def fetch_ohlcv(self, limit=500):
        """Fetch OHLCV data from exchange"""
        print(f"📥 Fetching {self.symbol} {self.timeframe} data...")
        logger.info(f"Fetching {self.symbol} {self.timeframe} data with limit {limit}")
        
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"Fetched {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def add_technical_features(self, df):
        """Add comprehensive technical indicators"""
        print("🧮 Calculating technical features...")
        logger.info("Adding technical features")
        
        # === MACD Histogram
        macd = ta.macd(df['close'])
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['macd_line'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        
        # === ADX & DI+/-
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        df['plus_di'] = adx['DMP_14']
        df['minus_di'] = adx['DMN_14']
        
        # === SuperTrend
        st = ta.supertrend(df['high'], df['low'], df['close'])
        df['supertrend'] = st['SUPERT_7_3.0']
        df['supertrend_dir'] = st['SUPERTd_7_3.0']
        
        # === Multiple EMAs
        df['ema9'] = ta.ema(df['close'], length=9)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)
        
        # === EMA Crossovers
        df['ema9_above_21'] = df['ema9'] > df['ema21']
        df['ema21_above_50'] = df['ema21'] > df['ema50']
        df['above_ema200'] = df['close'] > df['ema200']
        
        # === ROC (Rate of Change)
        df['roc'] = ta.roc(df['close'], length=10)
        df['roc_5'] = ta.roc(df['close'], length=5)
        
        # === RSI & StochRSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rsi_ma'] = ta.sma(df['rsi'], length=5)
        stochrsi = ta.stochrsi(df['close'], length=14)
        df['stochrsi_k'] = stochrsi['STOCHRSIk_14_14_3_3']
        df['stochrsi_d'] = stochrsi['STOCHRSId_14_14_3_3']
        
        # === Bollinger Bands
        bbands = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / df['close']
        df['bb_position'] = (df['close'] - bbands['BBL_20_2.0']) / (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0'])
        
        # === ATR & Volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_ma'] = ta.sma(df['atr'], length=5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # === Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # === Price momentum
        df['momentum'] = ta.mom(df['close'], length=10)
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # === Custom volatility percentile
        df['atr_percentile'] = df['atr'].rolling(100).apply(
            lambda x: (x.rank(pct=True).iloc[-1]) if len(x.dropna()) > 0 else np.nan
        )
        
        logger.info("Technical features added successfully")
        return df
    

    
    def add_additional_filters(self, df):
        """Add additional filter features"""
        print("🔧 Adding additional filters...")
        logger.info("Adding additional filters")
        
        # === Trend strength filter
        df['trend_strength'] = abs(df['adx']) * np.where(df['plus_di'] > df['minus_di'], 1, -1)
        
        # === Volatility filter
        df['high_volatility'] = (df['atr_percentile'] > 0.7).astype(int)
        df['low_volatility'] = (df['atr_percentile'] < 0.3).astype(int)
        
        # === Market structure filter
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(3).sum()
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(3).sum()
        
        # === Time-based filters
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_target_and_features(self, df, prediction_horizon=1):
        """Create target variable and feature matrix"""
        print("🎯 Creating target variable and features...")
        logger.info("Creating target variable and features")
        
        # Create target: 1 if price goes up, 0 if down
        df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Use a smaller threshold for more balanced targets
        price_change_threshold = 0.005  # 0.5% minimum price change
        df['target'] = np.where(df['future_return'] > price_change_threshold, 1, 
                               np.where(df['future_return'] < -price_change_threshold, 0, np.nan))
        
        # Remove ambiguous cases (small price changes)
        df = df.dropna(subset=['target'])
        
        # Check target distribution
        target_dist = df['target'].value_counts()
        print(f"📊 Target Distribution: Up={target_dist.get(1, 0)}, Down={target_dist.get(0, 0)}")
        print(f"📈 Target Balance: {target_dist.get(1, 0)/(len(df)):.1%} Up, {target_dist.get(0, 0)/(len(df)):.1%} Down")
        logger.info(f"Target distribution - Up: {target_dist.get(1, 0)}, Down: {target_dist.get(0, 0)}")
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                       'supertrend', 'ema9', 'ema21', 'ema50', 'ema200', 
                       'bb_upper', 'bb_lower', 'future_return', 'target']
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols and not pd.isna(df[col]).all()]
        
        # Create lagged features for time series prediction
        for lag in range(1, self.lookback_periods + 1):
            for col in ['close', 'volume', 'rsi', 'macd_hist']:
                if col in df.columns:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    if f'{col}_lag_{lag}' not in exclude_cols:
                        self.feature_cols.append(f'{col}_lag_{lag}')
        
        # Remove features with too many NaN values
        valid_features = []
        for col in self.feature_cols:
            if col in df.columns and df[col].notna().sum() > len(df) * 0.5:  # At least 50% valid data
                valid_features.append(col)
        
        self.feature_cols = valid_features
        print(f"🔧 Selected {len(self.feature_cols)} features for training")
        logger.info(f"Selected {len(self.feature_cols)} features for training")
        
        return df
    
    def train_models(self, df):
        """Train multiple ML models"""
        print("🤖 Training AI models...")
        logger.info("Starting model training")
        
        # Prepare data
        feature_data = df[self.feature_cols + ['target']].dropna()
        
        if len(feature_data) < 50:
            raise ValueError(f"Not enough data for training models. Got {len(feature_data)} samples, need at least 50")
        
        X = feature_data[self.feature_cols]
        y = feature_data['target']
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"Training data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (time series aware)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📚 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=50, 
                random_state=42, 
                max_depth=8,
                min_samples_split=5,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=50, 
                random_state=42, 
                max_depth=4,
                learning_rate=0.1
            )
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                train_score = accuracy_score(y_train, model.predict(X_train))
                test_score = accuracy_score(y_test, model.predict(X_test))
                print(f"📊 {name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
                logger.info(f"{name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
                
                if test_score > best_score:
                    best_score = test_score
                    best_model = model
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                logger.error(f"Error training {name}: {e}")
                continue
                
        if best_model is None:
            raise ValueError("No models could be trained successfully")
            
        self.model = best_model
        print(f"✅ Best model selected with test accuracy: {best_score:.3f}")
        logger.info(f"Best model selected with test accuracy: {best_score:.3f}")
        
        # Save model
        joblib.dump(self.model, "trained_model.pkl")

        # Optionally save feature column list if needed during prediction
        joblib.dump(self.feature_cols, "feature_cols.pkl")

        # Save other objects like scalers if used (e.g., StandardScaler)
        joblib.dump(self.scaler, "scaler.pkl")
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n🔝 Top 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
            logger.info(f"Top features: {importance_df.head(10)['feature'].tolist()}")
        
        return feature_data
    
    def backtest_strategy(self, df, initial_capital=10000, leverage=10, 
                        commission_rate=0.0006, funding_rate=0.0001, 
                        liquidation_threshold=0.8, margin_requirement=0.1):
        """Fixed leverage trading backtesting with proper accounting and entry/exit time tracking"""
        print("📈 Running leverage backtest simulation...")
        logger.info(f"Starting backtest - Capital: ${initial_capital}, Leverage: {leverage}x")
        
        # Prepare data for backtesting
        feature_data = df[self.feature_cols + ['target', 'close', 'timestamp']].dropna()
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Generate predictions
        X = feature_data[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create backtest dataframe
        backtest_df = feature_data.copy()
        backtest_df['prediction'] = predictions
        backtest_df['probability'] = probabilities
        backtest_df['actual'] = backtest_df['target']
        
        # Trading strategy signals
        confidence_threshold = 0.55
        backtest_df['signal'] = np.where(
            (backtest_df['probability'] > confidence_threshold) & (backtest_df['prediction'] == 1), 1,
            np.where((backtest_df['probability'] < (1 - confidence_threshold)) & (backtest_df['prediction'] == 0), -1, 0)
        )
        
        # Debug: Print signal distribution
        signal_counts = backtest_df['signal'].value_counts()
        print(f"📊 Signal Distribution: Long={signal_counts.get(1, 0)}, Short={signal_counts.get(-1, 0)}, No Signal={signal_counts.get(0, 0)}")
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        # Initialize trading variables
        capital = initial_capital
        position_size = 0  # Position size in base currency
        entry_price = 0
        margin_used = 0
        position_entry_time = None  # Track when position was opened
        trades = []
        equity_curve = []
        liquidations = 0
        total_funding_paid = 0
        total_commission_paid = 0
        trade_counter = 0
        just_opened = False  # Flag to prevent immediate closure
        
        # Track capital changes for debugging
        capital_changes = []
        
        for i in range(len(backtest_df)):
            row = backtest_df.iloc[i]
            current_price = row['close']
            signal = row['signal']
            timestamp = row['timestamp']
            
            # Reset just_opened flag at start of each iteration
            just_opened = False
            
            # Calculate unrealized PnL for existing position
            unrealized_pnl = 0
            if position_size != 0:
                unrealized_pnl = position_size * (current_price - entry_price)
            
            # Calculate current equity
            available_capital = capital
            total_equity = available_capital + margin_used + unrealized_pnl
            
            # Check for liquidation (only if position exists)
            if position_size != 0:
                margin_loss_pct = -unrealized_pnl / margin_used if margin_used > 0 else 0
                is_liquidated = margin_loss_pct >= liquidation_threshold
                
                if is_liquidated:
                    liquidation_loss = margin_used * liquidation_threshold
                    capital -= liquidation_loss
                    
                    trade_counter += 1
                    trade_record = {
                        'trade_id': trade_counter,
                        'entry_time': position_entry_time,  # Use stored entry time
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position_size,
                        'position_type': 'Long' if position_size > 0 else 'Short',
                        'pnl': -liquidation_loss,
                        'commission': 0,
                        'funding': 0,
                        'prediction': row['prediction'],
                        'actual': row['actual'],
                        'probability': row['probability'],
                        'liquidated': True,
                        'margin_used': margin_used,
                        'unrealized_pnl': unrealized_pnl,
                        'margin_loss_pct': margin_loss_pct
                    }
                    trades.append(trade_record)
                    
                    logger.warning(f"LIQUIDATION - Trade {trade_counter}: Loss ${liquidation_loss:.2f}, Margin Loss {margin_loss_pct:.1%}")
                    
                    # Reset position
                    position_size = 0
                    entry_price = 0
                    margin_used = 0
                    position_entry_time = None
                    liquidations += 1
                    
                    capital_changes.append({
                        'timestamp': timestamp,
                        'action': 'LIQUIDATION',
                        'amount': -liquidation_loss,
                        'capital_after': capital
                    })
                    continue
            
            # Pay funding fees for existing positions
            funding_fee = 0
            if position_size != 0:
                position_value = abs(position_size * current_price)
                funding_fee = position_value * funding_rate
                capital -= funding_fee
                total_funding_paid += funding_fee
                
                if funding_fee > 0:
                    capital_changes.append({
                        'timestamp': timestamp,
                        'action': 'FUNDING_FEE',
                        'amount': -funding_fee,
                        'capital_after': capital
                    })
            
            # Close existing position if signal changes or no signal (but not if just opened)
            if position_size != 0 and not just_opened and (signal == 0 or (signal > 0 and position_size < 0) or (signal < 0 and position_size > 0)):
                realized_pnl = unrealized_pnl
                position_value = abs(position_size * current_price)
                exit_commission = position_value * commission_rate
                total_commission_paid += exit_commission
                net_pnl = realized_pnl - exit_commission
                capital += margin_used + net_pnl
                
                trade_counter += 1
                trade_record = {
                    'trade_id': trade_counter,
                    'entry_time': position_entry_time,  # Use stored entry time
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size': position_size,
                    'position_type': 'Long' if position_size > 0 else 'Short',
                    'pnl': net_pnl,
                    'gross_pnl': realized_pnl,
                    'commission': exit_commission,
                    'funding': funding_fee,
                    'prediction': row['prediction'],
                    'actual': row['actual'],
                    'probability': row['probability'],
                    'liquidated': False,
                    'margin_used': margin_used,
                    'unrealized_pnl': unrealized_pnl,
                    'margin_loss_pct': 0
                }
                trades.append(trade_record)
                
                trade_result = "WIN" if net_pnl > 0 else "LOSS"
                #logger.info(f"TRADE {trade_counter} CLOSED - {trade_result}: "
                #       f"Entry: ${entry_price:.2f}, Exit: ${current_price:.2f}, "
                #       f"PnL: ${net_pnl:.2f}, Gross: ${realized_pnl:.2f}, "
                #       f"Commission: ${exit_commission:.2f}")
                
                capital_changes.append({
                    'timestamp': timestamp,
                    'action': 'CLOSE_POSITION',
                    'amount': margin_used + net_pnl,
                    'capital_after': capital
                })
                
                # Reset position
                position_size = 0
                entry_price = 0
                margin_used = 0
                position_entry_time = None
            
            # Open new position
            if signal != 0 and position_size == 0 and capital > 1000:
                risk_per_trade = capital * 0.02
                margin_to_use = min(capital * 0.1, risk_per_trade * leverage)
                
                if margin_to_use >= 100:
                    position_value = margin_to_use * leverage
                    position_size = (position_value / current_price) * signal
                    entry_price = current_price
                    margin_used = margin_to_use
                    position_entry_time = timestamp  # Store entry time
                    
                    entry_commission = position_value * commission_rate
                    total_commission_paid += entry_commission
                    capital -= (margin_to_use + entry_commission)

                    #logger.info(f"POSITION OPENED - Signal: {signal}, Size: {abs(position_size):.4f}, "
                    #        f"Value: ${position_value:.2f}, Margin: ${margin_to_use:.2f}, "
                    #        f"Commission: ${entry_commission:.2f}")

                    capital_changes.append({
                        'timestamp': timestamp,
                        'action': 'OPEN_POSITION',
                        'amount': -(margin_to_use + entry_commission),
                        'capital_after': capital
                    })
                    
                    just_opened = True  # Set flag to prevent immediate closure
            
            # Record equity curve
            available_capital = capital
            total_equity = available_capital + margin_used + unrealized_pnl
            
            equity_curve.append({
                'timestamp': timestamp,
                'available_capital': available_capital,
                'margin_used': margin_used,
                'unrealized_pnl': unrealized_pnl,
                'total_equity': total_equity,
                'position_size': position_size,
                'price': current_price
            })
        
        # Close final position if exists
        if position_size != 0:
            final_price = backtest_df['close'].iloc[-1]
            final_unrealized_pnl = position_size * (final_price - entry_price)
            position_value = abs(position_size * final_price)
            exit_commission = position_value * commission_rate
            final_net_pnl = final_unrealized_pnl - exit_commission
            capital += margin_used + final_net_pnl
            total_commission_paid += exit_commission
            
            trade_counter += 1
            trade_record = {
                'trade_id': trade_counter,
                'entry_time': position_entry_time,  # Use stored entry time
                'exit_time': backtest_df['timestamp'].iloc[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_size': position_size,
                'position_type': 'Long' if position_size > 0 else 'Short',
                'pnl': final_net_pnl,
                'gross_pnl': final_unrealized_pnl,
                'commission': exit_commission,
                'funding': 0,  # Note: Add funding fee if applicable
                'prediction': backtest_df['prediction'].iloc[-1],
                'actual': backtest_df['actual'].iloc[-1],
                'probability': backtest_df['probability'].iloc[-1],
                'liquidated': False,
                'margin_used': margin_used,
                'unrealized_pnl': final_unrealized_pnl,
                'margin_loss_pct': 0
            }
            trades.append(trade_record)
            #logger.info(f"FINAL POSITION CLOSED - PnL: ${final_net_pnl:.2f}")
        
        # Create results dataframes
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        capital_changes_df = pd.DataFrame(capital_changes)
        
        # Calculate performance metrics
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital
        num_trades = len(trades_df)
        
        # Save detailed logs
        if not trades_df.empty:
            trades_df.to_csv('detailed_trades.csv', index=False)
            logger.info(f"Saved {len(trades_df)} trades to detailed_trades.csv")
        
        if not capital_changes_df.empty:
            capital_changes_df.to_csv('capital_changes.csv', index=False)
            logger.info(f"Saved {len(capital_changes_df)} capital changes to capital_changes.csv")
        
        if not equity_df.empty:
            equity_df.to_csv('equity_curve.csv', index=False)
            logger.info(f"Saved equity curve to equity_curve.csv")
        
        # Calculate additional metrics
        if num_trades > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            win_rate = len(winning_trades) / num_trades
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            total_wins = total_losses = 0
        
        # Model accuracy
        accuracy = accuracy_score(backtest_df['actual'], backtest_df['prediction'])
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("📊 LEVERAGE TRADING BACKTEST RESULTS")
        print("="*70)
        print(f"\n💰 CAPITAL SUMMARY:")
        print(f"   Initial Capital:        ${initial_capital:,.2f}")
        print(f"   Final Capital:          ${final_capital:,.2f}")
        print(f"   Total Return:           {total_return:.2%}")
        print(f"   Net Profit/Loss:        ${final_capital - initial_capital:,.2f}")
        print(f"\n📈 TRADING PERFORMANCE:")
        print(f"   Total Trades:           {num_trades}")
        print(f"   Winning Trades:         {len(winning_trades) if num_trades > 0 else 0}")
        print(f"   Losing Trades:          {len(losing_trades) if num_trades > 0 else 0}")
        print(f"   Win Rate:               {win_rate:.2%}")
        print(f"   Average Win:            ${avg_win:.2f}")
        print(f"   Average Loss:           ${avg_loss:.2f}")
        print(f"\n🔥 LEVERAGE & RISK:")
        print(f"   Leverage Used:          {leverage}x")
        print(f"   Liquidations:           {liquidations}")
        print(f"   Liquidation Rate:       {liquidations/num_trades:.2%}" if num_trades > 0 else "   Liquidation Rate:       0.00%")
        print(f"\n💸 COSTS & FEES:")
        print(f"   Total Commission Paid:  ${total_commission_paid:.2f}")
        print(f"   Total Funding Paid:     ${total_funding_paid:.2f}")
        print(f"   Total Fees:             ${total_commission_paid + total_funding_paid:.2f}")
        print(f"   Fees as % of Capital:   {(total_commission_paid + total_funding_paid)/initial_capital:.2%}")
        print(f"\n🤖 AI MODEL PERFORMANCE:")
        print(f"   Prediction Accuracy:    {accuracy:.2%}")
        print(f"   Confidence Threshold:   {confidence_threshold:.1%}")
        
        if num_trades > 0:
            print(f"\n📊 TRADE BREAKDOWN:")
            print(f"   Regular Exits:          {num_trades - liquidations}")
            print(f"   Liquidated Positions:   {liquidations}")
            print(f"   Average Hold Time:      {(trades_df['exit_time'] - trades_df['entry_time']).mean()}")
        
            print("\n📋 RECENT TRADES SAMPLE:")
            print("-" * 70)
            recent_trades = trades_df.tail(5)[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'liquidated', 'probability']]
            for idx, trade in recent_trades.iterrows():
                status = "LIQUIDATED" if trade['liquidated'] else "CLOSED"
                print(f"   Entry: {trade['entry_time']} at ${trade['entry_price']:.2f} → "
                    f"Exit: {trade['exit_time']} at ${trade['exit_price']:.2f} | "
                    f"PnL: ${trade['pnl']:+.2f} | Conf: {trade['probability']:.2%} | {status}")
        
        print("\n" + "="*70)
        
        return {
            'trades': trades_df,
            'equity_curve': equity_df,
            'backtest_data': backtest_df,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'liquidations': liquidations,
            'total_commission_paid': total_commission_paid,
            'total_funding_paid': total_funding_paid,
            'profit_factor': profit_factor,
        }
        
    def run_full_analysis(self):
        """Run the complete trading system analysis"""
        print("🚀 Starting Full Trading System Analysis")
        print("="*50)
        
        try:
            # 1. Fetch data
            df = self.fetch_ohlcv(limit=1000)
            
            # 2. Add all features
            df = self.add_technical_features(df)
          
            df = self.add_additional_filters(df)
            df = self.create_target_and_features(df)
            
            # 3. Train models
            feature_data = self.train_models(df)
            
            # 4. Run backtest
            results = self.backtest_strategy(df)
            
            print("\n✅ Analysis Complete!")
            return results
            
        except Exception as e:
            print(f"❌ Error in analysis: {str(e)}")
            return None

def main():
    # Initialize trading system
    system = TradingSystem(symbol="ETH/USDT:USDT", timeframe="4h", lookback_periods=3)
    
    # Run full analysis with different leverage scenarios
    leverage_scenarios = [3]
    
    print("🚀 MULTI-SCENARIO LEVERAGE ANALYSIS")
    print("="*70)
    
    best_scenario = None
    best_return = -float('inf')
    
    for leverage in leverage_scenarios:
        print(f"\n🔥 TESTING LEVERAGE: {leverage}x")
        print("-" * 50)
        
        try:
            # Fetch and prepare data
            df = system.fetch_ohlcv(limit=1000)
            df = system.add_technical_features(df)
            df = system.add_additional_filters(df)
            df = system.create_target_and_features(df)
            
            # Train models (only once)
            if system.model is None:
                system.train_models(df)
            
            # Run backtest with current leverage
            results = system.backtest_strategy(
                df, 
                initial_capital=10000,
                leverage=leverage,
                commission_rate=0.0005,  # 0.05% commission
                funding_rate=0.0001,     # 0.01% funding per period
                liquidation_threshold=0.66,  # 66% threshold
                margin_requirement=0.1   # 10% margin requirement
            )
            
            if results and results['total_return'] > best_return:
                best_return = results['total_return']
                best_scenario = {
                    'leverage': leverage,
                    'results': results
                }
                
        except Exception as e:
            print(f"❌ Error with {leverage}x leverage: {str(e)}")
            continue
    
    # Print best scenario summary
    if best_scenario:
        print("\n" + "="*70)
        print("🏆 BEST PERFORMING SCENARIO")
        print("="*70)
        print(f"🔥 Optimal Leverage: {best_scenario['leverage']}x")
        print(f"💰 Best Return: {best_scenario['results']['total_return']:.2%}")
        print(f"📊 Accuracy: {best_scenario['results']['accuracy']:.2%}")
        print(f"🎯 Win Rate: {best_scenario['results']['win_rate']:.2%}")
        print(f"⚠️ Liquidations: {best_scenario['results']['liquidations']}")
        print(f"💸 Total Fees: ${best_scenario['results']['total_commission_paid'] + best_scenario['results']['total_funding_paid']:.2f}")
        
        print("\n🎉 Complete trading system analysis finished!")
        print("💡 Consider the risk-return tradeoff when choosing leverage.")
        print("⚠️ Higher leverage = Higher returns but also Higher liquidation risk!")
    else:
        print("❌ All scenarios failed. Please check your data and configuration.")

if __name__ == "__main__":
    main()