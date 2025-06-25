import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import time
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_simulation_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSimulator:
    def __init__(self, symbol="ETH/USDT", timeframe="4h", lookback_periods=5):
        self.symbol = symbol.upper().replace("/", "")
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        
        
        # Trading simulation variables
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.current_position = None
        self.position_size = 0
        self.entry_price = 0
        self.margin_used = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_capital = self.initial_capital
        
        # Load model if available
        try:
            self.model = joblib.load("trainesd_model.pkl")
            self.scaler = joblib.load('scaler.pkl')
            self.feature_cols = joblib.load('feature_cols.pkl')
            logger.info("Loaded saved model, scaler, and feature columns")
            self.use_model = True
        except FileNotFoundError:
            logger.warning("Model files not found. Using simple strategy instead.")
            self.use_model = False

    def fetch_latest_ohlcv(self, limit=500):
        """Fetch latest OHLCV data using Binance API with requests."""
        logger.info(f"Fetching latest {self.symbol} {self.timeframe} data via REST")
        base_url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": "4h",
            "limit": limit
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            raw_data = response.json()

            # Columns: open time, open, high, low, close, volume, close time, quote asset volume, num trades...
            df = pd.DataFrame(raw_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'num_trades', 'taker_base_volume',
                'taker_quote_volume', 'ignore'
            ])

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            logger.info(f"Fetched {len(df)} data points. Latest: {df['timestamp'].iloc[-1]}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None

    def add_technical_features(self, df):
        """Add same technical indicators as in training"""
        logger.info("Adding technical features")
        
        # Same technical indicators as in original code
        macd = ta.macd(df['close'])
        df['macd_hist'] = macd['MACDh_12_26_9']
        df['macd_line'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        df['plus_di'] = adx['DMP_14']
        df['minus_di'] = adx['DMN_14']
        
        st = ta.supertrend(df['high'], df['low'], df['close'])
        df['supertrend'] = st['SUPERT_7_3.0']
        df['supertrend_dir'] = st['SUPERTd_7_3.0']
        
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
        
        bbands = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / df['close']
        df['bb_position'] = (df['close'] - bbands['BBL_20_2.0']) / (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0'])
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_ma'] = ta.sma(df['atr'], length=5)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        df['momentum'] = ta.mom(df['close'], length=10)
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        df['atr_percentile'] = df['atr'].rolling(100).apply(
            lambda x: (x.rank(pct=True).iloc[-1]) if len(x.dropna()) > 0 else np.nan
        )
        
        df['trend_strength'] = abs(df['adx']) * np.where(df['plus_di'] > df['minus_di'], 1, -1)
        df['high_volatility'] = (df['atr_percentile'] > 0.7).astype(int)
        df['low_volatility'] = (df['atr_percentile'] < 0.3).astype(int)
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(3).sum()
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(3).sum()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add lagged features
        for lag in range(1, self.lookback_periods + 1):
            for col in ['close', 'volume', 'rsi', 'macd_hist']:
                if col in df.columns:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df

    def make_prediction(self, df):
        """Make trading prediction using the loaded model or simple strategy"""
        if self.use_model:
            try:
                latest_data = df[self.feature_cols].iloc[-1:].dropna()
                if latest_data.empty:
                    logger.warning("No valid data for prediction")
                    return None, None
                
                X_scaled = self.scaler.transform(latest_data)
                prediction = self.model.predict(X_scaled)[0]
                probability = self.model.predict_proba(X_scaled)[0, 1]
                return prediction, probability
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                return None, None
       
         

    def simulate_order(self, side, quantity, price):
        """Simulate placing an order"""
        try:
            order = {
                'id': f'sim_{int(time.time())}',
                'symbol': self.symbol,
                'side': side,
                'amount': abs(quantity),
                'price': price,
                'timestamp': datetime.now(),
                'status': 'filled'
            }
            logger.info(f"Simulated order: {side} {abs(quantity):.4f} {self.symbol} at {price:.2f}")
            return order
        except Exception as e:
            logger.error(f"Error simulating order: {e}")
            return None

    def calculate_position_size(self, capital, leverage, risk_per_trade=0.02):
        """Calculate position size based on risk management"""
        risk_amount = capital * risk_per_trade
        margin_to_use = min(capital * 0.1, risk_amount * leverage)
        return margin_to_use

    def update_statistics(self, pnl):
        """Update trading statistics"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def print_statistics(self):
        """Print current trading statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        print("\n" + "="*50)
        print("📊 TRADING STATISTICS")
        print("="*50)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Current Capital: ${self.current_capital:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total PnL: ${self.total_pnl:+,.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Max Drawdown: {self.max_drawdown:.1%}")
        print("="*50)

    def run_simulation(self, leverage=3, commission_rate=0.0005, polling_interval=300):
        """Run trading simulation with auto-flip support"""
        print("🎮 Starting Trading Simulation")
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Initial Capital: ${self.current_capital:,.2f}")
        print(f"Leverage: {leverage}x")
        print("-" * 50)

        logger.info("Starting trading simulation")

        try:
            while True:
                df = self.fetch_latest_ohlcv()
                if df is None:
                    time.sleep(polling_interval)
                    continue

                df = self.add_technical_features(df)
                prediction, probability = self.make_prediction(df)

                if prediction is None:
                    logger.warning("Skipping trading cycle due to prediction error")
                    time.sleep(polling_interval)
                    continue

                current_price = df['close'].iloc[-1]
                confidence_threshold = 0.55

                signal = 0
                if prediction == 1 and probability > confidence_threshold:
                    signal = 1
                elif prediction == 0 and probability < (1 - confidence_threshold):
                    signal = -1

                logger.info(f"Signal: {signal}, Probability: {probability:.2%}, Price: {current_price:.2f}")

                # Auto-flip logic
                if self.current_position is not None and (
                    signal == 0 or
                    (signal == 1 and self.current_position == 'short') or
                    (signal == -1 and self.current_position == 'long')
                ):
                    # Close current position
                    side = 'sell' if self.current_position == 'long' else 'buy'
                    order = self.simulate_order(side, abs(self.position_size), current_price)

                    if order:
                        realized_pnl = self.position_size * (current_price - self.entry_price)
                        commission = abs(self.position_size * current_price) * commission_rate
                        self.current_capital += (realized_pnl - commission)
                        self.update_statistics(realized_pnl)

                        logger.info(f"Closed {self.current_position} position: "
                                    f"PnL=${realized_pnl:.2f}, Commission=${commission:.2f}")

                        self.current_position = None
                        self.position_size = 0
                        self.entry_price = 0
                        self.margin_used = 0

                        # Auto-flip: Open new position immediately
                        if signal in [1, -1] and self.current_capital > 1000:
                            margin_to_use = self.calculate_position_size(self.current_capital, leverage)
                            position_value = margin_to_use * leverage
                            self.position_size = position_value / current_price * (1 if signal == 1 else -1)

                            side = 'buy' if signal == 1 else 'sell'
                            order = self.simulate_order(side, abs(self.position_size), current_price)

                            if order:
                                self.current_position = 'long' if signal == 1 else 'short'
                                self.entry_price = current_price
                                commission = position_value * commission_rate
                                self.margin_used = margin_to_use
                                self.current_capital -= commission

                                logger.info(f"Auto-flipped to {self.current_position} position: "
                                            f"Size={self.position_size:.4f}, Entry=${self.entry_price:.2f}, "
                                            f"Margin=${margin_to_use:.2f}")

                elif signal != 0 and self.current_position is None and self.current_capital > 1000:
                    # Open new position
                    margin_to_use = self.calculate_position_size(self.current_capital, leverage)
                    position_value = margin_to_use * leverage
                    self.position_size = position_value / current_price * (1 if signal == 1 else -1)

                    side = 'buy' if signal == 1 else 'sell'
                    order = self.simulate_order(side, abs(self.position_size), current_price)

                    if order:
                        self.current_position = 'long' if signal == 1 else 'short'
                        self.entry_price = current_price
                        commission = position_value * commission_rate
                        self.margin_used = margin_to_use
                        self.current_capital -= commission

                        logger.info(f"Opened {self.current_position} position: "
                                    f"Size={self.position_size:.4f}, Entry=${self.entry_price:.2f}, "
                                    f"Margin=${margin_to_use:.2f}")

                # Unrealized PnL and logging
                unrealized_pnl = 0
                if self.current_position:
                    unrealized_pnl = self.position_size * (current_price - self.entry_price)

                total_equity = self.current_capital + unrealized_pnl
                logger.info(f"Capital: ${self.current_capital:.2f}, Unrealized PnL: ${unrealized_pnl:.2f}, "
                            f"Total Equity: ${total_equity:.2f}")

                if self.total_trades % 10 == 0 or self.current_position is None:
                    self.print_statistics()

                print(f"⏳ Waiting {polling_interval//60} minutes for next update...")
                time.sleep(polling_interval)

        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")
            logger.info("Simulation stopped by user")

            if self.current_position:
                logger.info("Closing open position...")
                current_price = df['close'].iloc[-1] if df is not None else self.entry_price
                realized_pnl = self.position_size * (current_price - self.entry_price)
                self.current_capital += realized_pnl
                self.update_statistics(realized_pnl)

            self.print_statistics()

        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
            self.print_statistics()


if __name__ == "__main__":
    # Create and run simulation
    simulator = TradingSimulator(
        symbol="ETH/USDT",
        timeframe="4h",
        lookback_periods=3
    )
    
    simulator.run_simulation(
        leverage=3,
        commission_rate=0.0005,
        polling_interval=300  # 5 minutes for testing, use 14400 for 4h candles
    )