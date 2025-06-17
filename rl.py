# Install required libraries (if not already installed)
# You can uncomment these lines if running in a local environment
# !pip install stable-baselines3 gym pandas_ta yfinance matplotlib

import gym
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
import time
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Download historical 4H data for BTC/USD
def get_data(symbol='BTC/USDT', timeframe='4h', limit=500):
    binance = ccxt.binance({
        'options': {'defaultType': 'future'}
    })

    since = binance.milliseconds() - limit * 4 * 60 * 60 * 1000  # 4h candles
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    return df

# Custom Gym Environment
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.initial_balance = 1000.0
        self.reset()
        
        # Observation: [price, ema, rsi, macd, signal, atr, supertrend, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def reset(self):
        self.step_idx = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.equity_curve = []
        self.max_drawdown = 0
        self.max_equity = self.initial_balance
        logging.info(f"--- Environment reset --- Starting balance: ${self.balance}")
        return self._get_observation()

    def _get_observation(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['Close'],
            row['EMA_20'],
            row['RSI_14'],
            row['MACD_12_26_9'],
            row['MACDs_12_26_9'],
            row['ATRr_14'],
            row['SUPERT_7_3.0'],
            1 if self.shares > 0 else 0
        ], dtype=np.float32)

    def step(self, action):
        done = False
        self.current_step +=1
        reward = 0.0
        price = self.df.iloc[self.step_idx]['Close']

        # Execute action
        if action == 1 and self.shares == 0:  # Buy
            self.shares = self.balance / price
            self.balance = 0
        elif action == 2 and self.shares > 0:  # Sell
            self.balance = self.shares * price
            self.shares = 0

        # Update equity and reward
        total_equity = self.balance + self.shares * price
        self.max_equity = max(self.max_equity, total_equity)
        drawdown = (self.max_equity - total_equity) / self.max_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        self.equity_curve.append(total_equity)

        reward = (total_equity - self.initial_balance) / self.initial_balance
        reward -= drawdown  # penalize drawdowns

        self.step_idx += 1
        if self.step_idx >= self.max_steps:
            done = True
        logging.info(f"Step: {self.current_step} | Action: {action} | Price: {price:.2f} | Balance: {self.balance:.2f}")

        return self._get_observation(), reward, done, {}

    def render(self):
        plt.plot(self.equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Step")
        plt.ylabel("Equity")
        plt.show()

# Prepare data and environment
df = get_data()
# Add technical indicators
df['EMA_20'] = ta.ema(df['Close'], length=20)

# RSI 14
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# MACD 12/26/9
macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD_12_26_9'] = macd['MACD_12_26_9']
df['MACDs_12_26_9'] = macd['MACDs_12_26_9']

# ATR 14
df['ATRr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

# SuperTrend 7, 3.0
st = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
df['SUPERT_7_3.0'] = st['SUPERT_7_3.0']

# Drop warm-up NaNs
df.dropna(inplace=True)

env = TradingEnv(df)

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # steps
    save_path="./models/",
    name_prefix="ppo_trading_model",
    verbose=1,
)

model.learn(total_timesteps=5000)

# Test the 
model.save("ppo_trading_final")
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

env.render()
