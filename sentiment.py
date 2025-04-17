import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV and prepare it for analysis
    """
    df = pd.read_csv(file_path)
    
    # Ensure columns are properly named
    expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in expected_cols):
        # Try to map based on position
        df.columns = expected_cols[:len(df.columns)]
    
    # Convert timestamp if needed
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index if it exists
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    return df

def calculate_rsi(df: pd.DataFrame, length: int = 14, source_col: str = 'close') -> pd.Series:
    """Calculate RSI and normalize to 0-100 scale"""
    rsi = df.ta.rsi(close=source_col, length=length)
    
    # Normalize RSI values based on ranges similar to the original code
    normalized = pd.Series(index=rsi.index, dtype=float)
    
    # RSI > 70 => interpolate(rsi, 100, 70, 100, 75)
    mask = rsi > 70
    normalized[mask] = 75 + (rsi[mask] - 70) * (100 - 75) / (100 - 70)
    
    # RSI > 50 => interpolate(rsi, 70, 50, 75, 50)
    mask = (rsi > 50) & (rsi <= 70)
    normalized[mask] = 50 + (rsi[mask] - 50) * (75 - 50) / (70 - 50)
    
    # RSI > 30 => interpolate(rsi, 50, 30, 50, 25)
    mask = (rsi > 30) & (rsi <= 50)
    normalized[mask] = 25 + (rsi[mask] - 30) * (50 - 25) / (50 - 30)
    
    # RSI >= 0 => interpolate(rsi, 30, 0, 25, 0)
    mask = (rsi >= 0) & (rsi <= 30)
    normalized[mask] = 0 + (rsi[mask] - 0) * (25 - 0) / (30 - 0)
    
    return normalized

def calculate_stochastic(df: pd.DataFrame, length_k: int = 14, smooth_k: int = 3) -> pd.Series:
    """Calculate Stochastic %K and normalize to 0-100 scale"""
    stoch = df.ta.stoch(high='high', low='low', close='close', k=length_k, d=smooth_k)['STOCHk_' + str(length_k) + '_' + str(smooth_k) + '_' + str(smooth_k)]
    
    # Normalize Stochastic values
    normalized = pd.Series(index=stoch.index, dtype=float)
    
    # stoch > 80 => interpolate(stoch, 100, 80, 100, 75)
    mask = stoch > 80
    normalized[mask] = 75 + (stoch[mask] - 80) * (100 - 75) / (100 - 80)
    
    # stoch > 50 => interpolate(stoch, 80, 50, 75, 50)
    mask = (stoch > 50) & (stoch <= 80)
    normalized[mask] = 50 + (stoch[mask] - 50) * (75 - 50) / (80 - 50)
    
    # stoch > 20 => interpolate(stoch, 50, 20, 50, 25)
    mask = (stoch > 20) & (stoch <= 50)
    normalized[mask] = 25 + (stoch[mask] - 20) * (50 - 25) / (50 - 20)
    
    # stoch >= 0 => interpolate(stoch, 20, 0, 25, 0)
    mask = (stoch >= 0) & (stoch <= 20)
    normalized[mask] = 0 + (stoch[mask] - 0) * (25 - 0) / (20 - 0)
    
    return normalized

def calculate_stochastic_rsi(df: pd.DataFrame, rsi_length: int = 14, stoch_length: int = 14, 
                             stoch_smooth: int = 3, source_col: str = 'close') -> pd.Series:
    """Calculate Stochastic RSI and normalize to 0-100 scale"""
    # Calculate RSI first
    rsi = df.ta.rsi(close=source_col, length=rsi_length)
    
    # Calculate Stochastic of RSI
    stoch_rsi = pd.Series(index=rsi.index, dtype=float)
    for i in range(stoch_length, len(rsi)):
        window = rsi[i-stoch_length+1:i+1]
        if not window.isna().all():
            min_val = window.min()
            max_val = window.max()
            if max_val - min_val != 0:
                stoch_rsi[i] = (rsi[i] - min_val) / (max_val - min_val) * 100
            else:
                stoch_rsi[i] = 50  # Default value if range is zero
    
    # Apply smoothing
    stoch_rsi = stoch_rsi.rolling(window=stoch_smooth).mean()
    
    # Normalize Stochastic RSI values
    normalized = pd.Series(index=stoch_rsi.index, dtype=float)
    
    # stoch > 80 => interpolate(stoch, 100, 80, 100, 75)
    mask = stoch_rsi > 80
    normalized[mask] = 75 + (stoch_rsi[mask] - 80) * (100 - 75) / (100 - 80)
    
    # stoch > 50 => interpolate(stoch, 80, 50, 75, 50)
    mask = (stoch_rsi > 50) & (stoch_rsi <= 80)
    normalized[mask] = 50 + (stoch_rsi[mask] - 50) * (75 - 50) / (80 - 50)
    
    # stoch > 20 => interpolate(stoch, 50, 20, 50, 25)
    mask = (stoch_rsi > 20) & (stoch_rsi <= 50)
    normalized[mask] = 25 + (stoch_rsi[mask] - 20) * (50 - 25) / (50 - 20)
    
    # stoch >= 0 => interpolate(stoch, 20, 0, 25, 0)
    mask = (stoch_rsi >= 0) & (stoch_rsi <= 20)
    normalized[mask] = 0 + (stoch_rsi[mask] - 0) * (25 - 0) / (20 - 0)
    
    return normalized

def calculate_cci(df: pd.DataFrame, length: int = 20, source_type: str = 'hlc3') -> pd.Series:
    """Calculate CCI and normalize to 0-100 scale"""
    # Determine source data
    if source_type == 'hlc3':
        source = (df['high'] + df['low'] + df['close']) / 3
    else:
        source = df[source_type]
    
    cci = df.ta.cci(high='high', low='low', close='close', length=length)
    
    # Normalize CCI values
    normalized = pd.Series(index=cci.index, dtype=float)
    
    # cci > 100 => cci > 300 ? 100 : interpolate(cci, 300, 100, 100, 75)
    mask = cci > 300
    normalized[mask] = 100
    
    mask = (cci > 100) & (cci <= 300)
    normalized[mask] = 75 + (cci[mask] - 100) * (100 - 75) / (300 - 100)
    
    # cci >= 0 => interpolate(cci, 100, 0, 75, 50)
    mask = (cci >= 0) & (cci <= 100)
    normalized[mask] = 50 + (cci[mask] - 0) * (75 - 50) / (100 - 0)
    
    # cci < -100 => cci < -300 ? 0 : interpolate(cci, -100, -300, 25, 0)
    mask = cci < -300
    normalized[mask] = 0
    
    mask = (cci < -100) & (cci >= -300)
    normalized[mask] = 0 + (cci[mask] - (-300)) * (25 - 0) / (-100 - (-300))
    
    # cci < 0 => interpolate(cci, 0, -100, 50, 25)
    mask = (cci < 0) & (cci >= -100)
    normalized[mask] = 25 + (cci[mask] - (-100)) * (50 - 25) / (0 - (-100))
    
    return normalized

def calculate_bull_bear_power(df: pd.DataFrame, length: int = 13) -> pd.Series:
    """Calculate Bull Bear Power and normalize to 0-100 scale"""
    # Calculate EMA
    ema = df.ta.ema(length=length)
    
    # Calculate Bull Bear Power
    bbp = df['high'] + df['low'] - 2 * ema
    
    # Calculate Bollinger Bands on BBP for normalization
    std = bbp.rolling(window=100).std()
    upper = bbp.rolling(window=100).mean() + 2 * std
    lower = bbp.rolling(window=100).mean() - 2 * std
    
    # Normalize BBP values
    normalized = pd.Series(index=bbp.index, dtype=float)
    
    # bbp > upper => bbp > 1.5 * upper ? 100 : interpolate(bbp, 1.5 * upper, upper, 100, 75)
    mask = bbp > 1.5 * upper
    normalized[mask] = 100
    
    mask = (bbp > upper) & (bbp <= 1.5 * upper)
    normalized[mask] = 75 + (bbp[mask] - upper[mask]) * (100 - 75) / (1.5 * upper[mask] - upper[mask])
    
    # bbp > 0 => interpolate(bbp, upper, 0, 75, 50)
    mask = (bbp > 0) & (bbp <= upper)
    normalized[mask] = 50 + (bbp[mask] - 0) * (75 - 50) / (upper[mask] - 0)
    
    # bbp < lower => bbp < 1.5 * lower ? 0 : interpolate(bbp, lower, 1.5 * lower, 25, 0)
    mask = bbp < 1.5 * lower
    normalized[mask] = 0
    
    mask = (bbp < lower) & (bbp >= 1.5 * lower)
    normalized[mask] = 0 + (bbp[mask] - 1.5 * lower[mask]) * (25 - 0) / (lower[mask] - 1.5 * lower[mask])
    
    # bbp < 0 => interpolate(bbp, 0, lower, 50, 25)
    mask = (bbp < 0) & (bbp >= lower)
    normalized[mask] = 25 + (bbp[mask] - lower[mask]) * (50 - 25) / (0 - lower[mask])
    
    return normalized

def calculate_moving_average(df: pd.DataFrame, length: int = 20, ma_type: str = 'SMA', 
                             smooth: int = 3) -> pd.Series:
    """Calculate Moving Average indicator and normalize to 0-100 scale"""
    # Calculate the specified moving average
    if ma_type == 'SMA':
        ma = df.ta.sma(length=length)
    elif ma_type == 'EMA':
        ma = df.ta.ema(length=length)
    elif ma_type == 'WMA':
        ma = df.ta.wma(length=length)
    elif ma_type == 'HMA':
        ma = df.ta.hma(length=length)
    elif ma_type == 'VWMA':
        ma = df.ta.vwma(length=length)
    else:  # Default to SMA
        ma = df.ta.sma(length=length)
    
    # Calculate buy/sell signals
    buy = df['close'] > ma
    sell = df['close'] < ma
    
    # Normalize using the custom normalization function
    return normalize_trend_indicator(buy, sell, smooth)

def normalize_trend_indicator(buy: pd.Series, sell: pd.Series, smooth: int = 3) -> pd.Series:
    """Normalize trend indicators to 0-100 scale using the algorithm from Pine Script"""
    # Initialize output series
    normalized = pd.Series(index=buy.index, dtype=float)
    
    # Initialize state variables
    os = 0
    max_val = None
    min_val = None
    
    # Process each candle
    for i in range(len(buy)):
        # Determine order state (1 for buy, -1 for sell, previous state if neither)
        if buy.iloc[i]:
            new_os = 1
        elif sell.iloc[i]:
            new_os = -1
        else:
            new_os = os
        
        # Update max/min values based on order state changes
        close = buy.iloc[i].name if hasattr(buy.iloc[i], 'name') else i  # Index for tracking
        
        if new_os > os:  # Going from sell to buy or neutral to buy
            max_val = close
            if min_val is None:
                min_val = close
        elif new_os < os:  # Going from buy to sell or neutral to sell
            min_val = close
            if max_val is None:
                max_val = close
        else:  # No change in order state
            if new_os > 0:  # In buy state
                max_val = max(close, max_val) if max_val is not None else close
            elif new_os < 0:  # In sell state
                min_val = min(close, min_val) if min_val is not None else close
        
        # Update order state
        os = new_os
        
        # Calculate normalized value
        if max_val is not None and min_val is not None and max_val != min_val:
            normalized.iloc[i] = (close - min_val) / (max_val - min_val) * 100
        else:
            normalized.iloc[i] = 50  # Default value if we can't normalize yet
    
    # Apply smoothing
    return normalized.rolling(window=smooth).mean()

def calculate_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0, 
                              ma_type: str = 'SMA', source_col: str = 'close', 
                              smooth: int = 3) -> pd.Series:
    """Calculate Bollinger Bands indicator and normalize to 0-100 scale"""
    # Calculate Bollinger Bands
    bbands = df.ta.bbands(close=source_col, length=length, std=std_dev, mamode=ma_type)
    
    # Get upper and lower bands
    upper_band = bbands[f'BBU_{length}_{std_dev}']
    lower_band = bbands[f'BBL_{length}_{std_dev}']
    
    # Calculate buy/sell signals
    buy = df[source_col] > upper_band
    sell = df[source_col] < lower_band
    
    # Normalize using the custom normalization function
    return normalize_trend_indicator(buy, sell, smooth)

def calculate_supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0, 
                          smooth: int = 3) -> pd.Series:
    """
    Calculate Supertrend indicator and normalize to 0-100 scale
    Implementation adapted from the Pine Script
    """
    # Calculate ATR
    atr = df.ta.atr(length=length)
    
    # Calculate basic upper and lower bands
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize supertrend series
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    # Calculate Supertrend
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
            continue
            
        # Adjust bands based on previous values and direction
        if direction.iloc[i-1] == 1:
            # Previous trend was up
            if df['close'].iloc[i] <= supertrend.iloc[i-1]:
                # Price closed below supertrend - change to down trend
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                # Continue up trend
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = 1
        else:
            # Previous trend was down
            if df['close'].iloc[i] >= supertrend.iloc[i-1]:
                # Price closed above supertrend - change to up trend
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                # Continue down trend
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = -1
    
    # Calculate buy/sell signals
    buy = df['close'] > supertrend
    sell = df['close'] < supertrend
    
    # Normalize using the custom normalization function
    return normalize_trend_indicator(buy, sell, smooth)

def calculate_linear_regression(df: pd.DataFrame, length: int = 25, source_col: str = 'close') -> pd.Series:
    """Calculate Linear Regression indicator normalized to 0-100 scale"""
    # Initialize result series
    lr_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate correlation for each window
    for i in range(length, len(df)):
        window = df[source_col].iloc[i-length+1:i+1]
        indices = np.arange(length)
        correlation = np.corrcoef(indices, window.values)[0, 1]
        lr_values.iloc[i] = 50 * correlation + 50
    
    return lr_values

def calculate_market_structure(df: pd.DataFrame, length: int = 5, smooth: int = 3) -> pd.Series:
    """
    Calculate Market Structure indicator and normalize to 0-100 scale
    Implementation adapted from the Pine Script
    """
    ph_values = []  # Store pivot high values
    pl_values = []  # Store pivot low values
    
    # Calculate pivot points
    for i in range(length, len(df) - length):
        # Check for pivot high
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, length+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, length+1)):
            ph_values.append((i, df['high'].iloc[i]))
        
        # Check for pivot low
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, length+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, length+1)):
            pl_values.append((i, df['low'].iloc[i]))
    
    # Initialize result series
    bull_signals = pd.Series(False, index=df.index)
    bear_signals = pd.Series(False, index=df.index)
    
    # Track last pivot values and state
    last_ph = None
    last_pl = None
    ph_crossed = False
    pl_crossed = False
    
    # Process each candle
    for i in range(len(df)):
        # Update pivot highs
        for ph_idx, ph_val in ph_values:
            if i == ph_idx + length:  # Confirm pivot at right time
                last_ph = ph_val
                ph_crossed = False
        
        # Update pivot lows
        for pl_idx, pl_val in pl_values:
            if i == pl_idx + length:  # Confirm pivot at right time
                last_pl = pl_val
                pl_crossed = False
        
        # Check for crosses
        if last_ph is not None and df['close'].iloc[i] > last_ph and not ph_crossed:
            bull_signals.iloc[i] = True
            ph_crossed = True
        
        if last_pl is not None and df['close'].iloc[i] < last_pl and not pl_crossed:
            bear_signals.iloc[i] = True
            pl_crossed = True
    
    # Normalize using the custom normalization function
    return normalize_trend_indicator(bull_signals, bear_signals, smooth)

def calculate_vwap(df: pd.DataFrame, smooth: int = 3) -> pd.Series:
    """
    Calculate VWAP indicator and normalize to 0-100 scale
    Using a simple daily VWAP implementation
    """
    # Calculate VWAP
    df['vwap'] = df.ta.vwap()
    
    # Calculate standard deviations
    vwap_std = df['close'].rolling(window=20).std()
    upper_band = df['vwap'] + 2 * vwap_std
    lower_band = df['vwap'] - 2 * vwap_std
    
    # Calculate buy/sell signals
    buy = df['close'] > upper_band
    sell = df['close'] < lower_band
    
    # Normalize using the custom normalization function
    return normalize_trend_indicator(buy, sell, smooth)

# Add this function to your code to ensure values stay within 0-100 range
def bound_sentiment(value):
    """Ensure sentiment values stay within 0-100 range"""
    # Check if value is a Series and handle accordingly
    if isinstance(value, pd.Series):
        # Apply bounds to each value in the series
        return value.apply(lambda x: 50 if pd.isna(x) else max(0, min(100, x)))
    else:
        # Handle single value
        if pd.isna(value):
            return 50  # Default to neutral for NaN values
        return max(0, min(100, value))  # Clamp between 0 and 100

# ... existing code ...
def interpolate(value, value_high, value_low, range_high, range_low):
    """Exact implementation of Pine's interpolate function"""
    return range_low + (value - value_low) * (range_high - range_low) / (value_high - value_low)


def calculate_market_sentiment(df: pd.DataFrame, 
                              rsi_length: int = 14,
                              stoch_k_length: int = 14, stoch_k_smooth: int = 3,
                              stoch_rsi_length: int = 14, stoch_rsi_smooth: int = 3,
                              cci_length: int = 20,
                              bbp_length: int = 13,
                              ma_length: int = 20, ma_type: str = 'SMA',
                              bb_length: int = 20, bb_mult: float = 2.0, bb_type: str = 'SMA',
                              st_length: int = 10, st_factor: float = 3.0,
                              lr_length: int = 25,
                              ms_length: int = 5,
                              norm_smooth: int = 3) -> pd.Series:
    """
    Calculate overall market sentiment by averaging multiple indicators
    Returns a value between 0-100 where:
    - 0-20: Strong Bearish
    - 20-40: Bearish
    - 40-60: Neutral
    - 60-80: Bullish
    - 80-100: Strong Bullish
    """
    # Calculate individual indicators and ensure they're bounded
    df = df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    # 1. RSI (14)
    rsi = df.ta.rsi(length=14)
    df['rsi_score'] = rsi.apply(lambda x: 
        interpolate(x, 100, 70, 100, 75) if x > 70 else
        interpolate(x, 70, 50, 75, 50) if x > 50 else
        interpolate(x, 50, 30, 50, 25) if x > 30 else
        interpolate(x, 30, 0, 25, 0) if x >= 0 else 0
    )

    # 2. Stochastic (14, 3)
    stoch = df.ta.stoch(k=14, d=3)
    df['stoch_score'] = stoch['STOCHk_14_3_3'].apply(lambda x: 
        interpolate(x, 100, 80, 100, 75) if x > 80 else
        interpolate(x, 80, 50, 75, 50) if x > 50 else
        interpolate(x, 50, 20, 50, 25) if x > 20 else
        interpolate(x, 20, 0, 25, 0) if x >= 0 else 0
    )

    # 3. Stochastic RSI (14, 3)
    stoch_rsi = df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3)
    df['stoch_rsi_score'] = stoch_rsi['STOCHRSIk_14_14_3_3'].apply(lambda x: 
        interpolate(x, 100, 80, 100, 75) if x > 80 else
        interpolate(x, 80, 50, 75, 50) if x > 50 else
        interpolate(x, 50, 20, 50, 25) if x > 20 else
        interpolate(x, 20, 0, 25, 0) if x >= 0 else 0
    )

    # 4. CCI (20)
    cci = df.ta.cci(length=20)
    df['cci_score'] = cci.apply(lambda x: 
        100 if x > 300 else
        interpolate(x, 300, 100, 100, 75) if x > 100 else
        interpolate(x, 100, 0, 75, 50) if x >= 0 else
        interpolate(x, 0, -100, 50, 25) if x >= -100 else
        interpolate(x, -100, -300, 25, 0) if x >= -300 else 0
    )

    # 5. Bull Bear Power (13)
    ema = df.ta.ema(length=13)
    bbp = df['high'] + df['low'] - 2 * ema
    df['bbp_score'] = bbp.apply(lambda x: 
        interpolate(x, 100, 0, 75, 50) if x > 0 else
        interpolate(x, 0, -100, 50, 25)
    )

    # 6. Moving Average (20)
    sma = df.ta.sma(length=20)
    df['ma_score'] = ((df['close'] - sma) / sma * 100).clip(0, 100)

    # 7. Bollinger Bands (20, 2)
    bb = df.ta.bbands(length=20, std=2)
    bb_width = bb['BBU_20_2.0'] - bb['BBL_20_2.0']
    df['bb_score'] = ((df['close'] - bb['BBL_20_2.0']) / bb_width * 100).clip(0, 100)

    # Calculate final sentiment (average of all components)
    df['sentiment'] = (
        df['rsi_score'] + 
        df['stoch_score'] + 
        df['stoch_rsi_score'] + 
        df['cci_score'] + 
        df['bbp_score'] + 
        df['ma_score'] + 
        df['bb_score']
    ) / 7

    # Create output DataFrame
    result = pd.DataFrame({
        'timestamp': df.index if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'],
        'close': df['close'],
        'sentiment': df['sentiment']
    })

    # Remove NaN values
    result = result.dropna(subset=['sentiment'])
    
    return result

def get_sentiment_category(sentiment_value: float) -> str:
    """Convert numeric sentiment to categorical value"""
    if sentiment_value >= 80:
        return "Strong Bullish"
    elif sentiment_value >= 60:
        return "Bullish"
    elif sentiment_value >= 40:
        return "Neutral"
    elif sentiment_value >= 20:
        return "Bearish"
    else:
        return "Strong Bearish"

def analyze_btcusdt(file_path: str = 'btcusdt.csv') -> pd.Series:
    """
    Analyze BTCUSDT data and return market sentiment for each candle
    Returns a pandas Series with sentiment values (0-100)
    """
    # Load and prepare data
    df = load_and_prepare_data(file_path)
    
    # Calculate market sentiment
    sentiment = calculate_market_sentiment(df)
    
    return sentiment

def plot_sentiment_last_month(sentiment_df: pd.DataFrame, output_path: str = 'c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btc_sentiment_last_month.png'):
    """
    Plot the closing price and sentiment for the last month
    """
    # Get the last month of data
    last_month = sentiment_df.iloc[-30:] if len(sentiment_df) >= 30 else sentiment_df
    
    # Create the plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price
    ax1.plot(last_month['timestamp'], last_month['close'], color='white', linewidth=1.5)
    ax1.set_title('BTC-USD Price (Last Month)', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    
    # Plot sentiment
    ax2.plot(last_month['timestamp'], last_month['sentiment'], linewidth=2)
    ax2.set_title('Market Sentiment Oscillator', fontsize=14)
    ax2.set_ylabel('Sentiment (0-100)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    
    # Add horizontal reference lines
    ax2.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Extreme Bullish')
    ax2.axhline(y=60, color='green', linestyle=':', alpha=0.3, label='Bullish')
    ax2.axhline(y=50, color='white', linestyle='-', alpha=0.3, label='Neutral')
    ax2.axhline(y=40, color='red', linestyle=':', alpha=0.3, label='Bearish')
    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Extreme Bearish')
    
    # Color the sentiment line based on value
    for i in range(len(last_month)-1):
        color = plt.cm.RdYlGn(last_month['sentiment'].iloc[i]/100)
        ax2.plot(last_month['timestamp'].iloc[i:i+2], last_month['sentiment'].iloc[i:i+2], 
                color=color)
    
    # Add legend
    ax2.legend(loc='upper left')
    
    # Format x-axis dates
    fig.autofmt_xdate()
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(date_format)
    
    # Add current sentiment value
    latest_sentiment = last_month['sentiment'].iloc[-1]
    sentiment_category = get_sentiment_category(latest_sentiment)
    
    sentiment_color = 'green' if latest_sentiment > 60 else 'red' if latest_sentiment < 40 else 'white'
    
    ax2.text(0.98, 0.95, f"Current: {latest_sentiment:.1f} - {sentiment_category}", 
            transform=ax2.transAxes, fontsize=12, color=sentiment_color,
            horizontalalignment='right', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path)
    
    return fig

if __name__ == "__main__":
    # Load data and calculate sentiment
    sentiment = analyze_btcusdt('c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btcusdt.csv')
    
    # Plot the last month of sentiment
    fig = plot_sentiment_last_month(sentiment)
    
    # Display the plot
    plt.show()
    
    # Save results to CSV
    sentiment.to_csv('c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btcusdt_sentiment.csv')
    
    # Display statistics
    print("\nSentiment Statistics:")
    print(sentiment['sentiment'].describe())
    
    # Display categorical breakdown
    categories = sentiment['sentiment'].apply(get_sentiment_category)
    category_counts = categories.value_counts()
    print("\nSentiment Categories Distribution:")
    print(category_counts)
    
    print(f"\nChart saved to c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btc_sentiment_last_month.png")
    print(f"Data exported to c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btcusdt_sentiment.csv")