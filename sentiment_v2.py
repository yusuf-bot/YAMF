import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Define colors similar to TradingView
BULL_COLOR = '#085def'  # Blue
BEAR_COLOR = '#f10707'  # Red
NEUTRAL_COLOR = '#787b86'  # Gray

# Create a custom colormap for the sentiment gradient
colors = [(BEAR_COLOR, BEAR_COLOR), (NEUTRAL_COLOR, NEUTRAL_COLOR), (BULL_COLOR, BULL_COLOR)]
sentiment_cmap = LinearSegmentedColormap.from_list('sentiment', colors, N=100)

# Function to interpolate values (same as in Pine script)
def interpolate(value, value_high, value_low, range_high, range_low):
    return range_low + (value - value_low) * (range_high - range_low) / (value_high - value_low)

# RSI interpretation function (same as in Pine script)
def interpret_rsi(rsi_value):
    if rsi_value > 70:
        return interpolate(rsi_value, 100, 70, 100, 75)
    elif rsi_value > 50:
        return interpolate(rsi_value, 70, 50, 75, 50)
    elif rsi_value > 30:
        return interpolate(rsi_value, 50, 30, 50, 25)
    else:
        return interpolate(rsi_value, 30, 0, 25, 0)

# Stochastic interpretation function
def interpret_stoch(stoch_value):
    if stoch_value > 80:
        return interpolate(stoch_value, 100, 80, 100, 75)
    elif stoch_value > 50:
        return interpolate(stoch_value, 80, 50, 75, 50)
    elif stoch_value > 20:
        return interpolate(stoch_value, 50, 20, 50, 25)
    else:
        return interpolate(stoch_value, 20, 0, 25, 0)

# CCI interpretation function
def interpret_cci(cci_value):
    if cci_value > 100:
        return min(100, interpolate(cci_value, 300, 100, 100, 75))
    elif cci_value >= 0:
        return interpolate(cci_value, 100, 0, 75, 50)
    elif cci_value < -100:
        return max(0, interpolate(cci_value, -100, -300, 25, 0))
    else:
        return interpolate(cci_value, 0, -100, 50, 25)

# Bull Bear Power interpretation
def interpret_bbp(bbp_value, upper, lower):
    if bbp_value > upper:
        return min(100, interpolate(bbp_value, 1.5 * upper, upper, 100, 75))
    elif bbp_value > 0:
        return interpolate(bbp_value, upper, 0, 75, 50)
    elif bbp_value < lower:
        return max(0, interpolate(bbp_value, lower, 1.5 * lower, 25, 0))
    else:
        return interpolate(bbp_value, 0, lower, 50, 25)

# Moving Average interpretation
def interpret_ma(close, ma):
    return 100 if close > ma else 0

# Bollinger Bands interpretation
def interpret_bb(close, upper, lower):
    if close > upper:
        return 100
    elif close < lower:
        return 0
    else:
        return 50

# Supertrend interpretation
def calculate_supertrend(df, atr_period=10, multiplier=3):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate ATR
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    for i in range(1, len(df)):
        if close.iloc[i] > upperband.iloc[i-1]:
            supertrend.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1 and lowerband.iloc[i] < supertrend.iloc[i]:
                supertrend.iloc[i] = lowerband.iloc[i]
            elif direction.iloc[i] == -1 and upperband.iloc[i] > supertrend.iloc[i]:
                supertrend.iloc[i] = upperband.iloc[i]
    
    return supertrend, direction

# Market Structure interpretation
def calculate_market_structure(df, length=5):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Initialize arrays
    ph = np.zeros(len(df))
    pl = np.zeros(len(df))
    ph_y = np.zeros(len(df))
    pl_y = np.zeros(len(df))
    ph_cross = np.zeros(len(df), dtype=bool)
    pl_cross = np.zeros(len(df), dtype=bool)
    bull = np.zeros(len(df), dtype=bool)
    bear = np.zeros(len(df), dtype=bool)
    
    # Find pivot highs and lows
    for i in range(length, len(df)-length):
        if all(high.iloc[i] > high.iloc[i-length:i]) and all(high.iloc[i] > high.iloc[i+1:i+length+1]):
            ph[i] = high.iloc[i]
        if all(low.iloc[i] < low.iloc[i-length:i]) and all(low.iloc[i] < low.iloc[i+1:i+length+1]):
            pl[i] = low.iloc[i]
    
    # Process market structure
    for i in range(1, len(df)):
        # Update pivot high/low values
        if ph[i-1] > 0:
            ph_y[i] = ph[i-1]
        else:
            ph_y[i] = ph_y[i-1]
            
        if pl[i-1] > 0:
            pl_y[i] = pl[i-1]
        else:
            pl_y[i] = pl_y[i-1]
            
        # Check for crosses
        if close.iloc[i] > ph_y[i] and not ph_cross[i-1] and ph_y[i] > 0:
            ph_cross[i] = True
            bull[i] = True
        else:
            ph_cross[i] = ph_cross[i-1]
            
        if close.iloc[i] < pl_y[i] and not pl_cross[i-1] and pl_y[i] > 0:
            pl_cross[i] = True
            bear[i] = True
        else:
            pl_cross[i] = pl_cross[i-1]
    
    # Convert to normalized score (0-100)
    ms_score = pd.Series(50.0, index=df.index)
    
    for i in range(len(df)):
        if bull[i]:
            ms_score.iloc[i] = 100
        elif bear[i]:
            ms_score.iloc[i] = 0
        elif i > 0:
            ms_score.iloc[i] = ms_score.iloc[i-1]
    
    return ms_score

# Linear Regression interpretation
def calculate_linear_regression(series, length=25):
    result = pd.Series(index=series.index)
    x = np.arange(length)
    
    for i in range(length-1, len(series)):
        y = series.iloc[i-length+1:i+1].values
        slope, intercept = np.polyfit(x, y, 1)
        correlation = np.corrcoef(x, y)[0, 1]
        result.iloc[i] = 50 * correlation + 50
        
    return result

# Download data for BTC-USD
print("Downloading BTC-USD data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
data = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")

if data.empty:
    print("Failed to download data. Please check your internet connection.")
    exit()

print(f"Downloaded {len(data)} days of data")

# Calculate all indicators
print("Calculating indicators...")

# ... existing code ...

# RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
rsi_score = pd.Series(index=data.index)
for i in range(len(rsi)):
    if not pd.isna(rsi.iloc[i]).any():  # Changed from pd.notna to not pd.isna
        try:
            rsi_score.iloc[i] = interpret_rsi(float(rsi.iloc[i]))  # Convert to float
        except:
            rsi_score.iloc[i] = 50  # Default value if calculation fails

# Stochastic
k_period = 14
d_period = 3
stoch_k = 100 * ((data['Close'] - data['Low'].rolling(k_period).min()) / 
                (data['High'].rolling(k_period).max() - data['Low'].rolling(k_period).min()))
stoch_d = stoch_k.rolling(d_period).mean()
stoch_score = pd.Series(index=data.index)
for i in range(len(stoch_k)):
    if not pd.isna(stoch_k.iloc[i]).any():  # Changed from pd.notna to not pd.isna
        try:
            stoch_score.iloc[i] = interpret_stoch(float(stoch_k.iloc[i]))  # Convert to float
        except:
            stoch_score.iloc[i] = 50

# Stochastic RSI
stoch_rsi = 100 * ((rsi - rsi.rolling(14).min()) / 
                  (rsi.rolling(14).max() - rsi.rolling(14).min()))
stoch_rsi = stoch_rsi.rolling(3).mean()
stoch_rsi_score = pd.Series(index=data.index)
for i in range(len(stoch_rsi)):
    if not pd.isna(stoch_rsi.iloc[i]).any():  # Changed from pd.notna to not pd.isna
        try:
            stoch_rsi_score.iloc[i] = interpret_stoch(float(stoch_rsi.iloc[i]))  # Convert to float
        except:
            stoch_rsi_score.iloc[i] = 50

# CCI
typical_price = (data['High'] + data['Low'] + data['Close']) / 3
ma_tp = typical_price.rolling(window=20).mean()
mean_deviation = abs(typical_price - ma_tp).rolling(window=20).mean()
cci = (typical_price - ma_tp) / (0.015 * mean_deviation)
cci_score = pd.Series(index=data.index)
for i in range(len(cci)):
    if not pd.isna(cci.iloc[i]).any():  # Changed from pd.notna to not pd.isna
        try:
            cci_score.iloc[i] = interpret_cci(float(cci.iloc[i]))  # Convert to float
        except:
            cci_score.iloc[i] = 50

# ... rest of the code remains the same ...

# Bull Bear Power
ema13 = data['Close'].ewm(span=13, adjust=False).mean()
bbp = data['High'] + data['Low'] - 2 * ema13
bbp_std = bbp.rolling(100).std()
bbp_upper = bbp.rolling(100).mean() + 2 * bbp_std
bbp_lower = bbp.rolling(100).mean() - 2 * bbp_std
bbp_score = pd.Series(index=data.index)
for i in range(len(data)):
    bbp_score.iloc[i] = interpret_bbp(bbp.iloc[i], bbp_upper.iloc[i], bbp_lower.iloc[i])
    
# Moving Average
sma20 = data['Close'].rolling(window=20).mean()
ma_score = pd.Series(index=data.index)
for i in range(len(data)):
    ma_score.iloc[i] = interpret_ma(data['Close'].iloc[i], sma20.iloc[i])

# Bollinger Bands
bb_period = 20
bb_std = 2
bb_middle = data['Close'].rolling(window=bb_period).mean()
bb_std_dev = data['Close'].rolling(window=bb_period).std()
bb_upper = bb_middle + bb_std * bb_std_dev
bb_lower = bb_middle - bb_std * bb_std_dev
bb_score = pd.Series(index=data.index)
for i in range(len(data)):
    bb_score.iloc[i] = interpret_bb(data['Close'].iloc[i], bb_upper.iloc[i], bb_lower.iloc[i])

# Supertrend
supertrend, direction = calculate_supertrend(data, 10, 3)
st_score = pd.Series(index=data.index)
for i in range(len(data)):
    st_score.iloc[i] = 100 if direction.iloc[i] > 0 else 0

# Linear Regression
reg_score = calculate_linear_regression(data['Close'], 25)

# Market Structure
ms_score = calculate_market_structure(data, 5)

# Calculate final sentiment
indicators = [
    rsi_score, 
    stoch_score, 
    stoch_rsi_score, 
    cci_score, 
    bbp_score, 
    ma_score, 
    bb_score, 
    st_score, 
    reg_score, 
    ms_score
]

# Create a DataFrame with all indicators
sentiment_df = pd.DataFrame(index=data.index)
sentiment_df['RSI'] = rsi_score
sentiment_df['Stochastic'] = stoch_score
sentiment_df['StochRSI'] = stoch_rsi_score
sentiment_df['CCI'] = cci_score
sentiment_df['BBP'] = bbp_score
sentiment_df['MA'] = ma_score
sentiment_df['BB'] = bb_score
sentiment_df['Supertrend'] = st_score
sentiment_df['Regression'] = reg_score
sentiment_df['MarketStructure'] = ms_score

# Calculate average sentiment
sentiment_df['Sentiment'] = sentiment_df.mean(axis=1)

# Plot the results
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# Plot price
axs[0].plot(data.index, data['Close'], color='white', linewidth=1.5)
axs[0].set_title('BTC-USD Price', fontsize=14)
axs[0].grid(alpha=0.3)
axs[0].set_ylabel('Price (USD)', fontsize=12)

# Plot sentiment
axs[1].plot(sentiment_df.index, sentiment_df['Sentiment'], linewidth=2)
axs[1].set_title('Market Sentiment Oscillator', fontsize=14)
axs[1].set_ylabel('Sentiment (0-100)', fontsize=12)
axs[1].set_ylim(0, 100)
axs[1].grid(alpha=0.3)

# Add horizontal reference lines
axs[1].axhline(y=80, color=BULL_COLOR, linestyle='--', alpha=0.5, label='Extreme Bullish')
axs[1].axhline(y=60, color=BULL_COLOR, linestyle=':', alpha=0.3, label='Bullish')
axs[1].axhline(y=50, color=NEUTRAL_COLOR, linestyle='-', alpha=0.3, label='Neutral')
axs[1].axhline(y=40, color=BEAR_COLOR, linestyle=':', alpha=0.3, label='Bearish')
axs[1].axhline(y=20, color=BEAR_COLOR, linestyle='--', alpha=0.5, label='Extreme Bearish')

# Color the sentiment line based on value
for i in range(len(sentiment_df)-1):
    color = plt.cm.RdYlGn(sentiment_df['Sentiment'].iloc[i]/100)
    axs[1].plot(sentiment_df.index[i:i+2], sentiment_df['Sentiment'].iloc[i:i+2], 
                color=plt.cm.RdYlGn(sentiment_df['Sentiment'].iloc[i]/100))
    
    # Add background coloring
    if sentiment_df['Sentiment'].iloc[i] > 80:
        axs[1].axvspan(sentiment_df.index[i], sentiment_df.index[i+1], alpha=0.1, color=BULL_COLOR)
    elif sentiment_df['Sentiment'].iloc[i] > 60:
        axs[1].axvspan(sentiment_df.index[i], sentiment_df.index[i+1], alpha=0.05, color=BULL_COLOR)
    elif sentiment_df['Sentiment'].iloc[i] < 20:
        axs[1].axvspan(sentiment_df.index[i], sentiment_df.index[i+1], alpha=0.1, color=BEAR_COLOR)
    elif sentiment_df['Sentiment'].iloc[i] < 40:
        axs[1].axvspan(sentiment_df.index[i], sentiment_df.index[i+1], alpha=0.05, color=BEAR_COLOR)

# Add legend
axs[1].legend(loc='upper left')

# Format x-axis dates
fig.autofmt_xdate()
date_format = mdates.DateFormatter('%Y-%m-%d')
axs[1].xaxis.set_major_formatter(date_format)

# Add indicator values table
latest_date = sentiment_df.index[-1]
latest_values = sentiment_df.iloc[-1]

# Create a table with indicator values
indicator_names = ['RSI', 'Stochastic', 'StochRSI', 'CCI', 'BBP', 'MA', 'BB', 'Supertrend', 'Regression', 'Market Structure', 'Overall Sentiment']
indicator_values = [
    f"{rsi.iloc[-1]:.1f} → {latest_values['RSI']:.1f}",
    f"{stoch_k.iloc[-1]:.1f} → {latest_values['Stochastic']:.1f}",
    f"{stoch_rsi.iloc[-1]:.1f} → {latest_values['StochRSI']:.1f}",
    f"{cci.iloc[-1]:.1f} → {latest_values['CCI']:.1f}",
    f"{bbp.iloc[-1]:.1f} → {latest_values['BBP']:.1f}",
    f"{'Above' if data['Close'].iloc[-1] > sma20.iloc[-1] else 'Below'} → {latest_values['MA']:.1f}",
    f"{'Above' if data['Close'].iloc[-1] > bb_upper.iloc[-1] else 'Below' if data['Close'].iloc[-1] < bb_lower.iloc[-1] else 'Inside'} → {latest_values['BB']:.1f}",
    f"{'Bullish' if direction.iloc[-1] > 0 else 'Bearish'} → {latest_values['Supertrend']:.1f}",
    f"{latest_values['Regression']:.1f}",
    f"{'Bullish' if ms_score.iloc[-1] > 50 else 'Bearish'} → {latest_values['MarketStructure']:.1f}",
    f"{latest_values['Sentiment']:.1f}"
]

# Add text box with indicator values
textstr = '\n'.join([f'{name}: {value}' for name, value in zip(indicator_names, indicator_values)])
props = dict(boxstyle='round', facecolor='black', alpha=0.7)
axs[0].text(0.02, 0.05, textstr, transform=axs[0].transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

# Add sentiment category text
latest_sentiment = latest_values['Sentiment']
sentiment_category = "Extreme Bullish" if latest_sentiment > 80 else \
                    "Bullish" if latest_sentiment > 60 else \
                    "Neutral" if latest_sentiment > 40 else \
                    "Bearish" if latest_sentiment > 20 else "Extreme Bearish"

sentiment_color = BULL_COLOR if latest_sentiment > 60 else \
                 BEAR_COLOR if latest_sentiment < 40 else \
                 NEUTRAL_COLOR

axs[1].text(0.98, 0.95, f"Current: {latest_sentiment:.1f} - {sentiment_category}", 
            transform=axs[1].transAxes, fontsize=12, color=sentiment_color,
            horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.savefig('c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btc_sentiment_oscillator.png')
plt.show()

print(f"Current BTC-USD Sentiment: {latest_sentiment:.2f} - {sentiment_category}")
print(f"Chart saved to c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btc_sentiment_oscillator.png")

# Export data to CSV
sentiment_df.to_csv('c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btc_sentiment_data.csv')
print(f"Data exported to c:\\Users\\ASAA\\Desktop\\projects\\vibe\\btc_sentiment_data.csv")