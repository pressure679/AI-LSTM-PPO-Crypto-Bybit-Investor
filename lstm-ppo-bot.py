import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO
import random
from collections import deque
from datetime import datetime, timedelta
import requests
import threading
from multiprocessing import Process
import time
from time import timezone
from decimal import Decimal
from pybit.unified_trading import HTTP
from pybit.unified_trading import WebSocket
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
# from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import glob
import shutil

ready_event = threading.Event()
process_counter = 0
pd.set_option('future.no_silent_downcasting', True)

# import warnings
# import yfinance as yf

# warnings.filterwarnings("ignore", category=RuntimeWarning)
# Popular crypto: "DOGEUSDT", "HYPEUSDT", "FARTCOINUSDT", "SUIUSDT", "INITUSDT", "BABYUSDT", "NILUSDT"
# yf_symbols = ["BTC-USD", "BNB-USD", "ETH-USD", "XRP-USD", "XAUUSD=X"]
# bybit_symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT", "XAUTUSDT"]
# symbols = ["BTCUSD", "BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]
yf_symbols = ["BTC-USD", "BNB-USD", "ETH-USD", "XRP-USD"]
bybit_symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT", "XAUTUSDT"]
symbols = ["BTCUSD", "BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]
# yf_symbols = ["BNB-USD", "ETH-USD", "XRP-USD", "XAUUSD=X"]
# bybit_symbols = ["BNBUSDT", "ETHUSDT", "XRPUSDT", "XAUTUSDT"]
# symbols = ["BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]

ACTIONS = ['hold', 'long', 'short', 'close']
# alpha = 0.1
# gamma = 0.95
# epsilon = 0.1

capital = 1000

# Bybit Demo API Key and Secret - 1khTo0Bme2LA3YL4gU - JkDgEjC4O8pIiu9ysMKMiRVITE0Setwjf1I9
# Bybit API Key and Secret - PoP1ud3PuWajwecc4S - z9RXVMWpiOoE3TubtAQ0UtGx8I5SOiRp1KPU
# Bybit SubAccount API Key and Secret - UwZ6Br6QwinYJeDcf6 - Nai4QZfKVOU9756IsRpbm1d7gh70RAEwFd4K
api_key = "PoP1ud3PuWajwecc4S"
api_secret = "z9RXVMWpiOoE3TubtAQ0UtGx8I5SOiRp1KPU"
session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    demo=False,  # or False for mainnet
    recv_window=15000,
    timeout=60
)

# def load_last_mb(filepath, symbol, mb_size=20):
def load_last_mb(symbol, filepath="/mnt/chromeos/removable/sd_card/1m dataframes", mb_size=6*3):
    # Search for a file containing the symbol in its name
    matching_files = [f for f in os.listdir(filepath) if symbol.lower() in f.lower()]
    if not matching_files:
        raise FileNotFoundError(f"No file containing '{symbol}' found in {filepath}")
    
    # Use the first matching file
    fp = os.path.join(filepath, matching_files[0])
    bytes_to_read = mb_size * 1024 * 1024

    with open(fp, "rb") as f:
        f.seek(0, os.SEEK_END)
        start = max(0, f.tell() - bytes_to_read)
        f.seek(start)
        data = f.read().decode("utf-8", errors="ignore")

    lines = data.split("\n")[1:] if start else data.split("\n")
    df = pd.read_csv(StringIO("\n".join([l for l in lines if l.strip()])), header=None)
    
    df.columns = [
        "Open time","Open","High","Low","Close","Volume","Close time",
        "Quote asset vol","Trades","Taker buy base","Taker buy quote","Ignore"
    ]

    # Convert 'Open time' to datetime and set as index
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)

    # Keep only OHLC columns
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    df = df.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    ready_event.set()
    return df

def load_last_mb_xauusd(file_path="/mnt/chromeos/removable/sd_card/1m dataframes/XAUUSD_1m_data.csv", mb=3*2, delimiter=';', col_names=None):
    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)  # start position
    
    with open(file_path, 'rb') as f:
        # Seek to approximately 20 MB before EOF
        f.seek(offset)
        
        # Read to the end of file from that offset
        data = f.read().decode(errors='ignore')
        
        # If not at start of file, discard partial first line (incomplete)
        if offset > 0:
            data = data.split('\n', 1)[-1]
        
    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=None, engine='python')
    
    #if col_names:
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    # Convert columns if needed, e.g.:
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors='coerce')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    df = df.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    df = df.dropna()
    
    return df

def yf_get_ohlc_df(symbol: str, interval: str = "1m", period: str = "7d") -> pd.DataFrame:
    """
    Downloads OHLCV data using yfinance and returns a DataFrame.

    Args:
        symbol (str): The ticker symbol, e.g., "BTC-USD", "ETH-USD", "AAPL"
        interval (str): The interval (e.g., "1m", "5m", "15m", "1h", "1d")
        period (str): How much historical data to pull (e.g., "1d", "5d", "60d", "1y")

    Returns:
        pd.DataFrame: DataFrame with ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=True,
        threads=False
    )

    df = df.dropna()
    # df.columns = [col.lower() for col in df.columns]  # ['open', 'high', 'low', 'close', 'volume']
    # print(f"df.columns: {df.columns}")
    df.columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]
    df = df.rename_axis("Timestamp").reset_index()

    return df

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    """
    candles = list of [open, high, low, close, volume]
    Returns the latest ATR value.
    """
    trs = []
    for i in range(1, len(df)):
        high = df.iloc[i]['High']
        low = df.iloc[i]['Low']
        prev_close = df.iloc[i - 1]['Close']

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

    atr = sum(trs[-period:]) / period
    return atr

# def ATR(df, period=14):
#     trs = []
#     for i in range(1, len(df)):
#         high = df.iloc[i]['High']
#         low = df.iloc[i]['Low']
#         prev_close = df.iloc[i - 1]['Close']
# 
#         tr = max(
#             high - low,
#             abs(high - prev_close),
#             abs(low - prev_close)
#         )
#         trs.append(tr)
# 
#     atr_series = pd.Series(trs)
#     atr = atr_series.rolling(window=period).mean()
#     # atr = pd.Series([np.nan]).append(atr, ignore_index=True)  # To match original df length
#     atr = pd.concat([pd.Series([np.nan]), pd.Series(atr)], ignore_index=True)
#     return atr

def RSI(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def Bollinger_Bands(df, period=20, num_std=2):
    sma = df.rolling(window=period).mean()
    std = df.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def ADX(df, period=14):
    """
    Returns +DI, -DI and ADX using Wilder's smoothing.
    Columns required: High, Low, Close
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- directional movement -----------------------------------------
    # plus_dm  = (high.diff()  > low.diff())  * (high.diff()).clip(lower=0)
    # minus_dm = (low.diff()   > high.diff()) * (low.diff().abs()).clip(lower=0)̈́
    up  =  high.diff()
    dn  = -low.diff()

    plus_dm_array  = np.where((up  >  dn) & (up  > 0),  up,  0.0)
    minus_dm_array = np.where((dn  >  up) & (dn  > 0),  dn,  0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index) # ← wrap
    minus_dm = pd.Series(minus_dm_array, index=df.index) # ← wrap

    # --- true range ----------------------------------------------------
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # --- Wilder smoothing ---------------------------------------------
    atr       = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di  = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di

def Momentum(df, period=14):
    """
    Adds a 'momentum' column to the DataFrame based on Rate of Change (ROC).

    Args:
        df (pd.DataFrame): DataFrame with at least a 'close' column.
        period (int): Number of periods to calculate momentum over.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'momentum' column.
    """
    df = df.copy()
    df['Momentum'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
    return df
def detect_order_blocks(df, lookback=20, min_size=0.002):
    """
    Detect bullish (1) and bearish (-1) order blocks in OHLCV data.
    Once detected, the label persists until the opposite block occurs.

    Parameters:
        df (pd.DataFrame): Must have columns ['open', 'high', 'low', 'close'].
        lookback (int): Candles to look ahead for confirmation.
        min_size (float): Minimum percentage move to consider valid.
        
    Returns:
        pd.DataFrame: Original df with an added 'order_block' column.
    """
    df = df.copy()
    df['order_block'] = 0  # default: no block
    current_block = 0      # persistent state

    for i in range(lookback, len(df) - lookback):
        prev_open = df['Open'].iloc[i - 1]
        prev_close = df['Close'].iloc[i - 1]
        prev_high = df['High'].iloc[i - 1]
        prev_low = df['Low'].iloc[i - 1]

        body_size = abs(prev_close - prev_open) / prev_open

        new_block = 0  # reset detection each loop

        # Bullish order block
        if prev_close < prev_open and body_size >= min_size:
            future_high = df['High'].iloc[i:i + lookback].max()
            if future_high > prev_high * (1 + min_size):
                new_block = 1

        # Bearish order block
        elif prev_close > prev_open and body_size >= min_size:
            future_low = df['Low'].iloc[i:i + lookback].min()
            if future_low < prev_low * (1 - min_size):
                new_block = -1

        # If new block found, switch
        if new_block != 0:
            current_block = new_block

        # Apply current persistent state
        df.at[df.index[i], 'order_block'] = current_block

    # Forward fill any remaining zeros with the last known block
    # df['order_block'].replace(0, method='ffill', inplace=True)
    # df['order_block'].fillna(0, inplace=True)

    df['order_block'] = df['order_block'].replace(0, pd.NA)
    df['order_block'] = df['order_block'].ffill().fillna(0).astype(int)

    return df
def process_orderbook(message):
    """
    Parses the order book update message from Bybit and extracts key features.

    Parameters:
        message (dict): The order book update message from Bybit WebSocket.

    Returns:
        dict: A dictionary of processed order book features.
    """
    bids = message['data']['b']  # format: [['price', 'size'], ...]
    asks = message['data']['a']  # format: [['price', 'size'], ...]

    # Convert top 10 levels to floats and sum their volumes
    top_bid_price = float(bids[0][0])
    top_ask_price = float(asks[0][0])
    spread = top_ask_price - top_bid_price

    bid_volume = sum(float(bid[1]) for bid in bids[:10])
    ask_volume = sum(float(ask[1]) for ask in asks[:10])

    # Market imbalance: +1 = heavy buy pressure, -1 = heavy sell pressure
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)

    # Detect bid wall (adjust threshold as needed)
    wall_flag = 1 if any(float(bid[1]) > 500000 for bid in bids[:10]) else 0

    return {
        'top_bid': top_bid_price,
        'top_ask': top_ask_price,
        'spread': spread,
        'bid_vol': bid_volume,
        'ask_vol': ask_volume,
        'imbalance': imbalance,
        'wall_flag': wall_flag
    }
def update_orderbook(symbol, shared_features, lock):
    def handle_message(message):
        if message["topic"] == f"orderbook.50.{symbol}":
            features = process_orderbook(message)
            with lock:
                shared_features.clear()
                shared_features.update(features)

    ws = WebSocket(
        testnet=False,
        channel_type="linear",
        trace_logging=False,
        # ping_interval=20,  # seconds
        # ping_timeout=10,   # seconds
        # retry_attempts=5
    )
    ws.orderbook_stream(depth="50", symbol=symbol, callback=handle_message)

    while True:
        time.sleep(0.5)  # Keep the thread alive
# def fetch_daily_crypto_sentiment_diff(api_key, currencies=["BTC","ETH","XRP","BNB"]):
#     base_url = "https://cryptopanic.com/api/v1/posts/"
# 
#     def fetch_news(filter_str):
#         params = {
#             "auth_token": api_key,
#             "currencies": ",".join(currencies),
#             "filter": filter_str,
#             "kind": "news",
#             "public": "true",
#             "regions": "en"
#         }
#         resp = requests.get(base_url, params=params)
#         resp.raise_for_status()
#         return resp.json().get("results", [])
# 
#     # Helper to count today's news for a filter
#     def count_today_news(filter_str):
#         news = fetch_news(filter_str)
#         today_utc = datetime.now(timezone.utc).date()
#         count = 0
#         for item in news:
#             published_at = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00")).date()
#             if published_at == today_utc:
#                 count += 1
#         return count
# 
#     bullish_count = count_today_news("bullish")
#     bearish_count = count_today_news("bearish")
# 
#     sentiment_diff = bullish_count - bearish_count
#     return sentiment_diff
# def detect_order_blocks(df, lookback=50, move_strength=1.5, influence_window=10):
#     df = df.copy()
#     df['order_block'] = 0  # Neutral by default

#     for i in range(lookback, len(df) - 1):
#         current = df.iloc[i]
#         next_candle = df.iloc[i + 1]

#         body_size = abs(current['Close'] - current['Open'])
#         next_body_size = abs(next_candle['Close'] - next_candle['Open'])

#         # Bullish Order Block
#         if current['Close'] < current['Open'] and next_candle['Close'] > next_candle['Open']:
#             if next_body_size > move_strength * body_size:
#                 for j in range(i + 1, min(i + 1 + influence_window, len(df))):
#                     df.at[df.index[j], 'order_block'] = 1

#         # Bearish Order Block
#         elif current['Close'] > current['Open'] and next_candle['Close'] < next_candle['Open']:
#             if next_body_size > move_strength * body_size:
#                 for j in range(i + 1, min(i + 1 + influence_window, len(df))):
#                     df.at[df.index[j], 'order_block'] = -1

#     return df

# def detect_fvg(df, max_lookback=50):
#     """
#     Detects bullish and bearish FVGs in the dataframe and stores:
#     - fvg_upper
#     - fvg_lower
#     - fvg_direction (1 for bullish, -1 for bearish)
#     - fvg_active (1 if currently in an FVG zone)
#     """
#     df['fvg_upper'] = None
#     df['fvg_lower'] = None
#     df['fvg_direction'] = 0
#     df['fvg_active'] = 0

#     for i in range(2, len(df)):
#         # Bullish FVG: candle2.low > candle0.high
#         if df['Low'].iloc[i - 1] > df['High'].iloc[i - 3]:
#             lower = df['High'].iloc[i - 3]
#             upper = df['Low'].iloc[i - 1]

#             for j in range(i, min(i + max_lookback, len(df))):
#                 current_low = df['Low'].iloc[j]
#                 if current_low <= lower:
#                     break  # gap filled

#                 df.loc[j, 'fvg_upper'] = upper
#                 df.loc[j, 'fvg_lower'] = lower
#                 df.loc[j, 'fvg_direction'] = 1
#                 df.loc[j, 'fvg_active'] = 1

#         # Bearish FVG: candle2.high < candle0.low
#         elif df['High'].iloc[i - 1] < df['Low'].iloc[i - 3]:
#             upper = df['Low'].iloc[i - 3]
#             lower = df['High'].iloc[i - 1]

#             for j in range(i, min(i + max_lookback, len(df))):
#                 current_high = df['High'].iloc[j]
#                 if current_high >= upper:
#                     break  # gap filled

#                 df.loc[j, 'fvg_upper'] = upper
#                 df.loc[j, 'fvg_lower'] = lower
#                 df.loc[j, 'fvg_direction'] = -1
#                 df.loc[j, 'fvg_active'] = 1

#     return df

# def detect_bos(df, lookback=20):
#     """
#     Detect Break of Structure (BoS) where price breaks last swing high/low.
#     Bullish BoS: closes above last swing high
#     Bearish BoS: closes below last swing low
#     """
#     df['bos'] = 0  # 1 = bullish BoS, -1 = bearish BoS

#     for i in range(lookback, len(df)):
#         recent_highs = df['High'].iloc[i - lookback:i]
#         recent_lows = df['Low'].iloc[i - lookback:i]
#         swing_high = recent_highs.max()
#         swing_low = recent_lows.min()

#         close = df.loc[i, 'Close']

#         if close > swing_high:
#             df.loc[i, 'bos'] = 1  # Bullish BoS
#         elif close < swing_low:
#             df.loc[i, 'bos'] = -1  # Bearish BoS

#     return df

def get_klines_df(symbol, interval, limit=240):
    if interval == 15:
        limit = 104
    response = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
    data = response['result']['list']
    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    # df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.int64), unit='ms')
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.int64), unit='ms')

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Select only OHLC
    df = df[["Open", "High", "Low", "Close"]].astype(float)

    return df

def add_indicators(df):
    # df['H-L'] = df['High'] - df['Low']
    # df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    # df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    # tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    # df['ATR'] = tr.rolling(window=14).mean()
    df['ATR'] = ATR(df)

    df['macd_line'], df['macd_signal'], df['macd_osma'] = MACD(df['Close'])

    df['macd_line_diff'] = df['macd_line'].diff()
    df['macd_signal_diff'] = df['macd_signal'].diff()

    df['macd_line_slope'] = df['macd_line'] - df['macd_line'].shift(1)
    df['macd_signal_line_slope'] = df['macd_signal'] - df['macd_signal'].shift(1) 
    
    df['bb_sma'], df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])
    df['bb_sma_pos'] = (df['Close'] >= df['bb_sma']).astype(int)

    # Custom momentum (% of recent high-low range)
    # high_14 = df['High'].rolling(window=14).max()
    # low_14 = df['Low'].rolling(window=14).min()
    # price_range = high_14 - low_14

    df['RSI'] = RSI(df['Close'], 14)
    # df['BB_SMA'], df['BB_Upper_Band'], df['BB_Lower_Band'] = Bollinger_Bands(df)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    # df['DI_Diff'] = (df['+DI'] - df['-DI']).abs()

    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_28'] = df['Close'].ewm(span=28).mean()

    # df['log_return'] = np.log(df['Close'] / df['Close'].shift(1) + 1e-6)

    # df['Body'] = df['Close'] - df['Open']
    # df['Range'] = df['High'] - df['Low'] + 1e-6  # prevent divide by zero

    # df['Body_pct_range'] = df['Body'] / df['Range']

    # df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    # df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # def zscore(series, window=50):
    #     return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-6)

    # df['log_return_z'] = zscore(df['log_return'])
    # df['Body_pct_range_z'] = zscore(df['Body_pct_range'])
    # df['Upper_Wick_z'] = zscore(df['Upper_Wick'])
    # df['Lower_Wick_z'] = zscore(df['Lower_Wick'])

    # df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_signal_diff", "macd_histogram", "BB_SMA", "RSI", "ADX", "Bulls", "Bears", "+DI", "-DI", "ATR"]].copy()
    # df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_osma", "bb_sma", "bb_upper", "bb_lower", "RSI", "ADX", "+DI", "-DI", "ATR", "Body_pct_range_z", "Upper_Wick_z", "Lower_Wick_z"]].copy()

    # conditions_sma = [
    #     df['BB_SMA'] < df['Close'],
    #     df['BB_SMA'] > df['Close']
    # ]
    # choices_sma = [-1, 1]
    # df['BB_SMA'] = np.select(conditions_sma, choices_sma, default=0)

    conditions = [
        df['RSI'] < 30,
        (df['RSI'] >= 30) & (df['RSI'] < 50),
        (df['RSI'] >= 50) & (df['RSI'] < 70),
        df['RSI'] >= 70
    ]
    choices = [1, 2, 3, 4]
    df['RSI_zone'] = np.select(conditions, choices, default=-1)

    # +DI_val
    conditions_plus = [
        df['+DI'] < 5,
        (df['+DI'] >= 5) & (df['+DI'] < 10),
        (df['+DI'] >= 10) & (df['+DI'] < 20),
        (df['+DI'] >= 20) & (df['+DI'] < 30),
        df['+DI'] >= 30
    ]
    choices_plus = [1, 2, 3, 4, 5]
    df['+DI_val'] = np.select(conditions_plus, choices_plus, default=-1)

    # -DI_val
    conditions_minus = [
        df['-DI'] < 5,
        (df['-DI'] >= 5) & (df['-DI'] < 10),
        (df['-DI'] >= 10) & (df['-DI'] < 20),
        (df['-DI'] >= 20) & (df['-DI'] < 30),
        df['-DI'] >= 30
    ]
    choices_minus = [1, 2, 3, 4, 5]
    df['-DI_val'] = np.select(conditions_minus, choices_minus, default=-1)

    # ADX Zone: 0 (<20), 1 (20–29.99), 2 (≥30)
    adx_conditions = [
        df['ADX'] < 20,
        (df['ADX'] >= 20) & (df['ADX'] < 30),
        (df['ADX'] >= 30) & (df['ADX'] < 40),
        df['ADX'] >= 40
    ]
    adx_choices = [0, 1, 2, 3]
    df['ADX_zone'] = np.select(adx_conditions, adx_choices, default=-1)
    
    # EMA crossover: 1 if EMA_7 > EMA_14 > EMA_28, -1 if EMA_7 < EMA_14 < EMA_28, else 0
    df['EMA_crossover'] = np.select(
        [
            (df['EMA_7'] > df['EMA_14']) & (df['EMA_14'] > df['EMA_28']),
            (df['EMA_7'] < df['EMA_14']) & (df['EMA_14'] < df['EMA_28'])
        ],
        [1, -1],
        default=0
    )

    df['macd_zone'] = np.select(
        [
            df['macd_signal'] < 0,
            df['macd_signal'] > 0
        ],
        [-1, 1],
        default=0
    )

    df['macd_osma'] = np.select(
        [
            df['macd_osma'] < 0,
            df['macd_osma'] > 0
        ],
        [-1, 1],
        default=0
    )

    df['macd_crossover'] = np.select(
        [
            df['macd_line'] < df['macd_signal'],
            df['macd_line'] > df['macd_signal'] 
        ],
        [-1, 1],
        default=0
    )

    df['macd_line_diff'] = np.select(
        [
            df['macd_line_diff'] > 0,
            df['macd_line_diff'] < 0
        ],
        [1, -1],
        default=0
    )

    df['macd_signal_diff'] = np.select(
        [
            df['macd_signal_diff'] > 0,
            df['macd_signal_diff'] < 0
        ],
        [1, -1],
        default=0
    )

    # df["Open"]  = df["Open"] / df["Close"] - 1
    # df["High"]  = df["High"] / df["Close"] - 1
    # df["Low"]   = df["Low"] / df["Close"] - 1
    # df["Close"] = df["Close"].pct_change().fillna(0)  # as return

    df = detect_order_blocks(df)
    # df = detect_fvg(df)
    # df = detect_bos(df)
    df = Momentum(df)

    df['spread'] = 0
    df['imbalance'] = 0
    df['wall_flag'] = 0

    df = df[["Open", "High", "Low", "Close", "EMA_crossover", "macd_zone", "macd_line", "macd_signal", "macd_line_diff", "macd_signal_diff", "macd_line_slope", "macd_signal_line_slope" , "macd_osma", "macd_crossover", "bb_sma", "bb_upper", "bb_lower", "RSI_zone", "ADX_zone", "+DI_val", "-DI_val", "ATR", "Momentum", "spread", "imbalance", "wall_flag", "order_block"]].copy()
    # df = df[["Open", "High", "Low", "Close", "EMA_crossover", "macd_zone", "macd_line", "macd_signal", "macd_line_diff", "macd_signal_diff", "macd_line_slope", "macd_signal_line_slope" , "macd_osma", "macd_crossover", "bb_sma", "bb_upper", "bb_lower", "RSI_zone", "ADX_zone", "+DI_val", "-DI_val", "ATR", "order_block_type"]].copy()

    df.dropna(inplace=True)
    ready_event.set()
    return df

def generate_signals(df):
    """
    Return a Series of 'Buy', 'Sell', or '' using
        • flat‑market veto             (10‑bar High/Low range)
        • ADX strength > 20
        • MACD‑angle filter            (current slope > 5‑bar avg slope)
        • MACD hook skip               (no trade immediately after peak/valley)
        • DI / Bull‑Bear / OSMA logic  (your original direction rules)
    """

    signals = [0] * len(df)

    # --- pre‑compute helpers once ------------------------------------
    macd_slope      = df['macd_line'].diff()
    macd_slope_ma_5 = macd_slope.rolling(5).mean()

    # “hook” is True on the bar *right after* MACD slope changes sign
    macd_hook = (
        (macd_slope.shift(1) > 0) & (macd_slope <= 0) |   # bullish peak
        (macd_slope.shift(1) < 0) & (macd_slope >= 0)     # bearish valley
    )

    for i in range(len(df)):
        if i < 10:        # need at least 10 bars for range filter
            continue

        latest = df.iloc[i]

        # -------------- flat‑market veto ------------------------------
        # window = df.iloc[i - 10 : i]
        # price_range = window["High"].max() - window["Low"].min()
        # if (price_range / window["Close"].mean()) <= 0.005:
        #     continue

        # -------------- trend strength check --------------------------
        if latest.ADX_zone == 0:
            continue

        # -------------- MACD angle filter -----------------------------
        angle_ok = abs(macd_slope.iloc[i]) > abs(macd_slope_ma_5.iloc[i])
        if not angle_ok:
            continue

        # -------------- hook (peak/valley) skip -----------------------
        if macd_hook.iloc[i]:
            continue

        # if df['macd_crossover'].iloc[i] == 1:
        #    signals[i] = 1
        # df['EMA_crossover'].iloc[i] == 1:
        if df['macd_signal_diff'].iloc[i] == 1:
        # if df['order_block'].iloc[i] == 1:
           signals[i] = 1
        # elif df['macd_crossover'].iloc[i] == -1:
        # df['EMA_crossover'].iloc[i] == -1:
        elif df['macd_signal_diff'].iloc[i] == -1:
        # elif df['order_block'].iloc[i] == -1:
            signals[i] = -1
        else:
            signals[i] = 0


        # -------------- directional logic -----------------------------
        # if latest.macd_signal_diff > 0:
        #     if latest.macd_trending_up:
       #         signals[i] = "Buy"
        #     else:
        #         signals[i] = "Close"
        # if (latest.macd_signal_diff > 0 and
        #     latest['+DI'] > latest['-DI'] and
        #     latest.Bull_Bear_Diff > 0 and
        #     latest.OSMA_Diff > 0):
        #     signals[i] = "Buy"

        # elif (latest.macd_signal_diff < 0 and
        #       latest['+DI'] < latest['-DI'] and
        #       latest.Bull_Bear_Diff < 0 and
        #       latest.OSMA_Diff < 0):
        #     signals[i] = "Sell"
        # elif latest.macd_signal_diff < 0:
        #     if latest.macd_trending_down:
        #         signals[i] = "Sell"
        #     else:
        #         signals[i] = "Close"

    df["signal"] = signals
    ready_event.set()
    return df["signal"]

def append_new_candle_and_update(df, new_candle):
    # new_row = pd.DataFrame([new_candle])
    # new_row = pd.DataFrame([new_candle])
    df = pd.concat([df, new_candle], ignore_index=True)

    # Keep only the last N rows
    if len(df) > self.max_window_size:
        df = df.iloc[-self.max_window_size:].reset_index(drop=True)

    # Percentage changes
    for col in ["Open", "High", "Low", "Close"]:
        df[f"{col}_pct"] = df[col].pct_change().fillna(0)

    # MACD
    # macd, macd_signal, macd_histogram = MACD(df["close"])
    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    # df["macd"] = macd
    # df["macd_signal"] = macd_signal
    # df['macd_histogram'] = macd_histogram

    # ADX, +DI, -DI
    # adx, plus_di, minus_di = ADX(df)
    # df["ADX"] = adx
    # df["+DI"] = plus_di
    # df["-DI"] = minus_di
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)

    # EMA 7, 14, 28
    df["EMA_7"] = ema(df["Close"], span=7)
    df['EMA_14'] = ema(df["Close"], span=14)
    df["EMA_28"] = ema(df["Close"], span=28)

    # Bulls and Bears Power
    df["Bulls"] = df["High"] - df['EMA_14']
    df["Bears"] = df["Low"] - df['EMA_14']

    df.fillna(0, inplace=True)
    return df

def keep_session_alive():
    global session
    for attempt in range(30):
        try:
            # Example: Get latest position
            result = session.get_positions(category="linear", symbol="BTCUSDT")
            break  # If success, break out of loop
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on attempt {attempt+1}, retrying...")
            time.sleep(5)  # wait before retry
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
        finally:
            session = HTTP(api_key=api_key, api_secret=api_secret, demo=False, recv_window=15000, timeout=60)
            threading.Timer(120, keep_session_alive).start()

def get_balance():
    """
    Return total equity in USDT only.
    """
    try:
        resp = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"               # still useful – filters server‑side
        )

        coins = resp["result"]["list"][0]["coin"]   # ← go into the coin array
        usdt_row = next(c for c in coins if c["coin"] == "USDT")
        return float(usdt_row["equity"])            # or "availableToWithdraw"
    except (KeyError, StopIteration, IndexError) as e:
        print("[get_balance] parsing error:", e, resp)
        return 0.0

def get_mark_price(symbol):
    price_data = session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(symbol):
    info = session.get_instruments_info(category="linear", symbol=symbol)
    data = info["result"]["list"][0]
    step = float(data["lotSizeFilter"]["qtyStep"])
    min_qty = float(data["lotSizeFilter"]["minOrderQty"])
    return step, min_qty

def calc_order_qty(risk_amount: float,
                   entry_price: float,
                   min_qty: float,
                   qty_step: float) -> float:
    """
    Return a Bybit‑compliant order quantity, rounded *down* to the nearest
    step size.  If the rounded amount is below `min_qty`, return 0.0.
    """
    # 1️⃣  Convert everything to Decimal
    risk_amt   = Decimal(str(risk_amount))
    price      = Decimal(str(entry_price))
    step       = Decimal(str(qty_step))
    min_q      = Decimal(str(min_qty))

    # print(f"risk_amount: {risk_amount}, entry price: {entry_price}, qty_step: {qty_step}, min_qty: {min_qty}")

    # 2️⃣  Raw qty (no rounding yet)
    raw_qty = risk_amt / price           # still Decimal

    # 3️⃣  Round *down* to the nearest step
    qty = (raw_qty // step) * step       # floor division works (both Decimal)

    # 4️⃣  Enforce minimum
    if qty < min_q:
        return 0.0

    # 5️⃣  Cast once for the API
    return float(qty)

def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    print(f"Waiting {round(sleep_seconds / 60, 2)} minutes until next candle...")
    print()
    time.sleep(sleep_seconds)

def update_candles(df, symbol, interval=15):
    response = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=2)
    latest = response['result']['list'][-1]  # most recent candle
    latest_time = int(latest[0])

    latest_timestamp = df.index[-1]
    # if df.iloc[-1]['Timestamp'] < latest_time:
    if latest_timestamp < latest_time:
        # Add new completed candle
        # df = df.append({
        #     'Timestamp': latest_time,
        #     'Open': float(latest[1]),
        #     'High': float(latest[2]),
        #     'Low': float(latest[3]),
        #     'Close': float(latest[4]),
        #     'Volume': float(latest[5])
        # }, ignore_index=True)
        new_row = pd.DataFrame([{
            "Timestamp": latest_timestamp,
            "Open": latest['Open'],
            "High": latest['High'],
            "Low": latest['Low'],
            "Close": latest['Close'],
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)

        df = df.tail(240)  # Keep max 240
    else:
        # Candle still forming, optionally update last row
        df.iloc[-1] = {
            'Timestamp': latest_time,
            'Open': float(latest[1]),
            'High': float(latest[2]),
            'Low': float(latest[3]),
            'Close': float(latest[4]),
        }

    return df

def get_position(symbol):
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        positions = response.get("result", {}).get("list", [])
        
        if not positions:
            # print(f"[INFO] No positions found for {symbol}.")
            return None

        pos = positions[0]
        size = float(pos.get("size", 0))
        return pos
    except Exception as e:
        print(f"[get_position] Error fetching position for {symbol}: {e}")
        return None

def close_old_orders(symbol):
    """
    1) Close any live position at market (reduce‑only).
    2) Cancel all outstanding TP / SL / limit orders.
    """

    # ── 1) Close the live position ─────────────────────────────
    pos = get_position(symbol)                # returns dict or None
    # print(f"pos: get_position({symbol}): {pos}")
    if pos and float(pos.get("size", 0)) > 0:
        side        = pos["side"]             # "Buy" or "Sell"
        close_side  = "Sell" if side == "Buy" else "Buy"
        # qty         = round_qty(
        #                  symbol,
        #                  float(pos["size"]),
        #                  float(pos["avgPrice"])
        #              )
        qty = pos["size"]

        resp_close = session.place_order(
            category     = "linear",
            symbol       = symbol,
            side         = close_side,
            order_type   = "Market",
            qty          = qty,
            reduce_only  = True,
            time_in_force= "IOC"
        )
        if resp_close.get("retCode") == 0:
            print(f"[{symbol}] Position closed ({side} → {close_side}) qty={qty}")
        else:
            print(f"[{symbol}] Close‑position fail: {resp_close}")

        # tiny pause to let Bybit settle before we yank orders
        time.sleep(0.2)
class LSTMPPOAgent:
    def __init__(self, state_size, hidden_size, action_size, lr=1e-3, gamma=0.95, clip_ratio=0.2):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_epochs = 5
        self.batch_size = 32
        self.entropy_coef = 0.01

        # Initialize weights
        self.model = {
            # LSTM
            'Wx': np.random.randn(4 * hidden_size, state_size) * 0.1,
            'Wh': np.random.randn(4 * hidden_size, hidden_size) * 0.1,
            'b': np.zeros(4 * hidden_size),

            # Policy head
            'W_policy': np.random.randn(action_size, hidden_size) * 0.1,
            'W_policy_2': np.random.randn(1, hidden_size) * 0.1,
            'b_policy': np.zeros(action_size),
            'b_policy_2': np.zeros(1),

            # Value head
            'W_value': np.random.randn(1, hidden_size) * 0.1,
            'b_value': np.zeros(1),
        }

        self.reset_state()
        self.trajectory = []

    def reset_state(self):
        self.h = np.zeros((self.hidden_size,))
        self.c = np.zeros((self.hidden_size,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))  # prevent overflow

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def lstm_forward(self, x_seq):
        h, c = self.h.copy(), self.c.copy()
        for x in x_seq:
            x = np.asarray(x).reshape(-1)  # ensure shape (state_size,)
            assert x.shape[0] == self.state_size, f"x shape {x.shape} does not match state_size {self.state_size}"
            z = np.dot(self.model['Wx'], x) + np.dot(self.model['Wh'], h) + self.model['b']
            i = self.sigmoid(z[0:self.hidden_size])
            f = self.sigmoid(z[self.hidden_size:2*self.hidden_size])
            o = self.sigmoid(z[2*self.hidden_size:3*self.hidden_size])
            g = self.tanh(z[3*self.hidden_size:])
            c = f * c + i * g
            h = o * self.tanh(c)
        self.h, self.c = h, c
        return h

    def forward(self, x_seq):
        h = self.lstm_forward(x_seq)
        logits = np.dot(self.model['W_policy'], h) + self.model['b_policy']
        value = np.dot(self.model['W_value'], h) + self.model['b_value']
        probs = self.softmax(logits)
        return probs, value[0]

    def sl_tp_forward(self, x_seq):
        h = self.lstm_forward(x_seq)
        # logits = np.dot(self.model['W_policy'], h) + self.model['b_policy']
        logits = np.dot(self.model['W_policy_2'], h) + self.model['b_policy_2']
        # probs = self.softmax(logits)
        value = np.dot(self.model['W_value'], h) + self.model['b_value']
        continuous_action = self.sigmoid(logits)
        # return probs, value[0]
        return continuous_action, value[0] # (tp_raw, sl_raw)

    def scale_action(self, tp_raw, sl_raw):
        tp_min, tp_max = 0.0002, 0.002  # 0.5% to 5%, 0.2% to 2% with 50x leverage
        sl_min, sl_max = 0.0001, 0.001 # 0.2% to 3%, 
        tp_pct = tp_min + (tp_max - tp_min) * tp_raw
        sl_pct = sl_min + (sl_max - sl_min) * sl_raw
        return tp_pct, sl_pct

    def select_action(self, state_seq, in_position):
        valid_actions = []
        try:
            logits, value = self.forward(state_seq)

            # Convert logits to probabilities using softmax
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)

            # Handle invalid probabilities
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                # print("⚠️ Warning: NaN or zero-sum probabilities. Using uniform distribution.")
                probs = np.ones(self.action_size) / self.action_size

            probs = probs / np.sum(probs)  # Normalize again just in case

            if np.argmax(probs) >= 0.9:
                action = np.argmax(probs)
                logprob = np.log(probs[action] + 1e-8)
            else:
                action = np.random.choice(self.action_size, p=probs)
                logprob = np.log(probs[action] + 1e-8)

            # if in_position:
            #     action = 0
            if in_position:
                valid_actions = [0, 1]  # Hold or Close
            else:
                valid_actions = [0, 2, 3]  # Hold, Long, Short
                
            # Mask invalid actions:
            masked_probs = np.array([probs[a] if a in valid_actions else 0 for a in range(self.action_size)])
            if masked_probs.sum() == 0:
                masked_probs = np.ones(self.action_size) / self.action_size
            else:
                masked_probs = masked_probs / masked_probs.sum()
                    
            if np.argmax(masked_probs) >= 0.9:
                action = np.argmax(masked_probs)
            else:
                action = np.random.choice(self.action_size, p=masked_probs)
            logprob = np.log(masked_probs[action] + 1e-8)

            # if last_ema_crossover == current_ema_crossover:
            #     action = 0

            return action, logprob, value
        except Exception as e:
            # print(f"❌ select_action error: {e}")
            return None

    def store_transition(self, state_seq, action, logprob, value, reward, done):
        self.trajectory.append((state_seq, action, logprob, value, reward, done))

    def store_reward(self, reward):
        if self.trajectory:
            last = self.trajectory[-1]
            self.trajectory[-1] = (*last, reward)

    def _discount_rewards(self, rewards):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted.insert(0, R)
        return discounted

    def update_policy_and_value(self, states_seq, actions, old_action_probs, advantages, returns, lr=1e-3, epsilon=0.1):
        for i in range(len(states_seq)):
            state_seq = states_seq[i]
            action = actions[i]
            old_prob = old_action_probs[i]
            advantage = advantages[i]
            target_value = returns[i]

            # Initialize LSTM hidden state
            h = np.zeros((self.hidden_size, 1))

            # Forward pass through LSTM for the state sequence
            for x in state_seq:
                x = np.array(x, dtype=np.float32).reshape(self.state_size, 1)
                z = np.dot(self.model['Wx'], x) + np.dot(self.model['Wh'], h) + self.model['b'].reshape(-1, 1)
                # LSTM gates split
                i_gate = self.sigmoid(z[0:self.hidden_size])
                f_gate = self.sigmoid(z[self.hidden_size:2*self.hidden_size])
                o_gate = self.sigmoid(z[2*self.hidden_size:3*self.hidden_size])
                g_gate = np.tanh(z[3*self.hidden_size:4*self.hidden_size])

                # LSTM cell state update (you might want to maintain c state if you have it, here simplified)
                c = f_gate * np.zeros_like(h) + i_gate * g_gate  # Assume c initialized as zeros for simplicity
                h = o_gate * np.tanh(c)

            # Policy logits and probabilities
            logits = np.dot(self.model['W_policy'], h) + self.model['b_policy'].reshape(-1, 1)
            probs = self.softmax(logits.flatten())
            new_prob = probs[action]

            # Value prediction
            value = (np.dot(self.model['W_value'], h) + self.model['b_value']).item()

            # Value loss gradient
            v_error = value - target_value
            grad_W_value = v_error * h.T
            grad_b_value = v_error

            # PPO policy loss gradient
            ratio = new_prob / (old_prob + 1e-10)
            clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
            policy_grad_coef = -min(ratio * advantage, clipped_ratio * advantage)

            grad_logits = probs.copy()
            grad_logits[action] -= 1  # dSoftmax cross-entropy grad
            grad_logits *= policy_grad_coef

            grad_W_policy = np.dot(grad_logits.reshape(-1, 1), h.T)
            grad_b_policy = grad_logits.reshape(-1, 1)

            # Gradient descent update
            self.model['W_policy'] -= lr * grad_W_policy
            self.model['b_policy'] -= lr * grad_b_policy.flatten()
            self.model['W_value'] -= lr * grad_W_value
            self.model['b_value'] -= lr * grad_b_value

    def ppo_policy_loss(self, old_probs, new_probs, advantages, epsilon=0.1):
        old_probs = np.array(old_probs)
        new_probs = np.array(new_probs)
        advantages = np.array(advantages)
        
        ratios = new_probs / (old_probs + 1e-10)
        clipped = np.clip(ratios, 1 - epsilon, 1 + epsilon)
        loss = -np.mean(np.minimum(ratios * advantages, clipped * advantages))
        return loss
    def value_loss(self, values, returns):
        values = np.array(values)
        returns = np.array(returns)
        return 0.5 * np.mean((returns - values) ** 2)

    def compute_gae(self, rewards, values, dones, gamma=0.95, lam=0.95):
        advantages = []
        gae = 0
        values = np.append(values, 0)  # Bootstrap value after last step
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def train(self):
        # if len(self.trajectory) < 2:
        #     return  # Not enough data to train
        if self.trajectory:
            # Unpack trajectory
            states, actions, logprobs_old, values, rewards, dones = zip(*self.trajectory)

            # Convert to arrays
            values = np.array(values)
            rewards = np.array(rewards)

            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Compute GAE
            advantages = self.compute_gae(rewards, values, dones, gamma=0.95, lam=0.95)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
            returns = advantages + values

            # Clip returns and values
            returns = np.clip(returns, -1000, 1000)
            values = np.clip(values, -1000, 1000)
            advantages = np.clip(advantages, -10, 10)

            # Train for multiple epochs (optional)
            for _ in range(self.train_epochs):
                for i in range(len(states)):
                    state_seq = states[i]
                    action = actions[i]
                    old_logprob = logprobs_old[i]
                    # print(f"len(states): {len(states)}")
                    # print(f"len(actions): {len(actions)}")
                    # print(f"len(logprobs_old): {len(logprobs_old)}")
                    # print(f"len(values): {len(values)}")
                    # print(f"len(rewards): {len(rewards)}")
                    # print(f"len(dones): {len(dones)}")
                    # print(f"length of advantages: {len(advantages)}, i: {i}, length of states: {len(states)}")
                    advantage = advantages[i]
                    target_value = returns[i]

                    # Forward pass
                    probs, value = self.forward(state_seq)

                    # Entropy bonus
                    entropy = -np.sum(probs * np.log(probs + 1e-8))

                    # Policy loss
                    logprob = np.log(probs[action] + 1e-8)
                    ratio = np.exp(logprob - old_logprob)
                    clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

                    # Value loss
                    v_loss = 0.5 * ((target_value - value) ** 2)

                    # Total loss
                    loss = policy_loss + 0.5 * v_loss - 0.01 * entropy

                    # Gradient descent step (simplified)
                    for k in self.model:
                        self.model[k] -= self.lr * loss
                        self.model[k] = np.clip(self.model[k], -1000, 1000)  # Clamp weights

                self.trajectory.clear()

    def savecheckpoint(self, symbol):
        os.makedirs("LSTM-PPO-saves", exist_ok=True)
        filename = f"LSTM-PPO-saves/{datetime.now().strftime('%Y-%m-%d')}-{symbol}.checkpoint.lstm-ppo.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def loadcheckpoint(self, symbol):
        files = sorted(os.listdir("LSTM-PPO-saves"))
        files = [f for f in files if f.endswith(".checkpoint.lstm-ppo.pkl") and symbol in f]
        if not files:
            print(f"[!] No checkpoint found for {symbol}")
            return

        latest = os.path.join("LSTM-PPO-saves", files[-1])
        with open(latest, "rb") as f:
            self.model = pickle.load(f)

class WinRateKNN:
    def __init__(self, symbol, k=10):
        self.k = k
        self.symbol = symbol
        self.states = []
        self.labels = []  # 1 = win, 0 = loss
        self.model = None

    def add(self, state, is_win):
        try:
            state = np.array(state, dtype=np.float32).flatten()  # Force all elements to float
        except Exception as e:
            # print("❌ Could not convert state to float:", state, "| Error:", e)
            return

        if not np.all(np.isfinite(state)):
            # print("⚠️ Skipping state with NaN or Inf:", state)
            return

        self.states.append(state)
        self.labels.append(1 if is_win else 0)

        if len(self.states) >= 100:
            self._remove_redundant_neighbor()
            # self.states.pop(0)
            # self.labels.pop(0)

        if len(self.states) >= self.k:
            self._fit()

    def _remove_redundant_neighbor(self):
        if len(self.states) < 2:
            return  # Nothing to remove

        X = np.array(self.states)

        # Compute pairwise similarity (cosine, or use euclidean if you prefer)
        sim_matrix = cosine_similarity(X)

        # Zero out diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, 0)

        # Compute average similarity for each row (how redundant each entry is)
        redundancy_scores = sim_matrix.mean(axis=1)

        # Remove the most redundant (highest avg similarity)
        idx_to_remove = np.argmax(redundancy_scores)

        del self.states[idx_to_remove]
        del self.labels[idx_to_remove]
    def _fit(self):
        """
        Fit the KNN model with stored data.
        """
        if len(self.states) < 1:
            # print("⚠️ Not enough data to fit KNN.")
            return

        # Safety check
        k_neighbors = max(1, min(self.k, len(self.states)))

        self.model = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree")
        self.model.fit(self.states)

    def predict_win_rate(self, state_seq, k_near=5, k_far=5):
        """
        Return the win rate based on k nearest neighbors of the input state.
        """
        # if not self.model or len(self.states) < self.k:
        if len(self.states) < 1000:
            # return True  # Not enough data
            return 1  # Not enough data

        # Find the 100 nearest neighbors
        distances, indices = self.model.kneighbors(state.reshape(1, -1), n_neighbors=50)
        distances = distances[0]
        indices = indices[0]

        # Split into nearest and farthest groups
        nearest_idx = indices[:k_near]
        nearest_dist = distances[:k_near]

        farthest_idx = indices[-k_far:]
        farthest_dist = distances[-k_far:]

        # Combine indices and distances
        combined_idx = np.concatenate([nearest_idx, farthest_idx])
        combined_dist = np.concatenate([nearest_dist, farthest_dist])

        # Get win/loss labels for selected neighbors
        selected_labels = np.array([self.labels[i] for i in combined_idx])

        # Calculate weights (closer gets higher weight)
        weights = 1 / (combined_dist + 1e-6)  # Add epsilon to avoid div-by-zero

        # Normalize weights
        weights /= weights.sum()

        # Compute weighted win rate
        win_rate = np.dot(selected_labels, weights)

        return win_rate
    def save(self):
        """
        Save the KNN model to disk.
        """
        path = f"LSTM-PPO-saves/{datetime.now().strftime('%Y-%m-%d')}-{self.symbol}.win_rate_knn.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "states": self.states,
                "labels": self.labels,
                "model": self.model
            }, f)

    def load(self):
        """
        Load the KNN model from disk.
        """
        # path = f"LSTM-PPO-saves/win_rate_knn-{self.symbol}.pkl"
        files = sorted(os.listdir("LSTM-PPO-saves"))
        files = [f for f in files if f.endswith(".win_rate_knn.pkl") and self.symbol in f]
        if not files:
            print(f"[!] No checkpoint found for {self.symbol}")
            return

        latest = os.path.join("LSTM-PPO-saves", files[-1])

        try:
            with open(latest, "rb") as f:
                data = pickle.load(f)
                self.states = data["states"]
                self.labels = data["labels"]
                self.model = data["model"]
                # print(f"✅ Loaded WinRateKNN from {latest}")
        except FileNotFoundError:
            print(f"⚠️ No saved KNN found at {path}. Starting fresh.")
    
def sharpe_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return (mean_ret - risk_free_rate) / std_ret

def sortino_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    # Downside deviation: only consider returns below risk-free rate, and their square differences
    downside_diff = [(r - risk_free_rate)**2 for r in returns if r < risk_free_rate]
    
    if len(downside_diff) == 0:
        return 0  # Or float('inf') if you'd rather signal perfect performance
    
    downside_std = np.sqrt(np.mean(downside_diff))
    
    if downside_std == 0:
        return 0
    
    return (mean_ret - risk_free_rate) / downside_std

def calculate_position_size(balance, risk_pct, sl_dist, leverage, min_qty):
    risk_amount = max(balance * risk_pct, 15)
    position_size = risk_amount * leverage / sl_dist
    # position_size = round(position_size, 3)

    if position_size < min_qty:
        position_size = min_qty

    return position_size
def get_max_leverage(position_size, symbol):
    try:
        response = session.get_risk_limit(category="linear", symbol=symbol)
    except AttributeError:
        response = session._submit_request(
            method="GET",
            url="https://api.bybit.com/v5/position/risk-limit",
            params={"category": "linear", "symbol": symbol},
            auth=True
        )
    # print(response)
    risk_limits = response['result']['list']  # <-- get the whole list, not just first item's riskLimitValue
    for tier in risk_limits:
        if float(position_size) <= float(tier['riskLimitValue']):
            return float(tier['maxLeverage'])
    return float(risk_limits[-1]['maxLeverage'])


def train_bot(df, agent, symbol, bybit_symbol, window_size=20):
    global capital
    global process_counter
    invest = 0.0
    all_trade_pcts = []
    mean_pct = 0.0
    pct = 0.0
    daily_pnl = 0.0
    reward = 0.0
    position = 0
    sl_price = 0.0
    tp_price = 0.0
    entry_price = 0.0
    price = 0.0
    leverage = 20
    position_size = 0.0
    daily_trades = 0
    daily_returns = []
    action = 0
    current_day = None
    save_counter = 0
    atr = 0.0
    # trailing_sl_pct = 0.000875
    knn = WinRateKNN(symbol)
    entry_state = None
    trade_reward = 0
    in_position = False
    instrument_info = session.get_instruments_info(
        category="linear",
        symbol=bybit_symbol
    )
    info = instrument_info['result']['list'][0]
    qty_step = float(info['lotSizeFilter']['qtyStep'])  # Minimum qty increment
    min_qty = float(info['lotSizeFilter']['minOrderQty'])  # Minimum allowed qty
    price_precision = int(info['priceScale'])
    done = False
    pnl = 0.0

    for t in range(window_size, len(df) - 1):
        state_seq = df[t - window_size:t].values.astype(np.float32)
        if state_seq.shape != (window_size, agent.state_size):
            print("Shape mismatch:", state_seq.shape)
            continue

        result = agent.select_action(state_seq, in_position)
        if result is None:
            continue
        action, logprob, value = result

        invest = max(capital * 0.05, 15)

        position_size = invest
        # leverage = get_max_leverage(position_size, bybit_symbol)

        price = df.iloc[t]["Close"]
        day = str(df.index[t]).split(' ')[0]

        if current_day is None:
            current_day = day
        elif day != current_day:
            daily_return_pct = (daily_pnl / capital) * 100 if capital != 0 else 0
            daily_returns.append(daily_return_pct)

            avg_profit_per_trade = daily_pnl / daily_trades if daily_trades > 0 else 0
            sr = sharpe_ratio(daily_returns)
            sor = sortino_ratio(daily_returns)

            capital += daily_pnl
            if capital < 0:
                daily_pnl *= -1
            print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.2f}, {avg_profit_per_trade/position_size*100:.2f}% - PnL: {daily_pnl/capital*100:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.2f} - Sortino: {sor:.2f}")
            process_counter += 1
            if process_counter % 5 == 0:
                print(f"Above is day {process_counter / 5}")
                print()
            
            current_day = day

            daily_pnl = 0.0
            daily_trades = 0

        # Force close if ADX zone becomes 0
        if df["ADX_zone"].iloc[t] == 0 and position != 0:
            action = 3
        elif df["ADX_zone"].iloc[t] == 0:
            action = 0
        
        macd_zone = df.iloc[t]['macd_zone']
        plus_di = df.iloc[t]['+DI_val']
        minus_di = df.iloc[t]['-DI_val']
        ema_crossover = df.iloc[t]['EMA_crossover']
        macd_crossover = df.iloc[t]['macd_crossover']
        atr = df.iloc[t]['ATR']
                
        tp_dist = price * 0.1
        sl_dist = price * 0.05
        trailing_sl_pct = price * 0.05
        tp_hit = False
        sl_hit = False

        final_pct = 0.0

        max_price = 0.0
        min_price = 0.0

        profit_pct = 0.0

        if position == 0:
            if action == 1 and knn.predict_win_rate(state_seq) > 0.9 and df['signal'].iloc[t] == 1:
                entry_state = state_seq
                entry_price = price
                invest = max(capital * 0.05, 15)
                position_size = invest
                # leverage = get_max_leverage(position_size, bybit_symbol)
                # position_size = calculate_position_size(capital, 0.05, sl_dist, leverage, min_qty)
                entry_price = price
                capital -= position_size * 0.00075
                position = 1
                partial_tp_hit = [False, False, False]
                position_pct_left = 1.0
                in_position = True

            if action == 2 and knn.predict_win_rate(state_seq) > 0.9 and df['signal'].iloc[t] == -1:
                entry_state = state_seq
                entry_price = price
                invest = max(capital * 0.05, 15)
                position_size = invest
                # leverage = get_max_leverage(position_size, bybit_symbol)
                # position_size = calculate_position_size(capital, 0.05, sl_dist, leverage, min_qty)
                entry_price = price
                capital -= position_size * 0.00075
                position = -1
                partial_tp_hit = [False, False, False]
                position_pct_left = 1.0
                in_position = True

        if position == 1:
            profit_pct = (price - entry_price) / entry_price
            # if price > entry_price - position_size * 0.00075 * 2 - position_size * 0.00025 * (1 + trailing_sl_pct):
            min_price = max(min_price, price)
            if price <= (min_price + position_size * 0.00075 * 2 + position_size * 0.00025) * (1 - trailing_sl_pct):
                action = 3
                trailing_sl_hit = True
            # tp_levels = [
            #     entry_price + 0.382 * tp_dist,
            #     entry_price + 0.618 * tp_dist,
            #     entry_price + 0.786 * tp_dist,
            # ]
            # sl_price = entry_price - sl_dist
            # tp_price = entry_price + tp_dist
            # tp_shares = [0.4, 0.2, 0.2]

            # for i in range(3):
            #     if not partial_tp_hit[i] and price >= tp_levels[i]:
            #         pnl = position_size * tp_shares[i]
            #         capital += pnl
            #         daily_pnl += pnl
            #         position_size -= pnl
            #         position_pct_left -= tp_shares[i]
            #         partial_tp_hit[i] = True
            #         action = 0
            #         reward += 1
            #         daily_trades += 1

            # if price >= tp_price or price <= sl_price:
            #     if price >= tp_price:
            #         reward += 1
            #     action = 3
            #     if price >= tp_price:
            #         tp_hit = True
            #     elif price <= sl_price:
            #         sl_hit = True
            if df['signal'].iloc[t] == -1:
                action = 3
                                      
        elif position == -1:
            profit_pct = (entry_price - price) / entry_price
            # if price <= entry_price + position_size * 0.003 * 2 + position_size * 0.001 * (1 - trailing_sl_pct):
            max_price = min(max_price, price)
            if price >= (max_price - position_size * 0.0075 * 2 - position_size * 0.00025) * (1 + trailing_sl_pct):
                action = 3
                trailing_sl_hit = True
            # tp_levels = [
            #     entry_price - 0.382 * tp_dist,
            #     entry_price - 0.618 * tp_dist,
            #     entry_price - 0.786 * tp_dist
            # ]
            # sl_price = entry_price + sl_dist - position_size * 0.00075 * 2 - position_size * 0.00025
            # tp_price = entry_price - tp_dist
            # tp_shares = [0.4, 0.2, 0.2]

            # for i in range(3):
            #     if not partial_tp_hit[i] and price <= tp_levels[i]:
            #         pnl = position_size * tp_shares[i]
            #         capital += pnl
            #         daily_pnl += pnl
            #         position_pct_left -= tp_shares[i]
            #         partial_tp_hit[i] = True
            #         action = 0
            #         reward += 1
            #         daily_trades += 1

            # if price <= tp_price or price >= sl_price:
            #     if price <= tp_price:
            #         reward += 1
            #     action = 3
            #     if price <= tp_price:
            #         tp_hit = True
            #     elif price >= sl_price:
            #         sl_hit = True
            if df['signal'].iloc[t] == 1:
                action = 3

        if action == 3 and df['signal'].iloc[t] == -1 and position == 1 and entry_price != 0:
            final_pct = 0.0
            if sl_hit:
                final_pct = -sl_dist / entry_price
            elif tp_hit:
                final_pct = tp_dist / entry_price
            if trailing_sl_hit:
                final_pct = abs(min_price - price) / entry_price
            # final_pct = (entry_price - price) / entry_price
            capital -= position_size * 0.00075
            pnl = position_size * final_pct - position_size * 0.00075 * 2 - 0.00025 * position_size
            capital += pnl
            reward += final_pct - 0.00075 * 2 - 0.00025
            daily_pnl += pnl
            daily_trades += 1
            knn.add(entry_state, is_win=(final_pct - 0.00075 * 2 - 0.00025 > 0.00))
            entry_price = 0.0
            position = 0
            in_position = False
            done = True

        if action == 3 and df['signal'].iloc[t] == 1 and position == -1 and entry_price != 0:
            final_pct = 0.0
            if sl_hit:
                final_pct = -sl_dist / entry_price
            elif tp_hit:
                final_pct = tp_dist / entry_price
            if trailing_sl_hit:
                final_pct = abs(max_price - price) / entry_price
            # final_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            capital -= position_size * 0.00075
            pnl = position_size * final_pct - position_size * 0.00075 * 2 - 0.00025 * position_size
            capital += pnl
            reward += final_pct - 0.00075 * 2 - 0.00025
            daily_pnl += pnl
            daily_trades += 1
            knn.add(entry_state, is_win=(final_pct - 0.00025 * 2 - 0.00075 > 0.00))
            entry_price = 0.0
            position = 0
            in_position = False
            daily_trades += 1
            done = True

        if action == 0 and position != 0:
            profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
            pnl = calc_order_qty(profit_pct * position_size, entry_price, min_qty, qty_step)  # not margin
            if pnl > invest:
                action == 3
            reward += profit_pct - 0.00075 * 2 - 0.00025

        if action == 3 and pnl > invest and entry_price != 0:
            capital -= invest
            profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
            reward += profit_pct - 0.00075 * 2 - 0.00025
            position = 0
            in_position = False
            daily_pnl += pnl
            daily_trades += 1
            entry_price = 0.0
            done = True
                
        # === Store reward and update step ===
        agent.store_transition(state_seq, action, logprob, value, reward, done)
        save_counter += 1
        reward = 0
        trailing_sl_hit = False
        done = False
        sl_hit = False
        tp_hit = False
        # if save_counter % 10080 == 0: # 1 week timeframe training on 1min timeframe
        if save_counter % 672 == 0: # 1 week timeframe training on 15min timeframe
        # if save_counter % 168 == 0: # 1 week timeframe training on 4h timeframe
            print(f"[{symbol}] [INFO] Training PPO on step {save_counter}...")
            agent.train()
            # agent.savecheckpoint(symbol)
            knn._fit()
            # knn.save()
            # print(f"[{symbol}] [INFO] Saved checkpoint at step {save_counter}")
            # print()
    agent.train()
    agent.savecheckpoint(symbol)
    knn._fit()
    knn.save()
    print(f"[{symbol}] [INFO] Saved checkpoint at step {save_counter}")
    print(f"✅ [{symbol}] PPO training complete. Final capital: {capital:.2f}, Total accumulation: {capital/1000:.2f}x")

def test_bot(df, agent, symbol, bybit_symbol, window_size=20):
    knn = WinRateKNN(symbol)
    knn.load()
    agent.loadcheckpoint(symbol)
    capital = get_balance()
    position = 0
    entry_price = None
    response = session.get_positions(category="linear", symbol=bybit_symbol)
    session_positions = response['result']['list']
    for pos in session_positions:
        if 'side' in pos and pos['size'] != '0':
            # print(f"{pos}")
            # print(f"Side: {pos['side']}, Size: {pos['size']}")
            position = 1 if pos['side'] == "Buy" else -1 if pos['side'] == "Sell" else 0
            position_size = float(pos['size'])
        if 'avgPrice' in pos and pos['size'] != '0':
            entry_price = float(pos['avgPrice'])
    position_size = 0.0

    invest = 0.0
    total_qty = 0.0
    sl_price = 0.0
    tp_price = 0.0
    leverage = 20
    # qty_step, min_qty = get_qty_step(bybit_symbol)
    current_day = None
    reward = 0.0
    save_counter = 0
    atr = 0.0
    tp_shares = []
    daily_pnl = 0.0
    df = None
    partial_tp_hit = [False, False, False]
    position_size = 0.0
    in_position = False
    entry_state = None
    
    instrument_info = session.get_instruments_info(
        category="linear",
        symbol=bybit_symbol
    )
    info = instrument_info['result']['list'][0]
    qty_step = float(info['lotSizeFilter']['qtyStep'])  # Minimum qty increment
    min_qty = float(info['lotSizeFilter']['minOrderQty'])  # Minimum allowed qty
    price_precision = int(info['priceScale'])
    df = None
    profit_pct = 0.0
    partial_tp_hit = [False, False, False]
    position_pct_left = 1.0
    done = False
    daily_trades = 0

    orderbook_features = {}
    orderbook_lock = threading.Lock()
    orderbook_thread = threading.Thread(
        target=update_orderbook,
        args=(bybit_symbol, orderbook_features, orderbook_lock)
    )
    orderbook_thread.daemon = True
    orderbook_thread.start()

    while True:
        wait_until_next_candle(15)
        df = get_klines_df(bybit_symbol, 15)
        df = add_indicators(df)
        df['signal'] = generate_signals(df)
        atr = df['ATR'].iloc[-1]
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"=== [{bybit_symbol}] stats at {now} ===")
        print(f"price: {df['Close'].iloc[-1]}")
        print(f"adx: {df['ADX_zone'].iloc[-1]}")
        print(f"rsi: {df['RSI_zone'].iloc[-1]:.2f}")
        print(f"atr: {round(atr, price_precision)}")
        bias_ema_crossover = "bullish" if df['EMA_crossover'].iloc[-1] > 0  else "bearish" if df['EMA_crossover'].iloc[-1] < 0 else "neutral"
        print(f"ema7/14/28 crossover up/down: {df['EMA_crossover'].iloc[-1] > 0}/{df['EMA_crossover'].iloc[-1] < 0} ({bias_ema_crossover})")
        bias_macd_signal_line = "bullish" if df['macd_signal'].iloc[-1] > 0 else "bearish" if df["macd_signal"].iloc[-1] < 0 else "neutral"
        print(f"macd zone: {df['macd_zone'].iloc[-1]:.2f} ({bias_macd_signal_line})")
        # print(f"macd line: {df['macd_line'].iloc[-1]:.2f} - going up/down: {df['macd_line_diff'].iloc[-1]:.2f} ({bias_macd_signal_line})")
        bias_osma_diff = "bullish" if df['macd_osma'].iloc[-1] > 0 else "bearish" if df['macd_osma'].iloc[-1] < 0 else "neutral"
        print(f"OSMA zone: {df['macd_osma'].iloc[-1]:.2f} ({bias_osma_diff})")
        # di_diff = df['di_diff'].iloc[-1]
        bias_DI_DIff = "bullish" if df['+DI_val'].iloc[-1] > df['-DI_val'].iloc[-1] else "bearish" if df['+DI_val'].iloc[-1] < df['-DI_val'].iloc[-1] else "neutral"
        print(f"+DI_val/-DI_val: {df['+DI_val'].iloc[-1]:.2f}/{df['-DI_val'].iloc[-1]:.2f} ({bias_DI_DIff})")
        print()

        df['spread'] = orderbook_features.get('spread', 0.0)
        df['imbalance'] = orderbook_features.get('imbalance', 0.0)
        df['wall_flag'] = orderbook_features.get('wall_flag', 0)

        state_seq = df[-window_size:].values.astype(np.float32)
        if state_seq.shape != (window_size, agent.state_size):
            print("Shape mismatch:", state_seq.shape)
            continue

        response = session.get_positions(category="linear", symbol=bybit_symbol)
        positions = response["result"]["list"]
        if positions:
            session_position = positions[0]
            position = 1 if float(session_position["size"]) > 0 else -1 if float(session_position["size"]) < 0 else 0
            in_position = True
            if 'side' in pos and pos['size'] != '0':
                position = 1 if pos['side'] == "Buy" else -1 if pos['side'] == "Sell" else 0
            else:
                position = 0
            if 'avgPrice' in pos and pos['size'] != '0':
                entry_price = float(pos['avgPrice'])
            else:
                entry_price = 0.0
            
        result = agent.select_action(state_seq, in_position)
        if result is None:
            continue

        action, logprob, value = result

        price = df.iloc[-1]["Close"]
        day = str(df.index[-1]).split(' ')[0]

        if current_day is None:
            current_day = day
        elif day != current_day:
            daily_return_pct = (daily_pnl / capital) * 100 if capital != 0 else 0
            daily_returns.append(daily_return_pct)

            avg_profit_per_trade = daily_pnl / daily_trades if daily_trades > 0 else 0
            sr = sharpe_ratio(daily_returns)
            sor = sortino_ratio(daily_returns)

            capital = get_balance()

            print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.2f}, {avg_profit_per_trade/invest*100:.2f}% - PnL: {daily_pnl/capital*100:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.2f} - Sortino: {sor:.2f}")
            current_day = day

            daily_pnl = 0.0
            daily_trades = 0
            
        # Force close if ADX zone becomes 0
        if df["ADX_zone"].iloc[-1] == 0 and position != 0:
            action = 3
        elif df["ADX_zone"].iloc[-1] == 0:
            action = 0

        macd_zone = df.iloc[-1]['macd_zone']
        plus_di = df.iloc[-1]['+DI_val']
        minus_di = df.iloc[-1]['-DI_val']
        ema_crossover = df.iloc[-1]['EMA_crossover']
        macd_crossover = df.iloc[-1]['macd_crossover']
        atr = df.iloc[-1]['ATR']
        capital = get_balance()
                
        tp_dist = price * 0.1
        sl_dist = price * 0.05
        trailing_sl_pct = price * 0.05 / price

        if position == 0:
            if action == 1 and knn.predict_win_rate(state_seq) > 0.9 and df['signal'].iloc[-1] == 1:
                entry_price = price
                entry_seq = state_seq
                capital = get_balance()
                invest = max(capital * 0.05, 15)
                position_size = calc_order_qty(invest * leverage, entry_price, min_qty, qty_step)
                # position_size = calculate_position_size(capital, 0.05, sl_dist, leverage, min_qty=min_qty)
                position = 1
                in_position = True

                # session.set_leverage(
                #     category="linear",        # or "inverse", depending on the symbol type
                #     symbol=bybit_symbol,         # or any other trading pair
                #     buyleverage=str(leverage),         # leverage for long positions
                #     sellleverage=str(leverage)         # leverage for short positions
                # )
                session.place_order(
                    category="linear",
                    symbol=bybit_symbol,
                    side="Buy",
                    order_type="Market",
                    qty=str(round(position_size, 6)),
                    reduce_only=False,
                    buyLeverage=leverage,
                    sellLeverage=leverage,
                    # take_profit=str(round(entry_price + tp_dist, price_precision))
                )
                print(f"[{bybit_symbol}] Entered Buy order")
                # response = session.set_trading_stop(
                #     category="linear",  # or "inverse", depending on your market
                #     symbol=bybit_symbol,
                #     trailing_stop=str(round(sl_dist, price_precision)),  # Trailing stop in USD or quote currency
                #     # active_price=str(round(entry_price + entry_price * 0.001, price_precision)),
                #     position_idx=0
                # )
                # tp_levels = [
                #     entry_price + 0.382 * tp_dist,
                #     entry_price + 0.618 * tp_dist,
                #     entry_price + 0.786 * tp_dist
                # ]
                # tp_shares = [0.4, 0.2, 0.2]
                # for i in range(3):
                #     realized = calc_order_qty(position_size * tp_shares[i], entry_price, min_qty, qty_step)
                #     close_side = "Sell" if position == 1 else "Buy"
                #     response = session.place_order(
                #         category="linear",
                #         symbol=bybit_symbol,
                #         side=close_side,  # opposite side to close position
                #         order_type="Limit",
                #         qty=str(round(realized, 6)),
                #         price=str(round(tp_levels[i], price_precision)),
                #     )
                #     partial_tp_hit = [False, False, False]
                #     position_pct_left = 1.0
                #     daily_trades += 1
                            
            if action == 2 and knn.predict_win_rate(state_seq) > 0.9 and df['signal'].iloc[-1] == -1:
                entry_price = price
                entry_seq = state_seq
                capital = get_balance()
                invest = max(capital * 0.05, 15)
                position_size = calc_order_qty(invest * leverage, entry_price, min_qty, qty_step)
                # position_size = calculate_position_size(capital, 0.05, sl_dist, leverage, min_qty=min_qty)
                position = -1
                in_position = True

                # session.set_leverage(
                #     category="linear",        # or "inverse", depending on the symbol type
                #     symbol=bybit_symbol,         # or any other trading pair
                #     buyleverage=str(leverage),         # leverage for long positions
                #     sellleverage=str(leverage)         # leverage for short positions
                # )
                session.place_order(
                    category="linear",
                    symbol=bybit_symbol,
                    side="Sell",
                    order_type="Market",
                    qty=str(round(position_size, 6)),
                    reduce_only=False,
                    buyLeverage=leverage,
                    sellLeverage=leverage,
                    take_profit=str(round(entry_price+tp_dist, price_precision))
                )
                print(f"[{bybit_symbol}] Entered Sell order")
                # response = session.set_trading_stop(
                #     category="linear",  # or "inverse", depending on your market
                #     symbol=bybit_symbol,
                #     trailing_stop=str(round(sl_dist, price_precision)),  # Trailing stop in USD or quote currency
                #     # active_price=str(round(entry_price - entry_price * 0.001, price_precision)),
                #     position_idx=0
                # )
                # tp_levels = [
                #     entry_price - 0.382 * tp_dist,
                #     entry_price - 0.618 * tp_dist,
                #     entry_price - 0.786 * tp_dist
                # ]
                # tp_shares = [0.4, 0.2, 0.2]
                # for i in range(3):
                #     realized = calc_order_qty(position_size * tp_shares[i], entry_price, min_qty, qty_step)
                #     close_side = "Sell" if position == 1 else "Buy"
                #     response = session.place_order(
                #         category="linear",
                #         symbol=bybit_symbol,
                #         side=close_side,  # opposite side to close position
                #         order_type="Limit",
                #         qty=str(round(realized, 6)),
                #         price=str(round(tp_levels[i], price_precision)),
                #         buyLeverage=leverage,
                #         sellLeverage=leverage,
                #     )
                #     partial_tp_hit = [False, False, False]
                #     position_pct_left = 1.0
                #     daily_trades += 1

        if action == 0 and position != 0:
            if entry_price == 0.00:
                profit_pct = 0.00
            else:
                profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
                reward += profit_pct

        if position == 1:
            # tp_levels = [
            #     entry_price - 0.382 * tp_dist,
            #     entry_price - 0.618 * tp_dist,
            #     entry_price - 0.786 * tp_dist
            # ]
            # sl_price = price + sl_dist
            # tp_price = entry_price + tp_dist
            # tp_shares = [0.4, 0.2, 0.2]

            # for i in range(3):
            #     if not partial_tp_hit[i] and price >= tp_levels[i]:
            #         pnl = calc_order_qty(position_size * tp_shares[i], entry_price, min_qty, qty_step)
            #         daily_pnl += pnl
            #         position_size -= pnl
            #         position_pct_left -= tp_shares[i]
            #         partial_tp_hit[i] = True
            #         action = 0
            #         reward += 1
            #         daily_trades += 1

            # if price >= tp_price or price <= sl_price:
            #     final_pct = (entry_price - price) / entry_price
            #     if price >= tp_price:
            #         reward += 1
            #         action = 3
            if df['signal'].iloc[-1] == 1:
                action = 3

        if position == -1:
            # tp_levels = [
            #     entry_price - 0.382 * tp_dist,
            #     entry_price - 0.618 * tp_dist,
            #     entry_price - 0.786 * tp_dist
            # ]
            # sl_price = price + sl_dist - position_size * 0.00075 * 2 - position_size * 0.00025
            # tp_price = entry_price - tp_dist
            # tp_shares = [0.4, 0.2, 0.2]

            # for i in range(3):
            #     if not partial_tp_hit[i] and price <= tp_levels[i]:
            #         pnl = calc_order_qty(position_size * tp_shares[i], entry_price, min_qty, qty_step)
            #         capital += pnl
            #         daily_pnl += pnl
            #         position_size -= pnl
            #         position_pct_left -= tp_shares[i]
            #         partial_tp_hit[i] = True
            #         action = 0
            #         reward += 1

            # if price <= tp_price or price >= sl_price:
            #     partial_tp_hit = [False, False, False]
            #     if price <= tp_price:
            #         reward += 1
            #         action = 3
            if df['signal'].iloc[-1] == 1:
                action = 3

        if action == 3 and position == 1 and df['signal'].iloc[-1] == -1 and entry_price != 0:
            final_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            reward += final_pct - 0.00075 * 2 - 0.00025
            daily_pnl += profit_pct * position_size - position_size * 0.00075 * 2 - position_size * 0.00025
            knn.add(entry_state, is_win=(final_pct - 0.00075 * 2 - 0.00025 > 0.00))
            close_old_orders(bybit_symbol)
            in_position = False
            entry_price = 0
            done = True

        if action == 3 and position == -1 and df['signal'].iloc[-1] == 1 and entry_price != 0:
            final_pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            reward += final_pct - 0.00075 * 2 - 0.00025
            daily_pnl += profit_pct * position_size - position_size * 0.00075 * 2 - position_size * 0.00025
            knn.add(entry_state, is_win=(final_pct - 0.00075 * 2 - 0.00025 > 0.00))
            close_old_orders(bybit_symbol)
            in_position = False
            daily_trades += 1
            entry_price = 0.0
            done = True

        if action == 0:
            if entry_price != 0.00:
                profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
            else:
                profit_pct = 0.00
            reward += profit_pct - 0.00075 * 2 - 0.00025

        # === Store reward and update step ===
        agent.store_transition(state_seq, action, logprob, value, reward, done)
        save_counter += 1
        reward = 0
        done = False
        # if save_counter % 10080 == 0:
        if save_counter % 1 == 0:
            knn._fit()
            knn.save()
            agent.train()
            agent.savecheckpoint(symbol)
            # print()
            # print(f"[{bybit_symbol}] [INFO] Saved checkpoint at step {save_counter}")
    # rrKNN.train()
    # rrKNN.save()
    # agent.train()
    # agent.savecheckpoint(symbol)
    # print(f"✅ PPO training complete. Final capital: {capital:.2f}, Total PnL: {capital/1000:.2f}")

# MAX_WORKERS = 5

def main():
    counter = 0
    threads = []
    # processes = []
    train = False
    test = True
    counter = 0
    threading.Thread(target=keep_session_alive).start()

    # subprocess.run(["git", "clone", "https://github.com/pressure679/AI-LSTM-PPO-Crypto-Bybit-Investor.git"])
    # Pull latest changes in an existing repo
    # subprocess.run(["git", "-C", "~/", "pull"])

    # subprocess.run(["cd AI-LSTM-PPO-Crypto-Bybit-Investor; mkdir LSTM-PPO-saves; mv *XAUUSD.win_rate_knn.pkl LSTM-PPO-saves; mv *XAUUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves; mv *XRPUSD.win_rate_knn.pkl LSTM-PPO-saves; mv *XRPUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves; mv *BNBUSD.win_rate_knn.pkl LSTM-PPO-saves; mv *BNBUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves; mv *ETHUSD.win_rate_knn.pkl LSTM-PPO-saves; mv *ETHUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves; mv *BTCUSD.win_rate_knn.pkl LSTM-PPO-saves; mv *BTCUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves"])

    # Pattern to match today's file or similar
    # files = glob.glob("*pkl")
    
    # Destination folder
    # destination = "LSTM-PPO-saves"
    
    # Move matching files
    # for file in files:
    #     shutil.move(file, destination)
    
    if train:
        print("Starting training phase...")
        # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #     futures = []
        for i in range(0, len(symbols)):
            df = None
            print(f"Initialized looping over symbol, currently at {counter + 1}/{len(symbols)}, {symbols[i]}")
            if symbols[i] == "XAUUSD":
                df = load_last_mb_xauusd()
            else:
                # continue
                df = load_last_mb(symbols[i])
                # df = yf_get_ohlc_df(yf_symbol)
            df = add_indicators(df)
            df['signal'] = generate_signals(df)
            lstm_ppo_agent = LSTMPPOAgent(state_size=28, hidden_size=64, action_size=4)
            counter += 1
            # futures.append(executor.submit(train_bot, df, lstm_ppo_agent, symbols[i], bybit_symbols[i]))
            t = threading.Thread(target=train_bot, args=(df, lstm_ppo_agent, symbols[i], bybit_symbols[i]))
            # p = Process(target=train_bot, args=(df, lstm_ppo_agent, symbols[i], bybit_symbols[i]))
            # processes.append(p)
            # p.start()
            t.start()
            threads.append(t)
            # p.join()
        # for future in as_completed(futures):
        #     try:
        #         future.result()
        #     except Exception as e:
        #         print(f"Training thread error: {e}")
    for t in threads:
        t.join()
    # processes[0].start()
    # processes[1].start()
    # processes[2].start()
    # processes[3].start()
    # processes[4].start()
    # processes[0].join()
    # processes[1].join()
    # processes[2].join()
    # processes[3].join()
    # processes[4].join()

    # Reset for test phase
    counter = 0
    # threads = []
    processes = []

    if test:
        print("Starting test phase...")
        # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #     futures = []
        for i, bybit_symbol in enumerate(bybit_symbols):
            df = get_klines_df(bybit_symbol, 96)
            df = add_indicators(df)
            df['signal'] = generate_signals(df)
            lstm_ppo_agent = LSTMPPOAgent(state_size=28, hidden_size=64, action_size=4)
            # futures.append(executor.submit(test_bot, df, lstm_ppo_agent, symbols[i], bybit_symbol))
            t = threading.Thread(target=test_bot, args=(df, lstm_ppo_agent, symbols[i], bybit_symbols[i]))
            # t.start()
            t.start()
            threads.append(t)
            # p = Process(target=test_bot, args=(df, lstm_ppo_agent, symbols[i], bybit_symbols[i]))
            # processes.append(p)
        # for future in as_completed(futures):
        #     try:
        #         future.result()
        #     except Exception as e:
        #         print(f"Test thread error: {e}")
    # processes[0].start()
    # processes[1].start()
    # processes[2].start()
    # processes[3].start()
    # processes[4].start()
    # processes[0].join()
    # processes[1].join()
    # processes[2].join()
    # processes[3].join()
    # processes[4].join()

    for t in threads:
        t.join()

main()
