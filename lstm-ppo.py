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
import time
from decimal import Decimal
from pybit.unified_trading import HTTP
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
# import warnings
# import yfinance as yf

# warnings.filterwarnings("ignore", category=RuntimeWarning)
# Popular crypto: "DOGEUSDT", "HYPEUSDT", "FARTCOINUSDT", "SUIUSDT", "INITUSDT", "BABYUSDT", "NILUSDT"
# yf_symbols = ["BTC-USD", "BNB-USD", "ETH-USD", "XRP-USD", "XAUUSD=X"]
# bybit_symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT", "XAUTUSDT"]
# symbols = ["BTCUSD", "BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]
yf_symbols = ["BTC-USD", "BNB-USD", "ETH-USD", "XRP-USD"]
bybit_symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT"]
symbols = ["BTCUSD", "BNBUSD", "ETHUSD", "XRPUSD"]
# yf_symbols = ["BNB-USD", "ETH-USD", "XRP-USD", "XAUUSD=X"]
# bybit_symbols = ["BNBUSDT", "ETHUSDT", "XRPUSDT", "XAUTUSDT"]
# symbols = ["BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]

ACTIONS = ['hold', 'long', 'short', 'close']
# alpha = 0.1
# gamma = 0.95
# epsilon = 0.1

capital = 1000

# Bybit Demo API Key and Secret - K2OW1k9LlnjeQWYHUK - y3TZKS6KV6Yt5Y4SqdmN6NO8y6htZXiAmUeV
# Bybit API Key and Secret - PoP1ud3PuWajwecc4S - z9RXVMWpiOoE3TubtAQ0UtGx8I5SOiRp1KPU
api_key = "PoP1ud3PuWajwecc4S"
api_secret = "z9RXVMWpiOoE3TubtAQ0UtGx8I5SOiRp1KPU"
session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    demo=False,  # or False for mainnet
    recv_window=5000,
    timeout=30
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

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()

    return df

def load_last_mb_xauusd(file_path="/mnt/chromeos/removable/sd_card/1m dataframes/XAUUSD_1m_data.csv", mb=2*3, delimiter=';', col_names=None):
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

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()
    
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

def BullsPower(df, period=14):
    ema = EMA(df['Close'], period)
    return df['High'] - ema

def BearsPower(df, period=14):
    ema = EMA(df['Close'], period)
    return df['Low'] - ema

def get_klines_df(symbol, interval, limit=240):
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

    # df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_signal_diff", "macd_histogram", "BB_SMA", "RSI", "ADX", "Bulls", "Bears", "+DI", "-DI", "ATR"]].copy()
    df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_osma", "bb_sma", "bb_upper", "bb_lower", "RSI", "ADX", "+DI", "-DI", "ATR"]].copy()

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

    # df["Open"]  = df["Open"] / df["Close"] - 1
    # df["High"]  = df["High"] / df["Close"] - 1
    # df["Low"]   = df["Low"] / df["Close"] - 1
    # df["Close"] = df["Close"].pct_change().fillna(0)  # as return

    # df = df[["Open", "High", "Low", "Close", "EMA_crossover", "EMA_7_28_crossover", "macd_zone", "macd_direction", "BB_SMA", "RSI_zone", "ADX_zone", "Bulls", "Bears", "+DI_val", "-DI_val", "ATR"]].copy()
    df = df[["Open", "High", "Low", "Close", "EMA_crossover", "macd_zone", "macd_osma", "macd_crossover", "bb_sma", "bb_upper", "bb_lower", "RSI_zone", "ADX_zone", "+DI_val", "-DI_val", "ATR"]].copy()

    df.dropna(inplace=True)
    return df

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
    for attempt in range(30):
        try:
            # Example: Get latest position
            result = session.get_positions(category="linear", symbol="BTCUSDT")
            break  # If success, break out of loop
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on attempt {attempt+1}, retrying...")
            time.sleep(2)  # wait before retry
        finally:
            threading.Timer(1500, keep_session_alive, args=[session]).start()

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
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)

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
                print("⚠️ Warning: NaN or zero-sum probabilities. Using uniform distribution.")
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

    def store_transition(self, state_seq, action, logprob, value, reward):
        self.trajectory.append((state_seq, action, logprob, value, reward))

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

    def train(self):
            # print("🚨 Trajectory buffer is empty, skipping training step.")
        # Unpack trajectory
        # states, actions, logprobs_old, values, rewards = None, None, None, None, None
        if self.trajectory:
            states, actions, logprobs_old, values, rewards = zip(*self.trajectory)
            # states, actions, logprobs_old, values = zip(*self.trajectory)
            # self.trajectory.clear()

            # Compute advantages
            returns = self._discount_rewards(rewards)
            advantages = np.array(returns) - np.array(values)

            # PPO update (simplified without real gradients)
            for i in range(len(states)):
                state_seq = states[i]
                action = actions[i]
                old_logprob = logprobs_old[i]
                advantage = advantages[i]

                probs, _ = self.forward(state_seq)
                logprob = np.log(probs[action] + 1e-8)
                ratio = np.exp(logprob - old_logprob)
                clipped = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                loss = -min(ratio * advantage, clipped * advantage)

                # Simple weight update (pretend gradient descent)
                for k in self.model:
                    self.model[k] -= self.lr * loss  # This is just placeholder logic

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

def update_policy_and_value(agent, states_seq, actions, old_action_probs, advantages, returns, lr=1e-3, epsilon=0.1):
    for i in range(len(states_seq)-1):
        if i >= len(advantages):
            continue
        state_seq = states_seq[i]
        action = actions[i]
        old_prob = old_action_probs[i]
        advantage = advantages[i]
        target_value = returns[i]

        # Forward pass
        h = np.zeros((agent.hidden_size, 1))
        for x in state_seq:
            if isinstance(x, float): continue
            x = x.reshape(agent.state_size, 1)
            h = np.tanh(np.dot(agent.model['Wx'], x) + np.dot(agent.model['Wh'], h) + agent.model['b'])

        logits = np.dot(agent.model['Why_pi'], h) + agent.model['by_pi']
        probs = agent._softmax(logits).flatten()
        new_prob = probs[action]

        value = np.dot(agent.model['Why_v'], h) + agent.model['by_v']
        value = value.item()

        # Compute gradients for value loss
        v_error = value - target_value
        grad_Why_v = v_error * h.T
        grad_by_v = v_error

        # Compute gradients for policy loss
        ratio = new_prob / (old_prob + 1e-10)
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
        policy_grad_coef = -min(ratio * advantage, clipped_ratio * advantage)

        grad_logits = probs.copy()
        grad_logits[action] -= 1  # dSoftmax cross-entropy
        grad_logits *= policy_grad_coef

        grad_Why_pi = np.dot(grad_logits.reshape(-1, 1), h.T)
        grad_by_pi = grad_logits.reshape(-1, 1)

        # Update parameters
        agent.model['Why_pi'] -= lr * grad_Why_pi
        agent.model['by_pi'] -= lr * grad_by_pi
        agent.model['Why_v'] -= lr * grad_Why_v
        agent.model['by_v'] -= lr * grad_by_v

def compute_gae(rewards, values, gamma=0.95, lam=0.95):
    advantages = []
    returns = []
    gae = 0
    next_value = 0  # Value after the last step (bootstrap)
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
        returns.insert(0, gae + values[t])
    
    # Normalize advantages
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return np.array(advantages), np.array(returns)

def ppo_policy_loss(old_probs, new_probs, advantages, epsilon=0.1):
    ratios = new_probs / (old_probs + 1e-10)  # Avoid div by zero
    clipped = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    loss = -np.mean(np.minimum(ratios * advantages, clipped * advantages))
    return loss
def value_loss(values, returns):
    return 0.5 * np.mean((returns - values) ** 2)


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

        if len(self.states) >= 1000:
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

        self.model = NearestNeighbors(n_neighbors=k_neighbors)
        self.model.fit(self.states)

    def predict_win_rate(self, state_seq, k_near=25, k_far=25):
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
    downside_returns = [r for r in returns if r < risk_free_rate]
    if len(downside_returns) == 0:
        return 0
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0
    return (mean_ret - risk_free_rate) / downside_std
def calculate_position_size(balance, risk_pct, entry_price, stop_loss, leverage, min_qty=0.001):
    risk_amount = max(balance * risk_pct, 50)
    position_size = risk_amount / abs(entry_price - stop_loss) * leverage
    # position_size = round(position_size, 3)

    if position_size < min_qty:
        position_size = min_qty

    return position_size
def train_bot(df, agent, symbol, window_size=20):
    # agent.loadcheckpoint(symbol)
    capital_lock = threading.Lock()
    global capital
    peak_capital = capital
    invest = 0.0
    all_trade_pcts = []
    mean_pct = 0.0
    pct = 0.0
    daily_pnl = 0.0
    drawdown = 0.0
    reward = 0.0
    position = 0
    sl_price = 0.0
    tp_price = 0.0
    entry_price = 0.0
    price = 0.0
    leverage = 50
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
    last_ema_crossover = 0

    for t in range(window_size, len(df) - 1):
        state_seq = df[t - window_size:t].values.astype(np.float32)
        if state_seq.shape != (window_size, agent.state_size):
            print("Shape mismatch:", state_seq.shape)
            continue
        result = agent.select_action(state_seq, in_position)
        if result is None:
            continue
        action, logprob, value = result

        tp_raw, sl_raw = agent.sl_tp_forward(state_seq)
        tp_pct, sl_pct = agent.scale_action(tp_raw, sl_raw)
        # print(f"tp_raw: {tp_raw}, tp_pct: {tp_pct}")
        # if tp_pct < 0.0002:
        #     continue

        price = df.iloc[t]["Close"]
        day = str(df.index[t]).split(' ')[0]

        if current_day is None:
            current_day = day
        elif day != current_day:
            # print(f"[{symbol}] Day {current_day} - PnL: {(daily_pnl / capital) * 100:.2f}% - Balance: {capital:.2f}")
            daily_return_pct = (daily_pnl / capital) * 100 if capital != 0 else 0
            daily_returns.append(daily_return_pct)

            avg_profit_per_trade = daily_pnl / daily_trades if daily_trades > 0 else 0
            sr = sharpe_ratio(daily_returns)
            sor = sortino_ratio(daily_returns)

            capital += daily_pnl
            # df['HL_rolling_mean'] = (df['High'] - df['Low']).rolling(window=14).mean()
            # print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.4f} - PnL: {daily_pnl/capital*100:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.4f} - Sortino: {sor:.4f} - HL rolling mean: {df['HL_rolling_mean'].iloc[-1]}")
            # print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade/(capital*0.05)*100:.2f}% - PnL: {daily_pnl/capital*100:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.2f} - Sortino: {sor:.2f}")
            if capital < 0:
                daily_pnl *= -1
            print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.2f} - PnL: {daily_pnl/capital*100:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.2f} - Sortino: {sor:.2f}")
            
            current_day = day

            daily_pnl = 0.0
            daily_trades = 0

        # Force close if ADX zone becomes 0
        if df["ADX_zone"].iloc[t] == 0 and position != 0:
            action = 3
        elif df["ADX_zone"].iloc[t] == 0:
            action = 0
        # price_range = (df['High'].iloc[t-14:t] - df['Low'].iloc[t-14:t])
        # if df['High'].iloc[t-14:t] - df['Low'].iloc[t-14:t] / df['Close'].iloc[t] < 0.003:
        # if price_range / df['Close'].iloc[t] < 0.003:
        # price_range = df['High'].iloc[t-14:t] - df['Low'].iloc[t-14:t]
        # if (price_range / df['Close'].iloc[t]).mean() < 0.003:
        #     continue

        macd_zone = df.iloc[t]['macd_zone']
        plus_di = df.iloc[t]['+DI_val']
        minus_di = df.iloc[t]['-DI_val']
        ema_crossover = df.iloc[t]['EMA_crossover']
        macd_crossover = df.iloc[t]['macd_crossover']
        atr = df.iloc[t]['ATR']
                
        # tp_dist = atr * 10
        # sl_dist = atr * 5
        # trailing_sl_pct = atr * 2.5 / price
        tp_dist = df.iloc[t]['Close'] * 0.0035
        sl_dist = df.iloc[t]['Close'] * 0.00175
        trailing_sl_dist = df.iloc[t]['Close'] * 0.000875
        trailing_sl_pct = 0.000875

        final_pct = 0.0

        max_price = 0.0
        min_price = 0.0

        profit_pct = 0.0

        # last_ema_crossover == ema_crossover

        # ema_7_28_crossover_idx = 0
        # ema_7_28_crossover = False
        # ema_7_28_crossback = False

        # HL = []

        # qty_step, min_qty = get_qty_step(symbol)
        # if df['EMA_7_28_crossover'].iloc[t] == 1 and df['EMA_7_28_crossover'].iloc[t-1] == -1:
        #     # ema_7_28_crossover = True
        #     # ema_7_28_crossover += 1
        #     position_size = calculate_position_size(capital, 0.05, entry_price, df.iloc[-1]['Close'] * 0.00175)
        #     df['HL_rolling_mean'] = (df['High'] - df['Low']).rolling(window=14).mean()
        #     # session.place_order(
        #     #     category="linear",
        #     #     symbol=symbol,
        #     #     side="Buy",
        #     #     order_type="Market",
        #     #     qty=position_size,
        #     #     reduce_only=False,
        #     #     # time_in_force="IOC"
        #     #     # take_profit=str(round(entry_price + df.iloc[-1]['Close'] * 0.01, 6)),
        #     #     take_profit=str(round(entry_price + df.iloc[t]['Close'] * df['HL_rolling_mean'], 6)),
        #     #     stop_loss=str(round(entry_price - df.iloc[t]['Close'] * 0.005, 6))
        #     # )
        # if df['EMA_7_28_crossover'].iloc[t] == -1 and df['EMA_7_28_crossover'].iloc[t-1] == 1:
        #     position_size = calculate_position_size(capital, 0.05, entry_price, df.iloc[t]['Close'] * 0.005)
        #     # ema_7_28_crossover = False
        #     # print(f"ema_7_28_crossover_idx: {ema_7_28_crossover_idx}")
        #     # HL = (df['High'].iloc[-ema_7_28_crossover_idx:] - df['Low'].iloc[-ema_7_28_crossover_idx:]) / df['Close'].iloc[-ema_7_28_crossover_idx:]
        #     # HL = HL.dropna()
        #     # ema_7_28_crossover_idx = 0
        #     # HL = append((df['High'].iloc[-ema_7_28_crossover_idx:] - df['Low'].iloc[-ema_7_28_crossover_idx:])/df['Close'].iloc[-ema_7_28_crossover_idx:])
        #     # session.place_order(
        #     #     category="linear",
        #     #     symbol=symbol,
        #     #     side="Sell",
        #     #     order_type="Market",
        #     #     qty=position_size,
        #     #     reduce_only=False,
        #     #     # time_in_force="IOC"
        #     #     take_profit=str(round(entry_price + df.iloc[t]['Close'] * 0.01, 6)),
        #     #     stop_loss=str(round(entry_price - df.iloc[t]['Close'] * 0.005, 6))
        #     # )

        if position == 0:
            # print(f"Open Buy or Sell order")
            # if action == 1 and macd_direction == 1 and plus_di > minus_di and bulls > 0:
            # if action == 1 and macd_direction == 1 and plus_di > minus_di and bulls > 0  and knn.predict_win_rate(state_seq) > 0.9:
            # if action == 1 and knn.predict_win_rate(state_seq) > 0.9:
            # if action == 1 and knn.predict_win_rate(state_seq) > 0.9:
            if action == 1 and knn.predict_win_rate(state_seq) > 0.9 and minus_di < plus_di and macd_crossover == 1:
                entry_state = state_seq
                # if capital < 20:
                #     print(f"capital under minimum, ${capital:.2f}, waiting until amount is free")
                #     break
                invest = max(capital * 0.05, 15)
                # if invest < 15:
                #     print(f"investment amount under minimum, ${invest:.2f}, waiting until amount is free")
                #     break
                position_size = invest * leverage
                entry_price = price
                capital -= positions_size * 0.002
                position = 1
                partial_tp_hit = [False, False, False]
                position_pct_left = 1.0
                in_position = True
                # daily_trades += 1

            # elif action == 2 and macd_zone == -1 and minus_di > plus_di and bears > 0:
            # elif action == 2 and macd_direction == -1 and minus_di > plus_di and bears > 0  and knn.predict_win_rate(state_seq) > 0.9:
            # if action == 2 and knn.predict_win_rate(state_seq) > 0.9:
            if action == 2 and knn.predict_win_rate(state_seq) > 0.9 and minus_di > plus_di and macd_crossover == -1:
                entry_state = state_seq
                # if capital < 20:
                #     print(f"Free capital under minimum, ${capital:.2f}, waiting until amount is free")
                #     break
                invest = max(capital * 0.05, 15)
                # if invest < 15:
                #     print(f"Free investment amount under minimum, {invest:.2f}, waiting until amount is free")
                #     break
                position_size = invest * leverage
                entry_price = price
                capital -= position_size * 0.002
                position = -1
                partial_tp_hit = [False, False, False]
                position_pct_left = 1.0
                in_position = True
                # daily_trades += 1

        # if position == 1:
        #     profit_pct = (price - entry_price) / entry_price
        #     max_price = max(max_price, price)
        #     if price <= max_price * (1 - trailing_sl_pct):
        #         # daily_pnl += profit_pct * position_size
        #         action = 3
        #     pnl = profit_pct * position_size  # not margin
        #     tp_levels = [
        #         entry_price + 0.4 * tp_dist,
        #         entry_price + 0.6 * tp_dist,
        #         entry_price + 0.8 * tp_dist,
        #     ]
        #     sl_price = entry_price - sl_dist
        #     tp_price = entry_price + tp_dist
        #     tp_shares = [0.4, 0.2, 0.2]

        #     for i in range(3):
        #         if not partial_tp_hit[i] and price >= tp_levels[i]:
        #             realized = position_size * tp_shares[i]
        #             pnl = realized * profit_pct
        #             capital += pnl
        #             reward += pnl / capital
        #             daily_pnl += pnl
        #             invest -= realized
        #             position_pct_left -= tp_shares[i]
        #             partial_tp_hit[i] = True
        #             reward += 1
        #             # trade_reward += 1
        #             action = 0

        #     if price >= tp_price or price <= sl_price:
        #         if price >= tp_price:
        #             trade_reward += 1
        #             reward += 1
        #         else:
        #             trade_reward -= 1
        #             reward -= 1 
        #         action = 3
                                        
        # elif position == -1:
        #     profit_pct = (entry_price - price) / entry_price
        #     pnl = position_size * profit_pct
        #     min_price = min(min_price, price)
        #     if price >= min_price * (1 + trailing_sl_pct):
        #         # daily_pnl += profit_pct * position_size
        #         action = 3
        #     reward += pnl
        #     tp_levels = [
        #         entry_price - 0.4 * tp_dist,
        #         entry_price - 0.6 * tp_dist,
        #         entry_price - 0.8 * tp_dist
        #     ]
        #     sl_price = entry_price + sl_dist
        #     tp_price = entry_price - tp_dist
        #     tp_shares = [0.4, 0.2, 0.2]

        #     for i in range(3):
        #         if not partial_tp_hit[i] and price <= tp_levels[i]:
        #             realized = position_size * tp_shares[i]
        #             pnl = realized * profit_pct
        #             capital += pnl
        #             reward += pnl / capital
        #             daily_pnl += pnl
        #             invest -= realized
        #             position_pct_left -= tp_shares[i]
        #             partial_tp_hit[i] = True
        #             daily_trades += 1
        #             reward += 1
        #             # trade_reward += 1
        #             action = 0

        #     if price <= tp_price or price >= sl_price:
        #         # final_pct = (entry_price - price) / entry_price
        #         # pnl = position_size * final_pct
        #         # capital += pnl
        #         # reward += pnl / capital
        #         # daily_pnl += pnl
        #         # invest = 0
        #         # position = 0
        #         # position_pct_left = 1.0
        #         # partial_tp_hit = [False, False, False, False]
        #         if price >= tp_price:
        #             reward += 1
        #             trade_reward += 1
        #         else:
        #             reward -= 1
        #             trade_reward -= 1
        #         action = 3
        if action == 3 and position != 0:
            final_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
            capital -= position_size * 0.002
            pnl = position_size * final_pct - position_size * 0.002 * 2
            # print(f"position size: {position_size}, final pct: {final_pct}, invest: {invest:.2f}, pnl: {pnl:.2f}")
            # if pnl < 2:
            #     reward -= 1
            # if pnl > 2:
            #     reward += 1
            capital += pnl
            # reward += pnl / capital
            reward = final_pct - 0.002 * 2
            daily_pnl += pnl
            daily_trades += 1
            knn.add(entry_state, is_win=(final_pct - 0.002 * 2 > 0))
            entry_price = 0.0
            position = 0
            in_position = False
            daily_trades += 1
            if profit_pct < 0.002:
                reward -= 1
            else:
                reward += 1

        if action == 0 and position != 0:
            profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
            # pnl = calc_order_qty(profit_pct * position_size, entry_price, min_qty, qty_step)  # not margin
            # pnl = position_size * profit_pct
            # reward += pnl
            reward = profit_pct - 0.002 * 2
                
        # === Store reward and update step ===
        agent.store_transition(state_seq, action, logprob, value, reward)
        save_counter += 1
        reward = 0
        if save_counter % 10080 == 0:
        # if save_counter % 24 * 60 == 0:
            # begun = True
            print(f"[INFO] Training PPO on step {save_counter}...")
            agent.train()
            # agent.savecheckpoint(symbol)
            knn._fit()
            # knn.save()
            # print(f"[INFO] Saved checkpoint at step {save_counter}")
            # print()
    agent.train()
    agent.savecheckpoint(symbol)
    knn._fit()
    knn.save()
    print(f"[INFO] Saved checkpoint at step {save_counter}")
    print(f"✅ PPO training complete. Final capital: {capital:.2f}, Total accumulation: {capital/1000:.2f}x")

def test_bot(df, agent, symbol, bybit_symbol, window_size=20):
    capital_lock = threading.Lock()
    knn = WinRateKNN(symbol)
    knn.load()
    agent.loadcheckpoint(symbol)
    capital = get_balance()
    peak_capital = capital
    # session_position = session.get_positions(category="linear", symbol=symbol)
    position = 0
    # if session_position['side']:
    #     position = 1 if session_position['side'] == "Buy" else -1 if session_position['side'] == "Sell" else 0
    invest = 0.0
    entry_price = 0.0
    total_qty = 0.0
    sl_price = 0.0
    tp_price = 0.0
    leverage=50
    qty_step, min_qty = get_qty_step(bybit_symbol)
    current_day = None
    reward = 0.0
    save_counter = 0
    atr = 0.0
    tp_shares = []
    daily_pnl = 0.0
    df = None
    # df = get_klines_df(bybit_symbol, 5)
    # df = add_indicators(df)
    partial_tp_hit = []
    position_size = 0.0
    in_position = False
    
    instrument_info = session.get_instruments_info(
        category="linear",
        symbol=bybit_symbol
    )
    info = instrument_info['result']['list'][0]
    qty_step = float(info['lotSizeFilter']['qtyStep'])  # Minimum qty increment
    min_qty = float(info['lotSizeFilter']['minOrderQty'])  # Minimum allowed qty
    price_precision = int(info['priceScale'])

    while True:
        with capital_lock:
            wait_until_next_candle(15)
            print()
            df = get_klines_df(bybit_symbol, 15)
            df = add_indicators(df)
            atr = df['ATR'].iloc[-1]
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"=== {bybit_symbol} stats at {now} ===")
            print(f"price: {df['Close'].iloc[-1]}")
            print(f"adx: {df['ADX_zone'].iloc[-1]}")
            print(f"rsi: {df['RSI_zone'].iloc[-1]:.2f}")
            print(f"atr: {atr:.6f}")
            bias_ema_crossover = "bullish" if df['EMA_crossover'].iloc[-1] > 0  else "bearish" if df['EMA_crossover'].iloc[-1] < 0 else "neutral"
            print(f"ema7/14/28 crossover above/below: {df['EMA_crossover'].iloc[-1] > 0}/{df['EMA_crossover'].iloc[-1] < 0} ({bias_ema_crossover})")
            bias_macd_signal_line = "bullish" if df['macd_direction'].iloc[-1] > 0 else "bearish" if df["macd_direction"].iloc[-1] < 0 else "neutral"
            print(f"macd zone: {df['macd_zone'].iloc[-1]:.2f} - going up/down: {df['macd_direction'].iloc[-1]} ({bias_macd_signal_line})")
            # print(f"macd line: {df['macd_line'].iloc[-1]:.2f} - going up/down: {df['macd_line_diff'].iloc[-1]:.2f} ({bias_macd_signal_line})")
            bias_osma_diff = "bullish" if df['macd_osma_direction'].iloc[-1] > 0 else "bearish" if df['macd_osma_direction'].iloc[-1] < 0 else "neutral"
            print(f"OSMA zone: {df['macd_osma'].iloc[-1]:.2f} - direction: {df['macd_osma_direction'].iloc[-1]:.2f} ({bias_osma_diff})")
            # di_diff = df['di_diff'].iloc[-1]
            bias_DI_DIff = "bullish" if df['+DI_val'].iloc[-1] > df['-DI_val'].iloc[-1] else "bearish" if df['+DI_val'].iloc[-1] < df['-DI_val'].iloc[-1] else "neutral"
            print(f"+DI_val/-DI_val: {df['+DI_val'].iloc[-1]:.2f}/{df['-DI_val'].iloc[-1]:.2f} ({bias_DI_DIff})")
            print()
    
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

                print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.4f} - PnL: {daily_pnl:.2f} - Balance: {capital:.2f} - Sharpe: {sr:.4f} - Sortino: {sor:.4f}")
                current_day = day
                # capital += daily_pnl

                daily_pnl = 0.0
                daily_trades = 0
                
            # Force close if ADX zone becomes 0
            if df["ADX_zone"].iloc[-1] == 0 and position != 0:
                action = 3
            elif df["ADX_zone"].iloc[-1] == 0:
                action = 0

            # price_range = (df['High'].iloc[-14:].max() - df['Low'].iloc[-14:].min()) / df['Close'].iloc[-1]

            # if price_range < 0.003:
            #     continue

            macd_zone = df.iloc[-1]['macd_zone']
            plus_di = df.iloc[-1]['+DI_val']
            minus_di = df.iloc[-1]['-DI_val']
            bulls = df.iloc[-1]['Bulls']
            bears = df.iloc[-1]['Bears']
            osma_direction = df.iloc[-1]['macd_osma_direction']
            macd_direction = df.iloc[-1]['macd_direction']
            ema_crossover = df.iloc[-1]['EMA_crossover']
            atr = df.iloc[-1]['ATR']
            capital = get_balance()
                    
            # tp_dist = atr * 10
            # sl_dist = atr * 5
            # trailing_sl_dist = atr * 2.5
            tp_dist = df.iloc[-1]['Close'] * 0.0035
            sl_dist = df.iloc[-1]['Close'] * 0.00175
            trailing_sl_dist = df.iloc[-1]['Close'] * 0.000875
            # if df['EMA_7_28_crossover'].iloc[-1] == 1 and df['EMA_7_28_crossover'].iloc[-2] == -1:
            #     position_size = calculate_position_size(capital, 0.05, entry_price, df.iloc[-1]['Close'] * 0.0005, 50)
            #     session.place_order(
            #         category="linear",
            #         symbol=bybit_symbol,
            #         side="Buy",
            #         order_type="Market",
            #         qty=round(position_size, price_precision),
            #         reduce_only=False,
            #         buyLeverage=leverage,
            #         sellLeverage=leverage,
            #         # time_in_force="IOC"
            #         # take_profit=str(round(entry_price + df.iloc[-1]['Close'] * 0.005, 6)),
            #         take_profit=str(round(entry_price + df.iloc[-1]['Close'] + df.iloc[-1]['HL_rolling_mean'], price_precision)),
            #         stop_loss=str(round(entry_price - df.iloc[-1]['Close'] * 0.00175, price_precision))
            #     )
            # if df['EMA_7_28_crossover'].iloc[-1] == -1 and df['EMA_7_28_crossover'].iloc[-2] == 1:
            #     position_size = calculate_position_size(capital, 0.05, entry_price, sl_dist, 50)
            #     session.place_order(
            #         category="linear",
            #         symbol=bybit_symbol,
            #         side="Sell",
            #         order_type="Market",
            #         qty=round(position_size, price_precision),
            #         reduce_only=False,
            #         # time_in_force="IOC"
            #         buyLeverage=leverage,
            #         sellLeverage=leverage,
            #         take_profit=str(round(entry_price + df.iloc[-1]['Close'] - df.iloc[-1]['HL_rolling_mean'], price_precision)),
            #         stop_loss=str(round(entry_price - df.iloc[-1]['Close'] * 0.00175, price_precision))
            #     )
            if position == 0:
                # if action == 1 and macd_direction == 1 and plus_di > minus_di and bulls > 0:
                # if action == 1 and macd_direction == 1 and plus_di > minus_di and bulls > 0 and knn.predict_win_rate(state_seq) > 0.9:
                if action == 1 and knn.predict_win_rate(state_seq) > 0.9 and minus_di < plus_di and bulls > 0 and ema_crossover == 1:
                    if ema_crossover == 1:
                        reward += 1
                    elif ema_crossover == -1:
                        reward -= 1
                    if macd_direction == 1:
                        reward += 1
                    elif macd_direction == -1:
                        reward -= 1
                    if plus_di > minus_di:
                        reward += 1
                    elif plus_di < minus_di:
                        reward -= 1
                    if bulls > 0:
                        reward += 1
                    else:
                        reward -= 1
                    if osma_direction == 1:
                        reward += 1
                    elif osma_direction == -1:
                        reward -= 1
                    entry_price = price
                    entry_seq = state_seq
                    capital = get_balance()
                    if capital < 110:
                        print(f"Free capital under minimum, ${capital:.2f}, waiting until amount is free")
                        continue
                    invest = max(capital * 0.05, 100)
                    if invest < 50:
                        print(f"Free investment amount under minimum, ${invest:.2f}, waiting until amount is free")
                        continue
                    # position_size = calculate_position_size(capital, 0.15 * reward, entry_price, sl_dist, 50)
                    position_size = calculate_position_size(capital, 0.05, entry_price, sl_dist, 50)
                    position = 1
                    in_position = True
                    
                    print(f"[{bybit_symbol}] Entered Buy order")
                    session.place_order(
                        category="linear",
                        symbol=bybit_symbol,
                        side="Buy",
                        order_type="Market",
                        qty=str(round(position_size, price_precision)),
                        reduce_only=False,
                        buyLeverage=leverage,
                        sellLeverage=leverage,
                    )
                    # print(f"trailing sl - entry price: {entry_price}, base price: {entry_price * (1 + 0.000875)}, trailing sl: {entry_price * 0.000875:.2f}")
                    response = session.set_trading_stop(
                        category="linear",  # or "inverse", depending on your market
                        symbol=bybit_symbol,
                        trailing_stop=str(round(trailing_sl_dist, price_precision)),  # Trailing stop in USD or quote currency
                        # side="Buy",
                        # active_price=str(round(entry_price, price_precision)),
                        active_price=str(round(entry_price + trailing_sl_dist, price_precision)),
                        position_idx=0
                    )
                    tp_levels = [
                        entry_price + 0.4 * tp_dist,
                        entry_price + 0.6 * tp_dist,
                        entry_price + 0.8 * tp_dist
                    ]
                    tp_shares = [0.4, 0.2, 0.2]
                    for i in range(3):
                        realized = position_size * tp_shares[i]
                        pnl = calc_order_qty(realized, entry_price, min_qty, qty_step)
                        if pnl == 0:
                            continue
                        close_side = "Sell" if position == 1 else "Buy"
                        response = session.place_order(
                            symbol=bybit_symbol,
                            side=close_side,  # opposite side to close position
                            order_type="Limit",
                            qty=str(round(pnl, price_precision)),
                            price=str(round(tp_levels[i], price_precision)),
                            reduce_only=True
                        )
                        partial_tp_hit = [False, False, False]
                        # position_pct_left = 1.0
                        daily_trades += 1
                                
                # if action == 2 and macd_direction == -1 and minus_di > plus_di and bears > 0 and knn.predict_win_rate(state_seq) > 0.9:
                if action == 2 and knn.predict_win_rate(state_seq) > 0.9 and minus_di > plus_di and bears < 0 and ema_crossover == -1:
                    if ema_crossover == -1:
                        reward += 1
                    elif ema_crossover == 1:
                        reward -= 1
                    if macd_direction == -1:
                        reward += 1
                    elif macd_direction == 1:
                        reward -= 1
                    if plus_di < minus_di:
                        reward += 1
                    elif plus_di > minus_di:
                        reward -= 1
                    if bears < 0:
                        reward += 1
                    elif bulls > 0:
                        reward -= 1
                    if osma_direction == 1:
                        reward += 1
                    elif osma_direction == -1:
                        reward -= 1
                    entry_price = price
                    entry_seq = state_seq
                    # position_size = calculate_position_size(capital, 0.15 * reward, entry_price, sl_dist, 50)
                    capital = get_balance()
                    if capital < 110:
                        print(f"Free capital under minimum, ${capital:.2f}, waiting until amount is free")
                        continue
                    invest = max(capital * 0.05, 100)
                    if invest < 50:
                        print(f"Free investment amount under minimum, ${invest:.2f}, waiting until amount is free")
                        continue
                    position_size = calculate_position_size(capital, 0.05, entry_price, sl_dist, 50)
                    position = -1
                    in_position = True

                    print(f"[{bybit_symbol}] Entered Sell order")
                    session.place_order(
                        category="linear",
                        symbol=bybit_symbol,
                        side="Sell",
                        order_type="Market",
                        qty=str(round(position_size, price_precision)),
                        reduce_only=False,
                        buyLeverage=leverage,
                        sellLeverage=leverage,
                    )
                    # print(f"trailing sl - entry price: {entry_price}, base price: {entry_price * (1 - 0.000875)}, trailing sl: {entry_price * 0.000875:.2f}")
                    response = session.set_trading_stop(
                        category="linear",  # or "inverse", depending on your market
                        symbol=bybit_symbol,
                        trailing_stop=str(round(trailing_sl_dist, price_precision)),  # Trailing stop in USD or quote currency
                        # base_price=str(round(entry_price - atr * 2.5, price_precision)),
                        # base_price=str(round(entry_price, price_precision)),
                        active_price=str(round(entry_price - trailing_sl_dist, price_precision)),
                        position_idx=0
                    )
                    tp_levels = [
                        entry_price - 0.4 * tp_dist,
                        entry_price - 0.6 * tp_dist,
                        entry_price - 0.8 * tp_dist
                    ]
                    tp_shares = [0.4, 0.2, 0.2]
                    for i in range(3):
                        realized = position_size * tp_shares[i]
                        pnl = calc_order_qty(realized, entry_price, min_qty, qty_step)
                        if pnl == 0:
                            continue
                        close_side = "Sell" if position == 1 else "Buy"
                        response = session.place_order(
                            symbol=bybit_symbol,
                            side=close_side,  # opposite side to close position
                            order_type="Limit",
                            qty=str(round(pnl, price_precision)),
                            price=str(round(tp_levels[i], price_precision)),
                            reduce_only=True,
                            buyLeverage=leverage,
                            sellLeverage=leverage,
                        )
                        partial_tp_hit = [False, False, False]
                        position_pct_left = 1.0
                        daily_trades += 1

            if action == 3 and position != 0:
                profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
                pnl = calc_order_qty(profit_pct * position_size, entry_price, min_qty, qty_step)  # not margin
                reward = profit_pct
                # if pnl < 2:
                #     reward -= 1
                # if pnl > 2:
                #     reward += 1
                knn.add(entry_seq, is_win=(profit_pct > 0))
                close_old_orders(bybit_symbol)
                in_position = False
            if action == 0 and position != 0:
                if position == 1:
                    if ema_crossover == 1:
                        reward += 1
                    elif ema_crossover == -1:
                        reward -= 1
                    if macd_direction == 1:
                        reward += 1
                    elif macd_direction == -1:
                        reward -= 1
                    if plus_di > minus_di:
                        reward += 1
                    elif plus_di < minus_di:
                        reward -= 1
                    if bulls > 0:
                        reward += 1
                    elif bears < 0:
                        reward -= 1
                    if osma_direction == 1:
                        reward += 1
                    elif osma_direction == -1:
                        reward -= 1

                if position == -1:
                    if ema_crossover == -1:
                        reward += 1
                    elif ema_crossover == 1:
                        reward -= 1
                    if macd_direction == -1:
                        reward += 1
                    elif macd_direction == 1:
                        reward -= 1
                    if plus_di < minus_di:
                        reward += 1
                    elif plus_di > minus_di:
                        reward -= 1
                    if bears < 0:
                        reward += 1
                    elif bulls > 0:
                        reward -= 1
                    if osma_direction == 1:
                        reward += 1
                    elif osma_direction == -1:
                        reward -= 1
                profit_pct = (entry_price - price) / entry_price if position == 1 else (price - entry_price) / entry_price
                pnl = calc_order_qty(profit_pct * position_size, entry_price, min_qty, qty_step)  # not margin
                reward = profit_pct

        # === Store reward and update step ===
        agent.store_transition(state_seq, action, logprob, value, reward)
        save_counter += 1
        reward = 0
        # if not begun:
        #     for t in range(30, len(df)):
        #         if t % 60 == 0:
        #             print(f"[INFO] Training PPO on step {save_counter}...")
        #             agent.train()
        #             agent.savecheckpoint(symbol)
        #             # print()
        #             print(f"[INFO] Saved checkpoint at step {save_counter}")
        # if save_counter % 10080 == 0:
        if save_counter % 60 == 0:
            print(f"[INFO] Training PPO on step {save_counter}...")
            knn._fit()
            knn.save()
            agent.train()
            agent.savecheckpoint(symbol)
            # print()
            print(f"[INFO] Saved checkpoint at step {save_counter}")
    # rrKNN.train()
    # rrKNN.save()
    # agent.train()
    # agent.savecheckpoint(symbol)
    # print(f"✅ PPO training complete. Final capital: {capital:.2f}, Total PnL: {capital/1000:.2f}")

def main():
    # global lstm_ppo_agent
    counter = 0
    test_threads = []
    train = True
    test = False
    counter = 0
    
    if train:
        for symbol in symbols:
            df = None
            print(f"Initialized looping over symbols, currently at #{counter + 1}, {symbol}")
            if symbol == "XAUUSD":
                df = load_last_mb_xauusd()
            else:
                df = load_last_mb(symbol)
                # continue
            # df = yf_get_ohlc_df(yf_symbol)
            df = df[['Open', "High", "Low", "Close"]]
            df = add_indicators(df)
            lstm_ppo_agent = LSTMPPOAgent(state_size=16, hidden_size=64, action_size=4)
            t = threading.Thread(target=train_bot, args=(df, lstm_ppo_agent, symbols[counter]))
            t.start()
            test_threads.append(t)
            counter += 1
    for t in test_threads:
        t.join()

    # Reset for test phase
    counter = 0
    test_threads = []
    keep_session_alive()
    if test:
        for bybit_symbol in bybit_symbols:
            df = None
            # session = HTTP(demo=True, api_key=api_key, api_secret=api_secret)
            # keep_session_alive()
            df = get_klines_df(bybit_symbol, 1)
            df = add_indicators(df)
            # print(f"columns: {df.columns}")
            lstm_ppo_agent = LSTMPPOAgent(state_size=16, hidden_size=64, action_size=4)
            t = threading.Thread(target=test_bot, args=(df, lstm_ppo_agent, symbols[counter], bybit_symbol))
            t.start()
            test_threads.append(t)
            counter += 1
    for t in test_threads:
        t.join()

main()
