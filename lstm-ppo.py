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
# from sklearn.neighbors import KNeighborsRegressor
import warnings
# import yfinance

warnings.filterwarnings("ignore", category=RuntimeWarning)
# Popular crypto: "DOGEUSDT", "HYPEUSDT", "FARTCOINUSDT", "SUIUSDT", "INITUSDT", "BABYUSDT", "NILUSDT"
yf_symbols = ["BTC-USD", "BNB-USD", "ETH-USD", "XRP-USD", "XAUUSD=X"]
bybit_symbols = ["BTCUSDT", "BNBUSDT", "ETHUSDT", "XRPUSDT", "XAUTUSDT"]
symbols = ["BTCUSD", "BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]

ACTIONS = ['hold', 'long', 'short', 'close']
# alpha = 0.1
# gamma = 0.95
# epsilon = 0.1

# def load_last_mb(filepath, symbol, mb_size=20):
def load_last_mb(filepath, symbol, mb_size=5):
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

    return df

def load_last_mb_xauusd(file_path="/mnt/chromeos/removable/sd_card/XAUUSD_1m_data.csv", mb=3, delimiter=';', col_names=None):
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
    
    df = df.dropna()
    
    return df

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

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

def get_klines_df(symbol, interval, session, limit=240):
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

    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    df['macd_signal_diff'] = df['macd_signal'].diff()
    # df['macd_signal_angle_deg'] = np.degrees(np.arctan(df['macd_signal_diff']))
    # df['macd_histogram_diff'] = df['macd_histogram'].diff()

    # df['bb_sma'], df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])
    # df['bb_sma_pos'] = (df['Close'] >= df['bb_sma']).astype(int)

    # Custom momentum (% of recent high-low range)
    # high_14 = df['High'].rolling(window=14).max()
    # low_14 = df['Low'].rolling(window=14).min()
    # price_range = high_14 - low_14

    df['RSI'] = RSI(df['Close'], 14)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    # df['DI_Diff'] = (df['+DI'] - df['-DI']).abs()

    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_28'] = df['Close'].ewm(span=28).mean()

    df["Bulls"] = df["High"] - df['EMA_14']
    df["Bears"] = df["Low"] - df['EMA_14']
    
    df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_signal_diff", "macd_histogram", "RSI", "ADX", "Bulls", "Bears", "+DI", "-DI", "ATR"]].copy()

    conditions = [
        df['RSI'] < 30,
        (df['RSI'] >= 30) & (df['RSI'] < 50),
        (df['RSI'] <= 50) & (df['RSI'] < 70),
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

    # MACD signal direction: -1 if falling, 1 if rising, 0 if flat
    df['macd_direction'] = np.select(
        [
            df['macd_signal_diff'] < 0,
            df['macd_signal_diff'] > 0
        ],
        [-1, 1],
        default=0
    )
    
    # MACD trend (histogram): -1 bearish, 1 bullish, 0 neutral
    df['Bears'] = np.select(
        [
            df['Bears'] < 0,
            df['Bears'] > 0
        ],
        [-1, 1],
        default=0
    )

    # MACD trend (histogram): -1 bearish, 1 bullish, 0 neutral
    df['Bulls'] = np.select(
        [
            df['Bulls'] < 0,
            df['Bulls'] > 0
        ],
        [-1, 1],
        default=0
    )

    # df["Open"]  = df["Open"] / df["Close"] - 1
    # df["High"]  = df["High"] / df["Close"] - 1
    # df["Low"]   = df["Low"] / df["Close"] - 1
    # df["Close"] = df["Close"].pct_change().fillna(0)  # as return

    df = df[["Open", "High", "Low", "Close", "EMA_crossover", "macd_zone", "macd_direction", "RSI_zone", "ADX_zone", "Bulls", "Bears", "+DI_val", "-DI_val", "ATR"]].copy()

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

def keep_session_alive(session):
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

def get_balance(session):
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

def get_qty_step(symbol, session):
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
            'b_policy': np.zeros(action_size),

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

    def select_action(self, state_seq):
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

            action = np.random.choice(self.action_size, p=probs)
            logprob = np.log(probs[action] + 1e-8)

            return action, logprob, value
        except Exception as e:
            print(f"❌ select_action error: {e}")
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
        # Unpack trajectory
        states, actions, logprobs_old, values, rewards = zip(*self.trajectory)
        # states, actions, logprobs_old, values = zip(*self.trajectory)
        self.trajectory.clear()

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
        # print(f"files: {files}")
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
            break
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


# class RewardRateKNN:
#     def __init__(self, symbol, k=10, reward_threshold=0.003):
#         self.symbol = symbol
#         self.k = k
#         self.threshold = reward_threshold
#         self.model = KNeighborsRegressor(n_neighbors=k)
#         self.X = []
#         self.y = []

#     def add_experience(self, state, reward):
#         self.X.append(state)
#         self.y.append(reward)

#     def train(self):
#         if len(self.X) >= self.k:
#             self.model.fit(self.X, self.y)

#     def should_act(self, current_state):
#         if len(self.X) < self.k:
#             return True
#         predicted_reward = self.model.predict([current_state])[0]
#         return predicted_reward >= self.threshold

#     def save(self, path=None):
#         if path is None:
#             path = f"/mnt/chromeos/removable/sd_card/LSTM-PPO-saves/reward_rate_knn-{self.symbol}.pkl"
#         with open(path, "wb") as f:
#             pickle.dump({
#                 "X": self.X,
#                 "y": self.y,
#                 "model": self.model
#             }, f)

#     def load(self, path=None):
#         if path is None:
#             path = f"/mnt/chromeos/removable/sd_card/LSTM-PPO-saves/reward_rate_knn-{self.symbol}.pkl"
#         try:
#             with open(path, "rb") as f:
#                 data = pickle.load(f)
#                 self.X = data["X"]
#                 self.y = data["y"]
#                 self.model = data["model"]
#         except FileNotFoundError:
#             print(f"No saved KNN found at {path}. Starting fresh.")

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
capital = 96
def train_bot(df, agent, symbol, window_size=20):
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
    # rrKNN = RewardRateKNN(symbol)
    
    for t in range(window_size, len(df) - 1):
        # if position != 0:
        #     action = 3
        # else:
        #     action = 0
        # if df['ADX_zone'].iloc[t] == 0:
        #     continue

        state_seq = df[t - 14:t].values.astype(np.float32)
        if state_seq.shape != (14, agent.state_size):
            print("Shape mismatch:", state_seq.shape)
            continue

        result = agent.select_action(state_seq)
        if result is None:
            continue

        action, logprob, value = result

        # print(f"action: {action}")

        price = df.iloc[t]["Close"]
        next_price = df.iloc[t + 1]["Close"]
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

            print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.4f} - PnL: {daily_pnl:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.4f} - Sortino: {sor:.4f}")
            
            current_day = day
            capital += daily_pnl

            daily_pnl = 0.0

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
        bulls = df.iloc[t]['Bulls']
        bears = df.iloc[t]['Bears']
        atr = df.iloc[t]['ATR']
        
        tp_dist = atr * 3
        sl_dist = atr * 1.5

        final_pct = 0.0

        # qty_step, min_qty = get_qty_step(symbol)

        if position == 0:
            if action == 1 and macd_zone == 1 and plus_di > minus_di and bulls > 0:
                invest = max(capital * 0.05, 15)
                position_size = invest * leverage
                entry_price = price
                capital -= 0.0089
                position = 1
                partial_tp_hit = [False, False, False]
                position_pct_left = 1.0
                daily_trades += 1
                
            elif action == 2 and macd_zone == -1 and minus_di > plus_di and bears > 0:
                invest = max(capital * 0.05, 15)
                position_size = invest * leverage
                entry_price = price
                capital -= 0.0089
                position = -1
                partial_tp_hit = [False, False, False]
                position_pct_left = 1.0
                daily_trades += 1

        elif position == 1:
            profit_pct = (price - entry_price) / entry_price
            pnl = profit_pct * position_size  # not margin
            reward += pnl
            tp_levels = [
                entry_price + 0.4 * tp_dist,
                entry_price + 0.5 * tp_dist,
                entry_price + 0.6 * tp_dist
            ]
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
            tp_shares = [0.4, 0.2, 0.2]

            for i in range(3):
                if not partial_tp_hit[i] and price >= tp_levels[i]:
                    realized = position_size * tp_shares[i]
                    pnl = realized * profit_pct
                    capital += pnl
                    reward += pnl / capital
                    daily_pnl += pnl
                    invest -= realized
                    position_pct_left -= tp_shares[i]
                    partial_tp_hit[i] = True
                    action = 0

            if price >= tp_price or price <= sl_price:
                final_pct = (price - entry_price) / entry_price
                pnl = position_size * final_pct
                capital += pnl
                reward += pnl / capital
                daily_pnl += pnl
                invest = 0
                position = 0
                position_pct_left = 1.0
                partial_tp_hit = [False, False, False]
                action = 3
                    
        elif position == -1:
            profit_pct = (entry_price - price) / entry_price
            pnl = position_size * final_pct
            reward += pnl
            tp_levels = [
                entry_price - 0.4 * tp_dist,
                entry_price - 0.5 * tp_dist,
                entry_price - 0.6 * tp_dist
            ]
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
            tp_shares = [0.4, 0.2, 0.2]
            
            for i in range(3):
                if not partial_tp_hit[i] and price <= tp_levels[i]:
                    realized = position_size * tp_shares[i]
                    pnl = realized * profit_pct
                    capital += pnl
                    reward += pnl / capital
                    daily_pnl += pnl
                    invest -= realized
                    position_pct_left -= tp_shares[i]
                    partial_tp_hit[i] = True
                    action = 0

                if price <= tp_price or price >= sl_price:
                    final_pct = (entry_price - price) / entry_price
                    pnl = position_size * final_pct
                    capital += pnl
                    reward += pnl / capital
                    daily_pnl += pnl
                    invest = 0
                    position = 0
                    position_pct_left = 1.0
                    partial_tp_hit = [False, False, False]
                    action = 3
            
        # === Store reward and update step ===
        agent.store_transition(state_seq, action, logprob, value, reward)
        save_counter += 1
        # if save_counter % 10080 == 0:
        if save_counter % 24 * 60 == 0:
            print(f"[INFO] Training PPO on step {save_counter}...")
            agent.train()
            agent.savecheckpoint(symbol)
            print(f"[INFO] Saved checkpoint at step {save_counter}")
        # print()
    agent.train()
    agent.savecheckpoint(symbol)
    print(f"[INFO] Saved checkpoint at step {save_counter}")
    # rrKNN.train()
    # rrKNN.save()
    print(f"✅ PPO training complete. Final capital: {capital:.2f}, Total PnL: {capital/1000:.2f}")

# Bybit Demo API Key and Secret - eS2OePPbbpRvE1yHck - XFQB3NCBxpyHWxgYv8tef8l7McVcvCxRLR0X
# Bybit API Key and Secret - PoP1ud3PuWajwecc4S - z9RXVMWpiOoE3TubtAQ0UtGx8I5SOiRp1KPU
api_key = "PoP1ud3PuWajwecc4S"
api_secret = "z9RXVMWpiOoE3TubtAQ0UtGx8I5SOiRp1KPU"
def test_bot(df, agent, symbol, bybit_symbol, session, window_size=20):
    global api_key
    global api_secret
    capital_lock = threading.Lock()
    # max_window_size = 240
    # rrKNN = RewardRateKNN(symbol)
    # rrKNN.load()
    agent.loadcheckpoint(symbol)
    session = HTTP(
        api_key=api_key,
        api_secret=api_secret,
        demo=False  # or False for mainnet
    )
    capital = get_balance(session)
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
    # leverage=50
    qty_step, min_qty = get_qty_step(bybit_symbol, session)
    current_day = None
    reward = 0.0
    save_counter = 0
    atr = 0.0
    tp_shares = []
    daily_pnl = 0.0

    while True:
        with capital_lock:
            wait_until_next_candle(1)
            print()
            df = get_klines_df(bybit_symbol, 1, session)
            df = add_indicators(df)
            # print(f'price: {df['Close'].iloc[-1]}')
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"=== {bybit_symbol} stats at {now} ===")
            print(f"['{bybit_symbol}'] price: {df['Close'].iloc[-1]}")
            # print(f"close: {df['Close'].iloc[-1]:.6f}")
            print(f"adx: {df['ADX_zone'].iloc[-1]}")
            print(f"rsi: {df['RSI_zone'].iloc[-1]:.2f}")
            print(f"atr: {atr:.6f}")
            bias_ema_crossover = "bullish" if df['EMA_crossover'].iloc[-1] > 0  else "bearish" if df['EMA_crossover'].iloc[-1] < 0 else "neutral"
            print(f"ema7/14/28 crossover above/below: {df['EMA_crossover'].iloc[-1] > 0}/{df['EMA_crossover'].iloc[-1] < 0} ({bias_ema_crossover})")
            bias_macd_signal_line = "bullish" if df['macd_direction'].iloc[-1] > 0 else "bearish" if df["macd_direction"].iloc[-1] < 0 else "neutral"
            print(f"macd zone: {df['macd_zone'].iloc[-1]:.2f} - going up/down: {df['macd_direction'].iloc[-1]} ({bias_macd_signal_line})")
            # print(f"macd line: {df['macd_line'].iloc[-1]:.2f} - going up/down: {df['macd_line_diff'].iloc[-1]:.2f} ({bias_macd_signal_line})")
            # bias_osma_diff = "bullish" if df['osma_diff'].iloc[-1] > 0 else "bearish" if df['osma_diff'].iloc[-1] < 0 else "neutral"
            # print(f"osma zone: {df['osma'].iloc[-1]:.2f} - direction: {df['osma_diff'].iloc[-1]:.2f} ({bias_osma_diff})")
            # di_diff = df['di_diff'].iloc[-1]
            bias_DI_DIff = "bullish" if df['+DI_val'].iloc[-1] > df['-DI_val'].iloc[-1] else "bearish" if df['+DI_val'].iloc[-1] < df['-DI_val'].iloc[-1] else "neutral"
            print(f"+DI_val/-DI_val: {df['+DI_val'].iloc[-1]:.2f}/{df['-DI_val'].iloc[-1]:.2f} ({bias_DI_DIff})")
            print()
    
            if df['ADX_zone'].iloc[-1] == 0:
                continue
            # state_seq = df[-window_size:].values.astype(np.float32)
            state_seq = df[-14:].values.astype(np.float32)
            # if state_seq.shape != (14, agent.state_size):
            #     print("Shape mismatch:", state_seq.shape)
            #     continue

            result = agent.select_action(state_seq)
            if result is None:
                continue

            action, logprob, value = result

            # print(f"action: {action}")

            price = df.iloc[-1]["Close"]
            # next_price = df.iloc[-1]["Close"]
            day = str(df.index[-1]).split(' ')[0]

            if current_day is None:
                current_day = day
            elif day != current_day:
                # print(f"[{symbol}] Day {current_day} - PnL: {(daily_pnl / capital) * 100:.2f}% - Balance: {capital:.2f}")
                daily_return_pct = (daily_pnl / capital) * 100 if capital != 0 else 0
                daily_returns.append(daily_return_pct)

                avg_profit_per_trade = daily_pnl / daily_trades if daily_trades > 0 else 0
                sr = sharpe_ratio(daily_returns)
                sor = sortino_ratio(daily_returns)

                print(f"[{symbol}] Day {current_day} - Trades: {daily_trades} - Avg Profit: {avg_profit_per_trade:.4f} - PnL: {daily_pnl:.2f}% - Balance: {capital:.2f} - Sharpe: {sr:.4f} - Sortino: {sor:.4f}")
                
                current_day = day
                # capital += daily_pnl

                daily_pnl = 0.0
                
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
            atr = df.iloc[-1]['ATR']
                    
            tp_dist = atr * 3.0
            # tp_dist = df.iloc[-1]['Close'] * 0.006
            sl_dist = atr * 1.5
            # sl_dist = df.iloc[-1]['Close'] * 0.994
            # sl_dist = df.iloc[-1]['Close'] * 0.003

            if position == 0:
                if action == 1 and macd_zone == 1 and plus_di > minus_di and bulls > 0:
                    invest = max(capital * 0.05, 15)
                    position_size = calc_order_qty(float(invest), df['Close'].iloc[-1], min_qty, qty_step)
                    entry_price = price
                    # capital -= 0.0089
                    position = 1
                    print(f"[{bybit_symbol}] Entered Buy order")
                    session.place_order(
                        category="linear",
                        symbol=bybit_symbol,
                        side="Buy",
                        order_type="Market",
                        qty=position_size,
                        reduce_only=False,
                        # time_in_force="IOC"
                        take_profit=str(round(entry_price + tp_dist, 6)),
                        stop_loss=str(round(entry_price - sl_dist, 6))
                    )
                    tp_levels = [
                        entry_price + 0.4 * tp_dist,
                        entry_price + 0.5 * tp_dist,
                        entry_price + 0.6 * tp_dist
                    ]
                    for i in range(3):
                        # if not partial_tp_hit[i] and price >= tp_levels[i]:
                        realized = position_size * tp_levels[i]
                        pnl = calc_order_qty(realized, entry_price, min_qty, qty_step)
                        # capital += pnl
                        # reward += pnl / capital
                        # daily_pnl += pnl
                        # print(f"[{bybit_symbol}] Hit Partial TP {tp_levels[i]:.6f}, realized {pnl:.2f}, balance: {get_balance(session):.2f}")
                        close_side = "Sell" if position == 1 else "Buy"
                        response = session.place_order(
                            symbol=bybit_symbol,
                            side=close_side,  # opposite side to close position
                            order_type="Limit",
                            qty=str(round(pnl, 6)),
                            reduce_only=True,
                            # time_in_force="ImmediateOrCancel"
                            # time_in_force="IOC"
                            # leverage=leverage
                        )
                        # partial_tp_hit = [False, False, False]
                        # position_pct_left = 1.0
                        daily_trades += 1
                                
                elif action == 2 and macd_zone == -1 and minus_di > plus_di and bears > 0:
                    invest = max(capital * 0.05, 15)
                    position_size = calc_order_qty(float(invest), entry_price, min_qty, qty_step)
                    entry_price = price
                    # capital -= 0.0089
                    position = -1
                    print(f"[{bybit_symbol}] Entered Sell order")
                    session.place_order(
                        category="linear",
                        symbol=bybit_symbol,
                        side="Sell",
                        order_type="Market",
                        qty=str(position_size),
                        reduce_only=False,
                        # time_in_force="IOC",
                        take_profit=str(round(entry_price - tp_dist, 6)),
                        stop_loss=str(round(entry_price + sl_dist, 6))
                    )
                    tp_levels = [
                        entry_price - 0.4 * tp_dist,
                        entry_price - 0.5 * tp_dist,
                        entry_price - 0.6 * tp_dist
                    ]
                    for i in range(3):
                        if not partial_tp_hit[i] and price >= tp_levels[i]:
                            realized = position_size * tp_shares[i]
                            pnl = calc_order_qty(realized, entry_price, min_qty, qty_step)
                            # capital += pnl
                            reward += pnl / capital
                            daily_pnl += pnl
                            print(f"[{bybit_symbol}] Hit Partial TP {tp_levels[i]:.6f}, realized {pnl:.2f}, balance: {get_balance(session):.2f}")
                            close_side = "Sell" if position == 1 else "Buy"
                            response = session.place_active_order(
                                symbol=bybit_symbol,
                                side=close_side,  # opposite side to close position
                                order_type="Limit",
                                qty=str(round(pnl, 6)),
                                reduce_only=True,
                                # time_in_force="ImmediateOrCancel"
                                # time_in_force="IOC"
                                # leverage=leverage
                            )
                            partial_tp_hit = [False, False, False]
                            position_pct_left = 1.0
                            daily_trades += 1

            elif position == 1:
                profit_pct = (price - entry_price) / entry_price
                pnl = calc_order_qty(profit_pct * position_size, entry_price, min_size, qty_step)  # not margin
                reward += pnl
                # tp_levels = [
                #     entry_price + 0.4 * tp_dist,
                #     entry_price + 0.5 * tp_dist,
                #     entry_price + 0.6 * tp_dist
                # ]
                # sl_price = entry_price - sl_dist
                # tp_price = entry_price + tp_dist
                # tp_shares = [0.4, 0.2, 0.2]
                
                # for i in range(3):
                #     if not partial_tp_hit[i] and price >= tp_levels[i]:
                #         realized = position_size * tp_shares[i]
                #         pnl = calc_order_qty(realized, entry_price, min_qty, qty_step)
                #         # capital += pnl
                #         reward += pnl / capital
                #         daily_pnl += pnl
                #         print(f"[{bybit_symbol}] Hit Partial TP {tp_levels[i]:.6f}, realized {pnl:.2f}, balance: {get_balance():.2f}")
                #         close_side = "Sell" if position == 1 else "Buy"
                #         response = session.place_active_order(
                #             symbol=symbol,
                #             side=close_side,  # opposite side to close position
                #             order_type="Market",
                #             qty=pnl,
                #             reduce_only=True,
                #             time_in_force="ImmediateOrCancel",
                #             leverage=leverage
                #         )
                #         capital += pnl
                #         position_pct_left -= tp_shares[i]
                #         partial_tp_hit[i] = True
                #         action = 0
                
                # if price >= tp_price or price <= sl_price:
                #     final_pct = (price - entry_price) / entry_price
                #     pnl = position_size * final_pct
                #         pnl = calc_order_qty(position_size * final_pct, entry_price, min_qty, qty_step)
                #     # capital += pnl
                #     reward += pnl / capital
                #     daily_pnl += pnl
                #     print(f"[{bybit_symbol}] Hit SL or TP, pnl pct: {price:6f}, realized {realized:.2f}, balance: {get_balance():.2f}")
                #     close_side = "Sell" if position == 1 else "Buy"
                #     response = session.place_active_order(
                #         symbol=symbol,
                #         side=close_side,  # opposite side to close position
                #         order_type="Market",
                #         qty=pnl,
                #         reduce_only=True,
                #         time_in_force="ImmediateOrCancel",
                #         leverage=leverage
                #     )
                #     invest = 0
                #     position = 0
                #     position_pct_left = 1.0
                #     partial_tp_hit = [False, False, False]
                #     action = 3
                                    
            elif position == -1:
                profit_pct = (entry_price - price) / entry_price
                pnl = calc_order_qty(profit_pct * position_size, entry_price, min_size, qty_step)  # not margin
                reward += pnl
                # tp_levels = [
                # entry_price - 0.4 * tp_dist,
                # entry_price - 0.5 * tp_dist,
                # entry_price - 0.6 * tp_dist
                # ]
                # sl_price = entry_price + sl_dist
                # tp_price = entry_price - tp_dist
                # tp_shares = [0.4, 0.2, 0.2]
                
                # for i in range(3):
                #     if not partial_tp_hit[i] and price <= tp_levels[i]:
                #         realized = position_size * tp_shares[i]
                #         pnl = realized * profit_pct
                #         # capital += realized + pnl
                #         reward += pnl / capital
                #         daily_pnl += pnl
                #         print(f"[{bybit_symbol}] Hit TP {tp_levels[i]}, realized {realized}, balance: {get_balance()}")
                #         close_side = "Sell" if position == 1 else "Buy"
                #         response = session.place_active_order(
                #             symbol=symbol,
                #             side=close_side,  # opposite side to close position
                #             order_type="Market",
                #             qty=pnl,
                #             reduce_only=True,
                #             time_in_force="ImmediateOrCancel",
                #             leverage=leverage
                #         )
                #         invest -= realized
                #         position_pct_left -= tp_shares[i]
                #         partial_tp_hit[i] = True
                #         action = 0
                
                # if price <= tp_price or price >= sl_price:
                #     final_pct = (entry_price - price) / entry_price
                #     pnl = position_size * final_pct
                #     # capital += invest + pnl
                #     reward += pnl / capital
                #     daily_pnl += pnl
                #     print(f"[{bybit_symbol}] Hit SL {price}, realized {pnl}, balance: {get_balance()}")
                #     close_side = "Sell" if position == 1 else "Buy"
                #     response = session.place_active_order(
                #         symbol=symbol,
                #         side=close_side,  # opposite side to close position
                #         order_type="Market",
                #         qty=pnl,
                #         reduce_only=True,
                #         time_in_force="ImmediateOrCancel",
                #         leverage=leverage
                #     )
                #     invest = 0
                #     position = 0
                #     position_pct_left = 1.0
                #     partial_tp_hit = [False, False, False]
                #     action = 3
                
        # === Store reward and update step ===
        agent.store_transition(state_seq, action, logprob, value, reward)
        save_counter += 1
        # if save_counter % 10080 == 0:
        if save_counter % 60 == 0:
            print(f"[INFO] Training PPO on step {save_counter}...")
            agent.train()
            agent.savecheckpoint(symbol)
            # print()
            print(f"[INFO] Saved checkpoint at step {save_counter}")
    # rrKNN.train()
    # rrKNN.save()
    # agent.train()
    # agent.savecheckpoint(symbol)
    print(f"✅ PPO training complete. Final capital: {capital:.2f}, Total PnL: {capital/1000:.2f}")

def main():
    # global lstm_ppo_agent
    counter = 0
    test_threads = []
    train = False
    test = True
    
    if train:
        for symbol in symbols:
            df = None
            print(f"Initialized looping over symbols, currently at #{counter + 1}, {symbol}")
            if symbol == "XAUUSD":
                df = load_last_mb_xauusd("/mnt/chromeos/removable/sd_card/XAUUSD_1m_data.csv")
            else:
                df = load_last_mb("/mnt/chromeos/removable/sd_card", symbol)
            # df = yf.download(yf_symbols[counter], interval="1m", period="7d")
            # df = df[['Open', "High", "Low", "Close"]].values
            df = df[['Open', "High", "Low", "Close"]]
            # df = df.reshape(20, 4)
            df = add_indicators(df)
            # print(f"columns: {df.columns}")
            # df = df.reshape(20, 14)
            lstm_ppo_agent = LSTMPPOAgent(state_size=14, hidden_size=64, action_size=4)
            t = threading.Thread(target=train_bot, args=(df, lstm_ppo_agent, symbol))
            # train_bot(df, lstm_ppo_agent, symbol)
            t.start()
            test_threads.append(t)
            counter += 1
    for t in test_threads:
        t.join()

    # Reset for test phase
    counter = 0
    test_threads = []
    if test:
        for bybit_symbol in bybit_symbols:
            df = None
            session = HTTP(demo=False, api_key=api_key, api_secret=api_secret)
            keep_session_alive(session)
            df = get_klines_df(bybit_symbol, 1, session)
            df = add_indicators(df)
            # print(f"columns: {df.columns}")
            lstm_ppo_agent = LSTMPPOAgent(state_size=14, hidden_size=64, action_size=4)
            t = threading.Thread(target=test_bot, args=(df, lstm_ppo_agent, symbols[counter], bybit_symbol, session))
            t.start()
            test_threads.append(t)
            counter += 1
    for t in test_threads:
        t.join()

main()
