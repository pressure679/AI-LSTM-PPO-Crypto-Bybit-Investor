import os
from io import StringIO
import gym
import numpy as np
import pandas as pd
import torch
from gym import spaces
from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

CSV_PATH = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/BTCUSD_1m_Binance.csv"
def load_last_mb(fp, mb_size=25):
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
    return df

EMA  = lambda s,p: s.ewm(span=p,adjust=False).mean()
RSI  = lambda s,p: 100 - 100/(1 + s.diff().clip(lower=0).rolling(p).mean() /
                                -s.diff().clip(upper=0).rolling(p).mean())
def MACD(s,f=12,sl=26,sg=9):
    fast, slow = EMA(s,f), EMA(s,sl)
    macd = fast - slow
    return macd, EMA(macd,sg), macd-EMA(macd,sg)

def ATR(df,p=14):
    high_low = df.High-df.Low
    high_close = (df.High-df.Close.shift()).abs()
    low_close  = (df.Low -df.Close.shift()).abs()
    tr = pd.concat([high_low,high_close,low_close],axis=1).max(axis=1)
    return tr.rolling(p).mean()

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

def add_indicators(df):
    df["EMA_7"], df["EMA_14"], df["EMA_28"] = EMA(df.Close,7), EMA(df.Close,14), EMA(df.Close,28)
    df["macd"], df["macd_signal"], df["OSMA"] = MACD(df.Close)
    df['macd_signal_diff'] = df['macd_signal'].diff()
    df['OSMA_Diff'] = df['OSMA'].diff()
    df["RSI"] = RSI(df['Close'], 14)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    df['DI_Diff'] = (df['+DI'] - df['-DI'])
    df.dropna(inplace=True)
    return df

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=10, initial_capital=1000.0, reward_scaler=0.01):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.reward_scaler = reward_scaler
        self.initial_capital = initial_capital

        self.action_space = spaces.Discrete(4)  # hold, long, short, close
        self.observation_space = spaces.Box(low=-1, high=2, shape=(window_size, 5), dtype=np.int8)

        self.reset()

    def encode_state(self, row):
        ema_cross_7_14_28 = 1 if row['EMA_7'] > row['EMA_14'] > row['EMA_28'] else 0 if row['EMA_7'] < row['EMA_14'] < row['EMA_28'] else -1
        rsi_zone = 0 if row['RSI'] < 30 else 2 if row['RSI'] > 70 else 1
        osma_diff = 0 if row['OSMA_Diff'] < 0 else 1
        di_bucket = 0 if row['Di_Diff'] < 0 else 1
        macd_direction = 0 if row['macd_signal_diff'] < 0 else 1
        return np.array([ema_cross_7_14_28, rsi_zone, osma_diff, di_bucket, macd_direction], dtype=np.int8)

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        return np.array([self.encode_state(row) for _, row in window.iterrows()])

    def reset(self):
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.position = 0      # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.invest = 0
        self.trades = 0
        self.daily_pnl = 0
        return self._get_observation(), {"state": None}

    def step(self, action):
        done = False
        reward = 0
        info = {}
    
        row = self.df.iloc[self.current_step]
        price = row["Close"]

        # Parse and extract current day
        current_date = pd.to_datetime(row["Open time"]).date()

        if action == 0 and self.position == 0:  # Long
            self.entry = price
            self.invest = max(self.capital * 0.05, 5)
            self.position = 1
            self.trades += 1

        elif action == 1 and self.position == 0:  # Short
            self.entry = price
            self.invest = max(self.capital * 0.05, 5)
            self.position = -1
            self.trades += 1

        elif action == 2 and self.position != 0:  # Close
            pct = (price - self.entry) / self.entry if self.position == 1 else (self.entry - price) / self.entry
            pnl = pct * 20 * self.invest  # 20x leverage
            self.capital += pnl
            reward = pnl * self.reward_scaler
            self.daily_pnl += pnl
            self.position = 0
            self.entry = None

        # Print daily balance and pnl if date changed
        next_date = pd.to_datetime(self.df.iloc[self.current_step + 1]["Open time"]).date() \
            if self.current_step + 1 < len(self.df) else current_date

        if next_date != current_date:
            print(f"[{current_date}] Balance: ${self.capital:.2f} | Daily PnL: {self.daily_pnl:.2f}")
            self.daily_pnl = 0  # Reset for new day

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_observation()
        return obs, reward, done, info

CSV_PATH = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/BTCUSD_1m_Binance.csv"
df = load_last_mb(CSV_PATH)
df = add_indicators(df)
env = TradingEnv(df)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)
model.learn(total_timesteps=1000)

