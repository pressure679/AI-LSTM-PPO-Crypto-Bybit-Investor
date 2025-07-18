import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

CSV_PATH = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/BTCUSD_1m_Binance.csv"
Q_TABLE_PATH = "260625-q-learner-trading-bot-q_table.pkl"

ACTIONS = ['long', 'short', 'close', 'hold']
Q = {}
alpha = 0.1
gamma = 0.95
epsilon = 0.1

def load_last_mb(file_path, mb_size=75):
    bytes_to_read = mb_size * 1024 * 1024
    with open(file_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        start_pos = max(0, file_size - bytes_to_read)
        f.seek(start_pos, os.SEEK_SET)
        data = f.read().decode('utf-8', errors='ignore')  # Decode to string

    lines = data.split('\n')
    if start_pos != 0:
        lines = lines[1:]
    lines = [line for line in lines if line.strip() != '']
    df = pd.read_csv(StringIO('\n'.join(lines)), header=None)
    df.columns = [
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ]
        # Convert 'Open time' to datetime and set as index
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)

    # Keep only OHLC columns
    df = df[['Open', 'High', 'Low', 'Close']].copy()

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

def Bollinger_Bands(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
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

def add_indicators(df):
    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_28'] = df['Close'].ewm(span=28).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    df['macd_signal_diff'] = df['macd_signal'].diff()
    df['macd_signal_angle_deg'] = np.degrees(np.arctan(df['macd_signal_diff']))
    # df['macd_histogram_diff'] = df['macd_histogram'].diff()

    df['bb_sma'], df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])
    df['bb_sma_pos'] = (df['Close'] >= df['bb_sma']).astype(int)

    # Custom momentum (% of recent high-low range)
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    price_range = high_14 - low_14

    df['RSI'] = RSI(df['Close'], 14)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    # df['DI_Diff'] = (df['+DI'] - df['-DI']).abs()

    # df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "ATR", "macd_line", "macd_signal", "macd_histogram", "macd_signal_angle_deg", "bb_sma_pos", "RSI", "ADX"]]
    df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_histogram", "macd_signal_angle_deg", "bb_sma_pos", "RSI", "ADX"]].copy()

    df.dropna(inplace=True)
    return df

# ─── Hyperparameters ────────────────────────────────────────────────
STATE_SIZE   = 10       # Number of past time steps for LSTM
FEATURES     = 14        # Number of float indicators (adjust as needed)
ACTIONS      = ['hold', 'long', 'short', 'close']
LR           = 0.001
GAMMA        = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON  = 0.01
BATCH_SIZE   = 32
MEMORY_SIZE  = 1000

# ─── LSTM Model ─────────────────────────────────────────────────────
class LSTM_QNet(nn.Module):
    def __init__(self):
        super(LSTM_QNet, self).__init__()
        self.lstm = nn.LSTM(input_size=FEATURES, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, len(ACTIONS))

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        c0 = torch.zeros(1, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)

# ─── Replay Memory ──────────────────────────────────────────────────
class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# ─── Agent Class ────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.model = LSTM_QNet().to("cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory()
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS)-1)
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))
            return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)

        states, actions, rewards, next_states = zip(*batch)
        # for i, state in enumerate(states):
        #     print(f"State {i} shape: {state.shape}")
        # print(f"df.columns: {df.columns}")
        # print(f"df.iloc[0].index: {df.iloc[0].index}")
        # print(f"df.iloc[1].index: {df.iloc[1].index}")
        states      = torch.stack(states)
        next_states = torch.stack(next_states)
        actions     = torch.tensor(actions).unsqueeze(1).float()
        rewards     = torch.tensor(rewards).float()

        q_values = self.model(states).gather(1, actions.long()).squeeze()
        next_qs  = self.model(next_states).max(1)[0].detach()
        targets  = rewards + GAMMA * next_qs

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# ─── State Construction ─────────────────────────────────────────────
def get_state_sequence(df, idx, window_size=11):
    feature_dim = df.shape[1]
    start_idx = max(0, idx - window_size + 1)
    end_idx = idx + 1  # inclusive of current idx

    seq = df.iloc[start_idx:end_idx].values  # shape (seq_len, feature_dim)
    seq_len = seq.shape[0]

    # Convert to tensor
    tensor_seq = torch.tensor(seq).float()

    if seq_len == window_size:
        # Perfect size, just return
        return tensor_seq

    elif seq_len < window_size:
        # Pad at the front with zeros (or replicate first row if preferred)
        padding = torch.zeros((window_size - seq_len, feature_dim), dtype=torch.float32)
        # Concatenate padding + original sequence
        tensor_seq = torch.cat((padding, tensor_seq), dim=0)
        return tensor_seq

    else:
        # Longer sequence, truncate at the front
        tensor_seq = tensor_seq[-window_size:]
        return tensor_seq

# ─── Main Training Loop ─────────────────────────────────────────────
def train_bot(df):
    agent = Agent()
    capital = 1000.0
    position = 0
    entry_price = 0.0
    invest = 0.0
    peak_capital = capital
    all_pcts = []

    current_day   = None
    daily_pnl     = 0.0

    for i in range(len(df) - 1):
        state      = get_state_sequence(df, i)
        next_state = get_state_sequence(df, i + 1)
        action_idx = agent.act(state)
        action     = ACTIONS[action_idx]
        price      = df.iloc[i]['Close']
        reward     = 0.0
        day = str(df.index[i]).split(' ')[0]

        # ── New day ────────────────────────────────────────────────
        if current_day is None:
            current_day = day
        elif day != current_day:
            print(f"Day {current_day} - PnL: {(daily_pnl/capital)*100:.2f}% - Balance: {capital:.2f}")
            current_day = day
            daily_pnl   = 0.0

        if action == 'long' and position == 0:
            entry_price = price
            invest = max(capital * 0.05, 5)
            position = 1

        elif action == 'short' and position == 0:
            entry_price = price
            invest = max(capital * 0.05, 5)
            position = -1

        elif action == 'close' and position != 0:
            pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            pnl = pct * 50 * invest
            reward = pct
            capital += pnl
            daily_pnl += pnl
            all_pcts.append(pct)
            mean_pct = np.mean(all_pcts) if all_pcts else 0.0

            # Reward shaping
            if pct > mean_pct:
                reward *= 2.0
            elif pct < 0:
                reward *= 2.0  # drawdown or loss penalty

            # 🔻 Drawdown penalty
            if capital > peak_capital:
                peak_capital = capital
            else:
                drawdown = (peak_capital - capital) / peak_capital
                reward -= drawdown  # Scale if needed (e.g., reward -= drawdown * 2)

            position = 0
            invest = 0

        agent.memory.push(state, action_idx, reward, next_state)
        agent.train()

    print(f"Training done. Final balance: {capital:.2f}")

def test_bot(df):
    capital       = 1000.0
    peak_capital  = capital
    position      = 0
    entry_price   = 0.0
    invest        = 0.0
    trades        = 0
    all_pcts      = []
    current_day   = None
    daily_pnl     = 0.0

    print("Testing agent...\n")

    for i in range(len(df) - 11):  # Leave room for state window
        row = df.iloc[i]
        day = str(df.index[i]).split(" ")[0]
        state = get_state_sequence(df, i)  # 10-step window + indicators

        if current_day is None:
            current_day = day
        elif day != current_day:
            print(f"[{current_day}] Daily PnL: {(daily_pnl/capital)*100:.2f} | Balance: {capital:.2f}")
            current_day = day
            daily_pnl = 0.0

        action_idx = agent.act(state, greedy=True)
        action = ACTIONS[action_idx]
        price = row['Close']

        if action == 'long' and position == 0:
            entry_price = price
            invest = max(capital * 0.05, 5)
            position = 1
            trades += 1

        elif action == 'short' and position == 0:
            entry_price = price
            invest = max(capital * 0.05, 5)
            position = -1
            trades += 1

        elif action == 'close' and position != 0:
            pct = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            pnl = pct * 50 * invest
            capital += pnl
            daily_pnl += pnl
            all_pcts.append(pct)
            position = 0
            invest = 0

            print(f"Trade closed: PnL = {pnl:.2f}, Capital = {capital:.2f}")

        # Update peak capital to track max drawdown
        if capital > peak_capital:
            peak_capital = capital

    total_return = (capital - 1000.0) / 1000.0 * 100
    max_drawdown = (peak_capital - capital) / peak_capital * 100

    print("\n📊 FINAL RESULTS 📊")
    print(f"Final Capital: {capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {trades}")
    if all_pcts:
        print(f"Mean Trade PnL: {np.mean(all_pcts) * 100:.2f}%")

if __name__ == "__main__":
    df = load_last_mb(CSV_PATH)
    df = add_indicators(df)
    df = df.astype(float)
    train_bot(df)
    with open(Q_TABLE_PATH, 'wb') as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {Q_TABLE_PATH}")
    test_bot(Q_TABLE_PATH, df)
