import pandas as pd
import numpy as np
import os
import pickle
from io import StringIO

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
    df['EMA_7_Diff'] = df['EMA_7'].diff()
    df['EMA_14_Diff'] = df['EMA_14'].diff()
    df['EMA_28_Diff'] = df['EMA_28'].diff()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    df['macd_signal_diff'] = df['macd_signal'].diff()
    df['macd_signal_angle_deg'] = np.degrees(np.arctan(df['macd_signal_diff']))
    df['macd_histogram_diff'] = df['macd_histogram'].diff()

    df['bb_sma'], df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])
    df['bb_sma_pos'] = (df['Close'] >= df['bb_sma']).astype(int)

    # Custom momentum (% of recent high-low range)
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    price_range = high_14 - low_14

    df['RSI'] = RSI(df['Close'], 14)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    df['DI_Diff'] = (df['+DI'] - df['-DI']).abs()

    df.dropna(inplace=True)
    return df

def encode_state(row):
    """
    Turn a single indicator row into a compact, hashable tuple for the Q‑table.
    """

    # 1) EMA alignment 7 > 14 > 28
    ema_cross_7_14_28 = (
        1 if row['EMA_7'] > row['EMA_14'] > row['EMA_28'] 
        else 0 if row['EMA_7'] < row['EMA_14'] < row['EMA_28'] 
        else -1
    )

    # 2) Bollinger position
    bb_zone = 0 if row['Close'] < row['bb_sma'] else 1

    # 3) RSI zone
    rsi_zone = 0 if row['RSI'] < 30 else 2 if row['RSI'] > 70 else 1

    # 4) ADX strength
    adx_zone = 0 if row['ADX'] < 20 else 2 if row['ADX'] > 40 else 1

    # 6) MACD trend (fast > signal?)
    macd_histogram_diff = 0 if row['macd_histogram_diff'] < 0 else 1

    # 7) DI_Diff bucket (trend strength)
    di_bucket = 0 if row['DI_Diff'] < 0 else 1

    bulls_bears = 1 if row['Bulls'] > 0 else 0

    # macd_trend = 0 if row['macd_line'] < row['macd_signal'] else 1
    macd_trend = 0 if row['macd_signal'] < 0 else 1

    # 13) Macd signal line diff
    macd_direction = 0 if row['macd_signal_angle_deg'] < 0 else 1

    # Assemble the final discrete state
    return (
        ema_cross_7_14_28,    # 0 / 1
        bb_zone,              # 0 / 1 / 2
        rsi_zone,             # 0 / 1 / 2
        adx_zone,             # 0 / 1 / 2
        macd_histogram_diff,  # 0 / 1
        di_bucket,            # 0 / 1 / 2
        macd_trend,
        macd_direction
    )
        
def update_q(s, a, reward, s_next):
    if s not in Q:
        Q[s] = np.zeros(len(ACTIONS))
    if s_next not in Q:
        Q[s_next] = np.zeros(len(ACTIONS))
    idx = ACTIONS.index(a)
    Q[s][idx] += alpha * (reward + gamma * max(Q[s_next]) - Q[s][idx])

def choose_action(s):
    if s not in Q or np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    return ACTIONS[np.argmax(Q[s])]

def train_bot(df):
    """Q‑learning trainer
       • reward = PnL on each closed trade (leveraged)
       • extra boost if a trade’s PnL exceeds running mean PnL
       • penalty if drawdown occurs (capital falls below peak)
    """
    global Q

    capital   = 1000.0
    position  = 0
    entry     = 0.0
    invest    = 0.0
    trades    = 0

    current_day   = None
    daily_pnl     = 0.0
    reward_scaler = 1.0

    # ── NEW ──────────────────────────────────────────────────────────
    peak_capital   = capital     # for drawdown tracking
    all_trade_pcts = []          # store every trade’s raw PnL
    mean_pct       = 0.0         # running average

    for i in range(len(df) - 1):
        row, nxt = df.iloc[i], df.iloc[i + 1]
        day = str(row['Open time']).split(' ')[0]

        # ── New day ────────────────────────────────────────────────
        if current_day is None:
            current_day = day
        elif day != current_day:
            print(f"Day {current_day} - PnL: {(daily_pnl/capital)*100:.2f}% - Balance: {capital:.2f}")
            current_day = day
            daily_pnl   = 0.0

        # ── Action logic ───────────────────────────────────────────
        state      = encode_state(row)
        next_state = encode_state(nxt)
        action     = choose_action(state)
        reward     = 0.0
        price      = row['Close']

        if action == 'long' and position == 0:
            entry    = price
            invest   = max(capital * 0.05, 5)
            position = 1
            trades  += 1

        elif action == 'short' and position == 0:
            entry    = price
            invest   = max(capital * 0.05, 5)
            position = -1
            trades  += 1

        if position != 0:
            pct = (price - entry) / entry if position == 1 else (entry - price) / entry
            reward = pct * reward_scaler

        if action == "close" and position != 0:
            pct  = (price - entry) / entry if position == 1 else (entry - price) / entry
            pnl  = pct * 50 * invest  # leverage
            reward = pct * reward_scaler

            capital   += pnl
            daily_pnl += pnl
            position   = 0
            all_trade_pcts.append(pct)
            mean_pct = np.mean(all_trade_pcts)

            # ── Above-average reward boost ────────────────────────
            if pct > mean_pct:
                reward *= 1.5
            if pct < 0:
                reward *= 2

        # ── Drawdown penalty ─────────────────────────────────
        if capital > peak_capital:
            peak_capital = capital
        else:
            drawdown = (peak_capital - capital) / peak_capital
            # Apply penalty proportional to drawdown (scaling optional)
            reward -= drawdown  # e.g., 5% drawdown → reward -= 0.05

        # ── Q‑update ─────────────────────────────────────────────
        update_q(state, action, reward, next_state)

    print(f"Training finished. Trades: {trades}, Final balance: {capital:.2f}")

def test_bot(qtable_path, df):
    with open(qtable_path, "rb") as f:
        q_table = pickle.load(f)

    balance = 1000.0
    position = None
    trades = 0

    daily_balances = {}
    current_day = None

    for i in range(len(df)):
        row = df.iloc[i]
        state = encode_state(row)
        price = row['Close']
        day = str(row['Open time']).split(' ')[0]
        
        # Calculate daily balances
        if current_day is None:
            current_day = day
            daily_balances[current_day] = balance
        elif day != current_day:
            pnl = balance - daily_balances[current_day]
            print(f"Day: {current_day} - PnL: {(pnl/balance)*100:.2f}% - Balance: {balance:.2f}")
            current_day = day
            daily_balances[current_day] = balance

        # Get action from Q-table
        if state in q_table:
            action = np.argmax(q_table[state])
        else:
            action = 0  # Default to Hold

        investment = max(balance * 0.05, 5)

        # ---------- ACTION 1: Buy ----------
        if action == 1:
            if position is None:
                position = {
                    'entry_price': price,
                    'type': 'long',
                    'entry_step': i,
                    'margin': investment
                }
                trades += 1
            # elif position['type'] == 'short':
            #     entry_price = position['entry_price']
            #     invest = position['margin']
            #     profit = ((entry_price - price) / entry_price) * invest * 50
            #     balance += profit
            #     position = {
            #         'entry_price': price,
            #         'type': 'long',
            #         'entry_step': i,
            #         'margin': investment
            #     }
            #     trades += 1

        # ---------- ACTION 2: Sell ----------
        elif action == 2:
            if position is None:
                position = {
                    'entry_price': price,
                    'type': 'short',
                    'entry_step': i,
                    'margin': investment
                }
                trades += 1
            # elif position['type'] == 'long':
            #     entry_price = position['entry_price']
            #     invest = position['margin']
            #     profit = ((price - entry_price) / entry_price) * invest * 50
            #     balance += profit
            #     position = {
            #         'entry_price': price,
            #         'type': 'short',
            #         'entry_step': i,
            #         'margin': investment
            #     }
            #     trades += 1

        # ---------- ACTION 3: Close ----------
        elif action == 3 and position is not None:
            entry_price = position['entry_price']
            invest = position['margin']
            if position['type'] == 'long':
                profit = ((price - entry_price) / entry_price) * invest * 50
            else:
                profit = ((entry_price - price) / entry_price) * invest * 50
            balance += profit
            # print(f"Closed {position['type']} | PnL: {profit:.2f} | Balance: {balance:.2f}")
            position = None
            trades += 1

        # ---------- ACTION 0: Hold ----------
        # No-op

    # ---------- FINAL POSITION CLOSE ----------
    if position is not None:
        entry_price = position['entry_price']
        invest = position['margin']
        price = df.iloc[-1]['Close']
        if position['type'] == 'long':
            profit = ((price - entry_price) / entry_price) * invest * 50
        else:
            profit = ((entry_price - price) / entry_price) * invest * 50
        balance += profit
        print(f"Final balance after closing position: {balance:.2f}")
        position = None

    # Final day PnL
    if current_day is not None:
        pnl = balance - daily_balances[current_day]
        print(f"Day: {current_day} - PnL: {pnl:.2f} - Balance: {balance:.2f}")

    print(f"Total trades: {trades}")
    print(f"Final balance: {balance:.2f}")
    return balance, trades
if __name__ == "__main__":
    df = load_last_mb(CSV_PATH)
    df = add_indicators(df)
    train_bot(df)
    with open(Q_TABLE_PATH, 'wb') as f:
        pickle.dump(Q, f)
    print(f"Q-table saved to {Q_TABLE_PATH}")
    test_bot(Q_TABLE_PATH, df)
