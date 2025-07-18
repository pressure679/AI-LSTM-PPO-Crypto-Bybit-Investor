import os, math, time
from io import StringIO
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import yfinance
from datetime import datetime, timedelta

# CSV_PATH = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/BTCUSD_1m_Binance.csv"
yf_symbols = ["BNB-USD", "ETH-USD", "XRP-USD", "BTC-USD", "XAUUSD=X"]
bybit_symbols = ["BNBUSDT", "ETHUSDT", "XRPUSDT", "BTCUSDT", "XAUUSDT"]
symbols = ["BNBUSD", "ETHUSD", "XRPUSD", "BTCUSD", "XAUUSD"]

scaler = StandardScaler()
knn_memory = []

# ------------------------------------------------------------------#
# DATA LOADING & INDICATORS
# ------------------------------------------------------------------#
def fetch_recent_data_for_training(symbol, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = {}

    today = datetime.now()
    symbol_safe = symbol.replace("=", "")  # Remove Yahoo special chars like "="
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".checkpoint.lstm-ppo.pt") and f"-{symbol_safe}." in f
    ]


    needs_training = True
    for file in checkpoint_files:
        try:
            date_str = file.split("-")[0]
            checkpoint_date = datetime.strptime(date_str, "%Y%m%d")
            if today - checkpoint_date < timedelta(days=90):
                print(f"✅ Checkpoint for {symbol} is recent: {file}")
                needs_training = False
                break
        except Exception as e:
            print(f"⚠️ Skipping file {file}, error parsing date: {e}")
    if not needs_training:
        return None
        # continue


    print(f"⬇️ Fetching new data for {symbol} (no recent checkpoint found)...")
    df = yf.download(symbol, period="3mo", interval="1m")
    if df.empty:
        print(f"❌ No data returned for {symbol}. Skipping.")
        return None
        # continue

    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    results[symbol] = df

    return results

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
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.int64), unit='ms')

    return df

def add_indicators(df):
    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    df['macd_signal_diff'] = df['macd_signal'].diff()
    # df['macd_histogram_diff'] = df['macd_histogram'].diff()
    # df['macd_zone'] = np.where(df['macd_line'] < df['macd_signal'], 0, 1)
    # df['macd_direction'] = np.where(df['macd_signal_diff'] < 0, 0, 1)

    # df['bb_sma'], df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])
    # df['bb_zone'] = np.where(df['Close'] < df['bb_sma'], 0, 1)

    df['ATR'] = ATR(df)

    df['RSI'] = RSI(df['Close'], 14)
    # df['RSI_zone'] = np.where(df['RSI'] < 30, 0, np.where(df['RSI'] > 70, 2, 1))

    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    # df['adx_zone'] = np.where(df['ADX'] < 20, 0, np.where(df['ADX'] > 40, 2, 1))

    df['DI_Diff'] = (df['+DI'] - df['-DI']).abs()
    # df['DI_direction'] = np.where(df['+DI'] > df['-DI'], 1, 0)

    # df['Bulls'] = BullsPower(df)
    # df['Bears'] = BearsPower(df)
    # df['bulls_bears'] = np.where(df['Bulls'] > df['Bears'], 1, 0)

    df = df[["Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_histogram", "RSI", "ADX", "+DI", "-DI"]].copy()

    df.dropna(inplace=True)
    return df

def keep_session_alive(symbol):
    for attempt in range(30):
        try:
            # Example: Get latest position
            result = session.get_positions(category="linear", symbol=symbol)
            break  # If success, break out of loop
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on attempt {attempt+1}, retrying...")
            time.sleep(2)  # wait before retry
        finally:
            threading.Timer(1500, keep_session_alive, args(symbol,)).start()  # Schedule next call

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


def store_trade_result(obs_vector, pnl, side):
    global knn_memory
    side_val = 1 if side == "Buy" else -1
    outcome = 1 if pnl > 0 else 0
    input_vector = np.append(obs_vector, side_val)
    knn_memory.append((input_vector, outcome))

    # Optional: limit memory to prevent bloat
    if len(knn_memory) > 10000:
        knn_memory.pop(0)

def should_take_trade(current_obs, side, k=50, threshold=0.8):
    global knn_memory
    if len(knn_memory) < 1000:
        return True  # not enough data to make a reliable prediction
    side_val = 1 if side == "Buy" else -1
    obs_with_side = np.append(current_obs, side_val)

    # Load features and labels
    X = np.array([v for v, _ in memory])
    y = np.array([o for _, o in memory])

    # Scale and fit
    X_scaled = scaler.fit_transform(X)
    obs_scaled = scaler.transform([obs_with_side])

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_scaled)

    distances, indices = knn.kneighbors(obs_scaled)
    neighbor_outcomes = y[indices[0]]
    win_rate = np.mean(neighbor_outcomes)

    return win_rate >= threshold

def save_knn_memory(symbol, date_str):
    global knn_memory
    filename = f"{symbol}-{date_str}.KNN.pt"
    torch.save(knn_memory, filename)

def load_knn_memory(symbol):
    global knn_memory
    filename = f"{symbol}-{date_str}.KNN.pt"
    try:
        knn_memory = torch.load(filename)
        print(f"[KNN] Memory loaded: {len(knn_memory)} entries from {filename}")
    except FileNotFoundError:
        print(f"[KNN] Memory file not found: {filename}")
        knn_memory = []


class TradingEnv:
    FEATURES = [
        "Open", "High", "Low", "Close", "EMA_7", "EMA_14", "EMA_28", "macd_line", "macd_signal", "macd_histogram", "RSI", "ADX", "+DI", "-DI"
    ]

    def __init__(self, df,
                 window=30,
                 start_balance=1000.0,
                 fee=0.0089,
                 leverage=50,
                 pos_fraction=0.05):
        self.df = df.reset_index(drop=True)
        self.window = window
        self.leverage = leverage
        self.pos_frac = pos_fraction
        self.fee = fee
        # self.target_pct = profit_target_pct
        self.feat_dim = len(self.FEATURES)
        self.start_bal = start_balance
        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        self.idx = 0
        self.done = False
        self.position = 0
        self.entry_price = 0.0
        self.qty = 0.0
        self.balance = self.start_bal
        self.peak_balance = self.start_bal

        z = np.zeros(self.feat_dim, dtype=np.float32)
        self.buffer = deque([z.copy() for _ in range(self.window)],
                            maxlen=self.window)
        return self._obs()

    def _push_obs(self):
        row = self.df.iloc[self.idx]
        if self.position != 0:
            pct_move = (row['Close'] - self.entry_price) / self.entry_price
            if self.position == -1:  # if short, invert move
                pct_move = -pct_move
            upnl_ratio = (pct_move * self.qty * self.leverage) / self.qty
        else:
            upnl_ratio = 0.0
        # vec = np.append(row[self.FEATURES].values, [self.position, upnl_ratio]).astype(np.float32)
        vec = row[self.FEATURES].values.astype(np.float32)
        self.buffer.append(vec)

    def _obs(self):
        self._push_obs()
        return torch.tensor(np.stack(self.buffer), dtype=torch.float32)

    def step(self, action:int):
        price = float(self.df.Close.iloc[self.idx])
        atr   = float(self.df.ATR  .iloc[self.idx])
        reward = 0.0
        current_vec = self.buffer[-1]

        if self.position == 1:  # Long
            if price <= self.sl_price or price >= self.tp_price:
                self._close_pos(price)

        elif self.position == -1:  # Short
            if price >= self.sl_price or price <= self.tp_price:
                self._close_pos(price)

        if action == 1 and self.position == 0:
            # side = "Buy" if action == 1 else "Sell" if action == 2
            if should_take_trade(self.buffer[-1], "Buy"):
                self._open_pos(+1, price, atr)
        elif action == 2 and self.position == 0:
            # side = "Buy" if action == 1 else "Sell" if action == 2
            if should_take_trade(self.buffer[-1], "Sell"):
                self._open_pos(-1, price, atr)
        elif action == 3 and self.position != 0:
            
            reward = self._close_pos(price) 
        # action==0: hold

        self.idx += 1
        if self.idx >= len(self.df) - 1:
            self.done = True
            
        return self._obs(), reward, self.done, {"balance": self.balance}

    def _open_pos(self, side:int, price:float, atr:float):
        """
        side : +1 (long) or -1 (short)
        price: entry price
        atr  : current ATR value (must be finite & >0)
        """
        # -------- SL / TP distances -------------------------------
        # sl_dist = atr * 1.5
        # tp_dist = atr * 3.0

        equity_dd     = (self.peak_balance - self.balance) / self.peak_balance
        eff_frac      = self.pos_frac * max(1.0, 1.0 + equity_dd)  # risk up in DD
        risk_dollar   = self.balance * eff_frac
        self.qty      = max(risk_dollar, 15)               # enforce min size

        self.position    = side
        self.entry_price = price
        # if side == 1:
        #     self.sl_price = price - sl_dist
        #     self.tp_price = price + tp_dist
        # else:
        #     self.sl_price = price + sl_dist
        #     self.tp_price = price - tp_dist
        #     self.balance -= self.fee
        self.balance -= self.fee
        
    def _close_pos(self, price:float):
        pct_move = (price - self.entry_price) / self.entry_price
        pnl      = pct_move * self.qty * self.leverage * (1 if self.position==1 else -1)
        self.balance += pnl - self.fee
        # reward = pnl / self.qty
        side = "Buy" if self.position == 1 else "Sell"
        # store_trade_result(obs_vector, pnl, side)
        store_trade_result(obs_vector, pnl, side)
        reward = pct_move

        self.position = 0
        self.entry_price = self.qty = 0.0
        self.sl_price = self.tp_price = np.nan
        return reward

class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, action_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor  = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def reset_hidden(self, batch_size=1, device="cpu"):
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(self, x, hidden=None):
        """
        x shape accepted:
          • (batch, seq_len, feat)   — normal batched input
          • (seq_len, feat)          — single trajectory (adds batch dim)
        """
        # ---- ensure 3‑D [batch, seq_len, feat] ---------------------
        if x.dim() == 2:            # (seq_len, feat)
            x = x.unsqueeze(0)      # -> (1, seq_len, feat)
        elif x.dim() == 1:          # rare case (feat,)
            x = x.unsqueeze(0).unsqueeze(0)

        # ---- init hidden state if not provided ---------------------
        if hidden is None:
            batch = x.size(0)
            h0 = x.new_zeros(1, batch, self.hidden_dim)
            c0 = x.new_zeros(1, batch, self.hidden_dim)
            hidden = (h0, c0)

        # ---- forward through LSTM ---------------------------------
        out, hidden = self.lstm(x, hidden)         # out: (batch, seq_len, hidden)
        out = out[:, -1, :]                        # last time step

        # ---- actor & critic heads ---------------------------------
        logits = self.actor(out)                   # (batch, action_dim)
        value  = self.critic(out).squeeze(-1)      # (batch,)

        return logits, value, hidden


CHK_DIR = "checkpoints"
def save_checkpoint(policy, optimizer, episode, symbol):
    os.makedirs(CHK_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = os.path.join(CHK_DIR, f"{symbol}-{date_str}.data.pt")
    
    torch.save({
        "episode": episode,
        "policy_state": policy.state_dict(),
        "optim_state" : optimizer.state_dict(),
    }, path)

    print(f"[✔] saved checkpoint → {path} (ep {episode})")

def load_checkpoint(policy, optimizer, symbol, device="cpu"):
    # Find the most recent checkpoint for this symbol
    if not os.path.isdir(CHK_DIR):
        raise FileNotFoundError(f"Checkpoint directory '{CHK_DIR}' not found.")

    symbol_ckpts = [
        f for f in os.listdir(CHK_DIR)
        if f.startswith(symbol) and f.endswith(".data.pt")
    ]

    if not symbol_ckpts:
        raise FileNotFoundError(f"No checkpoint found for symbol '{symbol}'.")

    # Sort by date in filename (YYYYMMDD)
    symbol_ckpts.sort(reverse=True)
    latest_ckpt = os.path.join(CHK_DIR, symbol_ckpts[0])

    ckpt = torch.load(latest_ckpt, map_location=device)
    policy.load_state_dict(ckpt["policy_state"])
    optimizer.load_state_dict(ckpt["optim_state"])

    print(f"[✔] loaded checkpoint from → {latest_ckpt} (ep {ckpt['episode']})")
    return ckpt["episode"] + 1


def ppo_loss(old_logp, new_logp, adv, eps=0.2):
    ratio = torch.exp(new_logp - old_logp)
    return -(torch.min(ratio*adv, torch.clamp(ratio,1-eps,1+eps)*adv)).mean()

def compute_gae(rew, val, msk, γ=0.99, λ=0.95):
    val = torch.cat([val, torch.zeros(1, 1, device=val.device)])
    gae, ret = 0, []
    for i in reversed(range(len(rew))):
        delta = rew[i] + γ*val[i+1]*msk[i] - val[i]
        gae   = delta + γ*λ*msk[i]*gae
        ret.insert(0, gae + val[i])
    return torch.stack(ret)

def train_ppo(env, policy, optim, symbol,
              # episodes      = 20,
              episodes      = 20,
              # rollout_steps = 512,   # collect at most this many steps before an update
              rollout_steps = 512,   # collect at most this many steps before an update
              gamma         = 0.99,
              lam           = 0.95,
              clip_eps      = 0.2,
              epochs        = 2,
              # mb_size       = 256,
              mb_size       = 64,
              device        = "cpu"):

    policy.to(device)

    total_t = 0

    day_start_bal    = env.balance        # balance after last daily print
    week_start_bal   = env.balance        # balance after last weekly print
    month_start_bal  = env.balance        # balance after last monthly print
    episodes_per_day = 1    # ≈ one trading “day” if you like; tweak
    episodes_per_week  = 7
    episodes_per_month = 30
    
    for ep in range(1, episodes+1):
        S,A,LOGP,V,R,M = [],[],[],[],[],[]
        s   = env.reset()
        h   = policy.reset_hidden(device=device)
        done= False

        while len(R) < rollout_steps:
            logits, v, h  = policy(s.to(device), h)
            dist          = Categorical(logits=logits)
            a             = dist.sample()
            s2, r, done,_ = env.step(a.item())
            store_trade_result(s.squeeze(0).cpu().numpy(), r, a.item())

            S.append(s.squeeze(0))
            A.append(a)
            LOGP.append(dist.log_prob(a))
            V.append(v.squeeze(0))
            R.append(torch.tensor(r, device=device, dtype=torch.float32))
            M.append(torch.tensor(1-done, device=device, dtype=torch.float32))

            s = s2
            if done:
                s  = env.reset()
                h  = policy.reset_hidden(device=device)
                done = False

        total_t += len(R)
        V.append(torch.zeros(1, device=device))        # bootstrap 0
        gae, adv, ret = 0, [], []
        for i in reversed(range(len(R))):
            delta = R[i] + gamma * V[i+1] * M[i] - V[i]
            gae   = delta + gamma * lam * M[i] * gae
            adv.insert(0, gae)
            ret.insert(0, gae + V[i])
        ADV  = torch.stack(adv)
        RET  = torch.stack(ret)
        LOGP = torch.stack(LOGP).detach()
        A    = torch.stack(A)
        S    = torch.stack(S)

        ADV = (ADV-ADV.mean())/(ADV.std()+1e-8)         # normalize

        nbatch = len(S)
        idxs   = np.arange(nbatch)
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, nbatch, mb_size):
                mb = idxs[start : start + mb_size]

                batch_states = S[mb].to(device)          # (mb, window, feat)
                logits_mb, v_mb, _ = policy(batch_states)

                dist_mb   = Categorical(logits=logits_mb)
                logp_mb   = dist_mb.log_prob(A[mb].to(device))

                adv_mb  = ADV[mb].detach()
                ret_mb  = RET[mb].squeeze(-1).detach()      # <- size match

                ratio   = torch.exp(logp_mb - LOGP[mb])
                surr1   = ratio * adv_mb
                surr2   = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_mb
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v  = F.mse_loss(v_mb.squeeze(-1), ret_mb)

                loss    = loss_pi + 0.5 * loss_v - 0.01 * dist_mb.entropy().mean()

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()

        if (ep + 1) % 5 == 0:
            ep_ret = RET.sum().item()
            print(f"Ep {ep + 1:03d} | rollout {nbatch:4d} steps | episodic return {ep_ret:8.4f}")
            save_checkpoint(policy, optim, ep + 1)
        if (ep + 1) % episodes_per_day == 0:
            roi_day = (env.balance - day_start_bal) / day_start_bal * 100
            print(f"🗓  Day ROI: {roi_day:+6.2f}%  (Bal {env.balance:.2f})")
            day_start_bal = env.balance      # reset baseline

        if (ep + 1) % episodes_per_week == 0:
            roi_week = (env.balance - week_start_bal) / week_start_bal * 100
            print(f"🗓  Week ROI: {roi_week:+6.2f}%  (Bal {env.balance:.2f})")
            week_start_bal = env.balance      # reset baseline

        if (ep + 1) % episodes_per_month == 0:
            roi_month = (env.balance - month_start_bal) / month_start_bal * 100
            print(f"📆  Month ROI: {roi_month:+6.2f}% (Bal {env.balance:.2f})")
            month_start_bal = env.balance     # reset baseline
    save_tree_knn_memory(symbol) 

def evaluate_policy_live(
    session,
    symbol,
    device: str = "cpu",
    verbose: bool = False,
    risk_pct: float = 0.05,
    interval_minutes: int = 1,
):
    """
    Evaluates the trained PPO + LSTM policy in live mode on Bybit.

    Action mapping:
    0 - Hold
    1 - Open Long
    2 - Open Short
    3 - Close Position
    """

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    ckpt_path = os.path.join(CHK_DIR, f"{symbol}-{date_str}.data.pt")
    print(f"\n🚀 Starting LIVE trading on {symbol.upper()} using {ckpt_path}")

    # Load policy
    input_dim = len(TradingEnv.FEATURES)
    policy = LSTMPolicy(input_dim=input_dim, hidden_dim=64, action_dim=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    load_tree_knn_memory(symbol)

    hidden = None
    long_open = False
    short_open = False

    while True:
        wait_until_next_candle(interval_minutes)

        # Fetch and process latest data
        df = get_klines_df(symbol + "T", interval="1", limit=2880)
        df = add_indicators(df)
        env = TradingEnv(df)
        obs = env.reset()

        # Calculate qty based on risk and ATR
        latest = df.iloc[-1]
        risk_amount = max(get_balance(session) * risk_pct, 15)
        atr = latest['ATR']
        qty_step, min_qty = get_qty_step(symbol)
        qty = calc_order_qty(risk_amount, latest['Close'], min_qty, qty_step)

        with torch.no_grad():
            logits, _, hidden = policy(obs.to(device), hidden)
            action = logits.argmax(-1).item()

        obs, _, _, info = env.step(action)

        # Execute actions
        if action == 1 and not long_open:
            print(f"[{datetime.datetime.now()}] 🟢 Opening LONG")
            session.place_order(
                category="linear",
                symbol=symbol,
                side="Buy",
                order_type="Market",
                qty=qty,
                reduce_only=False,
                time_in_force="IOC"
            )
            long_open = True
            short_open = False

        elif action == 2 and not short_open:
            print(f"[{datetime.datetime.now()}] 🔻 Opening SHORT")
            session.place_order(
                category="linear",
                symbol=symbol,

                side="Sell",
                order_type="Market",
                qty=qty,
                reduce_only=False,
                time_in_force="IOC"
            )
            short_open = True
            long_open = False

        elif action == 3 and (long_open or short_open):
            close_side = "Sell" if long_open else "Buy"
            print(f"[{datetime.datetime.now()}] 🔴 Closing {'LONG' if long_open else 'SHORT'}")
            session.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                order_type="Market",
                qty=qty,
                reduce_only=True,
                time_in_force="IOC"
            )
            long_open = False
            short_open = False

        elif action == 0:
            if verbose:
                print(f"[{datetime.datetime.now()}] ⏸ HOLD")
            
def main():
    # df = add_indicators(load_last_mb(CSV_PATH))
    counter = 0
    for symbol in symbols:
        df = fetch_recent_data_for_training(yf_symbols[counter])
        if df == None:
            continue
        df = add_indicators(df)
        env     = TradingEnv(df)
        policy  = LSTMPolicy()
        optim   = torch.optim.Adam(policy.parameters(), lr=3e-4)
        train_ppo(env, policy, optim, symbol, episodes=10, device="cpu")
        counter += 1
    api_key = "8g4j5EW0EehZEbIaRD"
    api_secret = "ZocPJZUk8bTgNZUUkPfERCLTg001IY1XCCR4"
    session = HTTP(demo=True, api_key=api_key, api_secret=api_secret)
    keep_session_alive("BTCUSDT")
    # for symbol in bybit_symbols:
    for symbol in symbols:
        evaluate_policy(session, symbol)
main()
