import os, math, time
from io import StringIO
import json
import numpy as np
import pandas as pd
import joblib
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

CSV_PATH = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/BTCUSD_1m_Binance.csv"
memory = []

# ------------------------------------------------------------------#
# DATA LOADING & INDICATORS
# ------------------------------------------------------------------#
def load_last_mb(fp, mb_size=100):
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

def BullsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['High'] - ema

def BearsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['Low'] - ema

def add_indicators(df):
    df["EMA_7"], df["EMA_14"], df["EMA_28"] = EMA(df.Close,7), EMA(df.Close,14), EMA(df.Close,28)
    df["macd"], df["macd_signal"], df["OSMA"] = MACD(df.Close)
    df['macd_signal_diff'] = df['macd_signal'].diff()
    df['OSMA_Diff'] = df['OSMA'].diff()
    df["RSI"] = RSI(df['Close'], 14)
    df['ATR'] = ATR(df)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    df['DI_Diff'] = (df['+DI'] - df['-DI'])
    df['Bulls'] = BullsPower(df)     # High – EMA(close, 13)
    df['Bears'] = BearsPower(df)     # Low  – EMA(close, 13)
    df['Bull_Bear_Diff'] = (df['Bulls'] - df['Bears'])
    df.dropna(inplace=True)
    return df

def store_trade_result(obs_vector, pnl, side):
    outcome = 1 if pnl > 0 else 0
    memory.append((obs_vector, outcome))

    # Optional: limit memory to prevent bloat
    if len(memory) > 1000:
        memory.pop(0)

def should_take_trade(current_obs, k=10, threshold=0.7):
    if len(memory) < 100:
        return True  # Not enough history yet
    
    # Split stored memory into feature matrix and outcomes
    X = np.array([v for v, _ in memory])
    y = np.array([o for _, o in memory])
    
    # Train KNN model
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    
    # Find k-nearest neighbors
    distances, indices = knn.kneighbors([current_obs])
    neighbor_outcomes = y[indices[0]]
    
    # Calculate win ratio
    win_rate = np.mean(neighbor_outcomes)
    
    return win_rate >= threshold


class TradingEnv:
    FEATURES = [
        "Close", "ATR", "macd_signal", "macd_signal_diff", "OSMA", "OSMA_Diff", "RSI",
        "ADX", "+DI", "-DI", "DI_Diff", "Bulls", "Bears", "Bull_Bear_Diff", "EMA_7", "EMA_14", "EMA_28"
    ]

    def __init__(self, df,
                 window=30,
                 start_balance=100.0,
                 fee=0.0089,
                 leverage=50,
                 pos_fraction=0.1,
                 profit_target_pct=0.10):
        self.df = df.reset_index(drop=True)
        self.window = window
        self.leverage = leverage
        self.pos_frac = pos_fraction
        self.fee = fee
        self.target_pct = profit_target_pct
        self.feat_dim = len(self.FEATURES) + 2
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
        vec = np.append(row[self.FEATURES].values, [self.position, upnl_ratio]).astype(np.float32)
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
            self._open_pos(+1, price, atr)
        elif action == 2 and self.position == 0:
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
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0

        equity_dd     = (self.peak_balance - self.balance) / self.peak_balance
        eff_frac      = self.pos_frac * max(1.0, 1.0 + equity_dd)  # risk up in DD
        risk_dollar   = self.balance * eff_frac
        self.qty      = max(risk_dollar, 5)               # enforce min size

        self.position    = side
        self.entry_price = price
        if side == 1:
            self.sl_price = price - sl_dist
            self.tp_price = price + tp_dist
        else:
            self.sl_price = price + sl_dist
            self.tp_price = price - tp_dist
            self.balance -= self.fee
        
    def _close_pos(self, price:float):
        pct_move = (price - self.entry_price) / self.entry_price
        pnl      = pct_move * self.qty * self.leverage * (1 if self.position==1 else -1)
        self.balance += pnl - self.fee
        reward = pnl / self.qty

        self.position = 0
        self.entry_price = self.qty = 0.0
        self.sl_price = self.tp_price = np.nan
        return reward

class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, action_dim=4):
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
os.makedirs(CHK_DIR, exist_ok=True)
def save_checkpoint(policy, optimizer, episode, path=f"{CHK_DIR}/ppo_lstm.pt"):
    torch.save({
        "episode": episode,
        "policy_state": policy.state_dict(),
        "optim_state" : optimizer.state_dict(),
    }, path)
    # joblib.dump((self.scaler, self.tree), "tree.pkl")
    print(f"[✔] saved checkpoint → {path} (ep {episode})")

def load_checkpoint(policy, optimizer, path=f"{CHK_DIR}/ppo_lstm.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[✔] loaded checkpoint from ep {ckpt['episode']}")
    return ckpt["episode"] + 1            # resume from next episode


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


def train_ppo(env, policy, optim,
              episodes      = 20,
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
            print(f"Ep {ep + 1:03d} | rollout {nbatch:4d} steps | episodic return {ep_ret:8.4f}")
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

def evaluate_policy(df, env,
        ckpt_path:str = "checkpoints/ppo_lstm.pt",
        device:str = "cpu",
        verbose:bool = False):
    """
    Runs the trained PPO + LSTM policy over the CSV data once, no learning.

    Prints final balance and basic trade stats.
    """
    # ---------- load data & indicators -------------------------
    # df = add_indicators(load_last_mb(CSV_PATH))   # or full CSV
    # env = TradingEnv(df)

    # ---------- load policy ------------------------------------
    input_dim = len(TradingEnv.FEATURES) + 2
    policy = LSTMPolicy(input_dim=input_dim, hidden_dim=64, action_dim=4).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    # ---------- bookkeeping ------------------------------------
    hidden        = None
    done          = False
    step_counter  = 0
    day_steps     = 60 * 24
    week_steps    = 60 * 24 * 7            # 10080
    month_steps   = 60 * 24 * 30           # 43200
    day_steps     = 60 * 24
    day_start     = env.balance
    week_start    = env.balance
    month_start   = env.balance

    obs = env.reset()
    hidden = None
    done = False

    # ---------- loop -------------------------------------------
    while not done:
        with torch.no_grad():
            logits, _, hidden = policy(obs.to(device), hidden)
            action            = logits.argmax(-1).item()
        obs, _, done, info = env.step(action)
        step_counter += 1

        if step_counter % day_steps == 0 or done:
            pnl_d = info["balance"] - day_start
            print(f"📅 Day {step_counter // day_steps:02d}  "
                  f"Bal {info['balance']:.2f}   PnL {pnl_d:+.2f}")
            day_start = info["balance"]

        # -------- weekly checkpoint ----------------------------
        if step_counter % week_steps == 0 or done:
            pnl_w = info["balance"] - week_start
            print(f"📅 Week {step_counter // week_steps:02d}  "
                  f"Bal {info['balance']:.2f}   PnL {pnl_w:+.2f}")
            week_start = info["balance"]

        # -------- monthly checkpoint ---------------------------
        if step_counter % month_steps == 0 or done:
            pnl_m = info["balance"] - month_start
            print(f"🗓  Month {step_counter // month_steps:02d} "
                  f"Bal {info['balance']:.2f}   PnL {pnl_m:+.2f}")
            month_start = info["balance"]

    # ---------- final summary ----------------------------------
    final_bal = info["balance"]
    pnl_total = final_bal - env.start_bal
    print("===================================================")
    print(f"Backtest on {csv_path.split('/')[-1]}")
    print(f"Start balance : {env.start_bal:.2f}")
    print(f"Final balance : {final_bal:.2f}")
    print(f"Total  PnL    : {pnl_total:+.2f} "
          f"({pnl_total/env.start_bal*100:+.2f} %)")
    print("===================================================")


def main():
    df = add_indicators(load_last_mb(CSV_PATH))
    env     = TradingEnv(df)
    policy  = LSTMPolicy()
    optim   = torch.optim.Adam(policy.parameters(), lr=3e-4)
    train_ppo(env, policy, optim, episodes=500, device="cpu")
    env.reset()
    evaluate_policy(df, env)
main()
