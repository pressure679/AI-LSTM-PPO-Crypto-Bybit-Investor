import time
import math
from datetime import datetime
# import requests
import pandas as pd
# from pybit.unified_trading import HTTP  # pip install pybit
import numpy as np
from io import StringIO

# === SETUP ===
xrp = "XRPUSDT"
# balance = 50
# leverage = 75
# risk_pct = 0.3
csv_path = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/XRPUSD_1m_Binance.csv"
interval_minutes = 1440  # 1 day, can be 10080 (week), 43200 (month)

def load_last_n_mb_csv(filepath, max_mb=5):
    # Read header first (just first line
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()

    n_bytes = max_mb * 1024 * 1024
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        seek_pos = max(filesize - n_bytes, 0)
        f.seek(seek_pos)
        chunk = f.read()

        # Find first newline after seek_pos to skip partial line
        first_newline = chunk.find(b'\n')
        if first_newline == -1:
            chunk_start = 0
        else:
            chunk_start = first_newline + 1

        chunk_str = chunk[chunk_start:].decode('utf-8', errors='replace')

    # Combine header and chunk string (so we have header + last data)
    combined_str = header + chunk_str

    df = pd.read_csv(StringIO(combined_str), header=0)
    df.columns = ["Open time","Open","High","Low","Close","Volume","Close time","Quote asset volume","Number of trades","Taker buy base asset volume","Taker buy quote asset volume","Ignore"]

    # print("Columns in df:", df.columns.tolist())
    # print("First row of df:\n", df.head(1))
 
    return df

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_bollinger_bands(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

def calculate_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    return adx

def add_indicators(df):
    df['ema_7'] = calculate_ema(df['Close'], 7)
    df['ema_14'] = calculate_ema(df['Close'], 14)
    df['ema_28'] = calculate_ema(df['Close'], 28)
    # df['ema_45'] = calculate_ema(df['Close'], 45)
    # df['ema_75'] = calculate_ema(df['Close'], 75)
    # df['ema_100'] = calculate_ema(df['Close'], 100)
    # df['ema_200'] = calculate_ema(df['Close'], 200)
    df["ema_7_diff"] = df["ema_7"].diff()
    df["ema_14_diff"] = df["ema_14"].diff()
    df["ema_28_diff"] = df["ema_28"].diff()
    df['rsi'] = calculate_rsi(df['Close'])
    df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    df['macd_momentum'] = df['macd_line'].diff()
    # df['macd_signal_diff'] = df['macd_signal'].diff()
    df['macd_cross_up'] = (df['macd_line'].shift(1) < df['macd_signal'].shift(1)) & (df['macd_line'] > df['macd_signal'])
    df['macd_cross_down'] = (df['macd_line'].shift(1) > df['macd_signal'].shift(1)) & (df['macd_line'] < df['macd_signal'])

    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['Close'])
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['adx'] = calculate_adx(df)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['TR'] = tr
    return df

def generate_signals(df):
    signals = []
    modes = []
    # prev_hist = 0.0
    # curr_hist = 0.0
    # macd_increasing = False
    # macd_decreasing = False

    for i in range(len(df)):
        if i < 1:
            signals.append("")
            modes.append("")
            continue
        row = df.iloc[i]
        prev = df.iloc[i-1]
        signal = ""
        mode = ""

        # rsi_rising = row["rsi"] > prev["rsi"]
        # rsi_falling = row["rsi"] < prev["rsi"]

        # prev_hist = df.iloc[i - 1]["macd_hist"]
        # curr_hist = row["macd_hist"]
        # macd_increasing = curr_hist > prev_hist
        # macd_decreasing = curr_hist < prev_hist
        prev_macd_line = df["macd_line"].iloc[i - 3]
        curr_macd_line = row["macd_line"]
        macd_line_increasing = curr_macd_line > prev_macd_line
        macd_line_decreasing = curr_macd_line < prev_macd_line
        # print(f"price: {df['Close'].iloc[i]}")
        # print(f"macd line: {df['macd_line'].iloc[i]}")
        # print(f"price diff: {df['Close'].iloc[i] - df['Close'].iloc[i-5]}")
        # print(f"macd line diff: {df['macd_line'].iloc[i] - df['macd_line'].iloc[i-5]}")
        # print(f"macd line increasing: {macd_line_increasing}")
        # print(f"macd line decreasing: {macd_line_decreasing}")
            
        # if row["ema_28_diff"] > 0 and row['adx'] > 20:
        if macd_line_increasing and row['adx'] > 20:
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row["rsi"] < 70 :
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] > 0:
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] < 70 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs badly with high leverage
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row['macd_cross_up']: # performs badly with high leverage
        # if row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and row['macd_line'] > row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] > 0 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # has about a 1:2 RR, but performs badly with high leverage
            signal = "buy"
        # if row["ema_28_diff"] < 0 and row['adx'] > 20:
        if macd_line_decreasing and row['adx'] > 20:
        # if row["ema_28_diff"] < 0 and row["adx"] > 20 and row["rsi"] > 30:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] < 0:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] > 30 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # performs badly on high leverage
        # if row["ema_28_diff"] < 0 and row["adx"] > 20 and row['macd_cross_down']: # performs badly with high leverage
        # if row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and row['macd_line'] < row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] < 0 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # has about a 1:2 RR , but performs badly with high leverage
            signal = "sell"
            
        signals.append(signal)
        modes.append(mode)

    df['signal'] = signals
    df['mode'] = modes

    return df

def calculate_trade_parameters(entry_price, atr, balance, side, leverage=75, risk_pct=0.1):
    stop_loss_distance = atr * 0.5
    allocated_margin = balance * risk_pct
    position_value = allocated_margin * leverage
    position_size = position_value / entry_price

    # Example SL/TP based on ATR and levels
    stop_loss = entry_price - stop_loss_distance if side == "buy" else entry_price + stop_loss_distance
    tp_levels = [
        # entry_price + atr * (i + 0.5) if side == "buy" else entry_price - atr * (i + 0.5)
        entry_price + atr * i / 3 if side == "buy" else entry_price - atr * i / 3
        for i in range(1, 4)
    ]

    return {
        "position_size": round(position_size, 4),
        "stop_loss": round(stop_loss, 2),
        "tp_levels": [round(tp, 2) for tp in tp_levels]
    }

def update_trade(trade, current_price):
    if trade["status"] != "open":
        return trade, 0  # Return 0 PnL for consistency

    qty = trade["qty"]
    entry = trade["entry"]
    side = trade["side"]
    sl = trade["sl"]
    tp_levels = trade["tp_levels"]
    tp_hits = trade["tp_hits"]
    trail_active = trade.get("trail_active", False)
    trail_offset = trade.get("trail_offset", 0)
    realized_pnl = 0

    # Check SL or trailing SL
    if side == "buy":
        if trail_active and current_price <= trade["trail_sl"]:
            pnl = (trade["trail_sl"] - entry) * qty
            trade["status"] = "closed"
            trade["pnl"] += pnl
            return trade, pnl
        elif current_price <= sl:
            pnl = (sl - entry) * qty
            trade["status"] = "closed"
            trade["pnl"] += pnl
            return trade, pnl

    elif side == "sell":
        if trail_active and current_price >= trade["trail_sl"]:
            pnl = (entry - trade["trail_sl"]) * qty
            trade["status"] = "closed"
            trade["pnl"] += pnl
            return trade, pnl
        elif current_price >= sl:
            pnl = (entry - sl) * qty
            trade["status"] = "closed"
            trade["pnl"] += pnl
            return trade, pnl

    # Check TP levels (partial profits)
    for i, tp in enumerate(tp_levels):
        if not tp_hits[i]:
            if (side == "buy" and current_price >= tp) or (side == "sell" and current_price <= tp):
                portion = 0.25  # 25% closed per TP level
                pnl = (tp - entry) * qty * portion if side == "buy" else (entry - tp) * qty * portion
                realized_pnl += pnl
                trade["pnl"] += pnl
                tp_hits[i] = True

                # Activate trailing SL after TP2 hit
                if i == 1 and not trail_active:
                    trail_price = current_price - trail_offset if side == "buy" else current_price + trail_offset
                    trade["trail_sl"] = trail_price
                    trade["trail_active"] = True

    # Update trailing SL dynamically if active
    if trail_active:
        if side == "buy":
            new_trail_sl = current_price - trail_offset
            if new_trail_sl > trade["trail_sl"]:
                trade["trail_sl"] = new_trail_sl
        else:
            new_trail_sl = current_price + trail_offset
            if new_trail_sl < trade["trail_sl"]:
                trade["trail_sl"] = new_trail_sl

    # If all TP levels hit, close trade
    if all(tp_hits):
        trade["status"] = "closed"
        return trade, trade["pnl"]

    return trade, realized_pnl


def run_bot():
    df = load_last_n_mb_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)
    df = df.dropna()
    df = add_indicators(df)
    df = generate_signals(df)
    balance = 100
    risk_pct = 0.15
    # leverage = 20

    trade_results = []
    total_trades = 0
    num_active_trades = 0
    active_trade = None
    investment = 0.0
    index_placed_order = 0

    num_trades_active = 0

    for i in range(len(df)):
        # print(f"num trades active: {num_trades_active}")
        # investment = max(balance * risk_pct, 5)
        if balance < 10:
            break
        # 1440 1 day, can be 10080 (week), 43200 (month)
        if (i + 1) % 1440 == 0:
            if total_trades > 0:
                wins = [p for p in trade_results if p > 0]
                losses = [p for p in trade_results if p <= 0]
                win_rate = len(wins) / total_trades * 100
                avg_profit = sum(wins) / len(wins) if wins else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                
                print(f"Stats at day {((i + 1) / 1440):.0f}:")
                print(f"  Total trades: {total_trades}")
                print(f"  Win rate: {win_rate:.2f}%")
                print(f"  Avg profit: {avg_profit:.2f}")
                print(f"  Avg loss: {avg_loss:.2f}")
                print(f"  Balance: {balance:.2f}")

                # Reset stats for next interval if desired
                trade_results = []
                total_trades = 0

        current_price = df["Close"].iloc[i]

        # Update active trade if any
        if active_trade:
            active_trade, pnl = update_trade(active_trade, current_price)
            if pnl != 0:
                balance += pnl
                trade_results.append(pnl)
                if active_trade["status"] == "closed":
                    total_trades += 1
                    active_trade = None
                    # num_trades_active -= 1

        # Get current signal and mode
        signal = df["signal"].iloc[i]
        mode = df["mode"].iloc[i]
        atr = df["atr"].iloc[i]  # Make sure ATR is already calculated in your DataFrame
        # if signal != "":
        #     print(f"candle {i}, signal: {signal}, price: {current_price}")
        
        if active_trade and signal != active_trade["side"] and signal != "":
            # Close previous trade forcibly at current price
            active_trade["status"] = "closed"
            active_trade["exit"] = current_price
            active_trade["pnl"] = (current_price - active_trade["entry"]) * active_trade["qty"] if active_trade["side"] == "buy" else (active_trade["entry"] - current_price) * active_trade["qty"]
            active_trade["side"] = signal
            balance += active_trade["pnl"]
            trade_results.append(active_trade["pnl"])
            total_trades += 1
            # num_trades_active -= 1
            # print(f"Order closed due to opposite signal. PnL: {active_trade['pnl']:.2f}, New balance: {balance:.2f}")
            # print(f"atr: {df['atr'].iloc[i]}")
            # print(f"distance to last placed order: {i-index_placed_order}")
            # index_placed_order = i
            active_trade = None

        # Place new trade if no active trade and valid signal
        # Open a new trade if there's a signal, no active trade, and ATR is valid
        if not active_trade and signal in ["buy", "sell"] and atr > 0:
            entry_price = current_price
            trade_params = calculate_trade_parameters(entry_price, atr, balance, signal)
            
            active_trade = {
                "status": "open",
                "side": signal,
                "entry": entry_price,
                "qty": trade_params["position_size"],
                "sl": trade_params["stop_loss"],
                "tp_levels": trade_params["tp_levels"],
                "tp_hits": [False, False, False, False],
                "trail_active": False,
                "trail_offset": atr * 0.5,
                "pnl": 0
            }
            # num_trades_active +=1
            # print(f"Placed new {signal} order at {current_price:.4f} with Qty: {active_trade['qty']:.4f}")
            # print(f"macd line: {df['macd_line'].iloc[i]}")
            # print(f"macd momentum: {df['macd_momentum'].iloc[i]}")
            # index_placed_order = i


run_bot()
# print(f"Total trades placed: {total_trades}")
# print(f"Final balance: {balance}")
# print(f"Trade results: {trade_results}")
