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

def load_last_n_mb_csv(filepath, max_mb=50):
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
    df["ema_7_diff"] = df["ema_7"].diff()
    df["ema_14_diff"] = df["ema_14"].diff()
    df["ema_28_diff"] = df["ema_28"].diff()
    # df['ema_100'] = calculate_ema(df['Close'], 100)
    # df['ema_200'] = calculate_ema(df['Close'], 200)
    df['rsi'] = calculate_rsi(df['Close'])
    df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
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
    prev_hist = 0.0
    curr_hist = 0.0
    macd_increasing = None
    # macd_decreasing

    for i in range(len(df)):
        row = df.iloc[i]
        signal = ""
        mode = ""

        if i >= 1:
            prev_hist = df.iloc[i - 1]["macd_hist"]
            curr_hist = row["macd_hist"]
            macd_increasing = curr_hist > prev_hist
        else:
            macd_increasing = False  # not enough data to compare

        if row["adx"] > 20:
            mode = "trend"
            # if row["ema_7_diff"] > 0:
            #     signal = "buy"
            # elif row["ema_7_diff"] < 0:
            #     signal = "sell"
            if row["ema_7_diff"] > 0 and macd_increasing and row["rsi"] < 50 and row["ema_7_diff"] > 0:
                signal = "buy"
            elif row["ema_7_diff"] < 0 and macd_increasing is False and row["rsi"] > 50 and row["ema_7_diff"] < 0:
                signal = "sell"
        else:
            mode = "grid"
            signal = ""
        signals.append(signal)
        modes.append(mode)

    df['signal'] = signals
    df['mode'] = modes

    return df

def place_order(signal, entry_price, balance, leverage=75, qty_pct=0.10):
    # Allocate a portion of the balance
    allocated_margin = balance * qty_pct  # margin used
    position_value = allocated_margin * leverage  # total position value
    qty = position_value / entry_price  # size in contracts/coins

    return {
        "side": signal,
        "entry": entry_price,
        "qty": qty,
        "leverage": leverage,
        "margin": allocated_margin,
        "status": "open"
    }

def update_trade(trade, current_price):
    if trade["status"] != "open":
        return trade, None

    pnl = 0
    if trade["side"] == "buy":
        pnl = (current_price - trade["entry"]) * trade["qty"]

    elif trade["side"] == "sell":
        pnl = (trade["entry"] - current_price) * trade["qty"]

    trade["pnl"] = pnl
    return trade, pnl

def run_bot():
    df = load_last_n_mb_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)
    df = df.dropna()
    df = add_indicators(df)
    df = generate_signals(df)
    balance = 100
    risk_pct = 0.1
    leverage = 75

    trade_results = []
    total_trades = 0
    num_active_trades = 0
    active_trade = None

    # order tracker variables
    in_position = False
    position_size = 0.0
    order_side = ""

    # for index, row in df.iterrows():
    for i in range(len(df)):
        if balance * risk_pct < 5:
            break
        # print(f"active trades: {num_active_trades}")
        # if (i + 1) % interval_minutes == 0:
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
                print(f"  Avg profit: {avg_profit:.4f}")
                print(f"  Avg loss: {avg_loss:.4f}")
                print(f"  Balance: {balance:.2f}")

                # Reset stats for next interval if desired
                trade_results = []
                total_trades = 0

        current_price = df["Close"].iloc[i]

        # Update active trade if any
        if active_trade:
        # if in_position:
            # active_trade, pnl = update_trade(active_trade, current_price)
            active_trade, pnl = update_trade(active_trade, current_price)
            if active_trade["status"] == "closed":
                balance += pnl
                trade_results.append(pnl)
                total_trades += 1
                print(f"Trade closed. PnL: {pnl:.2f}, New balance: {balance:.2f}")
                active_trade = None
                num_active_trades -= 1

        # Get current signal and mode
        signal = df["signal"].iloc[i]
        mode = df["mode"].iloc[i]
        
        if active_trade and signal != active_trade["side"]:
        # if in_position == True and signal != active_trade["side"]:
            # Close previous trade forcibly at current price
            active_trade["status"] = "closed"
            active_trade["exit"] = current_price
            active_trade["pnl"] = (current_price - active_trade["entry"]) * active_trade["qty"] if active_trade["side"] == "buy" else (active_trade["entry"] - current_price) * active_trade["qty"]
            active_trade["side"] = signal
            balance += active_trade["pnl"]
            trade_results.append(active_trade["pnl"])
            total_trades += 1
            print(f"Grid order closed due to opposite signal. PnL: {active_trade['pnl']:.2f}, New balance: {balance:.2f}")
            active_trade = None
            in_position = False
            num_active_trades -= 1

        # Place new trade if no active trade and valid signal
        if active_trade == None and signal in ["buy", "sell"]:
        # if in_position == False and signal in ["buy", "sell"]:
            active_trade = place_order(
                signal=signal,
                entry_price=current_price,
                balance=balance,
                leverage=leverage,
                qty_pct=risk_pct
            )
            in_position = True
            num_active_trades += 1
            print(f"Placed new {signal} order at {current_price:.4f} with Qty: {active_trade['qty']:.4f}")

run_bot()
# print(f"Total trades placed: {total_trades}")
# print(f"Final balance: {balance}")
# print(f"Trade results: {trade_results}")
