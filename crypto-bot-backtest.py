import pandas as pd
import math
import time
from datetime import datetime, timedelta
import os
from io import StringIO

# --- Configuration ---
symbol = "XRPUSD"
csv_path = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/XRPUSD_1m_Binance.csv"
max_file_mb = 25  # Adjust how many MB to load from CSV file (e.g. last 10 MB)

# --- Globals ---
active_position = None
current_strategy = None
entry_price = None
entry_time = None
peak_balance = 1000  # starting simulated balance
balance = peak_balance
positions = []  # Simulated open positions: list of dicts
last_fetch_15m = 0
last_fetch_1h = 0
interval_15m = 900
interval_1h = 3600

trade_history = []  # Store closed trades: dict with pnl, entry, exit, side

last_week_print = None
last_month_print = None

# --- Helpers to read last N MB from CSV file ---
def read_last_n_mb_csv(filepath, n_mb=10):
    """
    Reads the last n_mb megabytes from a CSV file and returns a pandas DataFrame.
    Ensures it starts reading from the beginning of a line to avoid broken CSV rows.

    Args:
        filepath (str): Path to the CSV file.
        n_mb (int): Number of megabytes to read from the end of the file.

    Returns:
        pd.DataFrame: DataFrame containing the read CSV rows.
    """

    n_bytes = n_mb * 1024 * 1024  # Convert MB to bytes
    with open(filepath, 'rb') as f:
        f.seek(0, 2)  # Move to EOF
        filesize = f.tell()
        seek_pos = max(filesize - n_bytes, 0)
        f.seek(seek_pos)

        # Read a chunk of bytes from the file
        chunk = f.read()

        # Find first newline in chunk (to start reading from next full line)
        first_newline = chunk.find(b'\n')
        if first_newline == -1:
            # No newline found, read entire chunk
            chunk_start = 0
        else:
            # Start just after first newline to avoid partial line
            chunk_start = first_newline + 1

        # Decode the chunk from bytes to string starting from chunk_start
        chunk_str = chunk[chunk_start:].decode('utf-8', errors='replace')

    # Read CSV from the string chunk; assume header present in file, so skip first partial lines
    df = pd.read_csv(StringIO(chunk_str))

    return df


df_1m = read_last_n_mb_of_csv(csv_path, max_file_mb)

def resample_candles(df, interval):
    if interval == "15m":
        minutes = 15
    elif interval == "1h":
        minutes = 60
    else:
        raise ValueError("Unsupported interval")

    df = df.copy()
    df.set_index('Open time', inplace=True)

    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    df_resampled = df.resample(f'{minutes}T').apply(ohlc_dict).dropna()
    df_resampled.reset_index(inplace=True)
    return df_resampled

# Indicators simplified here, assume previous definitions...

def calculate_adx(df, period=14):
    df = df.copy()
    df['TR'] = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    df['+DM'] = df['High'].diff()
    df['-DM'] = df['Low'].diff() * -1

    df['+DM'] = df.apply(lambda x: x['+DM'] if (x['+DM'] > 0 and x['+DM'] > x['-DM']) else 0, axis=1)
    df['-DM'] = df.apply(lambda x: x['-DM'] if (x['-DM'] > 0 and x['-DM'] > x['+DM']) else 0, axis=1)

    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()

    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=period).mean()
    return df

def get_ema_signal(df, fast=12, slow=26):
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    if df['EMA_fast'].iloc[-2] < df['EMA_slow'].iloc[-2] and df['EMA_fast'].iloc[-1] > df['EMA_slow'].iloc[-1]:
        return "Buy"
    elif df['EMA_fast'].iloc[-2] > df['EMA_slow'].iloc[-2] and df['EMA_fast'].iloc[-1] < df['EMA_slow'].iloc[-1]:
        return "Sell"
    else:
        return None

def get_macd_signal(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    if df['MACD'].iloc[-2] < df['Signal'].iloc[-2] and df['MACD'].iloc[-1] > df['Signal'].iloc[-1]:
        return "buy"
    elif df['MACD'].iloc[-2] > df['Signal'].iloc[-2] and df['MACD'].iloc[-1] < df['Signal'].iloc[-1]:
        return "sell"
    else:
        return None

def get_balance():
    global balance
    return balance

def get_trade_qty():
    global balance
    usdt_balance = balance
    if usdt_balance < 5:
        return usdt_balance
    usd_amount = max(usdt_balance * 0.05, 5)
    price = float(df_1m['Close'].iloc[-1])
    qty = round(usd_amount / price, 3)
    return qty

def get_mark_price(symbol):
    return float(df_1m['Close'].iloc[-1])

def close_all_positions():
    global active_position, balance, entry_price, entry_time, positions, trade_history
    if active_position is None:
        print("[INFO] No active position to close.")
        return

    mark_price = get_mark_price(symbol)
    position = positions.pop() if positions else None
    if position:
        side = position['side']
        entry = position['entry_price']
        qty = position['qty']
        pnl = 0
        if side == "Buy":
            pnl = (mark_price - entry) * qty
        else:
            pnl = (entry - mark_price) * qty
        balance += pnl

        # Record trade history
        trade_history.append({
            "side": side,
            "entry_price": entry,
            "exit_price": mark_price,
            "qty": qty,
            "pnl": pnl,
            "entry_time": position['time'],
            "exit_time": datetime.utcnow()
        })

        print(f"[INFO] Closed {side} position at {mark_price:.4f}, PnL: ${pnl:.2f}, New balance: ${balance:.2f}")

    active_position = None
    entry_price = None
    entry_time = None

def enter_trade(signal, strategy, df):
    global current_strategy, peak_balance, active_position, entry_price, entry_time, balance, positions

    mark_price = get_mark_price(symbol)
    min_price = df["Low"].iloc[-10:].min()
    max_price = df["High"].iloc[-10:].max()
    range_pct = (max_price - min_price) / mark_price

    if range_pct < 0.015:
        print("[INFO] Market range too small (<1.5%), skipping trade.")
        return

    current_balance = get_balance()
    if current_balance > peak_balance:
        peak_balance = current_balance

    realized_gain = peak_balance - current_balance if peak_balance > current_balance else 0
    target_profit = realized_gain + peak_balance * 0.02

    expected_gain_pct = 0.3
    estimated_range_pct = range_pct * 3

    leverage = min(50, max(1, int((target_profit / (current_balance * expected_gain_pct)) * 10)))
    qty = get_trade_qty()

    if qty <= 0:
        print("[WARN] Trade quantity <= 0, skipping trade.")
        return

    print(f"[INFO] Entering {signal.upper()} trade with qty={qty}, leverage={leverage}, target_profit=${target_profit:.2f}")

    if active_position:
        close_all_positions()

    active_position = signal.capitalize()
    current_strategy = strategy
    entry_price = mark_price
    entry_time = datetime.utcnow()
    positions.append({
        "side": active_position,
        "entry_price": entry_price,
        "qty": qty,
        "leverage": leverage,
        "time": entry_time,
    })

def print_performance_summary():
    global trade_history
    now = datetime.utcnow()

    # Filter trades for last week and last month
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)

    trades_last_week = [t for t in trade_history if t['exit_time'] >= week_start]
    trades_last_month = [t for t in trade_history if t['exit_time'] >= month_start]

    def summarize(trades, period_name):
        if not trades:
            print(f"[{period_name} SUMMARY] No trades.")
            return

        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Calculate days span in trades
        first_trade = min(t['exit_time'] for t in trades)
        last_trade = max(t['exit_time'] for t in trades)
        days_span = (last_trade - first_trade).days + 1
        days_span = max(days_span, 1)

        avg_trades_per_day = total_trades / days_span

        print(f"[{period_name} SUMMARY]")
        print(f" Total PnL: ${total_pnl:.2f}")
        print(f" Win Rate: {win_rate:.2f}% ({wins}/{total_trades})")
        print(f" Avg PnL/Trade: ${avg_pnl:.2f}")
        print(f" Avg Trades/Day: {avg_trades_per_day:.2f}")

    summarize(trades_last_week, "WEEKLY")
    summarize(trades_last_month, "MONTHLY")

def run_bot():
    global df_1m, last_week_print, last_month_print

    while True:
        df_1m = read_last_n_mb_of_csv(csv_path, max_file_mb)
        df_15m = resample_candles(df_1m, "15m")
        df_1h = resample_candles(df_1m, "1h")

        ema_signal = get_ema_signal(df_1h)
        macd_signal = get_macd_signal(df_15m)
        adx_df = calculate_adx(df_1h)
        adx_value = adx_df['ADX'].iloc[-1] if not adx_df.empty else None

        print(f"[STATUS] EMA: {ema_signal}, MACD: {macd_signal}, ADX: {adx_value:.2f}")

        signal = None
        strategy = None
        if ema_signal and adx_value > 25:
            if ema_signal in ["Buy", "Sell"]:
                signal = ema_signal
                strategy = "EMA Trend"
        else:
            if macd_signal in ["Buy", "Sell"]:
                signal = macd_signal
                strategy = "MACD Range"

        if signal:
            enter_trade(signal, strategy, df_15m)
        # else:
        #     print("[INFO] No clear trading signal this iteration.")

        # Check if weekly/monthly summary should be printed
        now = datetime.utcnow()
        week_start = now - timedelta(days=now.weekday())  # Monday of current week
        month_start = now.replace(day=1)

        # Print weekly summary once per week
        if (last_week_print is None) or (week_start > last_week_print):
            print_performance_summary()
            last_week_print = week_start

        # Print monthly summary once per month
        if (last_month_print is None) or (month_start > last_month_print):
            print_performance_summary()
            last_month_print = month_start

        print(f"[INFO] Current balance: ${balance:.2f}, Active position: {active_position}")
        # print("---- Sleeping for 60 seconds before next iteration ----")
        # time.sleep(60)

if __name__ == "__main__":
    print("Starting trading bot simulation with weekly/monthly performance summary...")
    run_bot()
