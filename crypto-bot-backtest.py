import os
import io
import pandas as pd
import numpy as np

# Initialize global weekly tracking variables
weekly_trades = []
weekly_start = None
week_num = 0
balance = 500
positions = []
peak_balance = balance
realized_gain = 0.0

def get_15m_index_for_1m_timestamp(df_15m, timestamp_1m):
    # Find the latest 15m candle index where datetime <= timestamp_1m
    idx_15m = df_15m.index.get_loc(timestamp_1m, method='ffill')
    return idx_15m

def record_trade(trade_time, pnl):
    global weekly_trades, weekly_start, week_num

    if weekly_start is None:
        weekly_start = trade_time.to_period('W').start_time

    # Check if trade_time is still in current week; if not, print summary and reset
    current_week_start = trade_time.to_period('W').start_time
    if current_week_start != weekly_start:
        print_weekly_summary()
        weekly_trades = []
        weekly_start = current_week_start
        week_num += 1

    weekly_trades.append(pnl)

def print_weekly_summary():
    if not weekly_trades:
        return

    total_trades = len(weekly_trades)
    wins = sum(1 for pnl in weekly_trades if pnl > 0)
    losses = total_trades - wins
    total_pnl = sum(weekly_trades)
    avg_profit = total_pnl / total_trades if total_trades > 0 else 0
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    print(f"--- Weekly Summary {week_num} ---")
    print(f"Trades: {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Avg Profit per Trade: {avg_profit:.2f}")
    print(f"Balance at Week End: {balance:.2f}")  # assumes 'balance' is global current balance
    print("----------------------")

# --- Load last N MB of CSV ---
def load_last_n_mb_csv(filepath, n_mb=50):
    n_bytes = n_mb * 1024 * 1024
    filesize = os.path.getsize(filepath)
    start_pos = max(0, filesize - n_bytes)

    with open(filepath, 'rb') as f:
        f.seek(start_pos)
        data = f.read()

    text = data.decode('utf-8', errors='ignore')
    first_newline = text.find('\n')
    if first_newline == -1:
        raise ValueError("No newline found in chunk, file may be too small or corrupted.")
    text = text[first_newline+1:]

    # Read header line from original file
    with open(filepath, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()

    csv_data = header_line + '\n' + text
    df = pd.read_csv(io.StringIO(csv_data))
    return df

# --- Indicator functions (no TA-Lib) ---
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    return upper_band, lower_band

def calculate_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# --- Entry condition check ---
def check_entry_condition(df, idx, mark_price, rsi_threshold_low=30, rsi_threshold_high=70):
    """
    Returns 'long', 'short', or None based on combined conditions:
    - RSI below low threshold => long
    - RSI above high threshold => short
    - Bollinger Bands hit => long if below lower, short if above upper
    - Price below low/high over past 10 candles with direction confirmed by peak close price over last 3 candles
    """
    if idx < 20:  # Need enough data for indicators
        return None

    rsi = df['rsi'].iloc[idx]
    bb_upper = df['bb_upper'].iloc[idx]
    bb_lower = df['bb_lower'].iloc[idx]
    close = df['Close'].iloc[idx]

    window_10_low = df['Low'].iloc[max(0, idx-9):idx+1].min()
    window_10_high = df['High'].iloc[max(0, idx-9):idx+1].max()
    peak_close_3 = df['Close'].iloc[max(0, idx-2):idx+1].max()
    trough_close_3 = df['Close'].iloc[max(0, idx-2):idx+1].min()

    # RSI condition
    if rsi < rsi_threshold_low:
        return 'long'
    if rsi > rsi_threshold_high:
        return 'short'

    # Bollinger bands condition
    if close < bb_lower:
        return 'long'
    if close > bb_upper:
        return 'short'

    # Price relative to 10 candle low/high + confirm with 3 candle peak close
    if mark_price < window_10_low and close < trough_close_3:
        return 'long'
    if mark_price > window_10_high and close > peak_close_3:
        return 'short'
    # position = {
    #     'entry_price': price,
    #     'size': calculated_size,
    #     'side': 'long',
    #     'entry_time': idx,
    #     # add any other required keys here
    # }
    # positions.append(position)

    return None

# --- Backtest & trading logic ---
def check_exit_conditions(positions, idx, df):
    """
    Checks if any open positions should be closed based on:
      - MACD histogram peaks (local max/min)
      - RSI thresholds (overbought/oversold)
      - Price crossing Bollinger Bands

    Args:
        positions: list of open positions with keys: side, entry_price, size
        df: DataFrame with indicators (macd_hist, rsi, bb_upper, bb_lower)
        idx: int - index of current candle to check indicators on

    Returns:
        new_positions (list), gain (float) from closed trades
    """

    gain = 0
    new_positions = []

    # Parameters for thresholds
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    def is_macd_hist_peak(df, idx):
        if idx <= 0 or idx >= len(df)-1:
            return False
        prev_val = df['macd_hist'].iloc[idx-1]
        curr_val = df['macd_hist'].iloc[idx]
        next_val = df['macd_hist'].iloc[idx+1]
        return (curr_val > prev_val and curr_val > next_val) or (curr_val < prev_val and curr_val < next_val)

    # Current indicators
    macd_hist_val = df['macd_hist'].iloc[idx]
    rsi_val = df['rsi'].iloc[idx]
    close_price = df['Close'].iloc[idx]
    bb_upper = df['bb_upper'].iloc[idx]
    bb_lower = df['bb_lower'].iloc[idx]

    for pos in positions:
        # ✅ Validate required keys
        if not all(k in pos for k in ('side', 'entry_price', 'position_size')):
            print(f"[WARNING] Malformed position at idx={idx}: {pos}")
            continue

        side = pos['side'].lower()  # normalize to lowercase
        entry_price = pos['entry_price']
        position_size = pos['position_size']

        exit_trade = False

        # 1) MACD histogram peak
        if is_macd_hist_peak(df, idx):
            exit_trade = True
            # print("macd histogram peak possibly reached")
            # if side == "long":
            #     print(f"price range: {close_price - entry_price}")
            # else:
            #     print(f"price range: {entry_price - close_price}")
                

        # 2) RSI thresholds
        if side == "long" and rsi_val >= RSI_OVERBOUGHT:
            exit_trade = True
            # print("rsi indicates overbought condition")
            # print(f"price range: {close_price - entry_price}")
        elif side == "short" and rsi_val <= RSI_OVERSOLD:
            exit_trade = True
            # print("rsi indicates oversold condition")
            # print(f"price range: {entry_price - close_price}")

        # 3) Price crosses Bollinger Bands
        if side == "long" and close_price >= bb_upper:
            exit_trade = True
            # print("upper bollinger band reached")
            # print(f"price range: {close_price - entry_price}")
        elif side == "short" and close_price <= bb_lower:
            exit_trade = True
            # print("lower bollinger band reached")
            # print(f"price range: {entry_price - close_price}")
        if exit_trade:
            if side == "long":
                trade_gain = (close_price - entry_price) * position_size
            else:
                trade_gain = (entry_price - close_price) * position_size
            # print(f"{side.upper()} EXIT at {close_price}, gain: {trade_gain:.4f}")
            gain += trade_gain
        else:
            new_positions.append(pos)

    return new_positions, gain

# Example usage inside your close_all_positions or wherever a trade closes
def calculate_position_size(current_balance, peak_balance, realized_gain, mark_price, sl_price):
    # --- 1. Target profit ---
    realized_gain = current_balance - peak_balance if current_balance > peak_balance else 0
    target_profit = realized_gain + peak_balance * 0.05

    # --- 2. Price range to SL ---
    range_pct = abs(mark_price - sl_price) / mark_price
    if range_pct == 0:
        return 0, 0

    # --- 3. Required position size to reach target profit from that % move ---
    required_position_size = target_profit / range_pct

    # --- 4. Max position sizing ---
    max_position_value = current_balance * 0.35
    qty = max_position_value / mark_price
    position_value = qty * mark_price

    # --- 5. Leverage estimate ---
    # leverage = required_position_size / position_value
    # leverage = min(50, max(10, round(leverage)))
    leverage = 75

    # --- 6. Recalculate qty if over-allocating ---
    final_position_value = qty * mark_price * leverage
    if final_position_value > current_balance * leverage:
        qty = (current_balance * leverage) / (mark_price * leverage)
        final_position_value = qty * mark_price * leverage

    # print(f"qty: {qty}, leverage: {leverage}, final position value: {final_position_value}")
    # print(f"Calculated position size (qty): {qty}, leverage: {leverage}, final position value: {final_position_value}, balance: {balance}")

    return qty, leverage

def calculate_dynamic_sl_price(side, mark_price, rsi, bb_upper, bb_lower, high_series, low_series, window=10):
    # RSI distance from neutral 50 (normalized 0 to 1)
    rsi_distance_pct = abs(rsi - 50) / 50  # ranges 0 to 1

    # Bollinger Bands % distance
    bb_distance_pct = (bb_upper - bb_lower) / mark_price

    # Mean high-low range over window, as % of price
    mean_range = (high_series[-window:].mean() - low_series[-window:].mean())
    # print(f"mean range: {mean_range}")
    mean_range_pct = mean_range / mark_price

    # Average distance
    sl_distance_pct = (rsi_distance_pct + bb_distance_pct + mean_range_pct) / 3

    # Apply minimal and maximal boundaries if you want
    min_distance_pct = 0.005  # 0.5%
    max_distance_pct = 0.05   # 5%
    sl_distance_pct = max(min_distance_pct, min(sl_distance_pct, max_distance_pct))

    if side == 'long':
        sl_price = mark_price * (1 - sl_distance_pct)
    else:
        sl_price = mark_price * (1 + sl_distance_pct)

    return sl_price

# --- In run_bot(), modify trade entry to allow only 1 position ---
def run_bot(filepath, n_mb=25):
    global balance
    df_1m = load_last_n_mb_csv(filepath, n_mb)
    # print(df_1m.head())
    # print(df_1m.columns)

    if 'Open time' in df_1m.columns:
        df_1m['Open time'] = pd.to_datetime(df_1m['Open time'])
        df_1m.set_index('Open time', inplace=True)

    df_15m = df_1m.resample('15T').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    df_1m['ema'] = calculate_ema(df_1m['Close'], 7)
    df_1m['rsi'] = calculate_rsi(df_1m['Close'], 14)
    df_1m['macd_line'], df_1m['macd_signal'], df_1m['macd_hist'] = calculate_macd(df_1m['Close'])
    df_1m['bb_upper'], df_1m['bb_lower'] = calculate_bollinger_bands(df_1m['Close'])
    df_1m['atr'] = calculate_atr(df_1m)

    balance = 500.0
    peak_balance = balance
    realized_gain = 0
    positions = []

    for idx_1m in range(50, len(df_1m)):
        if np.isnan(df_1m['rsi'].iloc[idx_1m]):
            continue

        row = df_1m.iloc[idx_1m]
        mark_price = row['Close']
        # current_time = df_15m.index[idx]   # <---- ADD THIS HERE
        # idx_15m = df_15m.index.get_loc(timestamp_1m, method='ffill')
        timestamp_1m = df_1m.index[idx_1m]  # get 1m timestamp at current idx
        idx_15m = df_15m.index.get_loc(timestamp_1m, method='ffill')  # find closest 15m index at or before timestamp_1m
        current_time = df_15m.index[idx_15m]  # current 15m timestamp based on ffill mapping


        # Exit existing positions
        result = check_exit_conditions(positions, idx_1m, df_1m)
        if result:
            positions, gain = result
        else:
            positions, gain = positions, 0
        if gain != 0:
            realized_gain += gain
            balance += gain
            # print(f"Profit added to balance: {gain}")
            record_trade(current_time, gain)   # now current_time is defined

        if balance > peak_balance:
            peak_balance = balance

        if balance < 5:
            print("Balance too low, stopping.")
            break

        # ONLY enter trade if no open positions
        if len(positions) == 0:
            side = check_entry_condition(df_1m, idx_1m, mark_price)

            if side is not None:
                if side == 'long':
                    # sl_price = mark_price - 2 * row['atr'] if not np.isnan(row['atr']) else mark_price * 0.95
                    sl_price = calculate_dynamic_sl_price(
                        side,
                        mark_price,
                        row['rsi'],
                        row['bb_upper'],
                        row['bb_lower'],
                        # df_1m['High'],
                        # df_1m['Low'],
                        df_1m['High'],
                        df_1m['Low'],
                        window=10
                    )
                    tp_price = mark_price + (mark_price - sl_price) * 2
                else:
                    # sl_price = mark_price + 2 * row['atr'] if not np.isnan(row['atr']) else mark_price * 1.05
                    sl_price = calculate_dynamic_sl_price(
                        side,
                        mark_price,
                        row['rsi'],
                        row['bb_upper'],
                        row['bb_lower'],
                        # df_1m['High'],
                        # df_1m['Low'],
                        df_1m['High'],
                        df_1m['Low'],
                        window=10
                    )
                    tp_price = mark_price - (sl_price - mark_price) * 2

                pos_size, leverage = calculate_position_size(balance, peak_balance, realized_gain, mark_price, sl_price)
                if pos_size > 0:
                    positions.append({'side': side, 'entry_price': mark_price,
                                      'position_size': pos_size, 'sl_price': sl_price, 'tp_price': tp_price})
                    # print(f"{side.upper()} entry at {mark_price:.4f}, size {pos_size:.4f}, SL {sl_price:.4f}, TP {tp_price:.4f}, balance {balance:.2f}")

    print(f"Final balance: {balance:.2f}, Realized Gain: {realized_gain:.2f}")

# === Run ===
if __name__ == "__main__":
    filepath = "/mnt/chromeos/removable/SD Card/Linux-shared-files/crypto and currency pairs/XRPUSD_1m_Binance.csv"
    run_bot(filepath, n_mb=25)
