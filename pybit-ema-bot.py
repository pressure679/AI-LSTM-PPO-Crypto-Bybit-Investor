import time
import math
from datetime import datetime
import requests
import pandas as pd
from pybit.unified_trading import HTTP  # pip install pybit
import numpy as np

# === SETUP ===
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
xrp = "XRPUSDT"
# interval_15m = 15 * 60
# interval_1h = 60 * 60
max_risk_percentage = 0.2
leverage = 20
risk_pct = 0.15
entry_time = None
entry_price = 0.0

# Initialize session
session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    recv_window=5000,
    timeout=30
    # testnet=True  # uncomment if needed
)

# Global variables
# last_fetch_15m = 0
# last_fetch_1h = 0
# current_strategy = None
# active_position = None
# entry_price = None
# entry_time = None
# peak_balance = 0

# === UTILITY FUNCTIONS ===
def get_klines_df(symbol, interval_sec, limit=60*24):
    interval_map = {
        60: "1",
        900: "15",
        3600: "60"
    }
    interval = interval_map[interval_sec]
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={xrp}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200 or 'result' not in data or 'list' not in data['result']:
        raise RuntimeError(f"Failed to fetch klines: {data}")
    klines = data['result']['list']
    df = pd.DataFrame(klines, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    return df

def get_balance():
    balance_data = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    return float(balance_data[0]["totalEquity"])


def get_mark_price(symbol):
    price_data = session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(symbol):
    info = session.get_instruments_info(category="linear", symbol=symbol)
    data = info["result"]["list"][0]
    step = float(data["lotSizeFilter"]["qtyStep"])
    min_qty = float(data["lotSizeFilter"]["minOrderQty"])
    return step, min_qty

def get_trade_qty():
    wallet = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    usdt_balance = float(wallet[0]["totalEquity"])
    if usdt_balance <= 5:
        return usdt_balance  # or whatever fallback you want if balance too low

    # Calculate desired trade amount (5% or $5 minimum)
    desired_usd_amount = max(usdt_balance * 0.2, 5)

    # Cap desired amount to max_risk_pct * balance
    max_allowed_usd = usdt_balance * 0.15
    trade_usd_amount = min(desired_usd_amount, max_allowed_usd)

    price = float(session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["lastPrice"])
    raw_qty = trade_usd_amount / price

    step, min_qty = get_qty_step(symbol)
    qty = math.floor(raw_qty / step) * step
    qty = round(qty, 4)  # round for precision

    if qty < min_qty:
        qty = min_qty

    return qty

def close_all_positions():
    positions = session.get_positions(category="linear", symbol=xrp)
    position_lists = positions['result']['list']
    # for position in positions:
    for position in position_lists:
        # if "result" in positions and positions["result"]:
        # price_data = session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
        # exit_price = float(price_data["lastPrice"])
        # exit_time = datetime.now()
        # print(position)
        qty = float(position["size"])
        if qty == 0:
            continue
        side = "Sell" if position["side"] == "Buy" else "Buy"
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, reduce_only=True)
        # pnl = (entry_price - exit_price) * qty *  * leverage
        # pnl = (entry_price - exit_price) * qty
        # duration = (exit_time - entry_time).total_seconds() / 60  # minutes

        print(f"[TRADE CLOSED]")

        # entry_price = None
        # entry_time = None

def enter_trade(signal, df, symbol="XRPUSDT"):
    # global peak_balance, entry_price, entry_time
    global leverage, xrp

    mark_price = get_mark_price(symbol)
    val = df_1m['Close'].mean() * 0.02 / df_1m['atr'].mean()
    # print(f"atr's to reach 2% movement: {val:.2f}")
    offset = int(round(val, 0))
    min_price = df["Low"].iloc[offset:].min()
    max_price = df["High"].iloc[offset:].max()
    price_range_pct = (max_price - min_price) / mark_price

    if price_range_pct < 0.02:
    # if price_range_pct < 0.007:
        print("[INFO] Market range is too small (<1.5%), skipping trade.")
        # print(f"price range in pct: {price_range_pct:.6f}")
        # print(f"Max high prices: {max_price:.6f}")
        # print(f"Min low prices: {min_price:.6f}")
        return

    balance = get_balance()

    risk_amount = max(balance * risk_pct, 5)
    # print(risk_amount)
    # print(balance)

    step, min_qty = get_qty_step(symbol)
    qty = risk_amount / mark_price
    qty = math.floor(qty / step) * step
    if qty < min_qty:
        print(f"[WARNING] Calculated quantity {qty} is below minimum {min_qty}, skipping trade.")
        return

    # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Entering EMA trade | Signal: {signal} | Qty: {qty:.4f} | Leverage: {leverage} | Risk Amount: ${risk_amount:.2f}")

    side = "Buy" if signal == "Buy" else "Sell"

    # session.set_leverage(category="linear", symbol=symbol, buyLeverage=leverage, sellLeverage=leverage)
    response = session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, buyLeverage=leverage, sellLeverage=leverage)
    return response.get("result", {}).get("orderId")

    # entry_price = mark_price
    # entry_time = datetime.now()

def cancel_specific_order(order_id: str, symbol: str):
    if not order_id:
        print("No order ID to cancel")
        return
    try:
        response = session.cancel_order(order_id=order_id, symbol=symbol)
        print("Cancel order response:", response)
    except Exception as e:
        print("Error canceling order:", e)

def add_indicators(df):
    df['ema_7'] = calculate_ema(df['Close'], 7)
    df['ema_14'] = calculate_ema(df['Close'], 14)
    df['ema_100'] = calculate_ema(df['Close'], 100)
    df['ema_200'] = calculate_ema(df['Close'], 200)
    df['rsi'] = calculate_rsi(df['Close'])
    df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['Close'])
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['adx'] = calculate_adx(df['High'], df['Low'], df['Close'])
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['TR'] = tr
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

def calculate_adx(high, low, close, period=14):
    global df_1m
    df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
    df_1m['TR'] = df_1m[['High', 'Low', 'Close']].max(axis=1) - df_1m[['High', 'Low', 'Close']].min(axis=1)
    df_1m['+DM'] = np.where((df_1m['High'].diff() > df_1m['Low'].diff()) & (df_1m['High'].diff() > 0), df_1m['High'].diff(), 0)
    df_1m['-DM'] = np.where((df_1m['Low'].diff() > df_1m['High'].diff()) & (df_1m['Low'].diff() > 0), df_1m['Low'].diff(), 0)

    tr14 = df_1m['TR'].rolling(window=period).sum()
    plus_dm14 = df_1m['+DM'].rolling(window=period).sum()
    minus_dm14 = df_1m['-DM'].rolling(window=period).sum()

    plus_di = 100 * (plus_dm14 / tr14)
    minus_di = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx

def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)

def run_bot():
    global df_1m
    position_open = False
    slope = False
    trend_order_id = ""
    grid_order_id = ""
    while True:
        # print("In MA_diff decision logic")
        now = time.time()
        mark_price = get_mark_price(xrp)
        df_1m = get_klines_df(xrp, 60)
        df_1m = add_indicators(df_1m)
        # df_1m["EMA_diff"] = df_1m["ema_7"] - df_1m["ema14"]
        df_1m["MA_diff"] = df_1m["ema_7"].diff()
        df_1m['is_peak'] = (df_1m['Close'].shift(1) < df_1m['Close']) & (df_1m['Close'].shift(-1) < df_1m['Close'])
        df_1m['is_trough'] = (df_1m['Close'].shift(1) > df_1m['Close']) & (df_1m['Close'].shift(-1) > df_1m['Close'])
        # peak_indices = peaks.index
        # peak_distances = peak_indices.to_series().diff().dropna()
        # avg_peak_distance = peak_distances.mean()
        just_passed_peak = df_1m['is_peak'].iloc[-3]  # True if previous candle was a peak
        just_passed_trough = df_1m['is_trough'].iloc[-3]  # True if previous candle was a trough
        curr_diff = df_1m['MA_diff'].iloc[-1]
        prev_diff = df_1m['MA_diff'].iloc[-2]
        prev2_diff = df_1m['MA_diff'].iloc[-3]
        # enter_trade("Buy", df_1m)

        # Simple logic: increasing if each is greater than previous
        ma_diff_increasing = (prev2_diff < prev_diff) and (prev_diff < curr_diff)
        ma_diff_decreasing = (prev2_diff > prev_diff) and (prev_diff > curr_diff)

        # Slope approximation by difference between last two points
        # slope_short = ema_short.iloc[-1] - ema_short.iloc[-2]
        # slope_long = ema_long.iloc[-1] - ema_long.iloc[-2]
        slope_short = df_1m["ema_7"].iloc[-1] - df_1m["ema_14"].iloc[-2]
        slope_long = df_1m["ema_14"].iloc[-1] - df_1m["ema_7"].iloc[-2]

        df_1m["200madiff"] = df_1m["ema_200"].diff()
        df_1m["200_ma_diff_smooth"] = df_1m["200madiff"].rolling(window=200).mean()
        maslope = df_1m["200_ma_diff_smooth"].max() / df_1m["200_ma_diff_smooth"].min()
        # print(f"200 ema smoothened difference of slope's slope: {maslope:.2f}")
        # print(maslope * 200 / df_1m["atr"].mean())
        # print(f"atr: {df_1m['atr'].mean():.6f}")
        atr_to_30 = df_1m['Close'].mean()/df_1m['atr'].mean() * 0.03
        atr_to_30_distance = atr_to_30 * df_1m['atr'].mean()
        # print(f"atr to reach 30% roi: with x10: {atr_to_30:.2f}")
        # print(f"distance in atr to reach 30% with x10:: {atr_to_30_distance:.2f}")
        # print(f"distance in to reach 24h high-low: {df_1m['High'].max() - df_1m['Low'].min():.2f}")
        # print(f"distance in 24h high-low in xrp value: {(df_1m['High'].max() - df_1m['Low'].min()) / df_1m['Close'].mean():.2f}")
        # # print(maslope)
        # print(f"adx: {df_1m['adx'].iloc[90:].mean():.2f}")
        # print(f"maslope: {maslope:.2f}")
        # print(f"atr: {df_1m['atr'].rolling(window=999).mean():.2}")
        if not position_open:
            # if maslope < -1.5:
            if df_1m['adx'].iloc[int(round(df_1m['Close'].mean()/df_1m['atr'].mean(), 0)):].mean() > 25:
                if maslope < -1.5:
                    close_all_positions()
                    trend_order_id = enter_trade("Sell", df_1m)
                    print("Trend order filled")
                    position_open = True
                    entry_time = time.time()
                    entry_price = mark_price
                    roi = (entry_price - market_price) / entry_price
                elif maslope > 1.5:
                    close_all_positions()
                    trend_order_id = enter_trade("Buy", df_1m)
                    print("Trend order filled")
                    position_open = True
                    entry_time = time.time()
                    entry_price = mark_price
                    roi = (market_price - entry_price) / entry_price
        elif position_open:
            elapsed = now - entry_time
            roi = (mark_price - entry_price) / entry_price
            if elapsed >= 5400 or roi >= 0.25:
                   print("Closing position due to ROI or time limit.")
                   cancel_specific_order(trend_order_id, xrp)
                   position_open = False  # optional flag to prevent re-closing 
        if df_1m["MA_diff"].iloc[-2] <= 0 and df_1m["MA_diff"].iloc[-1] > 0 and slope_short > 0 and slope_long > 0:
            print("Cross above zero — entering long")
            print("Uptrend detected")
            cancel_specific_order(grid_order_id, xrp)
            grid_order_id = enter_trade("Buy", df_1m)
        elif df_1m["MA_diff"].iloc[-2] >= 0 and df_1m["MA_diff"].iloc[-1] < 0 and slope_short < 0 and slope_long < 0:
            print("Cross below zero — entering short")
            print("Downtrend detected")
            cancel_specific_order(grid_order_id, xrp)
            grid_order_id = enter_trade("Sell", df_1m)
        if just_passed_peak and ma_diff_decreasing:
            print("Just passed a peak — entering short")
            cancel_specific_order(grid_order_id, xrp)
            grid_order_id = enter_trade("Sell", df_1m)
        elif just_passed_trough and ma_diff_increasing:
            print("Just passed a trough — entering long")
            cancel_specific_order(grid_order_id, xrp)
            grid_order_id = enter_trade("Buy", df_1m)
        print()
        wait_until_next_candle(1)
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ADX: {adx_value:.2f} | EMA Signal")
        # time.sleep(5)
run_bot()
