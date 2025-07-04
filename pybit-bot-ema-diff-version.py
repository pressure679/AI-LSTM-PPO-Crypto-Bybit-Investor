from io import StringIO
import time
import math
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP  # pip install pybit
import threading

# === SETUP ===
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
coin = "XRPUSDT"
# balance = 100
# risk_pct = 0.1
leverage = 75
risk_pct = 0.1
current_trade_info = {}

session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    recv_window=5000,
    timeout=30
)
def keep_session_alive():
    try:
        response = session.get_server_time()  # Example lightweight call
        print("[KEEP-ALIVE] Session alive:", response)
    except Exception as e:
        print("[KEEP-ALIVE ERROR]", e)
    finally:
        threading.Timer(1500, keep_session_alive).start()  # Schedule next call
keep_session_alive()
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

def get_klines_df(coin, interval, limit=60):
    response = session.get_kline(
        category="linear",
        symbol=coin,
        interval=str(interval),
        limit=limit
    )
    data = response['result']['list']
    df = pd.DataFrame(data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"
    ])
    df = df.astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)
    return df

def get_balance():
    balance_data = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    return float(balance_data[0]["totalEquity"])

def get_mark_price(coin):
    price_data = session.get_tickers(category="linear", symbol=coin)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(coin):
    info = session.get_instruments_info(category="linear", symbol=coin)
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
    desired_usd_amount = max(usdt_balance * 0.1, 5)

    # Cap desired amount to max_risk_pct * balance
    max_allowed_usd = usdt_balance * 0.1
    trade_usd_amount = min(desired_usd_amount, max_allowed_usd)

    price = float(session.get_tickers(category="linear", symbol=coin)["result"]["list"][0]["lastPrice"])
    raw_qty = trade_usd_amount / price

    step, min_qty = get_qty_step(coin)
    qty = math.floor(raw_qty / step) * step
    qty = round(qty, 4)  # round for precision

    if qty < min_qty:
        qty = min_qty

    return qty

def cancel_specific_order(order_id: str, coin: str):
    if not order_id:
        print("No order ID to cancel")
        return
    try:
        response = session.cancel_order(category="linear", order_id=order_id, symbol=symbol)
        print("Cancel order response:", response)
    except Exception as e:
        print("Error canceling order:", e)

def setup_trade(df, signal, symbol):
    entry_price = df["Close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    adx = df["adx"].iloc[-1]

    high_range = df["High"].iloc[-14:].max()
    low_range = df["Low"].iloc[-14:].min()
    market_range_pct = (high_range - low_range) / entry_price * 100

    # Dynamic threshold based on ADX
    if adx < 20:
        print(f"[INFO] ADX too low ({adx:.2f}), ATR: {atr:.4f}, skipping trade.")
        return None
    elif adx < 30 and market_range_pct < 1.5:
        print(f"[INFO] ADX moderate ({adx:.2f}), ATR {atr:.4f}, but market range low ({market_range_pct:.2f}%), skipping.")
        return None
    elif adx >= 30 and market_range_pct < 1.0:
        print(f"[INFO] ADX strong ({adx:.2f}), ATR {atr:.4f}, but range too small ({market_range_pct:.2f}%), skipping.")
        return None

    # Passed filters
    print(f"[INFO] ADX {adx:.2f}, ATR {atr:.4f}, range {market_range_pct:.2f}%, trade OK.")
    return True
def enter_trade(signal, df):
    global leverage
    global current_trade_info

    if signal not in ["Buy", "Sell"]:
        print("[WARN] Invalid signal for enter_trade.")
        return None

    mark_price = get_mark_price(coin)
    entry_price = mark_price

    balance = get_balance()
    risk_amount = max(balance * risk_pct, 5)

    # Calculate qty
    step, min_qty = get_qty_step(coin)
    total_qty = risk_amount / mark_price
    total_qty = math.floor(total_qty / step) * step
    if total_qty < min_qty:
        print(f"[WARN] Quantity {total_qty} below minimum {min_qty}, skipping trade.")
        return None

    side = "Buy" if signal == "Buy" else "Sell"

    order_id = ""

    try:
        response = session.place_order(
            category="linear",
            symbol=coin,
            side=side,
            order_type="Market",
            qty=total_qty,
            buyLeverage=leverage,
            sellLeverage=leverage,
        )
        order_id = response.get("result", {}).get("orderId")
        # print(f"[INFO] Placed {side} order with TP={tp_price_str}, SL={sl_price_str}")
        print(f"[INFO] Placed {side} order")
    except Exception as e:
        print(f"[ERROR] Failed to place order: {e}")
        return None

    if order_id:
       current_trade_info = {
           "order_id": order_id,
           "total_qty": total_qty,
           "leverage": leverage,
           "side": side
       }

    return order_id

def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)

def generate_signals(df):
    signals = []
    # modes = []
    # prev_hist = 0.0
    # curr_hist = 0.0
    # macd_increasing = False
    # macd_decreasing = False

    for i in range(len(df)):
        if i < 3:
            signals.append("")
            # modes.append("")
            continue
        # row = df.iloc[i]
        prev = df.iloc[i-1]
        signal = ""
        # mode = ""

        # rsi_rising = row["rsi"] > prev["rsi"]
        # rsi_falling = row["rsi"] < prev["rsi"]

        prev_macd_line = df["macd_line"].iloc[i-5]
        curr_macd_line = df["macd_line"].iloc[i]
        # macd_line_diff = df["macd_line"].iloc[i] - df["macd_line"].iloc[i-3]
        # macd_line_increasing = macd_line_diff > 0
        # macd_line_decreasing = macd_line_diff < 0
        macd_line_increasing = curr_macd_line > prev_macd_line
        macd_line_decreasing = curr_macd_line < prev_macd_line
            
        if df["ema_7_diff"].iloc[i] > 0 and df['adx'].iloc[i] > 20:
        # if row["ema_14"] > row["ema_28"] and row['adx'] > 20:
        # if macd_line_increasing and df['adx'].iloc[i] > 20:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20:
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row["rsi"] < 70 :
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] > 0:
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] < 70 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs badly with high leverage
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row['macd_cross_up']: # performs badly with high leverage
        # if row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and row['macd_line'] > row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] > 0 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # has about a 1:2 RR, but performs badly with high leverage
            signal = "Buy"
        if df["ema_7_diff"].iloc[i] < 0 and df['adx'].iloc[i] > 20:
        # if macd_line_decreasing and df['adx'].iloc[i] > 20:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20:
        # if row["ema_28_diff"] < 0 and row["adx"] > 20 and row["rsi"] > 30:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] < 0:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] > 30 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # performs badly on high leverage
        # if row["ema_28_diff"] < 0 and row["adx"] > 20 and row['macd_cross_down']: # performs badly with high leverage
        # if row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and row['macd_line'] < row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] < 0 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # has about a 1:2 RR , but performs badly with high leverage
            signal = "Sell"
            
        signals.append(signal)
        # modes.append(mode)

    df['signal'] = signals
    # df['mode'] = modes

    return df

def run_bot():
    active_trade = False
    balance = get_balance()
    risk_pct = 0.10
    leverage = 75
    order_id = ""
    current_trade_side = ""

    while True:
        df = get_klines_df(coin, 120)
        df = add_indicators(df)
        df = generate_signals(df)
        # print(f"atr: {df['atr'].iloc[-1]}")

        mark_price = get_mark_price(coin)
        signal = df["signal"].iloc[-1]
        # print("in main loop")

        # print(f"price: {mark_price}")
        # print(f"current trade side: {current_trade_side}")
        # print(f"signal: {signal}")
        # print(f"macd line: {df['macd_line'].iloc[-1]}")
        # print(f"previous macd line: {df['macd_line'].iloc[-3]}")
        # print(f"macd line increasing: {df['macd_line'].iloc[-1] > df['macd_line'].iloc[-3]}")
        # print(f"price range: {df['High'].iloc[-15:].max() - df['Low'].iloc[-15:].min()}")
        # print(f"price range in %: {(df['High'].iloc[-5:].max() - df['Low'].iloc[-15:].min()) / mark_price}")
        # print(f"atr: {df['atr'].iloc[-1]}")
        # print(f"1/14 atr: {df['atr'].iloc[-1] / 14}")
        # print(f"1/14 atr compared to mark price: {df['atr'].iloc[-1] / 14 / mark_price}")
        # print(f"1/14 atr compared to mark price times leverage: {df['atr'].iloc[-1] / 14 / mark_price * 75}")
        # print("")

        if signal == "":
            print("no clear direction yet from macd line diff indicator")
        if active_trade:
            # Fetch open positions for symbol
            positions = session.get_positions(category="linear", symbol=coin)["result"]["list"]
            
            # Find the relevant open position (non-zero size, matching side)
            position = None
            for pos in positions:
                if float(pos["size"]) > 0 and pos["side"] == current_trade_side:
                    position = pos
                    break
            
            if not position:
                # Call trailing stop check with current ATR from df
                # current_atr = df['atr'].iloc[-1]
                # trailing_stop_check(position, current_atr)
            # else:
                # No open position found: reset state
                active_trade = False
                print("[INFO] Position closed or missing, resetting active_trade flag.")

            # Close position if signal flips
            if signal != current_trade_side:
                cancel_specific_order(order_id, coin)
                active_trade = False
                print(f"Order closed due to opposite signal")

        if not active_trade and signal != "":
            # print("in 'if not active_trade and signal != ''")
            setup = setup_trade(df, signal, symbol=coin)
            if setup is False:
                print("[INFO] Trade setup skipped due to low volatility or invalid conditions.")
                wait_until_next_candle(1)
                continue
            if signal == "Buy":
                total_qty = get_trade_qty()
                entry_price = get_mark_price(coin)
                order_id = enter_trade("Buy", df)
                current_trade_side = "Buy"
                active_trade = True
            elif signal == "Sell":
                total_qty = get_trade_qty()
                entry_price = get_mark_price(coin)
                order_id = enter_trade("Sell", df)
                current_trade_side = "Sell"
                active_trade = True

        wait_until_next_candle(1)

# set trailing stop
# session.set_trading_stop(
#     category="linear",
#     symbol=symbol,
#     stop_loss=new_stop_loss
# )

# partial close order
# response = session.place_active_order(
#     symbol=coin,
#     side="Sell",  # opposite side to close position
#     order_type="Market",
#     qty=qty_to_close,
#     reduce_only=True,
#     time_in_force="ImmediateOrCancel",
#     leverage=leverage
# )

run_bot()
