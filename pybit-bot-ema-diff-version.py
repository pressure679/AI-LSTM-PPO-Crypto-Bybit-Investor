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
    timeout=60
)
def keep_session_alive():
    for attempt in range(30):
        try:
            # Example: Get latest position
            result = session.get_positions(category="linear", symbol=coin)
            break  # If success, break out of loop
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on attempt {attempt+1}, retrying...")
            time.sleep(2)  # wait before retry
        finally:
            threading.Timer(1500, keep_session_alive).start()  # Schedule next call
keep_session_alive()
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period):
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
    df['rsi_24'] = calculate_rsi(df['Close'], 24)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['Close'])
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
def place_sl_and_tp(symbol, side, entry_price, atr, qty, levels=[1.5, 2.5, 3.5]):
    """
    Places initial SL and multiple TP orders based on ATR multiples.
    
    Args:
        symbol (str): trading symbol, e.g. "XRPUSDT"
        side (str): "Buy" or "Sell"
        entry_price (float): price at entry
        atr (float): current ATR value
        qty (float): total trade quantity
        levels (list): list of ATR multiples for TP levels
        
    Returns:
        dict with order IDs of SL and TP orders
    """
    orders = {}

    # SL price calculation
    if side == "Buy":
        sl_price = entry_price - 1.5 * atr
    else:  # Sell
        sl_price = entry_price + 1.5 * atr

    # Place stop loss order
    sl_response = session.set_trading_stop(
        category="linear",
        symbol=symbol,
        side=side,
        stop_loss=round(sl_price, 4),
        trailing_stop=None,  # we'll handle trailing later
        tp_trigger_by="LastPrice"
    )
    orders['sl'] = sl_response

    # Calculate and place TP limit orders
    qty_per_tp = qty / len(levels)
    tp_orders = []
    for i, mult in enumerate(levels):
        if side == "Buy":
            tp_price = entry_price + mult * atr
            tp_side = "Sell"
        else:  # Sell
            tp_price = entry_price - mult * atr
            tp_side = "Buy"

        # Place limit TP order with reduce_only flag
        resp = session.place_active_order(
            symbol=symbol,
            side=tp_side,
            order_type="Limit",
            price=round(tp_price, 4),
            qty=round(qty_per_tp, 4),
            time_in_force="PostOnly",  # avoid immediate execution
            reduce_only=True,
            close_on_trigger=False,
            leverage=leverage
        )
        tp_orders.append(resp)
    orders['tp'] = tp_orders

    return orders
def update_trailing_stop(symbol, side, atr, current_price):
    """
    Updates trailing stop loss to follow price with a buffer of 1.5 * ATR
    
    Args:
        symbol (str): trading symbol
        side (str): "Buy" or "Sell"
        atr (float): current ATR value
        current_price (float): current mark price
    """
    if side == "Buy":
        # Trailing SL = current price - 1.5 * ATR
        new_sl = current_price - 1.5 * atr
    else:
        # Trailing SL = current price + 1.5 * ATR
        new_sl = current_price + 1.5 * atr

    try:
        session.set_trading_stop(
            category="linear",
            symbol=symbol,
            side=side,
            trailing_stop=round(1.5 * atr, 4),  # set trailing stop distance
            stop_loss=None  # We let trailing stop handle SL now
        )
        print(f"[INFO] Trailing SL updated at {round(new_sl,4)}")
    except Exception as e:
        print(f"[ERROR] Failed to update trailing SL: {e}")

def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)

def run_bot():
    active_trade = False
    balance = get_balance()
    risk_pct = 0.10
    leverage = 20
    order_id = ""
    current_trade_side = ""
    # session.set_leverage(
    #     category="linear",  # or "inverse" depending on the contract
    #     symbol="XRPUSDT",
    #     buyLeverage="10",   # or your preferred long leverage
    #     sellLeverage="10",  # or your preferred short leverage
    #     marginMode="ISOLATED"  # <- switch to isolated here
    # )

    while True:
        df = get_klines_df(coin, 30)
        df = add_indicators(df)
        df = generate_signals(df)
        # print(f"atr: {df['atr'].iloc[-1]}")

        mark_price = get_mark_price(coin)
        signal = df["signal"].iloc[-1]
        # print("in main loop")

        print(f"signal: {signal}")
        print(f"macd cross up: {df['macd_cross_up'].iloc[-1]}")
        print(f"macd cross down: {df['macd_cross_down'].iloc[-1]}")
        print(f"atr: {df['atr'].iloc[-1]}")
        print(f"")

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
                current_price = get_mark_price(coin)
                atr = df['atr'].iloc[-1]
                update_trailing_stop(coin, current_trade_side, atr, current_price)

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
                order_id = enter_trade("Buy", df)  # Your existing entry function
                current_trade_side = "Buy"
                active_trade = True
                # Place SL and multiple TPs
                place_sl_and_tp(coin, "Buy", entry_price, df['atr'].iloc[-1], total_qty)
                
            elif signal == "Sell":
                total_qty = get_trade_qty()
                entry_price = get_mark_price(coin)
                order_id = enter_trade("Sell", df)
                current_trade_side = "Sell"
                active_trade = True
                # Place SL and multiple TPs
                place_sl_and_tp(coin, "Sell", entry_price, df['atr'].iloc[-1], total_qty)


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
