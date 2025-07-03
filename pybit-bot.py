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
    # testnet=True  # uncomment if needed
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
def calculate_position_size(balance, risk_pct, atr, price, atr_multiplier=1.5):
    """
    Calculate position size based on risk percentage and ATR.
    risk_amount = balance * risk_pct
    stop_loss_distance = atr * atr_multiplier
    qty = risk_amount / stop_loss_distance / price  # quantity in coins
    """
    risk_amount = balance * risk_pct
    # stop_loss_distance = atr * atr_multiplier
    stop_loss_distance = atr * 1.5
    if stop_loss_distance == 0:
        return 0  # avoid division by zero
    qty = risk_amount / (stop_loss_distance * price)
    step, min_qty = get_qty_step(coin)
    qty = math.floor(qty / step) * step
    qty = round(qty, 4)
    if qty < min_qty:
        print(f"[WARNING] Qty {qty} below minimum {min_qty}")
        return 0
    return qty

def get_tp_levels(entry_price, atr, leverage, direction):
    """
    Returns a list of TP price levels for partial profit taking.
    Example: 3 TPs at 0.5, 1.0, and 1.5 ATR multiples.
    """
    atr_value = float(atr) / float(entry_price)  # normalize ATR to % of price
    # tp_multipliers = [0.5, 1.0, 1.5]
    tp_multipliers = [0.02, 0.03, 0.04]
    # if direction.lower() == "buy":
    if direction == "Buy":
        return [round(entry_price * (1 + m * atr_value), 4) for m in tp_multipliers]
        # return [round(entry_price * (1 + m * atr * leverage), 4) for m in tp_multipliers]
    else:
        return [round(entry_price * (1 - m * atr_value), 4) for m in tp_multipliers]
        # return [round(entry_price * (1 - m * atr * leverage), 4) for m in tp_multipliers]
def get_sl_tp_levels(entry_price, leverage, direction, tp_roi=50, sl_roi=25):
    """
    Calculate TP and SL prices based on ROI percentage and leverage.
    
    :param entry_price: float, price at which position is entered
    :param leverage: int or float, leverage used
    :param direction: str, "Buy" or "Sell"
    :param tp_roi: float, take profit ROI percentage
    :param sl_roi: float, stop loss ROI percentage
    :return: (sl_price, tp_price) tuple
    """
    tp_move = (tp_roi / 100) / leverage
    sl_move = (sl_roi / 100) / leverage
    
    if direction == "Buy":
        tp_price = entry_price * (1 + tp_move)
        sl_price = entry_price * (1 - sl_move)
    else:  # Sell
        tp_price = entry_price * (1 - tp_move)
        sl_price = entry_price * (1 + sl_move)
        
    return round(sl_price, 2), round(tp_price, 4)

# A dictionary to keep track of TP levels for the current trade
# Structure example:
# {
#   "tp_levels": [tp1_price, tp2_price, tp3_price, tp4_price],
#   "tp_done": [False, False, False, False],
#   "order_id": "some_order_id",
#   "side": "Buy" or "Sell",
#   "total_qty": float,
# }
current_trade_info = {}
def validate_sl_tp(entry_price, sl_price, tp_price, side, atr=None):
    """
    Validates SL and TP, adjusting them if invalid.
    """
    atr = atr or entry_price * 0.01  # fallback: 1% of entry

    if side == "Buy":
        # SL must be < entry, TP must be > entry
        if sl_price >= entry_price:
            sl_price = entry_price - 0.5 * atr
        if tp_price <= entry_price:
            tp_price = entry_price + 1.5 * atr
    else:
        # SL must be > entry, TP must be < entry
        if sl_price <= entry_price:
            sl_price = entry_price + 0.5 * atr
        if tp_price >= entry_price:
            tp_price = entry_price - 1.5 * atr

    return round(sl_price, 4), round(tp_price, 4)
def setup_trade_tp_levels(order_id, side, entry_price, total_qty, leverage, atr):
    """
    Initialize the TP levels for a new trade.
    Example: 4 TP levels spaced by ATR or fixed %.
    You can customize as needed.
    """
    # For example, 4 TP levels with incremental profits:
    if side == "Buy":
        tp_levels = [
            entry_price * (1 + atr * 0.5),
            entry_price * (1 + atr * 1),
            entry_price * (1 + atr * 1.5),
            # entry_price * (1 + atr * 4),
        ]
    else:  # sell
        tp_levels = [
            entry_price * (1 - atr * 0.5),
            entry_price * (1 - atr * 1),
            entry_price * (1 - atr * 1.5),
            # entry_price * (1 - atr * 4),
        ]

    current_trade_info.update({
        "tp_levels": tp_levels,
        "tp_done": [False, False, False, False],
        "order_id": order_id,
        "side": side,
        "total_qty": total_qty,
        "leverage": leverage,
    })
def setup_trade(df, signal, symbol):
    entry_price = df["Close"].iloc[-1]
    atr = df["atr"].iloc[-1]
    adx = df["adx"].iloc[-1]

    high_range = df["High"].iloc[-14:].max()
    low_range = df["Low"].iloc[-14:].min()
    market_range_pct = (high_range - low_range) / entry_price * 100

    # Dynamic threshold based on ADX
    if adx < 20:
        print(f"[INFO] ADX too low ({adx:.2f}), skipping trade.")
        return None
    elif adx < 30 and market_range_pct < 1.5:
        print(f"[INFO] ADX moderate ({adx:.2f}) but market range low ({market_range_pct:.2f}%), skipping.")
        return None
    elif adx >= 30 and market_range_pct < 1.0:
        print(f"[INFO] ADX strong ({adx:.2f}) but range too small ({market_range_pct:.2f}%), skipping.")
        return None

    # Passed filters
    print(f"[INFO] ADX {adx:.2f}, range {market_range_pct:.2f}%, trade OK.")
    return True
    # return {
    #     "entry_price": entry_price,
    #     "atr": atr,
    #     "adx": adx,
    #     "range_pct": market_range_pct,
    # }
def partial_tp_check_and_reduce_position(order_id, side, mark_price):
    """
    Check if price reached any TP level not yet done, and place a partial close order.
    """
    if not current_trade_info or current_trade_info.get("order_id") != order_id:
        # No trade info or mismatched order id
        return

    tp_levels = current_trade_info["tp_levels"]
    tp_done = current_trade_info["tp_done"]
    total_qty = current_trade_info["total_qty"]
    leverage = current_trade_info["leverage"]

    # Define partial quantities for each TP level (sum to 1.0)
    partial_qty_fractions = [0.4, 0.4, 0.2]

    for i, tp_price in enumerate(tp_levels):
        if tp_done[i]:
            continue  # already closed partial for this TP

        if side.lower() == "buy" and mark_price >= tp_price:
            # Price reached TP level on a long position
            qty_to_close = total_qty * partial_qty_fractions[i]
            qty_to_close = round(qty_to_close, 4)  # round to allowed precision

            if qty_to_close <= 0:
                continue

            print(f"[INFO] TP level {i+1} reached. Closing {qty_to_close} qty on Buy position at {mark_price:.2f}")

            # Place a reduce position order - assuming Bybit supports reduce-only market order
            response = session.place_active_order(
                symbol=coin,
                side="Sell",  # opposite side to close position
                order_type="Market",
                qty=qty_to_close,
                reduce_only=True,
                time_in_force="ImmediateOrCancel",
                leverage=leverage
            )
            if response.get("ret_msg") == "OK":
                tp_done[i] = True
                # Update the total_qty after partial close
                current_trade_info["total_qty"] -= qty_to_close

        elif side.lower() == "sell" and mark_price <= tp_price:
            # Price reached TP level on a short position
            qty_to_close = total_qty * partial_qty_fractions[i]
            qty_to_close = round(qty_to_close, 4)

            if qty_to_close <= 0:
                continue

            print(f"[INFO] TP level {i+1} reached. Closing {qty_to_close} qty on Sell position at {mark_price:.2f}")

            response = session.place_active_order(
                symbol=coin,
                side="Buy",  # opposite side to close position
                order_type="Market",
                qty=qty_to_close,
                reduce_only=True,
                time_in_force="ImmediateOrCancel",
                leverage=leverage
            )
            if response.get("ret_msg") == "OK":
                tp_done[i] = True
                current_trade_info["total_qty"] -= qty_to_close
def trailing_stop_check(position, atr, trail_multiplier=0.01):
    """
    Adjusts trailing stop loss based on price movement and ATR.

    :param position: dict from session.get_positions()["result"]["list"][i]
    :param atr: current ATR value
    :param trail_multiplier: how far behind price the SL should trail
    """

    symbol = position["symbol"]
    side = position["side"]
    # print("Position data keys:", position.keys())
    # print(position)
    # entry_price = float(position["entryPrice"])
    entry_price = float(position["avgPrice"])
    qty = float(position["size"])
    current_stop_loss = float(position.get("stopLoss", 0))  # some Bybit keys use stopLoss
    mark_price = get_mark_price(symbol)

    trail_distance = atr * trail_multiplier

    if side == "Buy":
        new_stop_loss = mark_price * (1 - trail_distance)
        new_stop_loss = round(new_stop_loss, 2)
        if current_stop_loss == 0 or new_stop_loss > current_stop_loss:
            try:
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    stop_loss=new_stop_loss
                )
                print(f"[TRAILING STOP UPDATED] Buy SL: {new_stop_loss}")
            except Exception as e:
                print(f"[ERROR] Updating trailing stop (Buy): {e}")

    elif side == "Sell":
        new_stop_loss = mark_price * (1 + trail_distance)
        new_stop_loss = round(new_stop_loss, 2)
        if current_stop_loss == 0 or new_stop_loss < current_stop_loss:
            try:
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    stop_loss=new_stop_loss
                )
                print(f"[TRAILING STOP UPDATED] Sell SL: {new_stop_loss}")
            except Exception as e:
                print(f"[ERROR] Updating trailing stop (Sell): {e}")

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

def get_klines_df(coin, interval, limit=1000):
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

def close_all_positions():
    positions = session.get_positions(category="linear", symbol=coin)
    position_lists = positions['result']['list']
    # for position in positions:
    for position in position_lists:
        qty = float(position["size"])
        if qty == 0:
            continue
        side = "Sell" if position["side"] == "Buy" else "Buy"
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, reduce_only=True)
        print(f"[TRADE CLOSED]")

def cancel_specific_order(order_id: str, coin: str):
    if not order_id:
        print("No order ID to cancel")
        return
    try:
        response = session.cancel_order(category="linear", order_id=order_id, symbol=symbol)
        print("Cancel order response:", response)
    except Exception as e:
        print("Error canceling order:", e)

def enter_trade(signal, df, symbol="COINUSDT", n_tp=3):
    global leverage
    global current_trade_info

    if signal not in ["Buy", "Sell"]:
        print("[WARN] Invalid signal for enter_trade.")
        return None

    mark_price = get_mark_price(symbol)
    entry_price = mark_price

    balance = get_balance()
    risk_amount = max(balance * risk_pct, 10)

    # Calculate qty
    step, min_qty = get_qty_step(symbol)
    total_qty = risk_amount / mark_price
    total_qty = math.floor(total_qty / step) * step
    if total_qty < min_qty:
        print(f"[WARN] Quantity {total_qty} below minimum {min_qty}, skipping trade.")
        return None

    side = "Buy" if signal == "Buy" else "Sell"

    # === SL/TP logic based on ROI ===
    sl_price, tp_price = get_sl_tp_levels(entry_price, leverage, side, tp_roi=50, sl_roi=25)

    sl_price_str = f"{sl_price:.2f}"
    tp_price_str = f"{tp_price:.2f}"
    order_id = ""

    try:
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=total_qty,
            buyLeverage=leverage,
            sellLeverage=leverage,
            take_profit=tp_price_str,
            stop_loss=sl_price_str
        )
        order_id = response.get("result", {}).get("orderId")
        print(f"[INFO] Placed {side} order with TP={tp_price_str}, SL={sl_price_str}")
    except Exception as e:
        print(f"[ERROR] Failed to place order: {e}")
        return None

    # Generate TP levels
    tp_levels = [
        round(entry_price + (tp_price - entry_price) * i / n_tp, 2)
        if side == "Buy"
        else round(entry_price - (entry_price - tp_price) * i / n_tp, 2)
        for i in range(1, n_tp + 1)
    ]

    if order_id:
        current_trade_info = {
            "order_id": order_id,
            "tp_levels": tp_levels,
            "tp_done": [False] * len(tp_levels),
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
        if i < 1:
            signals.append("")
            # modes.append("")
            continue
        row = df.iloc[i]
        prev = df.iloc[i-1]
        signal = ""
        # mode = ""

        # rsi_rising = row["rsi"] > prev["rsi"]
        # rsi_falling = row["rsi"] < prev["rsi"]

        # prev_hist = df.iloc[i - 1]["macd_hist"]
        # curr_hist = row["macd_hist"]
        # # macd_increasing = curr_hist > prev_hist
        # # macd_decreasing = curr_hist < prev_hist
        # prev_macd_momentum = df.iloc[i - 1]["macd_momentum"]
        # curr_macd_momentum = row["macd_momentum"]
        # macd_momentum_increasing = curr_macd_momentum > prev_macd_momentum
        # macd_momentum_decreasing = curr_macd_momentum < prev_macd_momentum
            
        # if row["ema_14"] > row["ema_28"] and row['adx'] > 20:
        if row["ema_28_diff"] > 0 and row['adx'] > 20:
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row["rsi"] < 70 :
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] > 0:
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] < 70 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs badly with high leverage
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row['macd_cross_up']: # performs badly with high leverage
        # if row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and row['macd_line'] > row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] > 0 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # has about a 1:2 RR, but performs badly with high leverage
            signal = "Buy"
        # if row["ema_14"] < row["ema_28"] and row['adx'] > 20:
        if row["ema_28_diff"] < 0 and row['adx'] > 20:
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
        df = get_klines_df(coin, 60)
        df = add_indicators(df)
        df = generate_signals(df)

        mark_price = get_mark_price(coin)
        signal = df["signal"].iloc[-1]

        if active_trade:
            # Fetch open positions for symbol
            positions = session.get_positions(category="linear", symbol=coin)["result"]["list"]
            
            # Find the relevant open position (non-zero size, matching side)
            # position = None
            for pos in positions:
                if float(pos["size"]) > 0 and pos["side"] == current_trade_side:
                    position = pos
                    break
            
            if position:
                # Call trailing stop check with current ATR from df
                current_atr = df['atr'].iloc[-1]
                trailing_stop_check(position, current_atr)
            else:
                # No open position found: reset state
                active_trade = False
                print("[INFO] Position closed or missing, resetting active_trade flag.")

            # Example partial TP check (your existing logic)
            # partial_tp_check_and_reduce_position(order_id, current_trade_side, mark_price)

            # Close position if signal flips
            if signal != current_trade_side:
                cancel_specific_order(order_id, coin)
                active_trade = False
                print(f"Order closed due to opposite signal")

        if not active_trade and signal != "":
            setup = setup_trade(df, signal, symbol=coin)
            if setup is False:
                print("[INFO] Trade setup skipped due to low volatility or invalid conditions.")
                wait_until_next_candle(1)
                continue
            if signal == "Buy":
                order_id = enter_trade("Buy", df, symbol=coin)
                entry_price = get_mark_price(coin)
                total_qty = get_trade_qty()
                setup_trade_tp_levels(order_id, "Buy", entry_price, total_qty, leverage, df['atr'].iloc[-1])
                current_trade_side = "Buy"
                active_trade = True
                # print(f"Placed new Buy order")
            elif signal == "Sell":
                order_id = enter_trade("Sell", df, symbol=coin)
                entry_price = get_mark_price(coin)
                total_qty = get_trade_qty()
                setup_trade_tp_levels(order_id, "Sell", entry_price, total_qty, leverage, df['atr'].iloc[-1])
                current_trade_side = "Sell"
                active_trade = True
                # print(f"Placed new Sell order")

        wait_until_next_candle(1)

run_bot()
