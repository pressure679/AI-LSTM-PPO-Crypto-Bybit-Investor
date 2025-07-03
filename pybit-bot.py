import time
import math
from datetime import datetime
import requests
import pandas as pd
from pybit.unified_trading import HTTP  # pip install pybit
import numpy as np
from io import StringIO
import math

# === SETUP ===
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
bnb = "BNBUSDT"
# balance = 100
# risk_pct = 0.1
leverage = 75
risk_pct = 0.1

session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    recv_window=5000,
    timeout=30
    # testnet=True  # uncomment if needed
)

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
    step, min_qty = get_qty_step(bnb)
    qty = math.floor(qty / step) * step
    qty = round(qty, 4)
    if qty < min_qty:
        print(f"[WARNING] Qty {qty} below minimum {min_qty}")
        return 0
    return qty

def get_tp_levels(entry_price, atr, leverage, direction="Buy"):
    """
    Returns a list of TP price levels for partial profit taking.
    Example: 3 TPs at 0.5, 1.0, and 1.5 ATR multiples.
    """
    atr_value = atr / entry_price  # normalize ATR to % of price
    # tp_multipliers = [0.5, 1.0, 1.5]
    tp_multipliers = [1.5, 2.5, 3.5, 4.5]
    if direction.lower() == "buy":
        return [round(entry_price * (1 + m * atr_value * leverage), 4) for m in tp_multipliers]
    else:
        return [round(entry_price * (1 - m * atr_value * leverage), 4) for m in tp_multipliers]

def get_sl_level(entry_price, atr, leverage, direction="Buy"):
    """
    Stop loss level based on ATR multiplier, tighter than TP.
    """
    atr_value = atr / entry_price
    sl_multiplier = 1.0  # e.g. 1 ATR stop loss
    if direction.lower() == "buy":
        # 1.5 is atr multiplier
        return round(entry_price * (1 - 1.5 * atr_value * leverage), 4)
    else:
        return round(entry_price * (1 + 1.5 * atr_value * leverage), 4)

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
def setup_trade_tp_levels(order_id, side, entry_price, total_qty, leverage, atr):
    """
    Initialize the TP levels for a new trade.
    Example: 4 TP levels spaced by ATR or fixed %.
    You can customize as needed.
    """
    # For example, 4 TP levels with incremental profits:
    if side.lower() == "buy":
        tp_levels = [
            entry_price * (1 + atr),
            entry_price * (1 + atr * 2),
            entry_price * (1 + atr * 3),
            entry_price * (1 + atr * 4),
        ]
    else:  # sell
        tp_levels = [
            entry_price * (1 - atr),
            entry_price * (1 - atr * 2),
            entry_price * (1 - atr * 3),
            entry_price * (1 - atr * 4),
        ]

    current_trade_info.update({
        "tp_levels": tp_levels,
        "tp_done": [False, False, False, False],
        "order_id": order_id,
        "side": side,
        "total_qty": total_qty,
        "leverage": leverage,
    })
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
    partial_qty_fractions = [0.4, 0.2, 0.2, 0.2]

    for i, tp_price in enumerate(tp_levels):
        if tp_done[i]:
            continue  # already closed partial for this TP

        if side.lower() == "buy" and mark_price >= tp_price:
            # Price reached TP level on a long position
            qty_to_close = total_qty * partial_qty_fractions[i]
            qty_to_close = round(qty_to_close, 4)  # round to allowed precision

            if qty_to_close <= 0:
                continue

            print(f"[INFO] TP level {i+1} reached. Closing {qty_to_close} qty on Buy position at {mark_price:.4f}")

            # Place a reduce position order - assuming Bybit supports reduce-only market order
            response = session.place_active_order(
                symbol=bnb,
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

            print(f"[INFO] TP level {i+1} reached. Closing {qty_to_close} qty on Sell position at {mark_price:.4f}")

            response = session.place_active_order(
                symbol=bnb,
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
def trailing_stop_check(position, atr, trail_multiplier=0.5):
    """
    Adjust trailing stop loss based on price movement and ATR.
    Should be called periodically to update stop loss.

    position: dict from session.get_positions
    atr: current ATR value
    trail_multiplier: fraction of ATR to trail stop by

    This example only updates stop loss price if price moves favorably.
    """

    symbol = position["symbol"]
    side = position["side"]
    entry_price = float(position["entryPrice"])
    current_stop_loss = float(position.get("stopLossPrice", 0))
    qty = float(position["size"])
    mark_price = get_mark_price(symbol)

    trail_distance = atr * trail_multiplier

    if side == "Buy":
        new_stop_loss = max(current_stop_loss, mark_price - trail_distance) if current_stop_loss > 0 else mark_price - trail_distance
        new_stop_loss = round(new_stop_loss, 4)
        if new_stop_loss > current_stop_loss:
            # update stop loss
            try:
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    stop_loss=new_stop_loss
                )
                print(f"[TRAILING STOP UPDATED] New SL: {new_stop_loss}")
            except Exception as e:
                print(f"[ERROR] Updating trailing stop: {e}")

    elif side == "Sell":
        new_stop_loss = min(current_stop_loss, mark_price + trail_distance) if current_stop_loss > 0 else mark_price + trail_distance
        new_stop_loss = round(new_stop_loss, 4)
        if new_stop_loss < current_stop_loss or current_stop_loss == 0:
            try:
                session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    stop_loss=new_stop_loss
                )
                print(f"[TRAILING STOP UPDATED] New SL: {new_stop_loss}")
            except Exception as e:
                print(f"[ERROR] Updating trailing stop: {e}")

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

def get_klines_df(bnb, interval, limit=1000):
    response = session.get_kline(
        category="linear",
        symbol=bnb,
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

def get_mark_price(bnb):
    price_data = session.get_tickers(category="linear", symbol=bnb)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(bnb):
    info = session.get_instruments_info(category="linear", symbol=bnb)
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

    price = float(session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["lastPrice"])
    raw_qty = trade_usd_amount / price

    step, min_qty = get_qty_step(symbol)
    qty = math.floor(raw_qty / step) * step
    qty = round(qty, 4)  # round for precision

    if qty < min_qty:
        qty = min_qty

    return qty

def close_all_positions():
    positions = session.get_positions(category="linear", symbol=bnb)
    position_lists = positions['result']['list']
    # for position in positions:
    for position in position_lists:
        qty = float(position["size"])
        if qty == 0:
            continue
        side = "Sell" if position["side"] == "Buy" else "Buy"
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, reduce_only=True)
        print(f"[TRADE CLOSED]")

def cancel_specific_order(order_id: str, bnb: str):
    if not order_id:
        print("No order ID to cancel")
        return
    try:
        response = session.cancel_order(category="linear", order_id=order_id, symbol=symbol)
        print("Cancel order response:", response)
    except Exception as e:
        print("Error canceling order:", e)

def enter_trade(signal, df, symbol="BNBUSDT", n_tp=4):
    global leverage
    
    if signal.lower() not in ["buy", "sell"]:
        print("[WARN] Invalid signal for enter_trade.")
        return None

    mark_price = get_mark_price(symbol)
    
    # Skip trading if ATR range too low
    if df['atr'].iloc[-1] / df['Close'].iloc[-1] < 0.001:
        print("[INFO] Market range too small, skipping trade.")
        return None

    balance = get_balance()
    risk_amount = max(balance * risk_pct, 10)

    # Calculate qty based on risk and price
    step, min_qty = get_qty_step(symbol)
    total_qty = risk_amount / mark_price
    total_qty = math.floor(total_qty / step) * step
    if total_qty < min_qty:
        print(f"[WARN] Quantity {total_qty} below minimum {min_qty}, skipping trade.")
        return None

    # Split qty into n partial orders
    partial_qty = math.floor(total_qty / n_tp / step) * step
    if partial_qty < min_qty:
        print(f"[WARN] Partial qty {partial_qty} below minimum {min_qty}, reducing TP levels.")
        # If partial qty too small, reduce n_tp accordingly
        n_tp = max(1, int(total_qty // min_qty))
        partial_qty = math.floor(total_qty / n_tp / step) * step

    side = "Buy" if signal.lower() == "buy" else "Sell"

    # Get multiple TP levels, e.g. [tp1, tp2, tp3, tp4]
    tp_levels = get_tp_levels(df, signal)
    # if not tp_levels or len(tp_levels) < n_tp:
    #     print("[WARN] Not enough TP levels returned, using default calculations.")
    #     # fallback example TP levels (simple increments)
    #     # increment = 0.001  # 0.1%
    #     # tp_levels = []
    #     for i in range(1, n_tp+1):
    #         if side == "Buy":
    #             tp_levels.append(mark_price * (1 + increment * i))
    #         else:
    #             tp_levels.append(mark_price * (1 - increment * i))
    
    # Get SL level
    sl_price = get_sl_level(df, signal)
    if sl_price is None:
        # fallback SL, e.g. ATR based or fixed % SL
        if side == "Buy":
            # sl_price = mark_price * (1 - 0.0015)
            sl_price = mark_price * (1 - df['atr'].iloc[-1])
        else:
            sl_price = mark_price * (1 + df['atr'].iloc[-1])
    
    sl_price_str = f"{sl_price:.4f}"

    # Place partial market orders for each TP level
    order_ids = []
    for i in range(n_tp):
        tp_price_str = f"{tp_levels[i]:.4f}"
        try:
            response = session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                order_type="Market",
                qty=partial_qty,
                buyLeverage=leverage,
                sellLeverage=leverage,
                take_profit=tp_price_str,
                stop_loss=sl_price_str
            )
            order_id = response.get("result", {}).get("orderId")
            order_ids.append(order_id)
            print(f"[INFO] Placed partial order {i+1}/{n_tp} with TP={tp_price_str}, SL={sl_price_str}")
        except Exception as e:
            print(f"[ERROR] Failed to place partial order {i+1}: {e}")

    # After placing entries, set trading stop (trailing stop) for the full position
    try:
        # This adjusts SL and TS on the entire position (you may tune trailing_stop params)
        response = session.set_trading_stop(
            symbol=symbol,
            side=side,
            trailing_stop=0.0015,  # example trailing stop distance, adjust to your liking
            stop_loss=sl_price_str
        )
        print("[INFO] Set trailing stop and SL for position.")
    except Exception as e:
        print(f"[ERROR] Failed to set trailing stop: {e}")

    return order_ids

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
            
        if row["ema_28_diff"] > 0 and row['adx'] > 20:
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row["rsi"] < 70 :
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] > 0:
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] < 70 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs badly with high leverage
        # if row["ema_28_diff"] > 0 and row["adx"] > 20 and row['macd_cross_up']: # performs badly with high leverage
        # if row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] > 0 and row['adx'] > 20 and row['macd_line'] > row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] > 0 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # has about a 1:2 RR, but performs badly with high leverage
            signal = "buy"
        if row["ema_28_diff"] < 0 and row['adx'] > 20:
        # if row["ema_28_diff"] < 0 and row["adx"] > 20 and row["rsi"] > 30:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and macd_momentum_increasing and row['macd_line'] < 0:
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and macd_momentum_increasing and row['rsi'] > 30 and row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # performs badly on high leverage
        # if row["ema_28_diff"] < 0 and row["adx"] > 20 and row['macd_cross_down']: # performs badly with high leverage
        # if row['macd_line'] < row['macd_signal'] and curr_hist < prev_hist: # performs ok with high leverage
        # if row["ema_28_diff"] < 0 and row['adx'] > 20 and row['macd_line'] < row['macd_signal']: # has up to 1:2 RR, but performs badly with high leverage
        # if row['ema_28_diff'] < 0 and row['macd_line'] > row['macd_signal'] and curr_hist > prev_hist: # has about a 1:2 RR , but performs badly with high leverage
            signal = "sell"
            
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
        df = get_klines_df(bnb, 60)
        df = add_indicators(df)
        df = generate_signals(df)
        
        mark_price = get_mark_price(bnb)
        signal = df["signal"].iloc[-1]

        if active_trade:
            # Example logic for partial TP orders
            # This is where you call your function to reduce position by partial amounts per TP level
            # For example, something like:
            partial_tp_check_and_reduce_position(order_id, current_trade_side, mark_price)
            
            # If signal flipped opposite direction, close current position
            if signal.lower() != current_trade_side.lower():
                cancel_specific_order(order_id, bnb)
                active_trade = False
                print(f"Order closed due to opposite signal")

        # If no active trade, open one based on signal
        if not active_trade:
            if signal.lower() == "buy":
                order_id = enter_trade("Buy", df, symbol=bnb)
                # Assume you get entry price and qty from response or from mark price & calculation
                entry_price = get_mark_price(bnb)
                total_qty = calculate_qty_for_trade(...)  # your qty calculation function or reuse risk logic
                setup_trade_tp_levels(order_id, "Buy", entry_price, total_qty, leverage)
                current_trade_side = "Buy"
                active_trade = True
                print(f"Placed new Buy order")
            elif signal.lower() == "sell":
                order_id = enter_trade("Sell", df, symbol=bnb)
                entry_price = get_mark_price(bnb)
                # total_qty = calculate_qty_for_trade(...)
                total_qty = get_trade_qty()
                setup_trade_tp_levels(order_id, "Sell", entry_price, total_qty, leverage)
                current_trade_side = "Sell"
                active_trade = True
                print(f"Placed new Sell order")
        # Wait until the next candle to avoid over-trading within the same candle
        wait_until_next_candle(1)

run_bot()
