import time
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import threading
import requests
import math
from math import floor
import traceback
from decimal import Decimal, ROUND_DOWN
import os
import time

pd.set_option('future.no_silent_downcasting', True)

# API setup
api_key = ""
api_secret = ""
session = HTTP(demo=True, api_key=api_key, api_secret=api_secret)
SYMBOLS = ["SHIB1000USDT", "XRPUSDT", "DOGEUSDT", "FARTCOINUSDT", "SUIUSDT",
           "HYPEUSDT", "INITUSDT", "BABYUSDT", "NILUSDT", "XAUTUSDT"]
risk_pct = 0.1  # Example risk per trade
leverage=50
current_trade_info = []
# balance = 160

atr_sl_multiplier = 0.5
atr_tp_multiplier = 1

def keep_session_alive(symbol):
    for attempt in range(30):
        try:
            # Example: Get latest position
            result = session.get_positions(category="linear", symbol=symbol)
            break  # If success, break out of loop
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on attempt {attempt+1}, retrying...")
            time.sleep(2)  # wait before retry
        finally:
            threading.Timer(1500, keep_session_alive, args(symbol,)).start()  # Schedule next call
def get_klines_df(symbol, interval, limit=208):
    response = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
    data = response['result']['list']
    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.int64), unit='ms')
    return df

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def RSI(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    # red_line = ema_fast - ema_slow
    return macd_line, signal_line, histogram

def Bollinger_Bands(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band, sma

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

def calculate_indicators(df):
    # df.dropna(inplace=True)
    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_28'] = df['Close'].ewm(span=28).mean()
    df['EMA_7_Diff'] = df['EMA_7'].diff()
    df['EMA_14_Diff'] = df['EMA_14'].diff()
    df['EMA_28_Diff'] = df['EMA_28'].diff()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    # === MACD Crossovers ===
    # df['macd_cross_up'] = (df['macd_histogram'] > 0) & (df['macd_histogram'].shift(1) <= 0)
    # df['macd_cross_down'] = (df['macd_histogram'] < 0) & (df['macd_histogram'].shift(1) >= 0)
    df['macd_cross_up'] = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
    df['macd_signal_cross_up'] = (df['macd_signal'] > 0) & (df['macd_signal'].shift(1) <= 0)
    df['macd_signal_cross_down'] = (df['macd_signal'] < 0) & (df['macd_signal'].shift(1) >= 0)
    df['macd_signal_diff'] = df['macd_signal'].diff()
    df['OSMA'] = df['macd_line'] - df['macd_signal']
    df['OSMA_Diff'] = df['OSMA'].diff()

    # === MACD Trend Status ===
    df['macd_trending_up'] = df['macd_line'] > df['macd_signal']
    df['macd_trending_down'] = df['macd_line'] < df['macd_signal']
    df['macd_histogram_increasing'] = df['macd_histogram'].diff() > 0
    df['macd_histogram_decreasing'] = df['macd_histogram'].diff() < 0

    df['bb_upper'], df['bb_lower'], df['bb_sma'] = Bollinger_Bands(df['Close'])
    df['bb_sma_pos'] = (df['Close'] >= df['bb_sma']).astype(int)

    # df['Momentum'] = df['Close'] - df['Close'].shift(10)
    # Custom momentum (% of recent high-low range)
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    price_range = high_14 - low_14

    df['RSI'] = RSI(df['Close'], 14)
    adx, plus_di, minus_di = ADX(df)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    df['DI_Diff'] = (df['+DI'] - df['-DI']).abs()

    df['Bulls'] = BullsPower(df)     # High – EMA(close, 13)
    df['Bears'] = BearsPower(df)     # Low  – EMA(close, 13)

    return df

def generate_signals(df):
    """
    Return a Series of 'Buy', 'Sell', or '' using
        • flat‑market veto             (10‑bar High/Low range)
        • ADX strength > 20
        • MACD‑angle filter            (current slope > 5‑bar avg slope)
        • MACD hook skip               (no trade immediately after peak/valley)
        • DI / Bull‑Bear / OSMA logic  (your original direction rules)
    """

    signals = [""] * len(df)

    # --- pre‑compute helpers once ------------------------------------
    macd_slope      = df['macd_line'].diff()
    macd_slope_ma_5 = macd_slope.rolling(5).mean()

    # “hook” is True on the bar *right after* MACD slope changes sign
    macd_hook = (
        (macd_slope.shift(1) > 0) & (macd_slope <= 0) |   # bullish peak
        (macd_slope.shift(1) < 0) & (macd_slope >= 0)     # bearish valley
    )

    for i in range(len(df)):
        if i < 10:        # need at least 10 bars for range filter
            continue

        latest = df.iloc[i]

        # -------------- flat‑market veto ------------------------------
        window = df.iloc[i - 10 : i]
        price_range = window["High"].max() - window["Low"].min()
        if (price_range / window["Close"].mean()) <= 0.005:
            continue

        # -------------- trend strength check --------------------------
        if latest.ADX < 20:
            continue

        # -------------- MACD angle filter -----------------------------
        angle_ok = abs(macd_slope.iloc[i]) > abs(macd_slope_ma_5.iloc[i])
        if not angle_ok:
            continue

        # -------------- hook (peak/valley) skip -----------------------
        if macd_hook.iloc[i]:
            continue

        # -------------- directional logic -----------------------------
        # if (latest.macd_signal_diff > 0 and
        #     latest.Close > latest.bb_sma and
        #     latest['+DI'] > latest['-DI'] and
        #     latest.Bulls > 0 and
        #     latest.OSMA_Diff > 0):
        # if (latest.Close > latest.bb_sma and
        #     latest["+DI"] > latest["-DI"] and
        #     latest.OSMA_Diff > 0):
        # if latest.Close > latest.bb_sma:
        # if (latest.Bulls > 0 and latest.Bears > 0):
        if latest.macd_line > latest.macd_signal:
            signals[i] = "Buy"

        # elif latest.Close < latest.bb_sma:
        # elif (latest.Close < latest.bb_sma and
        #     latest["+DI"] < latest["-DI"] and
        #     latest.OSMA_Diff < 0):
        # elif (latest.Bulls < 0 and latest.Bears < 0):
        if latest.macd_line < latest.macd_signal:
            signals[i] = "Sell"

    df["signal"] = signals
    return df["signal"]

def get_balance():
    """
    Return total equity in USDT only.
    """
    try:
        resp = session.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"               # still useful – filters server‑side
        )

        coins = resp["result"]["list"][0]["coin"]   # ← go into the coin array
        usdt_row = next(c for c in coins if c["coin"] == "USDT")
        return float(usdt_row["equity"])            # or "availableToWithdraw"
    except (KeyError, StopIteration, IndexError) as e:
        print("[get_balance] parsing error:", e, resp)
        return 0.0

def get_mark_price(symbol):
    price_data = session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(symbol):
    info = session.get_instruments_info(category="linear", symbol=symbol)
    data = info["result"]["list"][0]
    step = float(data["lotSizeFilter"]["qtyStep"])
    min_qty = float(data["lotSizeFilter"]["minOrderQty"])
    return step, min_qty
def calc_order_qty(risk_amount: float,
                   entry_price: float,
                   min_qty: float,
                   qty_step: float) -> float:
    """
    Return a Bybit‑compliant order quantity, rounded *down* to the nearest
    step size.  If the rounded amount is below `min_qty`, return 0.0.
    """
    # 1️⃣  Convert everything to Decimal
    risk_amt   = Decimal(str(risk_amount))
    price      = Decimal(str(entry_price))
    step       = Decimal(str(qty_step))
    min_q      = Decimal(str(min_qty))

    # 2️⃣  Raw qty (no rounding yet)
    raw_qty = risk_amt / price           # still Decimal

    # 3️⃣  Round *down* to the nearest step
    qty = (raw_qty // step) * step       # floor division works (both Decimal)

    # 4️⃣  Enforce minimum
    if qty < min_q:
        return 0.0

    # 5️⃣  Cast once for the API
    return float(qty)

def round_qty(symbol, qty, mark_price):
    # Simple static example; ideally fetch from exchange info
    step, min_qty = get_qty_step(symbol)
    return floor(qty / mark_price) * step
def calculate_dynamic_qty(symbol, risk_amount, atr):
    price = get_mark_price(symbol)
    stop_distance = atr * 1.5
    qty = risk_amount / stop_distance
    return round(qty, 6)
def _round_down(qty, step):
    return math.floor(qty / step) * step
def place_sl_and_tp(symbol, side, entry_price, atr, qty):
    """
    SL at 1.5×ATR.
    Four TP limits at Fibonacci ratios < 1 of a 3×ATR base distance.
    Or four TP limits at [1.5, 2.5, 3.5, 4.5] × ATR
    """
    # fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
    # fib = [0.236, 0.382, 0.5, 0.618]      # ratios < 1
    # base = atr_tp_multiplier * atr
    # tp_distances = [base * f for f in fib]
    tp_levels = [
        0.5,
        1.0,
        1.5,
        2.0
    ]

    orders = {'sl': None, 'tp': []}

    # ───────── Stop‑loss (1.5×ATR) ─────────
    # sl_price = entry_price - atr_sl_multiplier * atr if side == "Buy" else entry_price + atr_sl_multiplier * atr
    # try:
    #     orders['sl'] = session.set_trading_stop(
    #         category="linear",
    #         symbol=symbol,
    #         side=side,
    #         stop_loss=str(round(sl_price, 6))
    #     )
    #     if orders['sl'].get('retCode') == 0:
    #         print(f"[{symbol}] SL placed @ {sl_price:.6f}")
    #     else:
    #         print(f"[{symbol}] SL failed: {orders['sl']}")
    # except Exception as e:
    #     if "ErrCode: 10001" not in str(e):
    #         print(f"[{symbol}] SL exception: {e}")

    # ───────── Take‑profits ─────────
    qty_split = [qty * 0.4, qty * 0.2, qty * 0.2, qty * 0.2]
    tp_side = "Sell" if side == "Buy" else "Buy"

    # for i, dist in enumerate(tp_distances):
    for i, dist in enumerate(tp_levels):
        try:
            # tp_price = entry_price + dist * atr if side == "Buy" else entry_price - dist * atr
            tp_price = entry_price + dist * atr if side == "Buy" else entry_price - dist * atr
            rounded_qty = round_qty(symbol, qty_split[i], entry_price)
            # rounded_qty = round(qty_split[i], 6)

            print(f"[{symbol}] TP {i+1}: {tp_side} {rounded_qty} @ {tp_price:.6f}")
            tp_resp = session.place_order(
                category="linear",
                symbol=symbol,
                side=tp_side,
                order_type="Limit",
                qty=rounded_qty,
                price=tp_price,
                time_in_force="GTC",
                reduce_only=True
            )
            # print(f"tp response: {tp_resp}")
            if tp_resp.get("retCode") != 0:
                print(f"[{symbol}] TP {i+1} failed: {tp_resp}")
            orders['tp'].append(tp_resp)
        except Exception as e:
            if "ErrCode: 110017" not in str(e):
                print(f"[{symbol}] TP {i+1} exception: {e}")
            orders['tp'].append(None)

    return orders

def enter_trade(signal, df, symbol, risk_pct):
    global balance
    global leverage
    global current_trade_info

    if signal not in ["Buy", "Sell"]:
        print("[WARN] Invalid signal for enter_trade.")
        return None

    entry_price = get_mark_price(symbol)  # or df['Close'].iloc[-1] — pick one
    balance = get_balance()
    risk_amount = max(balance * risk_pct, 6)  # minimum risk amount 6

    qty_step, min_qty = get_qty_step(symbol)
    atr = df['ATR'].iloc[-1]
    # print("atr in enter_trade: {atr}")
    # adx = df['ADX'].iloc[0]

    total_qty = calc_order_qty(risk_amount, entry_price, min_qty, qty_step)

    if total_qty < min_qty:
        print(f"[WARN] Quantity {total_qty} below minimum {min_qty}, skipping trade.")
        return None

    side = "Buy" if signal == "Buy" else "Sell"

    # Calculate SL price before placing order
    # roi_target_pct = 15.0  # 15% ROI
    # price_diff = (roi_target_pct / 100) * entry_price / leverage
    # sl_price = entry_price - price_diff if side == "Buy" else entry_price + price_diff
    sl_price = entry_price - atr_sl_multiplier * atr if side == "Buy" else entry_price + atr_sl_multiplier * atr
    sl_price = round(sl_price, 6)  # round to appropriate precision

    try:
        print(f"Placing order for {symbol} side={side} qty={total_qty}")
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=total_qty,
            buyLeverage=leverage,
            sellLeverage=leverage,
            stop_loss=str(sl_price)
        )
        # print(f"ORDER RESPONSE: {response}")

        if response['retCode'] != 0:
            if response['retCode'] == 110007:
                print(f"[{symbol}] Skipping order: Insufficient balance (ErrCode: 110007)")
            else:
                print(f"[{symbol}] Order failed: {response['retMsg']} (ErrCode: {response['retCode']})")
            return None

        order_id = response['result'].get("orderId")
        print(f"[INFO] Placed {side} order")
    except Exception as e:
        if "110007" in str(e):
            print(f"[{symbol}] Suppressed balance error: {str(e)}")
        else:
            print(f"[ERROR] Failed to place order: {e}")
        return None

    # Place SL and TPs separately if needed
    orders = {}
    try:
        orders = place_sl_and_tp(symbol, side, entry_price, atr, total_qty)
    except Exception as e:
        print(f"Error placing SL/TP for {symbol}: {e}")

    trade_info = {
        "symbol": symbol,
        "entry_price": entry_price,
        "signal": side,
        "qty": total_qty,
        "remaining_qty": total_qty,
        "tps": orders.get('tp', []),
        "atr": atr,
        "active_tp_index": 0,
        "order_id": order_id
    }
    current_trade_info = trade_info
    return trade_info

def cancel_old_orders(symbol):
    """
    1) Close any live position at market (reduce‑only).
    2) Cancel all outstanding TP / SL / limit orders.
    """

    # ── 1) Close the live position ─────────────────────────────
    pos = get_position(symbol)                # returns dict or None
    if pos and float(pos.get("size", 0)) > 0:
        side        = pos["side"]             # "Buy" or "Sell"
        close_side  = "Sell" if side == "Buy" else "Buy"
        # qty         = round_qty(
        #                  symbol,
        #                  float(pos["size"]),
        #                  float(pos["avgPrice"])
        #              )
        qty = pos["size"]

        resp_close = session.place_order(
            category     = "linear",
            symbol       = symbol,
            side         = close_side,
            order_type   = "Market",
            qty          = qty,
            reduce_only  = True,
            time_in_force= "IOC"
        )
        if resp_close.get("retCode") == 0:
            print(f"[{symbol}] Position closed ({side} → {close_side}) qty={qty}")
        else:
            print(f"[{symbol}] Close‑position fail: {resp_close}")

        # tiny pause to let Bybit settle before we yank orders
        time.sleep(0.3)

def manage_trailing_sl(current_price):
    global current_trade_info
    atr = current_trade_info['atr']
    side = current_trade_info['signal']
    entry_price = current_trade_info['entry_price']
    sl = current_trade_info['sl']

    if side == "Buy":
        new_sl = max(sl, current_price - atr * 1.5)
    else:
        new_sl = min(sl, current_price + atr * 1.5)

    current_trade_info['sl'] = new_sl

def take_partial_profit(symbol, side, qty):
    close_side = "Sell" if side == "Buy" else "Buy"
    try:
        response = session.place_active_order(
            symbol=symbol,
            side=close_side,  # opposite side to close position
            order_type="Market",
            qty=qty,
            reduce_only=True,
            time_in_force="ImmediateOrCancel",
            leverage=leverage
        )
        # response = session.place_order(
        #     category="linear",
        #     symbol=symbol,
        #     side=close_side,
        #     order_type="Market",
        #     qty=qty,
        #     reduce_only=True
        # )
        print(f"[INFO] Partially closed {qty} at TP.")
        return response
    except Exception as e:
        print(f"[ERROR] Failed to close partial position: {e}")
        return None
def check_exit_conditions(current_price, atr):
    side = current_trade_info['side']
    tps = current_trade_info['tps']
    active_index = current_trade_info['active_tp_index']
    sl = current_trade_info['sl']

    # Check SL hit
    if side == "Buy" and current_price <= sl:
        print("[INFO] Stop loss hit (Buy)")
        cancel_old_orders(current_trade_info['symbol'])
        return 'stop'
    elif side == "Sell" and current_price >= sl:
        print("[INFO] Stop loss hit (Sell)")
        cancel_old_orders(current_trade_info['symbol'])
        return 'stop'

    portion = [0.4, 0.2, 0.2, 0.2]

    # Check TP hit
    if active_index < 4:
        target = tps[active_index]
        if (side == "Buy" and current_price >= target) or (side == "Sell" and current_price <= target):
            print(f"[INFO] Take profit level {active_index + 1} hit")
            current_trade_info['active_tp_index'] += 1

            # exit_condition = manage_trailing_sl(current_price)
            update_trailing_sl(symbol, current_price, atr, side, current_trade_info)

            if active_index < len(portion):  # avoid index error
                qty_to_close = portion[active_index] * current_trade_info['qty']
                take_partial_profit(current_trade_info['symbol'], current_trade_info['side'], qty_to_close)
                active_index +=1
        exit_condition = "partial tp"
        return exit_condition

    if current_trade_info['active_tp_index'] >= len(tps):
        print("[INFO] All TP levels hit.")
        cancel_old_orders(current_trade_info['symbol'])
        return 'complete'

    return None
def get_position(symbol):
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        positions = response.get("result", {}).get("list", [])
        
        if not positions:
            print(f"[INFO] No positions found for {symbol}.")
            return None

        # Debug print full position to inspect structure
        # print(f"[DEBUG] Raw position data for {symbol}: {positions[0]}")

        pos = positions[0]
        # print("OPEN POSITION:", pos)
        size = float(pos.get("size", 0))
        return pos
    # try:
    #     response = session.get_positions(category="linear", symbol=symbol)
    #     print("[DEBUG] get_positions response:", response)
    #     return response
    except Exception as e:
        print(f"[get_position] Error fetching position for {symbol}: {e}")
        return None
def calculate_avg_pnl(df, symbol):
    global current_trade_info
    df['PnL'] = df['Close'] - current_trade_info['entry_price']
    df['PnL'] = df['PnL'] * np.where(df['Close'] > df['EMA_14'], 1, -1)
    df['adj_PnL'] = df['PnL'] / df['ATR']
    # avg_pnl = df['adj_PnL'].tail(360).mean()

    # df[symbol].append(avg_pnl)
    # if len(df[symbol]) > 20:
        # df[symbol] = df[symbol][-20:]

    # df['avg_adj_PnL'] = df['adj_PnL'].rolling(window=360).mean()
    # df['avg_PnL'] = df['adj_PnL'].rolling(window=360).mean()
    avg_pnl = df['adj_PnL'].rolling(window=360).mean()
    # return avg_pnl
    return avg_pnl
def update_trailing_sl(symbol, close, atr, side, current_trade_info):
    position = []
    try:
        position = current_trade_info[0] if current_trade_info else None
        print(f"POSITION in update_trailing_sl: {position}")
        if position is None:
            print(f"[{symbol}] No position info available.")
            return
        current_sl = float(position.get('sl', 0)) or 0.0

        new_sl = close - atr * atr_sl_multiplier if side == "Buy" else close + atr * atr_sl_multiplier
        new_sl = round(new_sl, 6)  # Precision

        should_update = (
            (side == "Buy" and (current_sl == 0 or new_sl > current_sl)) or
            (side == "Sell" and (current_sl == 0 or new_sl < current_sl))
        )

        if should_update:
            print(f"[{symbol}] Updating trailing SL to {new_sl}")
            response = session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stop_loss=new_sl
            )
            if response['retCode'] != 0:
                # Suppress error code 34040 ("not modified")
                if response['retCode'] == 34040:
                    print(f"[{symbol}] SL already set to {new_sl}. No update needed.")
                else:
                    print(f"[{symbol}] SL update failed: {response['retMsg']} (ErrCode: {response['retCode']})")
        else:
            print(f"[{symbol}] No SL update needed (new_sl: {new_sl}, current_sl: {current_sl})")

    except Exception as e:
        if "orderQty will be truncated to zero" in str(e):
            print(f"[{symbol}] Skipping SL update: Qty too small. (Suppressed Error 110017)")
        else:
            print(f"[{symbol}] Exception in trailing SL update: {str(e)}")

def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)
def run_bot():
    global balance
    global current_trade_info
    previous_signals = {symbol: "" for symbol in SYMBOLS}  # track last signal per symbol
    risk_pct = 0.1
    while True:
        for symbol in SYMBOLS:
            try:
                trade_info = None
                df = get_klines_df(symbol, "5")
                df = df.sort_values("Timestamp")      # oldest → newest
                df.reset_index(drop=True, inplace=True)
                df = calculate_indicators(df)
                df['signal'] = generate_signals(df)
                latest = df.iloc[-1]
                # print(f"Latest price: {latest_price['Close']}")
                # total_qty = math.floor(risk_amount / df['Close'].iloc[-1]) * step
                # total_qty = calc_order_qty(math.floor(risk_amount, df['Close'].iloc[-1], step))
                # total_qty = calc_order_qty(risk_amount, df['Close'].iloc[-1], min_qty, qty_step)
                df['symbol'] = symbol
                # latest = df.iloc[-1]
                risk_amount = max(get_balance() * risk_pct, 6) 
                atr = latest['ATR']
                qty_step, min_qty = get_qty_step(symbol)
                total_qty = calc_order_qty(risk_amount, latest['Close'], min_qty, qty_step)
                balance = get_balance()
                print(f"=== {symbol} Stats ===")
                # print(f"Open time: {latest['Timestamp']}")
                print(f"signal: {latest['signal']}")
                print(f"Close: {latest['Close']:.6f}")
                print(f"ADX: {latest['ADX']:.2f}")
                print(f"RSI: {latest['RSI']:.2f}")
                print(f"ATR: {atr:.6f}")
                # print(f"ADX over 14: {latest['ADX'] / 14:.2f}")
                # print(f"Balance: {balance:.2f}")
                # total_qty = calculate_dynamic_qty(symbol, risk_amount, latest['atr_bearish_multiplier'])
                # print(f"position size: {total_qty}")
                # print(f"EMA7: {latest['EMA_7']:.6f}")
                # print(f"EMA14: {latest['EMA_14']:.6f}")
                # print(f"EMA28: {latest['EMA_28']:.6f}")
                bias_ema_crossover = "Bullish" if latest['EMA_7'] > latest['EMA_14'] > latest['EMA_28'] else "Bearish" if latest['EMA_7'] < latest['EMA_14'] < latest['EMA_28'] else "Neutral"
                print(f"EMA7/14/28 crossover above/below: {latest['EMA_7'] > latest['EMA_14']}/{latest['EMA_7'] < latest['EMA_14']} ({bias_ema_crossover})")
                # bias_macd_trend = "Bullish" if latest['macd_trending_up'] else "Bearish" if latest['macd_trending_down'] else "Neutral"
                # print(f"MacD trend up/down: {latest['macd_trending_up']}/{latest['macd_trending_down']} ({bias_macd_trend})")
                # print(f"MacdD cross up/down: {latest['macd_cross_up']}/{latest['macd_cross_down']}")
                # bias_macd_hist = "Bullish" if latest['macd_histogram_increasing'] else "Bearish" if latest['macd_histogram_decreasing'] else "Neutral"
                # print(f"MacdD zone: {latest['macd_histogram']} - histogram diff: {latest['OSMA_Diff']} ({bias_macd_hist})")
                bias_macd_signal_diff = "Bullish" if latest['macd_signal_diff'] > 0 else "Bearish" if latest["macd_signal_diff"] < 0 else "Neutral"
                print(f"Macd signal line: {latest['macd_signal']:.2f} - going up/down: {latest['macd_signal_diff']:.2f} ({bias_macd_signal_diff})")
                bias_osma_diff = "Bullish" if latest['OSMA_Diff'] > 0 else "Bearish" if latest['OSMA_Diff'] < 0 else "Neutral"
                print(f"OSMA zone: {latest['OSMA']:.2f} - Direction: {latest['OSMA_Diff']:.2f} ({bias_osma_diff})")
                di_diff = latest['DI_Diff']
                bias_di_diff = "Bullish" if latest['+DI'] > latest['-DI'] else "Bearish" if latest['+DI'] < latest['-DI'] else "Neutral"
                print(f"+DI/-DI: {latest['+DI']:.2f}/{latest['-DI']:.2f} - Diff: {latest['DI_Diff']:.2f} ({bias_di_diff})")
                # power_diff = latest['Bulls'] - latest['Bears']
                # bias_power_diff = "Bullish" if power_diff > 0 else "Bearish" if power_diff < 0 else "Neutral"
                # print(f"Bulls Power/Bears Power: {latest['Bulls']:.6f}/{latest['Bears']:.6f} - Diff: {power_diff:.6f} ({bias_power_diff})")
                bias_price_sma = "Bullish" if latest.Close > latest.bb_sma else "Bearish" if latest.Close < latest.bb_sma else "Neutral"
                print(f"Price bullish or bearish according to BB SMA and price: {bias_price_sma}")
                # print(f"Trade Qty (calculated): {total_qty}")
                # print(f"Balance: {balance:.2f}")
                position = get_position(symbol)
                # Update current_trade_info from live position if not already set
                if (not current_trade_info or current_trade_info == []) and position:
                    if isinstance(position, list) and len(position) > 0:
                        current_trade_info = position[0]
                    elif isinstance(position, dict):
                        current_trade_info = position
                open_orders = session.get_open_orders(category="linear", symbol=symbol)['result']['list']
                # print(f"OPEN POSITION: {position}")
                # print(f"OPEN ORDERS: {open_orders}")
                in_position = bool(position and float(position.get("size", 0)))
                if in_position:
                    # print(f"Open Position: Side={position['side']} Entry={position['entry_price']:.6f} Qty={position['qty']} PnL={position['unrealizedPnl']:.2f}")
                    print(f"Open Position: Side={position['side']} Entry={position['avgPrice']} Qty={position['size']}")
                    # update_trailing_sl(symbol, df['Close'].iloc[-1], df['ATR'].iloc[-1])
                    # exit_condition = check_exit_conditions(symbol, df['ATR'].iloc[-1])
                    # --- AUTO-REVERSE LOGIC ---
                    if latest['signal'] != "" and latest['signal'] != position['side'] and position['side'] != "":
                        print(f"[AUTO-REVERSE] Opposite signal detected: Closing {position['side']} and entering {latest['signal']}")
                        # Exit current position
                        cancel_old_orders(symbol)
                        time.sleep(2)  # Give time for exit to process
                        # Enter opposite trade
                        trade_info = enter_trade(latest['signal'], df, symbol, risk_pct)
                        if trade_info:
                            print(f"[TRADE] Reversed position to {latest['signal']}")
                            # counters["totals"][this_mode] += 1
                            # save_counters(counters)                 # persist immediately
                            # count_str = ", ".join(f"{k}:{v}" for k, v in counters["totals"].items())
                            # msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {symbol}  "
                            #        f"{this_mode.upper()}  #{counters['totals'][this_mode]}/5  "
                            #        f"Totals → {count_str}\n")
                            # with open(LOG_FILE, "a") as log:
                            #         log.write(msg)
                            #         print(msg.strip())
                            current_trade_info = trade_info
                        else:
                            print(f"[WARN] Failed to enter trade for {symbol}, skipping update.")
                            # continue
                        # Update last known signal
                        # previous_signals[symbol] = latest['signal']
                    else:
                        print("[INFO] Holding current position — no reversal needed.")
                        update_trailing_sl(symbol, df['Close'].iloc[-1], df['ATR'].iloc[-1], latest['signal'], current_trade_info)
                        # exit_condition = check_exit_conditions(symbol, df['ATR'].iloc[-1])
                else:
                    if latest['signal'] != "":
                        print(f"[INFO] No open position — entering new {latest['signal']} trade.")
                        trade_info = enter_trade(latest['signal'], df, symbol, risk_pct)
                        if trade_info:
                            print(f"[INFO] Entered trade for {symbol} at {trade_info['entry_price']}")
                            current_trade_info = trade_info
                        else:
                            print(f"[WARN] Failed to enter trade for {symbol}, skipping update.")
                            # continue  # or continue
                        # print(f"current_trade_info: {current_trade_info}")
                        # avg_pnl = calculate_avg_pnl(df, symbol)
                        # print(f"avg_pnl: {avg_pnl}")
                        # previous_signals[symbol] = latest['signal']
                time.sleep(2)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        print("Cycle complete. Waiting for next candle...\n")
        wait_until_next_candle(5)
if __name__ == "__main__":
    run_bot()
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
