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

COUNTER_FILE = "pybit-log-counter.txt.log"

# ───────── daily per‑mode caps ─────────
MODE_CAP      = 5               # 5 trades per mode per day
mode_counter  = {"ema": 0, "macd": 0, "sar": 0}
day_anchor    = int(time.time() // 86400)  # UTC day‑number
# ────────────────────────────────────────

SYMBOLS = ["SHIB1000USDT", "XAUTUSDT"]

# ── trade‑frequency throttle ─────────────────────────
MIN_GAP = 60 * 96   # 5 760 s ≈ 1 h 36 m
last_trade_time = {sym: 0 for sym in SYMBOLS}
# ─────────────────────────────────────────────────────

# ───────── daily trade cap ─────────
DAILY_CAP      = 15          # max new positions in 24 h
trade_counter  = 0           # how many entries placed so far today
day_anchor     = int(time.time() // 86400)  # day number since Unix epoch
# ───────────────────────────────────

# API setup
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
# SYMBOLS = ["BNBUSDT", "SOLUSDT", "XRPUSDT", "FARTCOINUSDT", "DOGEUSDT"]
# SYMBOLS = ["BNBUSDT", "XRPUSDT", "SHIB1000USDT", "BROCCOLIUSDT"]
# SYMBOLS = ["BNBUSDT", "SOLUSDT", "XRPUSDT", "FARTCOINUSDT", "DOGEUSDT", "SHIB1000USDT", "BROCCOLIUSDT"]
# SYMBOLS = ["FARTCOINUSDT", "DOGEUSDT", "1000PEPEUSDT", "SHIB1000USDT", "BROCCOLIUSDT"]
SYMBOLS = ["SHIB1000USDT", "XAUTUSDT"]
# SYMBOLS = ["XRPUSDT", "FARTCOINUSDT", "DOGEUSDT", "SHIB1000USDT"]
risk_pct = 0.1  # Example risk per trade
leverage=25
current_trade_info = []
balance = 168
now = time.time()

def load_counters():
    """Return dict {"day": <utc‑day‑int>, "totals": {"ema":0,"macd":0,"sar":0}}."""
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass  # treat as corrupt → start fresh
    # default structure
    return {"day": int(time.time() // 86400),
            "totals": {"ema": 0, "macd": 0, "sar": 0}}

def save_counters(data):
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)

counters = load_counters() 
        
def reset_and_log_if_new_day():
    global counters
    today = int(time.time() // 86400)

    # — roll over to new day —
    if today != counters["day"]:
        counters = {"day": today, "totals": {"ema": 0, "macd": 0, "sar": 0}}
        save_counters(counters)
        with open(LOG_FILE, "a") as log:
            log.write(f"\n======== NEW UTC DAY {today} ========\n")

def reset_daily_counter_if_new_day():
    global trade_counter, day_anchor
    today = int(time.time() // 86400)
    if today != day_anchor:           # crossed into a new UTC day
        trade_counter = 0
        day_anchor    = today
        print("─── New day detected: trade counter reset to 0 ───")
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
    return macd_line, signal_line, histogram

def Bollinger_Bands(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band
def ADX(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = ATR(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di

def BullsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['High'] - ema

def BearsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['Low'] - ema

def Momentum(series, period=10):
    return series - series.shift(period)

def calculate_indicators(df):
    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_28'] = df['Close'].ewm(span=28).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    # === MACD Crossovers ===
    df['macd_cross_up'] = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
    # === MACD Trend Status ===
    df['macd_trending_up'] = df['macd_line'] > df['macd_signal']
    df['macd_trending_down'] = df['macd_line'] < df['macd_signal']
    df['macd_histogram_increasing'] = df['macd_histogram'].diff() > 0
    df['macd_histogram_decreasing'] = df['macd_histogram'].diff() < 0

    df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])

    # df['Momentum'] = df['Close'] - df['Close'].shift(10)
    # Custom momentum (% of recent high-low range)
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    price_range = high_14 - low_14
    df['Momentum'] = 100 * (df['Close'] - df['Close'].shift(14)) / price_range
    # Momentum trend signals: compare current Momentum with previous
    df['Momentum_increasing'] = df['Momentum'] > df['Momentum'].shift(2)
    df['Momentum_decreasing'] = df['Momentum'] < df['Momentum'].shift(2)

    df['RSI'] = RSI(df['Close'], 14)
    adx, plus_di, minus_di = ADX(df)
    df['ADX'] = adx
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['Bulls'] = df['High'] - df['Close'].shift(1)
    df['Bears'] = df['Close'].shift(1) - df['Low']

    df['atr_bullish_multiplier'] = (df['Bulls'] - df['Bears']) / df['ATR']
    df['atr_bearish_multiplier'] = (df['Bears'] - df['Bulls']) / df['ATR']
    df['Di_Diff'] = df['+DI'] - df['-DI']
    df['Bullish_DI'] = df['+DI'] - df['-DI']
    df['Bearish_DI'] = df['-DI'] - df['+DI']
    # df['Bull_Bear_Diff'] = (df['Bulls'] - df['Bears']) / df['ATR']
    df['Bull_Bear_Diff'] = (df['Bulls'] - df['Bears'])
    df['ADX_to_ATR_ratio'] = df['ADX'] / 100 / df['ATR']

    # df = calculate_fractals(df)

    df.dropna(inplace=True)
    return df
def calculate_signal_scores(latest):
    """
    Calculate bullish and bearish signal strength scores (0.0 to 1.0).
    Uses smoothed indicators if available.
    """

    # ---- Bearish Conditions ----
    bearish_conditions = {
        'momentum_decreasing': latest.get('Momentum_decreasing', False),
        'di_relation': latest.get('+DI', 0) < latest.get('-DI', 0) or latest.get('Bulls', 0) < latest.get('Bears', 0),
        'di_diff': latest.get('Di_Diff', latest.get('Di_Diff', 0)) < 0,
        'bull_bear_diff': latest.get('Bull_Bear_Diff', latest.get('Bull_Bear_Diff', 0)) < 0,
        'macd_trending_down': latest.get('macd_trending_down', False),
        'macd_histogram': latest.get('macd_histogram', latest.get('macd_histogram', 0)) < 0,
        'ema_cross': latest.get('EMA_7', 0) < latest.get('EMA_14', 0),
        'momentum': latest.get('Momentum', latest.get('Momentum', 0)) < 0
    }

    # ---- Bullish Conditions ----
    bullish_conditions = {
        'momentum_increasing': latest.get('Momentum_increasing', False),
        'di_relation': latest.get('+DI', 0) > latest.get('-DI', 0) or latest.get('Bulls', 0) > latest.get('Bears', 0),
        'di_diff': latest.get('Di_Diff', latest.get('Di_Diff', 0)) > 0,
        'bull_bear_diff': latest.get('Bull_Bear_Diff', latest.get('Bull_Bear_Diff', 0)) > 0,
        'macd_trending_up': latest.get('macd_trending_up', False),
        'macd_histogram': latest.get('macd_histogram', latest.get('macd_histogram', 0)) > 0,
        'ema_cross': latest.get('EMA_7', 0) > latest.get('EMA_14', 0),
        'momentum': latest.get('Momentum', latest.get('Momentum', 0)) > 0
    }

    bearish_score = sum(bool(v) for v in bearish_conditions.values()) / len(bearish_conditions)
    bullish_score = sum(bool(v) for v in bullish_conditions.values()) / len(bullish_conditions)
    return bearish_score, bullish_score
def generate_signals(df):
    """
    Three exclusive modes with EMA priority:
      1) ema_up  / ema_down
      2) macd_up / macd_down
      3) sar_up  / sar_down
    EMA alignment instantly overrides any other mode.
    """

    signals = [""] * len(df)
    mode = None  # None or one of the six mode strings

    for i in range(len(df)):
        latest = df.iloc[i]
        signal = ""

        # ---------- 0) GLOBAL EMA PRIORITY ----------
        ema_up   = latest.EMA_7 > latest.EMA_14 > latest.EMA_28
        ema_down = latest.EMA_7 < latest.EMA_14 < latest.EMA_28
        if ema_up or ema_down:
            mode = "ema_up" if ema_up else "ema_down"
            signal = "Buy" if ema_up else "Sell"
            signals[i] = signal
            continue  # skip rest; EMA outranks everything

        # ---------- 1) Maintain / exit non‑EMA modes ----------
        if mode == "macd_up":
            if not (latest.macd_trending_up and latest.macd_histogram >= df["macd_histogram"].iloc[i-1]):
                mode = None
        elif mode == "macd_down":
            if not (latest.macd_trending_down and latest.macd_histogram <= df["macd_histogram"].iloc[i-1]):
                mode = None
        elif mode == "sar_up":
            if latest.SAR >= latest.Close:
                mode = None
        elif mode == "sar_down":
            if latest.SAR <= latest.Close:
                mode = None

        # ---------- 2) If a mode is still active, keep its signal ----------
        if mode:
            signal = "Buy" if mode.endswith("_up") else "Sell"
            signals[i] = signal
            continue

        # ---------- 3) No active mode: evaluate fresh entries ----------
        # Basic filters + flat‑market gate
        if latest.ADX > 20 and 30 < latest.RSI < 70 and i >= 10:
            window = df.iloc[i-10:i]
            price_range = window["High"].max() - window["Low"].min()
            ratio = price_range / window["Close"]
            if (ratio <= 0.003).all():
                signals[i] = ""
                continue

            # 3a) MACD‑momentum entry
            if (latest.Bull_Bear_Diff > 0 or latest.Di_Diff > 0) and latest.macd_cross_up:
                mode, signal = "macd_up", "Buy"
            elif (latest.Bull_Bear_Diff < 0 or latest.Di_Diff < 0) and latest.macd_cross_down:
                mode, signal = "macd_down", "Sell"

            # 3b) SAR‑glide entry (only if MACD didn’t trigger)
            elif "SAR" in df.columns:
                if latest.SAR < latest.Close:
                    mode, signal = "sar_up", "Buy"
                elif latest.SAR > latest.Close:
                    mode, signal = "sar_down", "Sell"

        signals[i] = signal

    df["signal"] = signals
    return df["signal"]
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
def calc_order_qty(balance_usdt: float, entry_price: float, min_qty: float, qty_step: float) -> float:
    """
    Calculate a Bybit‑compliant futures quantity.

    Parameters
    ----------
    balance_usdt : float
        Amount of USDT you’re willing to commit.
    entry_price  : float
        Expected fill price of the contract.
    leverage     : int
        Leverage to apply (>= 1).
    min_qty      : float
        Exchange minimum order size for the symbol (e.g. 0.001).
    qty_step     : float
        Exchange quantity step for the symbol (e.g. 0.001).

    Returns
    -------
    float
        Order quantity rounded *down* to the nearest step, or 0.0
        if the wallet is too small to meet `min_qty`.
    """

    if leverage <= 0:
        raise ValueError("Leverage must be positive")

    # 1) raw contract size (before rounding)
    # raw_qty = (balance_usdt * leverage) / entry_price
    min_qty, qty_step = get_qty_step(symbol)
    raw_qty = (risk_amount * df['Close'].iloc[-1]) * qty_step
    order_qty = calc_order_qty(risk_amount, entry_price, min_qty, qty_step)

def calc_order_qty(risk_amount, entry_price, min_qty, qty_step):
    """Return Bybit‑compliant qty rounded *down* or 0.0 if below min."""
    raw_qty = risk_amount / entry_price
    step = Decimal(str(qty_step))
    min_qty  = Decimal(str(raw_qty)).quantize(step, rounding=ROUND_DOWN)
    return float(min_qty) if min_qty >= Decimal(str(min_qty)) else 0.0
# ---------------------------------------------------------------------
def round_qty(symbol, qty, mark_price):
    # Simple static example; ideally fetch from exchange info
    step, min_qty = get_qty_step(symbol)
    return floor(qty / mark_price) * step
def calculate_dynamic_qty(symbol, risk_amount, atr):
    price = get_mark_price(symbol)
    stop_distance = atr * 1.5
    qty = risk_amount / stop_distance
    return round(qty, 6)
def place_sl_and_tp(symbol, side, entry_price, atr, qty):
    """
    Place initial SL plus four TP limit orders.
    Silent‑skip ErrCode 10001 (zero‑position on SL) and
    ErrCode 110017 (zero‑position on reduce‑only TP).
    """
    levels  = [1.5, 2.5, 3.5, 4.5]
    orders  = {'sl': None, 'tp': []}

    # ───────────── Stop‑loss ─────────────
    sl_price = entry_price - 1.5 * atr if side == "Buy" else entry_price + 1.5 * atr
    try:
        orders['sl'] = session.set_trading_stop(
            category="linear",
            symbol=symbol,
            side=side,
            stop_loss=str(round(sl_price, 6))
        )
        if orders['sl'].get('retCode') == 0:
            print(f"[{symbol}] SL placed.")
        else:
            print(f"[{symbol}] SL failed: {orders['sl']}")
    except Exception as e:
        if "ErrCode: 10001" in str(e):
            pass                              # silent
        else:
            msg = str(e).split("Request")[0].strip()
            print(f"[{symbol}] SL exception: {msg}")

    # ───────────── Take‑profits ─────────────
    qty_split = [qty * 0.4, qty * 0.2, qty * 0.2, qty * 0.2]

    for i, mult in enumerate(levels):
        try:
            tp_price = entry_price + mult * atr if side == "Buy" else entry_price - mult * atr
            tp_side  = "Sell" if side == "Buy" else "Buy"
            rounded_qty = round_qty(symbol, qty_split[i], entry_price)

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
            orders['tp'].append(tp_resp)
        except Exception as e:
            if "ErrCode: 110017" in str(e):
                # silent skip: no position yet for reduce‑only order
                orders['tp'].append(None)
            else:
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
    risk_amount = max(balance * risk_pct, 6)  # minimum risk amount 5

    qty_step, min_qty = get_qty_step(symbol)
    atr = df['ATR'].iloc[-1]
    adx = df['ADX'].iloc[-1]
    # adxtoatr_ratio = adx / 100 / atr
    # total_qty = calculate_dynamic_qty(symbol, risk_amount, atr_bearish_multiplier)
    # print(total_qty)
    total_qty = calc_order_qty(risk_amount, entry_price, min_qty, qty_step)

    # Round total_qty down to nearest step
    order_qty = math.floor(risk_amount / entry_price) * qty_step

    order_id = ""

    if total_qty < min_qty:
        print(f"[WARN] Quantity {total_qty} below minimum {min_qty}, skipping trade.")
        return None

    side = "Buy" if signal == "Buy" else "Sell"

    sl_price = round_qty(symbol, entry_price - atr * 1.5, entry_price) if side == "Buy" else round_qty(symbol, entry_price + atr * 1.5, entry_price)
    tp_price = round_qty(symbol, entry_price + atr * 4.5, entry_price) if side == "Buy" else round_qty(symbol, entry_price - atr * 4.5, entry_price)
    # print(f"tp_price: {tp_price}")

    try:
        print(f"Placing order for {symbol} side={side} qty={total_qty}")
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=total_qty,
            leverage=leverage,
            take_profit=str(tp_price),
            stop_loss=str(sl_price)
        )

        # ✅ Validate API response before accessing
        # if not response or 'result' not in response or not response['result']:
        #     print(f"[ERROR] Bad order response for {symbol}: {response}")
        #     return None
        if response['retCode'] != 0:
            if response['retCode'] == 110007:
                print(f"[{symbol}] Skipping order: Insufficient balance (ErrCode: 110007)")
            else:
                print(f"[{symbol}] Order failed: {response['retMsg']} (ErrCode: {response['retCode']})")
        order_id = response['result'].get("orderId")
        print(f"[INFO] Placed {side} order with TP={tp_price}, SL={sl_price}")
    except Exception as e:
        if "110007" in str(e):
            print(f"[{symbol}] Suppressed balance error: {str(e)}")
        # else:
        #     print(f"[{symbol}] Unexpected exception in order: {str(e)}")
        # print(f"[ERROR] Failed to place order: {e}")
        # return None

    # Optionally place additional SL/TP if your function returns more orders/info
    orders = {}
    try:
        orders = place_sl_and_tp(symbol, side, entry_price, atr, total_qty)
    except Exception as e:
        print(f"Error placing SL/TP for {symbol}: {e}")

    # if orders:
    trade_info = {
        "symbol": symbol,
        "entry_price": entry_price,
        "signal": side,
        "qty": total_qty,
        "remaining_qty": total_qty,
        "sl": sl_price,
        # "tps": [tp_price],  # or orders['tps'] if you use multi-level TPs
        # "tps": orders['tp'],  # or orders['tps'] if you use multi-level TPs
        "atr": atr,
        "active_tp_index": 0,
        # "order_id": order_id or orders.get('order_id')
        "order_id": order_id
    }
    current_trade_info = trade_info
    return trade_info
def cancel_old_orders(symbol):
    try:
        # Fetch active orders for the symbol (category may be needed depending on API)
        # open_orders = session.get_active_order(category="linear", symbol=symbol)['result']['list']
        open_orders = session.get_open_orders(category="linear", symbol=symbol)['result']['list']
        for order in open_orders:
            order_id = order['orderId']
            # session.cancel_order(category="linear", symbol=symbol, orderId=order_id)
            session.cancel_order(category="linear", symbol=symbol)
        print(f"Cancelled old orders for {symbol}")
    except Exception as e:
        print(f"Failed to cancel orders for {symbol}: {e}")
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
        size = float(pos.get("size", 0))
        return pos
    # try:
    #     response = session.get_positions(category="linear", symbol=symbol)
    #     print("[DEBUG] get_positions response:", response)
    #     return response
    except Exception as e:
        print(f"[get_position] Error fetching position for {symbol}: {e}")
        return None
def update_trailing_sl(symbol, close, atr, side, current_trade_info):
    try:
        # pos = get_position(symbol)
        # if not pos or float(pos['size']) == 0:
        #     print(f"[{symbol}] No open position to update trailing SL.")
        #     return

        # current_sl = float(pos.get('sl', 0)) or 0.0
        # print(f"[{symbol}] current_trade_info: {current_trade_info}")
        position = current_trade_info[0] if current_trade_info else None
        if position is None:
            print(f"[{symbol}] No position info available.")
            return
        
        current_sl = float(position.get('sl', 0)) or 0.0

        # current_sl = float(current_trade_info['sl']) or 0.0
        new_sl = close - atr * 1.5 if side == "Buy" else close + atr * 1.5

        # Ensure proper rounding to avoid invalid order
        new_sl = round(new_sl, 6)  # Adjust precision as needed per symbol


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
                print(f"[{symbol}] SL update failed: {response['retMsg']} (ErrCode: {response['retCode']})")
        else:
            print(f"[{symbol}] No SL update needed (new_sl: {new_sl}, current_sl: {current_sl})")

    except Exception as e:
        if "orderQty will be truncated to zero" in str(e):
            print(f"[{symbol}] Skipping SL update: Qty too small. (Suppressed Error 110017)")
        else:
            print(f"[{symbol}] Exception in trailing SL update: {str(e)}")
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
        # reset_mode_counters_if_new_day()
        reset_and_log_if_new_day()
        reset_daily_counter_if_new_day()
        for symbol in SYMBOLS:
            try:
                # trade_info = []
                trade_info = None
                df = get_klines_df(symbol, "15")
                df = calculate_indicators(df)
                df['signal'] = generate_signals(df)
                risk_amount = max(get_balance() * risk_pct, 6) 
                atr = df['ATR'].iloc[-1]
                qty_step, min_qty = get_qty_step(symbol)
                # total_qty = math.floor(risk_amount / df['Close'].iloc[-1]) * step
                # total_qty = calc_order_qty(math.floor(risk_amount, df['Close'].iloc[-1], step))
                total_qty = calc_order_qty(risk_amount, df['Close'].iloc[-1], min_qty, qty_step)
                df['symbol'] = symbol
                latest = df.iloc[-1]
                # ── open a fresh trade only if signal present & cool‑down met ──
                if latest['signal'] and (now - last_trade_time[symbol] >= MIN_GAP):
                    print(f"[INFO] Cool‑down satisfied → opening {latest['signal']} on {symbol}")
                    trade_info = enter_trade(latest['signal'], df, symbol, risk_pct)
                    if trade_info:
                        last_trade_time[symbol] = now           # stamp the fill time
                        current_trade_info = trade_info

                        counters["totals"][this_mode] += 1
                        save_counters(counters)                 # persist immediately
                        count_str = ", ".join(f"{k}:{v}" for k, v in counters["totals"].items())
                        msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {symbol}  "
                               f"{this_mode.upper()}  #{counters['totals'][this_mode]}/5  "
                               f"Totals → {count_str}\n")
                        with open(LOG_FILE, "a") as log:
                            log.write(msg)
                            print(msg.strip())                      # echo to console
                            current_trade_info = trade_info

                    else:
                        if latest['signal']:
                            wait_left = MIN_GAP - (now - last_trade_time[symbol])
                            print(f"[{symbol}] Cool‑down active: {wait_left/60:.1f} min left")
                balance = get_balance()
                print(f"=== {symbol} Stats ===")
                print(f"signal: {latest['signal']}")
                print(f"Close: {latest['Close']:.6f}")
                print(f"ADX: {latest['ADX']:.2f}")
                print(f"RSI: {latest['RSI']:.2f}")
                print(f"ATR: {atr:.6f}")
                # print(f"ADX over 14: {latest['ADX'] / 14:.2f}")
                print(f"Balance: {balance:.2f}")
                # total_qty = calculate_dynamic_qty(symbol, risk_amount, latest['atr_bearish_multiplier'])
                total_qty = calc_order_qty(risk_amount, latest['Close'], min_qty, qty_step)
                print(f"position size: {total_qty}")
                # print(f"EMA7: {latest['EMA_7']:.6f}")
                # print(f"EMA14: {latest['EMA_14']:.6f}")
                # print(f"EMA28: {latest['EMA_28']:.6f}")
                bias_ema_crossover = "Bullish" if latest['EMA_7'] > latest['EMA_14'] else "Bearish" if latest['EMA_7'] < latest['EMA_14'] else "Neutral"
                print(f"EMA7/14 crossover above/below: {latest['EMA_7'] > latest['EMA_14']}/{latest['EMA_7'] < latest['EMA_14']} ({bias_ema_crossover})")
                bias_macd_trend = "Bullish" if latest['macd_trending_up'] else "Bearish" if latest['macd_trending_down'] else "Neutral"
                print(f"MacD trend up/down: {latest['macd_trending_up']}/{latest['macd_trending_down']} ({bias_macd_trend})")
                # print(f"MacdD cross up/down: {latest['macd_cross_up']}/{latest['macd_cross_down']}")
                # print(f"MacdD histogram increasing/decreasing: {latest['macd_histogram_increasing']}/{latest['macd_histogram_decreasing']}")
                di_diff = latest['+DI'] - latest['-DI']
                bias_di_diff = "Bullish" if latest['Di_Diff'] > 0 else "Bearish" if latest['Di_Diff'] < 0 else "Neutral"
                print(f"+DI/-DI: {latest['+DI']:.2f}/{latest['-DI']:.2f} - Diff: {di_diff:.2f} ({bias_di_diff})")
                # print(f"Bullish/Bearish DI: {latest['Bullish_DI']}/{latest['Bearish_DI']}")
                power_diff = latest['Bulls'] - latest['Bears']
                bias_power_diff = "Bullish" if power_diff > 0 else "Bearish" if power_diff < 0 else "Neutral"
                print(f"Bulls Power/Bears Power: {latest['Bulls']:.6f}/{latest['Bears']:.6f} - Diff: {power_diff:.6f} ({bias_power_diff})")
                # bullish_diff = latest['Bulls'] - latest['Bears']
                # bearish_diff = latest['Bears'] - latest['Bulls']
                # bias_bullish_bearish_diff = "Bullish" if bullish_diff > 0.5 else "Bearish" if bearish_diff < -0.5 else "Neutral"
                # print(f"Bullish/Bearish diffs: {latest['+DI']-latest['-DI']:.2f}/{latest['-DI']-latest['+DI']:.2f} ({bias_bullish_bearish_diff})")
                bias_momentum = "Bullish" if latest['Momentum'] > 0 else "Bearish" if latest['Momentum'] < 0 else "Neutral"
                print(f"Momentum: {latest['Momentum']:.2f} - ({bias_momentum})")
                bias_momentum_incr_decr = "Bullish" if latest['Momentum_increasing'] else "Bearish" if latest['Momentum_decreasing'] else "Neutral"
                print(f"Momentum increasing/decreasing: {latest['Momentum_increasing']}/{latest['Momentum_decreasing']} ({bias_momentum_incr_decr})")
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
                # print(f"position: {position}")
                # print(f"open orders: {open_orders}")
                if open_orders:
                    # print(f"Open Position: Side={position['side']} Entry={position['entry_price']:.6f} Qty={position['qty']} PnL={position['unrealizedPnl']:.2f}")
                    # update_trailing_sl(symbol, df['Close'].iloc[-1], df['ATR'].iloc[-1])
                    # exit_condition = check_exit_conditions(symbol, df['ATR'].iloc[-1])
                    # --- AUTO-REVERSE LOGIC ---
                    if latest['signal'] != "" and latest['signal'] != previous_signals[symbol] and previous_signals[symbol] != "":
                        print(f"[AUTO-REVERSE] Opposite signal detected: Closing {previous_signals[symbol]} and entering {latest['signal']}")
                        # Exit current position
                        cancel_old_orders(symbol)
                        time.sleep(2)  # Give time for exit to process
                        # Enter opposite trade
                        trade_info = enter_trade(latest['signal'], df, symbol, risk_pct)
                        if trade_info:
                            print(f"[TRADE] Reversed position to {latest['signal']}")
                            counters["totals"][this_mode] += 1
                            save_counters(counters)                 # persist immediately
                            count_str = ", ".join(f"{k}:{v}" for k, v in counters["totals"].items())
                            msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {symbol}  "
                                   f"{this_mode.upper()}  #{counters['totals'][this_mode]}/5  "
                                   f"Totals → {count_str}\n")
                            with open(LOG_FILE, "a") as log:
                                    log.write(msg)
                                    print(msg.strip())
                            current_trade_info = trade_info
                        else:
                            print(f"[WARN] Failed to enter trade for {symbol}, skipping update.")
                            # continue
                        # Update last known signal
                        previous_signals[symbol] = latest['signal']
                    else:
                        print("[INFO] Holding current position — no reversal needed.")
                        # update_trailing_sl(symbol, df['Close'].iloc[-1], df['ATR'].iloc[-1], latest['signal'], current_trade_info)
                        # exit_condition = check_exit_conditions(symbol, df['ATR'].iloc[-1])
                else:
                    if latest['signal'] != "":
                        print(f"[INFO] No open position — entering new {latest['signal']} trade.")
                        trade_info = enter_trade(latest['signal'], df, symbol, risk_pct)
                        if trade_info:
                            print(f"[INFO] Entered trade for {symbol} at {trade_info['entry_price']}")
                            counters["totals"][this_mode] += 1
                            save_counters(counters)                 # persist immediately
                            count_str = ", ".join(f"{k}:{v}" for k, v in counters["totals"].items())
                            msg = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {symbol}  "
                                   f"{this_mode.upper()}  #{counters['totals'][this_mode]}/5  "
                                   f"Totals → {count_str}\n")
                            with open(LOG_FILE, "a") as log:
                                    log.write(msg)
                                    print(msg.strip())
                            current_trade_info = trade_info
                        else:
                            print(f"[WARN] Failed to enter trade for {symbol}, skipping update.")
                            # continue  # or continue
                        # print(f"current_trade_info: {current_trade_info}")
                        # avg_pnl = calculate_avg_pnl(df, symbol)
                        # print(f"avg_pnl: {avg_pnl}")
                        previous_signals[symbol] = latest['signal']
                time.sleep(2)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        print("Cycle complete. Waiting for next candle...\n")
        wait_until_next_candle(1)
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
