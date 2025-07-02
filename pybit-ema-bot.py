import time
import math
from datetime import datetime
import requests
import pandas as pd
from pybit.unified_trading import HTTP  # pip install pybit
import numpy as np
from io import StringIO

# === SETUP ===
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
xrp = "XRPUSDT"
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

def get_klines_df(xrp, interval, limit=1000):
    response = session.get_kline(
        category="linear",
        symbol=xrp,
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

def get_mark_price(xrp):
    price_data = session.get_tickers(category="linear", symbol=xrp)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(xrp):
    info = session.get_instruments_info(category="linear", symbol=xrp)
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
    positions = session.get_positions(category="linear", symbol=xrp)
    position_lists = positions['result']['list']
    # for position in positions:
    for position in position_lists:
        qty = float(position["size"])
        if qty == 0:
            continue
        side = "Sell" if position["side"] == "Buy" else "Buy"
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, reduce_only=True)
        print(f"[TRADE CLOSED]")

def cancel_specific_order(order_id: str, xrp: str):
    if not order_id:
        print("No order ID to cancel")
        return
    try:
        response = session.cancel_order(category="linear", order_id=order_id, symbol=symbol)
        print("Cancel order response:", response)
    except Exception as e:
        print("Error canceling order:", e)

def enter_trade(signal, df, symbol="XRPUSDT"):
    # global peak_balance, entry_price, entry_time
    global leverage, xrp

    # mark_price = get_mark_price(symbol)
    # val = df['Close'].mean() * 0.02 / df['atr'].mean()
    # # print(f"atr's to reach 2% movement: {val:.2f}")
    # offset = int(round(val, 0))
    # min_price = df["Low"].iloc[offset:].min()
    # max_price = df["High"].iloc[offset:].max()
    # price_range_pct = (max_price - min_price) / mark_price
    # bb_width = df['bb_upper'] - df['bb_lower']
    # bb_width_pct = bb_width / mark_price

    # if price_range_pct < 0.002 or bb_width < 0.002:
    # if price_range_pct < 0.007:
    if df['atr'].iloc[-1] / df['Close'].iloc[-1] < 0.001:
        print("[INFO] Market range is too small (<2%), skipping trade.")
        # print(f"price range in pct: {price_range_pct:.6f}")
        # print(f"Max high prices: {max_price:.6f}")
        # print(f"Min low prices: {min_price:.6f}")
        # return

    balance = get_balance()

    risk_amount = max(balance * risk_pct, 10)

    mark_price = get_mark_price(symbol)

    step, min_qty = get_qty_step(symbol)
    qty = risk_amount / mark_price
    qty = math.floor(qty / step) * step
    if qty < min_qty:
        print(f"[WARNING] Calculated quantity {qty} is below minimum {min_qty}, skipping trade.")
        return

    side = "Buy" if signal == "Buy" else "Sell"

    tp_price = mark_price * (1 + 0.3 / 75) if signal == "buy" else mark_price * ((1 - 0.3) / leverage) # with leverage
    sl_price = mark_price * (1 - 0.15 / 75) if signal == "buy" else mark_price * ((1 + 0.3) / leverage) # with leverage


    # session.set_leverage(category="linear", symbol=symbol, buyLeverage=leverage, sellLeverage=leverage)
    response = session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, buyLeverage=leverage, sellLeverage=leverage)
    return response.get("result", {}).get("orderId")

def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)

def generate_signals(df):
    signals = []
    modes = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal = None
        mode = None

        if row["adx"] > 25:
            mode = "Trend"
            if row["ema_14_diff"] > 0:
                signal = "Buy"
            elif row["ema_14_diff"] < 0:
                signal = "Sell"
        else:
            mode = "Grid"
            price = row["Close"]
            band_length = row["bb_upper"] - row["bb_lower"]
            if band_length == 0:
                # avoid division by zero if bands coincide
                dist_upper = 1.0
                dist_lower = 1.0
            else:
                dist_upper = abs(price - row["bb_upper"]) / band_length
                dist_lower = abs(price - row["bb_lower"]) / band_length

            if dist_lower < 0.10 and row["ema_7_diff"] > 0:
                signal = "Buy"
            elif dist_upper < 0.10 and row["ema_7_diff"] < 0:
                signal = "Sell"

        signals.append(signal)
        modes.append(mode)

    df['signal'] = signals
    df['mode'] = modes
    return df

def run_bot():
    active_trade = False
    balance = get_balance()
    risk_pct = 0.10
    leverage = 75
    order_id = ""
    current_trade_side = ""

    # for index, row in df.iterrows():
    while True:
        df = get_klines_df(xrp, 60)
        df = add_indicators(df)
        df = generate_signals(df)
        
        now = time.time()
        mark_price = get_mark_price(xrp)

        # Get current signal and mode
        signal = df["signal"].iloc[-1]
        mode = df["mode"].iloc[-1]

        if active_trade and signal != df["signal"].iloc[-1]:
            cancel_specific_order(order_id, xrp)
            active_trade = False
            print(f"Order closed due to opposite signal")
            # print(f"Grid order closed due to opposite signal. PnL: {active_trade['pnl']:.2f}, New balance: {balance:.2f}")
        # Place new trade if no active trade and valid signal
        # if not active_trade and signal in ["buy", "sell"]:
        # if not active_trade and df["signal"] in ["Buy", "Sell"]:
        if not active_trade:
            if df["signal"].iloc[-1] == "Buy":
                order_id = enter_trade("Buy",  df)
                active_trade = True
                print(f"Placed new order")
            else:
                order_id = enter_trade("Sell", df)
                active_trade = True
                print(f"Placed new order")

run_bot()
