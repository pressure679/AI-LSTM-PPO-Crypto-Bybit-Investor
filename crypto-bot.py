import time
import math
from datetime import datetime
import requests
import pandas as pd
from pybit import HTTP  # pip install pybit

# === SETUP ===
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
symbol = "BTCUSDT"
interval_15m = 15 * 60
interval_1h = 60 * 60

# Initialize session
session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    recv_window=5000,
    timeout=30
    # testnet=True  # uncomment if needed
)

# Global variables
last_fetch_15m = 0
last_fetch_1h = 0
current_strategy = None
active_position = None
entry_price = None
entry_time = None
peak_balance = 0


# === UTILITY FUNCTIONS ===

def get_klines_df(symbol, interval_sec, limit=150):
    interval_map = {
        60: "1",
        900: "15",
        3600: "60"
    }
    interval = interval_map[interval_sec]
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    klines = response.json()['result']['list']
    df = pd.DataFrame(klines, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    return df

def get_macd_signal(df):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    if histogram.iloc[-1] > histogram.iloc[-2] > 0:
        return "buy"
    elif histogram.iloc[-1] < histogram.iloc[-2] < 0:
        return "sell"
    return None

def get_ema_signal(df):
    ema_short = df['Close'].ewm(span=9, adjust=False).mean()
    ema_long = df['Close'].ewm(span=21, adjust=False).mean()
    if ema_short.iloc[-2] < ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]:
        return "buy"
    elif ema_short.iloc[-2] > ema_long.iloc[-2] and ema_short.iloc[-1] < ema_long.iloc[-1]:
        return "sell"
    return None

def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    df['ADX'] = adx
    return df

def get_balance():
    balance_data = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    return float(balance_data[0]["totalEquity"])

def get_mark_price(symbol):
    price_data = session.get_ticker(category="linear", symbol=symbol)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(symbol):
    info = session.get_symbols()
    for item in info['result']:
        if item['name'] == symbol:
            step = float(item['lotSizeFilter']['qtyStep'])
            min_qty = float(item['lotSizeFilter']['minOrderQty'])
            return step, min_qty
    return 0.001, 0.001

def get_trade_qty():
    return 0.01  # placeholder, adjust your qty logic

def close_all_positions():
    global active_position, current_strategy, entry_price, entry_time

    if active_position:
        side = "Sell" if active_position == "Buy" else "Buy"
        print(f"Closing {active_position} position")

        price_data = session.get_ticker(category="linear", symbol=symbol)["result"]["list"][0]
        exit_price = float(price_data["lastPrice"])
        exit_time = datetime.now()

        qty = get_trade_qty()
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, reduce_only=True)

        direction = 1 if active_position == "Buy" else -1
        pnl = (exit_price - entry_price) * qty * direction
        duration = (exit_time - entry_time).total_seconds() / 60  # minutes

        print(f"[TRADE CLOSED] {active_position} | Entry: {entry_price:.4f} | Exit: {exit_price:.4f} | PnL: ${pnl:.2f} | Duration: {duration:.1f} min")

        active_position = None
        current_strategy = None
        entry_price = None
        entry_time = None

def enter_trade(signal, strategy, df):
    global current_strategy, peak_balance, active_position, entry_price, entry_time

    mark_price = get_mark_price(symbol)
    min_price = df["Low"].iloc[-10:].min()
    max_price = df["High"].iloc[-10:].max()
    range_pct = (max_price - min_price) / mark_price

    if range_pct < 0.015:
        print("[INFO] Market range is too small (<1.5%), skipping trade.")
        return

    balance = get_balance()
    if balance > peak_balance:
        peak_balance = balance

    target_profit = peak_balance * 0.02
    expected_gain_pct = 0.3
    estimated_range_pct = range_pct * 3

    leverage = min(75, max(1, math.floor((target_profit / balance) / estimated_range_pct)))

    max_risk_pct = 0.35
    risk_pct = min(max_risk_pct, max(target_profit / (balance * expected_gain_pct), 0.05))
    risk_amount = balance * risk_pct

    step, min_qty = get_qty_step(symbol)
    qty = risk_amount * leverage / mark_price
    qty = math.floor(qty / step) * step
    if qty < min_qty:
        print(f"[WARNING] Calculated quantity {qty} is below minimum {min_qty}, skipping trade.")
        return

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Entering {strategy} trade | Signal: {signal} | Qty: {qty:.4f} | Leverage: {leverage} | Risk Amount: ${risk_amount:.2f}")

    side = "Buy" if signal == "buy" else "Sell"
    # session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, leverage=leverage)

    active_position = "Buy" if signal == "buy" else "Sell"
    current_strategy = strategy
    entry_price = mark_price
    entry_time = datetime.now()

def run_bot():
    global last_fetch_15m, last_fetch_1h, current_strategy

    while True:
        now = time.time()

        if now - last_fetch_15m >= interval_15m:
            df_15m = get_klines_df(symbol, interval_15m, limit=150)
            macd_signal = get_macd_signal(df_15m)
            last_fetch_15m = now

            mark_price = get_mark_price(symbol)
            if macd_signal and current_strategy != "EMA":
                df_1h = get_klines_df(symbol, interval_1h, limit=150)
                df_1h = calculate_adx(df_1h)
                adx_value = df_1h['ADX'].iloc[-1]

                if adx_value < 25 and macd_signal:
                    close_all_positions()
                    enter_trade(macd_signal, "MACD", df_15m)

        if now - last_fetch_1h >= interval_1h:
            df_1h = get_klines_df(symbol, interval_1h, limit=150)
            df_1h = calculate_adx(df_1h)

            adx_value = df_1h['ADX'].iloc[-1]
            ema_signal = get_ema_signal(df_1h)
            last_fetch_1h = now

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ADX: {adx_value:.2f} | EMA Signal
