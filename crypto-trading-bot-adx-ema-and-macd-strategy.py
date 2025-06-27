import time
import pandas as pd
from pybit.unified_trading import HTTP
from datetime import datetime

# Bybit credentials
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"

session = HTTP(api_key=api_key, api_secret=api_secret)
symbol = "XRPUSDT"
interval_15m = 900   # 15 minutes in seconds
interval_1h = 3600   # 1 hour in seconds
leverage = 10

# For tracking last fetched times
last_fetch_15m = 0
last_fetch_1h = 0

# Current trade status
active_position = None
current_strategy = None

# ---------------------------------------------
def get_klines_df(symbol, interval_sec, limit=150):
    interval_map = {
        60: "1",
        180: "3",
        300: "5",
        900: "15",
        1800: "30",
        3600: "60",
        14400: "240",
        86400: "D"
    }

    candles = session.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval_map[interval_sec],
        limit=limit
    )["result"]["list"]

    df = pd.DataFrame(candles)
    df.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume", "turnover", "confirm", "cross_seq", "timestamp_e", "trade_num", "taker_base_vol"]
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df = df.astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    return df

# ---------------------------------------------
def calculate_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.shift() - low

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    df["PlusDI"] = plus_di
    df["MinusDI"] = minus_di
    df["ADX"] = adx
    return df

# ---------------------------------------------
def get_ema_signal(df):
    fast = df["Close"].ewm(span=5).mean()
    slow = df["Close"].ewm(span=15).mean()

    if fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
        return "buy"
    elif fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
        return "sell"
    return None

def get_macd_signal(df):
    close = df["Close"]
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return "buy"
    elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return "sell"
    return None

# ---------------------------------------------
def close_all_positions():
    global active_position, current_strategy
    if active_position:
        side = "Sell" if active_position == "Buy" else "Buy"
        print(f"Closing {active_position} position")
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=10, reduce_only=True)
        active_position = None
        current_strategy = None

def enter_trade(signal, strategy):
    global active_position, current_strategy
    if active_position:
        print("Already in a trade. Ignoring signal.")
        return
    print(f"Entering {signal.upper()} trade using {strategy} strategy")
    session.place_order(category="linear", symbol=symbol, side=signal.capitalize(), order_type="Market", qty=10)
    active_position = signal.capitalize()
    current_strategy = strategy

# ---------------------------------------------
def run_bot():
    global last_fetch_15m, last_fetch_1h

    session.set_leverage(category="linear", symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))

    while True:
        now = time.time()

        # Update 15m data
        if now - last_fetch_15m >= interval_15m:
            df_15m = get_klines_df(symbol, interval_15m, limit=150)
            df_15m['Close'] = df_15m['Close'].astype(float)
            ema_signal = get_ema_signal(df_15m)
            last_fetch_15m = now

            if ema_signal:
                # If already in a trade with MACD, close it before entering
                if current_strategy == "MACD":
                    close_all_positions()
                enter_trade(ema_signal, "EMA")

        # Update 1h data
        if now - last_fetch_1h >= interval_1h:
            df_1h = get_klines_df(symbol, interval_1h, limit=150)
            df_1h[["High", "Low", "Close"]] = df_1h[["High", "Low", "Close"]].astype(float)
            df_1h = calculate_adx(df_1h)

            adx_value = df_1h['ADX'].iloc[-1]
            macd_signal = get_macd_signal(df_1h)
            last_fetch_1h = now

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ADX: {adx_value:.2f} | MACD Signal: {macd_signal}")

            if adx_value > 25 and macd_signal:
                # Close EMA trade first
                if current_strategy == "EMA":
                    close_all_positions()
                enter_trade(macd_signal, "MACD")

        time.sleep(5)

# ---------------------------------------------
if __name__ == "__main__":
    run_bot()
