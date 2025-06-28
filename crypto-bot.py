import time
import pandas as pd
from pybit.unified_trading import HTTP
from datetime import datetime
from datetime import datetime, timedelta

# Bybit credentials
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"

session = HTTP(
    api_key=api_key,
    api_secret=api_secret,
    recv_window=5000,
    timeout=30  # Increased timeout
    # testnet=True
)
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

entry_price = None
entry_time = None

def wait_for_next_candle(interval):
    now = datetime.utcnow()

    if interval == "15m":
        next_minute = (now.minute // 15 + 1) * 15
        if next_minute == 60:
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)
    elif interval == "1h":
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        raise ValueError("Unsupported interval")

    wait_seconds = (next_time - now).total_seconds()
    print(f"Waiting {int(wait_seconds)}s until next {interval} candle at {next_time.strftime('%H:%M:%S')} UTC")
    time.sleep(wait_seconds)
    
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
    # df.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume", "turnover", "confirm", "cross_seq", "timestamp_e", "trade_num", "taker_base_vol"]
    df.columns = ["timestamp", "Open", "High", "Low", "Close", "Volume", "turnover"]
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
    if df is None or df.empty or len(df) < 35:
        print("[ERROR] Not enough data to compute MACD")
        return None

    df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    macd_val = df['macd'].iloc[-1]
    signal_val = df['signal'].iloc[-1]

    if pd.isna(macd_val) or pd.isna(signal_val):
        print("[ERROR] MACD or Signal is NaN")
        return None

    if macd_val > signal_val:
        return "long"
    elif macd_val < signal_val:
        return "short"
    else:
        return "neutral"  # or just keep returning None if you want to skip sideways signals

def get_trade_qty():
    wallet = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    # wallet = session.get_wallet_balance(accountType="CONTRACT")["result"]["list"]
    usdt_balance = float(wallet[0]["totalEquity"])
    if usdt_balance < 5:
        return usdt_balance
    usd_amount = max(usdt_balance * 0.05, 5)  # 5% or $5 minimum
    price = float(session.get_ticker(category="linear", symbol=symbol)["result"]["list"][0]["lastPrice"])
    qty = round(usd_amount / price, 3)
    return qty

# ---------------------------------------------
def close_all_positions():
    global active_position, current_strategy, entry_price, entry_time

    if active_position:
        side = "Sell" if active_position == "Buy" else "Buy"
        print(f"Closing {active_position} position")

        # Get exit price
        price_data = session.get_ticker(category="linear", symbol=symbol)["result"]["list"][0]
        exit_price = float(price_data["lastPrice"])
        exit_time = datetime.now()

        # Place close order
        qty = get_trade_qty()
        session.place_order(category="linear", symbol=symbol, side=side, order_type="Market", qty=qty, reduce_only=True)

        # Calculate profit (approximate, not accounting for fees/slippage)
        direction = 1 if active_position == "Buy" else -1
        pnl = (exit_price - entry_price) * qty * direction
        duration = (exit_time - entry_time).total_seconds() / 60  # minutes

        print(f"[TRADE CLOSED] {active_position} | Entry: {entry_price:.4f} | Exit: {exit_price:.4f} | PnL: ${pnl:.2f} | Duration: {duration:.1f} min")

        # Reset state
        active_position = None
        current_strategy = None
        entry_price = None
        entry_time = None

import time  # make sure you have this import at the top of your script

def enter_trade(signal, strategy):
    global active_position, current_strategy

    if signal is None:
        print(f"[WARN] Tried to enter trade with None signal using {strategy}. Skipping.")
        return

    signal_cap = signal.capitalize()  # "buy" -> "Buy", "sell" -> "Sell"

    # If already in a position but signal differs, close existing before entering new
    if active_position and active_position != signal_cap:
        # Close existing position first
        side_to_close = "Sell" if active_position == "Buy" else "Buy"
        print(f"Closing existing {active_position} position before entering new {signal_cap} position.")
        session.place_order(
            category="linear",
            symbol=symbol,
            side=side_to_close,
            order_type="Market",
            qty=quantity,  # Quantity will be calculated below, but you can set or reuse here if needed
            reduce_only=True
        )
        active_position = None
        current_strategy = None

        time.sleep(1)  # wait 1 second to ensure order processing

    # If already in the same position, do nothing
    if active_position == signal_cap:
        print(f"Already in a {signal_cap} position. No action taken.")
        return

    # Fetch wallet balance
    balance_data = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    usdt_balance = 0
    for item in balance_data:
        if item["coin"] == "USDT":
            usdt_balance = float(item["availableToTrade"])
            break

    if usdt_balance == 0:
        print("[ERROR] USDT balance is 0")
        return

    # Fetch market price
    price_data = session.get_ticker(category="linear", symbol=symbol)["result"]
    mark_price = float(price_data["markPrice"])

    # Calculate 5% or $5, whichever is higher
    trade_value = max(usdt_balance * 0.05, 5)
    if trade_value > usdt_balance:
        print(f"[WARN] Not enough balance for minimum trade. Available: ${usdt_balance:.2f}, Needed: ${trade_value:.2f}")
        return

    # Calculate quantity, e.g., XRP allows 2 decimal places
    quantity = round(trade_value / mark_price, 2)

    print(f"Entering {signal_cap} trade using {strategy} strategy with qty {quantity} (~${trade_value:.2f})")

    session.place_order(
        category="linear",
        symbol=symbol,
        side=signal_cap,
        order_type="Market",
        qty=quantity
    )

    active_position = signal_cap
    current_strategy = strategy

# ---------------------------------------------
def run_bot():
    while True:
        # Wait for next full 15m candle (00, 15, 30, 45)
        wait_for_next_candle("15m")

        # --- Fetch 15m MACD signal ---
        df_15m = get_klines_df(symbol, interval_15m, limit=150)
        df_15m['Close'] = df_15m['Close'].astype(float)
        macd_signal = get_macd_signal(df_15m)

        if macd_signal:
            # If already in a trade with EMA, close it before switching to MACD
            if current_strategy == "EMA":
                close_all_positions()
            enter_trade(macd_signal, "MACD")

        # --- Every hour, also fetch ADX + EMA ---
        now = datetime.utcnow()
        if now.minute == 0:  # Only run on full hour
            df_1h = get_klines_df(symbol, interval_1h, limit=150)
            df_1h[["High", "Low", "Close"]] = df_1h[["High", "Low", "Close"]].astype(float)
            df_1h = calculate_adx(df_1h)

            adx_value = df_1h['ADX'].iloc[-1]
            ema_signal = get_ema_signal(df_1h)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] ADX: {adx_value:.2f} | MACD Signal: {macd_signal}")

            if adx_value > 25 and ema_signal:
                if current_strategy == "MACD":
                    close_all_positions()
                enter_trade(ema_signal, "EMA")

        # Small pause before next iteration
        time.sleep(1)

# ---------------------------------------------
if __name__ == "__main__":
    run_bot()
