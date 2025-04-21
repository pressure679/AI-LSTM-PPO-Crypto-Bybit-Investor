import time
import math
import requests
from pybit import HTTP

# Constants
symbol = "XRPUSDT"
interval = 1  # in minutes
risk_amount = 0.01  # risk per trade as a fraction of balance
ema_fast_period = 7
ema_slow_period = 21
atr_period = 14
atr_threshold = 0.005  # 0.5% volatility filter threshold

# Auth
session = HTTP(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")

# Utility functions
def get_klines(symbol, interval, limit=100):
    res = session.get_kline(symbol=symbol, interval=interval, limit=limit)
    return res

def ema(data, period):
    k = 2 / (period + 1)
    ema = data[0]
    for price in data[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_atr(highs, lows, closes, period):
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(highs))]
    return sum(trs[-period:]) / period

def get_balance():
    return float(session.get_wallet_balance(coin="USDT")["USDT"]["available_balance"])

def get_last_price():
    return float(session.latest_information_for_symbol(symbol=symbol)["last_price"])

def open_trade(side, qty):
    session.place_active_order(
        symbol=symbol,
        side=side,
        order_type="Market",
        qty=qty,
        time_in_force="GoodTillCancel",
        reduce_only=False,
        close_on_trigger=False
    )

# Main strategy loop
def run_bot():
    while True:
        try:
            klines = get_klines(symbol, interval, limit=100)
            closes = [float(c[4]) for c in klines]
            highs = [float(c[2]) for c in klines]
            lows = [float(c[3]) for c in klines]

            ema_fast = ema(closes[-ema_fast_period:], ema_fast_period)
            ema_slow = ema(closes[-ema_slow_period:], ema_slow_period)
            atr = calculate_atr(highs, lows, closes, atr_period)

            last_price = closes[-1]
            volatility_ratio = atr / last_price

            print(f"EMA Fast: {ema_fast}, EMA Slow: {ema_slow}, ATR: {atr}, Ratio: {volatility_ratio}")

            if volatility_ratio < atr_threshold:
                print("Volatility too low. Skipping trade.")
            else:
                balance = get_balance()
                qty = balance * risk_amount / last_price
                side = "Buy" if ema_fast > ema_slow else "Sell"
                open_trade(side, qty)
                print(f"Opened {side} trade for qty {qty}")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(60)  # Sleep for 1 minute

# Run
run_bot()
