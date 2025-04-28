import time
import math
import requests
from pybit.unified_trading import HTTP

# Constants
symbol = "XRPUSDT"
interval = 1  # in minutes
risk_amount = 0.8  # risk per trade as a fraction of balance
ema_fast_period = 7
ema_slow_period = 14
atr_period = 14
atr_threshold = 0.005  # 0.5% volatility filter threshold

# Auth
session = HTTP(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET", testnet=False)

# Utility functions
def get_klines(symbol, interval, limit=100):
  res = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
    return res

def ema(data, period):
  k = 2 / (period + 1)
  ema_val = data[0]
  for price in data[1:]:
    ema_val = price * k + ema_val * (1 - k)
  return ema_val

def calculate_atr(highs, lows, closes, period):
  trs = [
    max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        for i in range(1, len(highs))
  ]
    return sum(trs[-period:]) / period

def get_balance():
  res = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    return float(res["result"]["list"][0]["coin"][0]["walletBalance"])

def get_last_price():
    return float(session.latest_information_for_symbol(symbol=symbol)["last_price"])

def open_trade(side, qty):
  session.place_order(
    category="linear",
    symbol=symbol,
    side=side,
    order_type="Market",
    qty=round(qty, 3),
    time_in_force="GoodTillCancel",
    reduce_only=False,
    close_on_trigger=False
  )

def close_trade(current_side, qty):
  close_side = "Sell" if current_side == "Buy" else "Buy"
  session.place_order(
    category="linear",
    symbol=symbol,
    side=close_side,
    order_type="Market",
    qty=round(qty, 3),
    reduce_only=True,
    time_in_force="GoodTillCancel",
    close_on_trigger=False
  )
  print(f"Closing {current_side} position with {qty} units")

def get_position():
  res = session.get_positions(category="linear", symbol=symbol)
  pos = res["result"]["list"][0]
  size = float(pos["size"])
  side = pos["side"]  # "Buy", "Sell", or "None"
    return side, size

def set_leverage(symbol, leverage):
    try:
      response = session.set_leverage(
        category="linear",
        symbol=symbol,
        buyLeverage=str(leverage),
        sellLeverage=str(leverage)
      )
      print(f"Leverage set response: {response}")
    except Exception as e:
      print(f"Error setting leverage for {symbol}: {e}")

def get_qty_step(symbol):
  info = session.get_instruments_info(category="linear", symbol=symbol)
  lot_size_filter = info["result"]["list"][0]["lotSizeFilter"]
  step = float(lot_size_filter["qtyStep"])
  min_qty = float(lot_size_filter["minOrderQty"])
    return step, min_qty

# Main strategy loop
def run_bot():
  while True:
    try:
      klines = get_klines(symbol=symbol, interval=str(interval), limit=100)["result"]["list"]
      closes = [float(c[4]) for c in klines]
      highs = [float(c[2]) for c in klines]
      lows = [float(c[3]) for c in klines]
      ema_fast = ema(closes[-ema_fast_period:], ema_fast_period)
      ema_slow = ema(closes[-ema_slow_period:], ema_slow_period)
      atr = calculate_atr(highs, lows, closes, atr_period)
      last_price = closes[-1]
      volatility_ratio = atr / last_price

      side = "Buy" if ema_fast > ema_slow else "Sell"

      current_side, current_size = get_position()

      if current_side != side:
        if current_size > 0:
          # Close previous position first
          close_trade(current_side, current_size)
          # Wait until the position is closed
          retries = 10
          while retries > 0:
            time.sleep(1)
            new_side, new_size = get_position()
            if new_size == 0:
              break
            retries -= 1
            if retries == 0:
              print("Warning: Position not closed after retries!")

        # Now open new position
        balance = get_balance()
        step, min_qty = get_qty_step(symbol)
        qty = balance * risk_amount / last_price
        qty = math.floor(qty / step) * step
        
        if qty < min_qty:
          print(f"Qty {qty} too small, skipping.")
          return

        open_trade(side, qty)
        print(f"Opened {side} trade for qty {qty}")
      else:
        print(f"No position change (still in {current_side}). Skipping trade.")

    except Exception as e:
      print(f"Error: {e}")

    now = datetime.utcnow()
    sleep_time = 60 - now.second
    if sleep_time == 60:
        sleep_time = 0  # Edge case: exactly at 00 seconds
    time.sleep(sleep_time)
    
# Example usage
set_leverage(symbol=symbol, leverage=10)
run_bot()
