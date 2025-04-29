import time
import math
import requests
from pybit.unified_trading import HTTP

# Constants
symbol = "SHIB1000USDT"
interval = 1  # in minutes
risk_amount = 1.0  # risk per trade as a fraction of balance
ema_fast_period = 7
ema_slow_period = 14

# Auth
session = HTTP(api_key="wLqYZxlM27F01smJFS", api_secret="tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7", testnet=False)

# Utility functions
def get_klines(symbol, interval, limit=100):
  res = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
  return res

def ema(data, period):
  ema_values = []
  k = 2 / (period + 1)

  # Start with SMA for first EMA value
  sma = sum(data[:period]) / period
  ema_values.append(sma)

  # Apply EMA formula from the next point
  for price in data[period:]:
    ema_prev = ema_values[-1]
    ema_new = price * k + ema_prev * (1 - k)
    ema_values.append(ema_new)

  return ema_values


def get_balance():
  res = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
  return float(res["result"]["list"][0]["coin"][0]["walletBalance"])

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

def wait_until_next_candle(interval_minutes):
  now = time.time()
  seconds_per_candle = interval_minutes * 60
  sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
  print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
  time.sleep(sleep_seconds)

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
      last_price = closes[-1]  # We still use the actual last close price for position sizing etc.

      side = "Buy" if ema_fast > ema_slow else "Sell"

      current_side, current_size = get_position()

      if current_side != side:
        if current_size > 0:
          # Close previous position first
          close_trade(current_side, current_size)
          
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

    # Wait smartly
    wait_until_next_candle(interval)

    
# Example usage
set_leverage(symbol=symbol, leverage=10)
run_bot()
