import time
import math
import requests
from pybit.unified_trading import HTTP

# Constants
symbol = "XRPUSDT"
interval = 1  # in minutes
risk_amount = 0.3  # risk per trade as a fraction of balance
ema_fast_period = 7
ema_slow_period = 14
atr_period = 14
atr_threshold = 0.005  # 0.5% volatility filter threshold

# Auth
# session = HTTP(api_key="wLqYZxlM27F01smJFS", api_secret="tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7")
session = HTTP(api_key="", api_secret="", testnet=True)

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
    max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
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
    qty=round(qty),
    time_in_force="GoodTillCancel",
    reduce_only=False,
    close_on_trigger=False
    
  )

# Set leverage to 10x
def set_leverage(symbol, leverage):
  try:
    response = session.get_instruments_info(category="linear")
    # for item in response["result"]["list"]:
      # print(item["symbol"])
          
    response = session.set_leverage(
      category="linear",  # required: 'linear' for USDT Perpetual, 'inverse' for Inverse Perpetual
      symbol=symbol,
      buyLeverage=str(leverage),
      sellLeverage=str(leverage)
    )
    print(f"Leverage set response: {response}")
  except Exception as e:
    print(f"Error setting leverage for {symbol}: {e}")          

# Main strategy loop
def run_bot():
  current_position = None  # None, "Buy", or "Sell"
  
  while True:
    try:
      klines = get_klines(symbol=symbol, interval=str(interval), limit=100)["result"]["list"]
      # print("klines")
      closes = [float(c[4]) for c in klines]
      # print("closes")
      highs = [float(c[2]) for c in klines]
      # print("highs")
      lows = [float(c[3]) for c in klines]
      # print("lows")
      ema_fast = ema(closes[-ema_fast_period:], ema_fast_period)
      ema_slow = ema(closes[-ema_slow_period:], ema_slow_period)
      atr = calculate_atr(highs, lows, closes, atr_period)
      last_price = closes[-1]
      volatility_ratio = atr / last_price
      # print(f"EMA Fast: {ema_fast}, EMA Slow: {ema_slow}, ATR: {atr}, Ratio: {volatility_ratio}")
      
      # if volatility_ratio < atr_threshold:
        # print("Volatility too low. Skipping trade.")
      # else:
      side = "Buy" if ema_fast > ema_slow else "Sell"
      
      if current_position != side:
        balance = get_balance()
        qty = balance * risk_amount / last_price
        open_trade(side, qty)
        current_position = side
        print(f"Opened {side} trade for qty {qty}")
      else:
        print(f"No position change (still in {current_position}). Skipping trade.")
    except Exception as e:
      print(f"Error: {e}")
  
    time.sleep(60)  # Sleep for 1 minute

# Example usage: Set leverage before opening a position
# set_leverage(symbol=symbol, leverage=10)

# Run
run_bot()
