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
soft_stoploss_pct = 0.003  # 0.3% stop loss

# Auth
session = HTTP(api_key="", api_secret="", testnet=False)

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
    qty=round(qty, 2),
    time_in_force="GoodTillCancel",
    reduce_only=False,
    close_on_trigger=False
    
  )
  
def close_position(current_position, qty):
  close_side = "Sell" if current_position == "Buy" else "Buy"
  session.place_order(
    category="linear",
    symbol=symbol,
    side=close_side,
    order_type="Market",
    qty=round(qty, 2),
    time_in_force="GoodTillCancel",
    reduce_only=True,
    close_on_trigger=False
    
  )

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
    print(f"Error setting leverage: {e}")
    
def get_qty_step(symbol):
  info = session.get_instruments_info(category="linear", symbol=symbol)
  lot_size_filter = info["result"]["list"][0]["lotSizeFilter"]
  step = float(lot_size_filter["qtyStep"])
  min_qty = float(lot_size_filter["minOrderQty"])
  return step, min_qty

def get_position_info(symbol):
  res = session.get_positions(category="linear", symbol=symbol)
  data = res["result"]["list"]
  for position in data:
    if float(position["size"]) > 0:
      return {
        "side": position["side"],
        "size": float(position["size"]),
        "entry_price": float(position["entryPrice"]),
        "unrealized_pnl": float(position["unrealisedPnl"])
        
      }
    return None
  
# Main Strategy
def run_bot():
  current_position = None  # None, "Buy", or "Sell"
  cooldown_until = 0
  recovery_trade = False
  
  while True:
    try:
      current_time = time.time()
      if current_time < cooldown_until:
        remaining = int(cooldown_until - current_time)
        print(f"Cooldown active... waiting {remaining} seconds")
        time.sleep(10)
        continue
      
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
      
      # Check soft stoploss
      position_info = get_position_info(symbol)
      if position_info:
        unrealized_pnl = position_info["unrealized_pnl"]
        position_size = position_info["size"]
        entry_price = position_info["entry_price"]
        
        if unrealized_pnl < - (entry_price * position_size * soft_stoploss_pct):
          print(f"Soft stop-loss hit! Unrealized PnL = {unrealized_pnl:.2f}")
          close_position(position_info["side"], position_size)
          current_position = None
          cooldown_until = current_time + (2 * 60)  # 2 min cooldown
          recovery_trade = True  # Activate recovery
          time.sleep(5)
          continue
        
        if current_position != side:
          balance = get_balance()
          step, min_qty = get_qty_step(symbol)
          qty = balance * risk_amount / last_price
          
          if recovery_trade:
            qty = qty * 0.5
            print("Recovery mode active: reducing position size by 50%")
            
            qty = math.floor(qty / step) * step
            
          if qty < min_qty:
            print(f"Qty {qty} too small, skipping trade.")
            continue
          
          open_trade(side, qty)
          current_position = side
          print(f"Opened {side} trade for qty {qty}")
          
          if recovery_trade:
            print("Recovery trade complete. Returning to normal size.")
            recovery_trade = False
            
          else:
            print(f"No position change (still {current_position}). Skipping trade.")
            
    except Exception as e:
      print(f"Error: {e}")
      
    time.sleep(60)
    
# Example: Set leverage first
set_leverage(symbol=symbol, leverage=10)

# Start the bot
run_bot()
