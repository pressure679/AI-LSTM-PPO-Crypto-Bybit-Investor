import time
import numpy as np
import pandas as pd
import ta
import logging
from datetime import datetime
from pybit.unified_trading import HTTP

# === CONFIGURATION ===
API_KEY = "wLqYZxlM27F01smJFS"
API_SECRET = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
SYMBOL = "SPXUSDT"
RISK_PERCENT = 1
CANDLE_LIMIT = 100
INTERVAL = "1"
ATR_MULTIPLIER = 1.5
STOP_LOSS_PERCENT = 0.1
VOLATILITY_THRESHOLD = 0.5
TREND_THRESHOLD = 0.01

# === INIT ===
client = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
position = {"side": None, "qty": 0}

# Setup logging
logging.basicConfig(filename="trading_bot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def log(msg):
  print(msg)
  logging.info(msg)
  
  # === HELPERS ===
def get_ohlcv():
  response = client.get_kline(category="linear", symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT)
  if response["retCode"] == 0:
    data = response["result"]["list"][::-1]
    return np.array(data, dtype=float)
  else:
    log(f"OHLCV error: {response}")
  return None

def moving_average(values, period):
  return np.convolve(values, np.ones(period) / period, mode='valid')

def calculate_atr(df):
  atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
  return atr.iloc[-1]

def place_order(side, qty, price=None, sl=None, tp=None):
  try:
    order_type = "Market" if price is None else "Limit"
    params = {
      "category": "linear",
      "symbol": SYMBOL,
      "side": side,
      "order_type": order_type,
      "qty": str(qty),
      "time_in_force": "GTC"
      
    }
    if price: params["price"] = str(price)
    if sl: params["stop_loss"] = str(sl)
    if tp: params["take_profit"] = str(tp)
    
    res = client.place_order(**params)
    log(f"{side} order placed: {res}")
  except Exception as e:
    log(f"Order error: {e}")
    
def cancel_all_orders():
  try:
    res = client.cancel_all_orders(category="linear", symbol=SYMBOL)
    log(f"Cancel orders response: {res}")
  except Exception as e:
    log(f"Cancel order error: {e}")
    
def get_balance():
  try:
    balance = client.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"]
    return float(balance)
  except Exception as e:
    log(f"Balance error: {e}")
    return 0

# === STRATEGY ===
def scalping_bot():
  global position
  ohlcv = get_ohlcv()
  if ohlcv is None:
    log("Failed to fetch OHLCV")
    return

  timestamps, open_, high, low, close, vol = ohlcv[:, :6].T
  ma7 = moving_average(close, 7)
  ma14 = moving_average(close, 14)
  last_price = close[-1]
  
  if len(ma7) < 2 or len(ma14) < 2:
    log("Insufficient MA data")
    return
  
  df = pd.DataFrame({"high": high, "low": low, "close": close})
  atr = calculate_atr(df)
  grid_size = atr * ATR_MULTIPLIER
  log(f"ATR: {atr:.4f}, Grid Size: {grid_size:.2f}")
  
  if atr < VOLATILITY_THRESHOLD:
    log("Volatility too low. Skipping.")
    return
  
  price_range = max(close[-10:]) - min(close[-10:])
  range_pct = price_range / last_price
  is_trending = range_pct > TREND_THRESHOLD
  log(f"Market Type: {'Trending' if is_trending else 'Consolidating'}")

  ma7_prev, ma7_curr = ma7[-2], ma7[-1]
  ma14_prev, ma14_curr = ma14[-2], ma14[-1]
  buy_signal = ma7_prev < ma14_prev and ma7_curr > ma14_curr
  sell_signal = ma7_prev > ma14_prev and ma7_curr < ma14_curr
  
  # === TREND STRATEGY ===
  if is_trending:
    cancel_all_orders()
    
    if position["side"] == "Buy" and sell_signal:
      log("Closing Buy due to Sell Signal")
      place_order("Sell", position["qty"], price=round(last_price, 2))
      position = {"side": None, "qty": 0}
      
    elif position["side"] == "Sell" and buy_signal:
      log("Closing Sell due to Buy Signal")
      place_order("Buy", position["qty"], price=round(last_price, 2))
      position = {"side": None, "qty": 0}
      
    elif position["side"] is None:
      balance = get_balance()
      qty = round(balance * (RISK_PERCENT / 100.0) / last_price, 2)
      if buy_signal:
        log("Opening Buy")
        place_order("Buy", qty, price=round(last_price, 2), sl=round(last_price * (1 - STOP_LOSS_PERCENT), 2))
        position = {"side": "Buy", "qty": qty}
      elif sell_signal:
        log("Opening Sell")
        place_order("Sell", qty, price=round(last_price, 2), sl=round(last_price * (1 + STOP_LOSS_PERCENT), 2))
        position = {"side": "Sell", "qty": qty}
      else:
        log("No crossover trend signal.")
        
        # === GRID SCALPING ===
    else:
      if position["side"] is None:
        upper_price = last_price + grid_size
        lower_price = last_price - grid_size
        log(f"Placing grid orders: Buy {lower_price:.2f}, Sell {upper_price:.2f}")
        place_order("Buy", 1, price=round(lower_price, 2),
                    sl=round(lower_price - lower_price * STOP_LOSS_PERCENT, 2))
