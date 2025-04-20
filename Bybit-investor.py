import time
import hmac
import hashlib
import json
import urllib.parse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import ta
from pybit.unified_trading import HTTP

# --- Configuration ---
API_KEY = "w"
API_SECRET = ""
BASE_URL = "https://api.bybit.com"  # Mainnet
SYMBOL = "SPXUSDT"
INTERVAL = "1"
CANDLE_LIMIT = 100
RISK_PERCENT = 1
ATR_MULTIPLIER = 1.5
STOP_LOSS_PERCENT = 0.1
LOG_FILE = "scalping_bot.log"

# --- Logging ---
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def generate_signature(api_key, api_secret, timestamp, params):
  params["timestamp"] = timestamp
  params["api_key"] = api_key
  params["recv_window"] = "5000"
  sorted_params = sorted(params.items())
  query_string = urllib.parse.urlencode(sorted_params)
  payload = f"{timestamp}{api_key}{params['recv_window']}{query_string}"
  return hmac.new(api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

def send_signed_request(http_method, endpoint, params):
  timestamp = str(int(time.time() * 1000))
  sign = generate_signature(API_KEY, API_SECRET, timestamp, params)
  headers = {
    "X-BAPI-API-KEY": API_KEY,
    "X-BAPI-SIGN": sign,
    "X-BAPI-TIMESTAMP": timestamp,
    "X-BAPI-RECV-WINDOW": "5000",
    "Content-Type": "application/json"
    
  }
  url = f"{BASE_URL}{endpoint}"
  logging.debug(f"Request URL: {url}")
  logging.debug(f"Request Params: {params}")
  if http_method == "GET":
    return requests.get(url, params=params, headers=headers)
  else:
    return requests.request(http_method, url, json=params, headers=headers)
  
def get_ohlcv():
  url = f"{BASE_URL}/v5/market/kline"
  params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": CANDLE_LIMIT}
  response = requests.get(url, params=params)
  data = response.json()
  if "result" in data and "list" in data["result"]:
    return np.array(data["result"]["list"][::-1], dtype=float)
  return None

def calculate_atr(df):
  df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
  return df['atr'].iloc[-1]

def place_order(side, qty, price=None, sl=None, tp=None):
  endpoint = "/v5/order/create"
  req_time = int(time.time() * 1000)
  params = {
    "category": "linear",
    "symbol": SYMBOL,
    "side": side,
    "order_type": "Market" if price is None else "Limit",
    "qty": str(qty),
    "time_in_force": "GTC"
    
  }
  if price: params["price"] = str(price)
  if tp: params["take_profit"] = str(tp)
  if sl: params["stop_loss"] = str(sl)
  
  signature = generate_signature(API_KEY, API_SECRET, req_time, params)
  headers = {
    "Content-Type": "application/json",
    "X-BAPI-API-KEY": API_KEY,
    "X-BAPI-SIGN": signature,
    "X-BAPI-TIMESTAMP": str(req_time)
    
  }
  response = requests.post(BASE_URL + endpoint, headers=headers, json=params)
  logging.info(f"Order response: {response.text}")
  
def cancel_all_orders():
  endpoint = "/v5/order/cancel"
  params = {"symbol": SYMBOL, "category": "linear"}
  response = send_signed_request("POST", endpoint, params)
  logging.info(f"Cancel orders response: {response.text}")
  
# --- Trading Logic ---
position = {"side": None, "qty": 0}

def scalping_bot():
  global position
  ohlcv = get_ohlcv()
  if ohlcv is None:
    logging.warning("Failed to fetch OHLCV")
    return
  
  timestamps, open_, high, low, close, vol = ohlcv[:, :6].T
  ma7 = np.convolve(close, np.ones(7) / 7, mode='valid')
  ma14 = np.convolve(close, np.ones(14) / 14, mode='valid')
  
  if len(ma7) < 2 or len(ma14) < 2:
    logging.warning("Insufficient MA data")
    return
  
  df = pd.DataFrame({"high": high, "low": low, "close": close})
  current_atr = calculate_atr(df)
  logging.info(f"Current ATR: {current_atr}")
  
  if current_atr < 0.5:
    logging.info("Volatility is too low. Skipping.")
    return
  
  grid_size = current_atr * ATR_MULTIPLIER
  last_price = close[-1]
  price_range = max(close[-10:]) - min(close[-10:])
  is_trending = (price_range / last_price) > 0.01
  logging.info(f"Market Type: {'Trending' if is_trending else 'Consolidating'}")

  if is_trending and position["side"] is None:
    cancel_all_orders()
    upper_price = last_price + grid_size
    lower_price = last_price - grid_size
    place_order("Buy", 1, price=round(lower_price, 2), sl=round(lower_price - lower_price * STOP_LOSS_PERCENT, 2))
    place_order("Sell", 1, price=round(upper_price, 2), sl=round(upper_price + upper_price * STOP_LOSS_PERCENT, 2))
    
    ma7_prev, ma7_curr = ma7[-2], ma7[-1]
    ma14_prev, ma14_curr = ma14[-2], ma14[-1]
    buy_signal = ma7_prev < ma14_prev and ma7_curr > ma14_curr
    sell_signal = ma7_prev > ma14_prev and ma7_curr < ma14_curr
    
    client = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
    balance = float(client.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"])
    qty = round(balance * (RISK_PERCENT / 100.0) / last_price, 2)
    
    if position["side"] == "Buy" and sell_signal:
      place_order("Sell", position["qty"], last_price)
      position = {"side": None, "qty": 0}
    elif position["side"] == "Sell" and buy_signal:
      place_order("Buy", position["qty"], last_price)
      position = {"side": None, "qty": 0}
    elif position["side"] is None:
      if buy_signal:
        place_order("Buy", qty, last_price, sl=round(last_price - last_price * STOP_LOSS_PERCENT, 2))
        position = {"side": "Buy", "qty": qty}
      elif sell_signal:
        place_order("Sell", qty, last_price, sl=round(last_price + last_price * STOP_LOSS_PERCENT, 2))
        position = {"side": "Sell", "qty": qty}
      else:
        logging.info("No clear crossover signal")
        
if __name__ == "__main__":
  while True:
    try:
      scalping_bot()
    except Exception as e:
      logging.error(f"Error: {e}")
    time.sleep(60)
