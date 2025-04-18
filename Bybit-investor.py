import requests
import time
import hmac
import hashlib
import json
import numpy as np
from datetime import datetime

# Constants
API_KEY = "pFjYU5BrYyM6yzWkgx"
API_SECRET = "CfExgMdMqzZ8CcpmWozrhOlyBrvpBaNKGaiQ"
BASE_URL = "https://api-testnet.bybit.com"
SYMBOL = "XRPUSDT"
RISK_PERCENT = 2
CANDLE_LIMIT = 100
INTERVAL = "1"

def fetch_balance():
  endpoint = "/v5/account/wallet-balance"
  params = {"accountType": "UNIFIED"}
  response = send_signed_request("GET", endpoint, params)
  if response.status_code == 200:
    try:
      balance = float(response.json()["result"]["list"][0]["totalEquity"])
      return balance
    except:
      print("Error parsing balance.")
      return 0
    return 0

def send_signed_request(http_method, endpoint, params):
  params["api_key"] = API_KEY
  params["timestamp"] = int(time.time() * 1000)
  params["recv_window"] = 5000
  params["sign"] = generate_signature(params, API_SECRET)
  url = f"{BASE_URL}{endpoint}"
  response = requests.request(http_method, url, params=params)
  return response

# Signature
def generate_signature(api_key, secret, req_time, sign_params):
  sign_params["api_key"] = api_key
  sign_params["timestamp"] = str(req_time)
  sorted_params = sorted(sign_params.items())
  sign_string = "&".join([f"{key}={value}" for key, value in sorted_params])
  return hmac.new(secret.encode(), sign_string.encode(), hashlib.sha256).hexdigest()

# Get OHLCV data
def get_ohlcv():
  url = f"{BASE_URL}/v5/market/kline"
  params = {
    "category": "linear",
    "symbol": SYMBOL,
    "interval": INTERVAL,
    "limit": CANDLE_LIMIT
    
  }
  response = requests.get(url, params=params)
  data = response.json()
  if "result" in data and "list" in data["result"]:
    ohlcv = data["result"]["list"][::-1]  # newest last
    return np.array(ohlcv, dtype=float)
  return None

# Moving Average
def moving_average(values, period):
  return np.convolve(values, np.ones(period)/period, mode='valid')

# Candlestick patterns
def is_bullish_engulfing(open_, close):
  return close[-2] < open_[-2] and close[-1] > open_[-1] and close[-1] > open_[-2] and open_[-1] < close[-2]

def is_bearish_engulfing(open_, close):
  return close[-2] > open_[-2] and close[-1] < open_[-1] and close[-1] < open_[-2] and open_[-1] > close[-2]

# Position Size Calculation
def calculate_position_size(balance, price, risk_percent):
  risk_amount = balance * (risk_percent / 100.0)
  position_size = risk_amount / price
  return round(position_size)

# Order placement
def place_order(side, qty, price=None, sl=None, tp=None):
  endpoint = "/v5/order/create"
  url = BASE_URL + endpoint
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
  response = requests.post(url, headers=headers, json=params)
  print("Order response:", response.text)
  

# Track position globally
position = {"side": None, "qty": 0}

# Main strategy
def scalping_bot():
  global position
  ohlcv = get_ohlcv()
  if ohlcv is None:
    print("Failed to fetch OHLCV")
    return
  
  timestamps, open_, high, low, close, vol = ohlcv[:, :6].T
  ma7 = moving_average(close, 7)
  ma14 = moving_average(close, 14)

  if len(ma7) < 2 or len(ma14) < 2:
    return
  
  ma7_prev, ma7_curr = ma7[-2], ma7[-1]
  ma14_prev, ma14_curr = ma14[-2], ma14[-1]
  last_price = close[-1]
  balance = fetch_balance()
  qty = calculate_position_size(balance, last_price, RISK_PERCENT)

  buy_signal = ma7_prev < ma14_prev and ma7_curr > ma14_curr
  sell_signal = ma7_prev > ma14_prev and ma7_curr < ma14_curr

  if position["side"] == "Buy" and sell_signal:
    print("Closing Buy due to Sell Signal")
    place_order("Sell", position["qty"], last_price, None, None)
    position = {"side": None, "qty": 0}
    
  elif position["side"] == "Sell" and buy_signal:
    print("Closing Sell due to Buy Signal")
    place_order("Buy", position["qty"], last_price, None, None)
    position = {"side": None, "qty": 0}
    
  elif position["side"] is None:
    if buy_signal:
      print("Opening Buy Position")
      place_order("Buy", qty, last_price, None, None)
      position = {"side": "Buy", "qty": qty}
    elif sell_signal:
      print("Opening Sell Position")
      place_order("Sell", qty, last_price, None, None)
      position = {"side": "Sell", "qty": qty}
    else:
      print("Holding current position")
      
      
if __name__ == "__main__":
  while True:
    try:
      scalping_bot()
    except Exception as e:
      print("Error:", e)
    time.sleep(60)
