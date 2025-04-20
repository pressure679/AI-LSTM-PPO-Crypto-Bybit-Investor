import requests
import time
import hmac
import hashlib
import json
import urllib.parse
import numpy as np
import pandas as pd
import ta  # Ensure you have the 'ta' package installed
from datetime import datetime
from pybit.unified_trading import HTTP

# Constants
API_KEY = ""
API_SECRET = ""
BASE_URL = "https://api-testnet.bybit.com"
SYMBOL = "BTCUSDT"
RISK_PERCENT = 1
CANDLE_LIMIT = 100
INTERVAL = "1"
TIMEOUT_THRESHOLD = 600  # in seconds, for cancelling old orders
ATR_MULTIPLIER = 1.5  # ATR change threshold to re-evaluate grid spacing
STOP_LOSS_PERCENT = 0.1  # Stop Loss at 10% of the capital

# Helper Functions
def send_signed_request(http_method, endpoint, params):
  timestamp = str(int(time.time() * 1000))  # current timestamp in milliseconds
  params["timestamp"] = timestamp
  params["api_key"] = API_KEY
  params["recv_window"] = "5000"

  sign = generate_signature(API_KEY, API_SECRET, timestamp, params)

  headers = {
    "X-BAPI-API-KEY": API_KEY,
    "X-BAPI-SIGN": sign,
    "X-BAPI-TIMESTAMP": timestamp,
    "X-BAPI-RECV-WINDOW": "5000",
    "Content-Type": "application/json"
    
  }

  url = f"{BASE_URL}{endpoint}"
  print("Request URL:", url)
  print("Request Params:", params)
  
  if http_method == "GET":
    response = requests.get(url, params=params, headers=headers)
  else:
    response = requests.request(http_method, url, json=params, headers=headers)
    
  return response


def generate_signature(api_key, api_secret, timestamp, params):
  # Add necessary parameters to the sign request
  params["timestamp"] = timestamp
  params["api_key"] = api_key
  params["recv_window"] = "5000"
  
  # Sort parameters
  sorted_params = sorted(params.items())
  query_string = urllib.parse.urlencode(sorted_params)
  
  # Create the payload for HMAC SHA256 hashing
  payload = f"{timestamp}{api_key}{params['recv_window']}{query_string}"
  
  # Generate HMAC SHA256 signature
  return hmac.new(api_secret.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()


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
  return np.convolve(values, np.ones(period) / period, mode='valid')


# ATR Calculation (using ta)
def calculate_atr(df):
  df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
  return df['atr'].iloc[-1]


# Order placement with Stop Loss
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
  if price:
    params["price"] = str(price)
  if tp:
    params["take_profit"] = str(tp)
  if sl:
    params["stop_loss"] = str(sl)
    
  signature = generate_signature(API_KEY, API_SECRET, req_time, params)
  headers = {
    "Content-Type": "application/json",
    "X-BAPI-API-KEY": API_KEY,
    "X-BAPI-SIGN": signature,
    "X-BAPI-TIMESTAMP": str(req_time)
    
  }
  response = requests.post(url, headers=headers, json=params)
  print("Order response:", response.text)
  
  
# Cancel all orders
def cancel_all_orders():
  endpoint = "/v5/order/cancel"
  url = BASE_URL + endpoint
  params = {"symbol": SYMBOL, "category": "linear"}
  response = send_signed_request("POST", endpoint, params)
  print("Cancel orders response:", response.text)
  
  
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
  last_price = close[-1]
  
  if len(ma7) < 2 or len(ma14) < 2:
    print("Insufficient MA data")
    return
  
  # Market Type Detection (Volatility)
  df = pd.DataFrame({"high": high, "low": low, "close": close})
  current_atr = calculate_atr(df)
  print("Current ATR:", current_atr)
  
  # Check if volatility is within acceptable limits
  if current_atr < 0.5:  # Example threshold for low volatility
    print("Volatility is too low. Skipping order placement.")
    return
  
  # Dynamic Grid Size Based on ATR
  grid_size = current_atr * ATR_MULTIPLIER
  print(f"Grid Size: {grid_size:.2f}")
  
  # Market Type (Trend vs. Consolidating)
  price_range = max(close[-10:]) - min(close[-10:])
  range_pct = price_range / last_price
  is_trending = range_pct > 0.01  # Adjust this threshold as needed
  print("Market Type:", "Trending" if is_trending else "Consolidating")
  
  # Cancel Grid Orders if Trend is Detected
  if is_trending and position["side"] is None:
    print("Trend detected, canceling all grid orders")
    cancel_all_orders()
    
    # Grid Scalping Logic (Simple Version)
    upper_price = last_price + grid_size
    lower_price = last_price - grid_size
    print(f"Placing grid orders at: Buy {lower_price:.2f}, Sell {upper_price:.2f}")
    
    if position["side"] is None:
      # Place new grid orders
      place_order("Buy", 1, price=round(lower_price, 2), sl=round(lower_price - (lower_price * STOP_LOSS_PERCENT), 2))
      place_order("Sell", 1, price=round(upper_price, 2), sl=round(upper_price + (upper_price * STOP_LOSS_PERCENT), 2))
      
      # MA Crossover Logic (Trend Strategy)
      ma7_prev, ma7_curr = ma7[-2], ma7[-1]
      ma14_prev, ma14_curr = ma14[-2], ma14[-1]
      buy_signal = ma7_prev < ma14_prev and ma7_curr > ma14_curr
      sell_signal = ma7_prev > ma14_prev and ma7_curr < ma14_curr
      
      if position["side"] == "Buy" and sell_signal:
        print("Closing Buy due to Sell Signal")
        place_order("Sell", position["qty"], last_price)
        position = {"side": None, "qty": 0}
        
      elif position["side"] == "Sell" and buy_signal:
        print("Closing Sell due to Buy Signal")
        place_order("Buy", position["qty"], last_price)
        position = {"side": None, "qty": 0}
        
      elif position["side"] is None:
        client = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
        balance = float(client.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"])
        qty = round(balance * (RISK_PERCENT / 100.0) / last_price, 2)
        
        if buy_signal:
          print("Opening Buy Position")
          place_order("Buy", qty, last_price, sl=round(last_price - (last_price * STOP_LOSS_PERCENT), 2))
          position = {"side": "Buy", "qty": qty}
        elif sell_signal:
          print("Opening Sell Position")
          place_order("Sell", qty, last_price, sl=round(last_price + (last_price * STOP_LOSS_PERCENT), 2))
          position = {"side": "Sell", "qty": qty}
        else:
          print("No clear crossover signal")
          
          
if __name__ == "__main__":
  while True:
    try:
      scalping_bot()
    except Exception as e:
      print("Error:", e)
    time.sleep(60)
