# ai_scalping_bot.py

import requests
import time
import hmac
import hashlib
import json
import uuid
import sys
import os
from datetime import datetime

# === CONFIGURATION ===
API_KEY = "pFjYU5BrYyM6yzWkgx"
API_SECRET = "CfExgMdMqzZ8CcpmWozrhOlyBrvpBaNKGaiQ"
BASE_URL = "https://api-testnet.bybit.com"
SYMBOL = "XRPUSDT"
INTERVAL = "1"
SL_PERCENT = 30.0
TP_PERCENT = 30.0
TRAILING_STOP_PERCENT = 0.5
RISK_PERCENT = 5.0

# === UTILITY FUNCTIONS ===
def generate_signature(params, api_secret):
  sorted_params = sorted(params.items())
  query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
  return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def send_signed_request(http_method, endpoint, params):
  params["api_key"] = API_KEY
  params["timestamp"] = int(time.time() * 1000)
  params["recv_window"] = 5000
  params["sign"] = generate_signature(params, API_SECRET)
  url = f"{BASE_URL}{endpoint}"
  response = requests.request(http_method, url, params=params)
  return response

def fetch_latest_price(symbol):
  endpoint = f"/v5/market/tickers"
  params = {"category": "linear", "symbol": symbol}
  response = requests.get(f"{BASE_URL}{endpoint}", params=params)
  if response.status_code == 200:
    try:
      last_price = float(response.json()["result"]["list"][0]["lastPrice"])
      return last_price
    except:
      print("Error parsing last price.")
      return None
    print("Error fetching price.")
    return None

def calculate_position_size(balance, price, risk_percent):
  risk_amount = balance * (risk_percent / 100.0)
  return round(risk_amount / price)

def calculate_sl_tp(entry_price, sl_percent, tp_percent, side):
  if side == "Buy":
    sl = entry_price * (1 - sl_percent / 100)
    tp = entry_price * (1 + tp_percent / 100)
  else:
    sl = entry_price * (1 + sl_percent / 100)
    tp = entry_price * (1 - tp_percent / 100)
  return round(sl, 4), round(tp, 4)

def place_order(symbol, side, qty, price, sl, tp):
  endpoint = "/v5/order/create"
  params = {
    "category": "linear",
    "symbol": symbol,
    "side": side,
    "order_type": "Limit",
    "qty": str(qty),
    "price": str(price),
    "take_profit": str(tp),
    "stop_loss": str(sl),
    "time_in_force": "GTC"
  }
  response = send_signed_request("POST", endpoint, params)
  if response.status_code == 200 and response.json()["retCode"] == 0:
    print(f"{side} order placed at {price}, order ID: {response.json()['result']['orderId']}")
    return response.json()["result"]["orderId"]
  else:
    print("Error placing order:", response.text)
    return None

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

# === MAIN TRADING FUNCTION ===
def scalping_bot():
  last_price = fetch_latest_price(SYMBOL)
  if last_price is None:
    return
  side = "Buy"  # This should be calculated by your MA logic in full version
  balance = fetch_balance()
  qty = calculate_position_size(balance, last_price, RISK_PERCENT)
  sl, tp = calculate_sl_tp(last_price, SL_PERCENT, TP_PERCENT, side)
  qty = round(qty, 1)
  last_price = round(last_price, 4)
  tp = round(tp, 4)
  sl = round(sl, 4)
  place_order(SYMBOL, side, qty, last_price, sl, tp)
  

# === START ===
if __name__ == "__main__":
  scalping_bot()
