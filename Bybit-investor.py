import time
from pybit.unified_trading import HTTP
import os
import sys
import urllib.request
import json
import pandas as pd
# from datetime import datetime
import requests
import math

# === CONFIG ===
api_key = "pFjYU5BrYyM6yzWkgx"
api_secret = "CfExgMdMqzZ8CcpmWozrhOlyBrvpBaNKGaiQ"
symbol = "XRPUSDT"
order_side = "Sell"  # "Buy" or "Sell"
qty = 0
entry_price = 0.0  # Example entry price
tp_percent = 0.002  # 0.2%
sl_percent = 0.002  # 0.2%
timeout_seconds = 180  # 3 minutes
trailing_distance = 0.002  # 0.2% trailing stop
position_active = False
highest_price = None
current_stop_order_id = None


# === BYBIT CONNECTION ===
client = HTTP(
    testnet=True,
    api_key=api_key,
    api_secret=api_secret
)

# Bybit API endpoint for market data
BYBIT_API_URL = "https://api.bybit.com/v5/market/kline"

# === FUNCTIONS ===
# Function to get the latest price using the Bybit API
def get_latest_price(symbol):
  url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
  try:
    response = requests.get(url, timeout=5)
    print(f"HTTP status code: {response.status_code}")
    
    if response.status_code != 200:
      print(f"Error: Bybit returned non-200 status code. Full response: {response.text}")
      return None
    
    data = response.json()
    if data.get("retCode") == 0 and "result" in data:
      tickers = data["result"].get("list", [])
      for ticker in tickers:
        if ticker["symbol"] == symbol:
          last_price = float(ticker["lastPrice"])
          print(f"Extracted last_price: {last_price}")
          return last_price
        print("Error: Could not find valid price data in response.")
        return None
  except Exception as e:
    print(f"Exception in get_latest_price: {e}")
    return None

def place_order(side, entry_price, tp, sl, qty):
  # Determine price precision (e.g. 4 decimal places)
  price_precision = len(str(entry_price).split('.')[1]) if '.' in str(entry_price) else 0

  # Format values properly as strings
  entry_price_str = f"{round(entry_price, price_precision):.{price_precision}f}"
  tp_str = f"{round(tp, price_precision):.{price_precision}f}"
  sl_str = f"{round(sl, price_precision):.{price_precision}f}"
  # Round down qty to nearest 0.1
  qty = math.floor(qty)
  qty_str = str(qty)
                                          
  try:
    response = client.place_order(
      category="linear",
      symbol=symbol,
      side=side,
      order_type="Limit",
      qty=qty_str,
      price=entry_price_str,
      take_profit=tp_str,
      stop_loss=sl_str,
      time_in_force="GTC"
    )

    order_id = response["result"]["orderId"]
    print(f"Buy order placed at {entry_price}, order ID: {order_id}")

    # Set trailing stop state
    position_active = True
    highest_price = entry_price
    current_stop_order_id = None  # reset

    # Check the response for errors
    if 'retCode' in response and response['retCode'] != 0:
      print(f"Error in placing order: {response.get('retMsg', 'No error message available')}")
      return {"error": response.get('retMsg', 'Unknown error')}
    
    return response
  except Exception as e:
    print(f"Error in placing order: {e}")
    return {"error": str(e)}

def get_market_data(symbol='XRPUSDT', interval='1', limit=200):
  # Prepare the URL with the query parameters
  url = f"{BYBIT_API_URL}?symbol={symbol}&interval={interval}&limit={limit}"

  # Debug print the URL to ensure it's correct
  # print(f"Requesting URL: {url}")
  
  try:
    # Make the GET request using urllib
    with urllib.request.urlopen(url) as response:
      data = json.loads(response.read())
      
    # Debug print the raw response to check the structure
    # print(f"Response: {data}")
    
    # Check if the response has a 'result' key
    if 'result' not in data:
      raise ValueError(f"API error: {data}")
    
    # Extract the list of candles
    ohlcv_data = data['result']['list']
    
    # If data is empty or no candles are returned, handle it
    if not ohlcv_data:
      print("No data returned.")
      return {}
    
    # Extract data into lists
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for candle in ohlcv_data:
      timestamps.append(candle[0])  # timestamp
      opens.append(float(candle[1]))  # open
      highs.append(float(candle[2]))  # high
      lows.append(float(candle[3]))  # low
      closes.append(float(candle[4]))  # close
      volumes.append(float(candle[5]))  # volume
      
    return {
      'timestamps': timestamps,
      'opens': opens,
      'highs': highs,
      'lows': lows,
      'closes': closes,
      'volumes': volumes    
    }
  
  except Exception as e:
    print(f"Error fetching market data: {e}")
    return {}

def get_closes(market_data):
  closes = []
  for item in market_data:
    # Ensure the item has at least 5 elements (index 4 must exist for close price)
    if len(item) >= 5:
      try:
        # Try to convert the closing price (index 4) to float
        close_price = float(item[4])
        closes.append(close_price)
      except ValueError:
        print(f"Skipping invalid closing price: {item[4]}")
        continue  # Skip invalid entries
      else:
        print(f"Skipping incomplete data: {item}")
  return closes

# Function to calculate Moving Average (MA)
def calculate_moving_average(data, period):
    return sum(data[-period:]) / period

def check_entry_signals(prices):
  # Calculate moving averages
  ma7 = calculate_moving_average(prices, 7)
  ma14 = calculate_moving_average(prices, 14)
  ma28 = calculate_moving_average(prices, 28)
    
  # Logic for buy/sell signals
  if ma7 > ma14 > ma28:
    print("Buy signal: 7-min MA crossed above 14-min MA")
    return "Buy"
  elif ma7 < ma14 < ma28:
    print("Sell signal: 7-min MA crossed below 14-min MA")
    return "Sell"
  else:
    print("No signal")
    return None

def calculate_volatility(closes, period=10):
  price_changes = []
    
  for i in range(1, period):
    price_changes.append(abs(closes[-i] - closes[-(i + 1)]))
    
    avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
  return avg_price_change

def set_sl_tp(current_price, volatility, side):
  if current_price is None or not isinstance(current_price, (int, float)):
    raise ValueError("current_price must be a number")

  min_gap = current_price * 0.3  # 30% buffer
  if side == "Buy":
    sl_price = round(current_price * (1 - volatility) - min_gap, 4)
    tp_price = round(current_price * (1 + volatility) + min_gap, 4)
  elif side == "Sell":
    sl_price = round(current_price * (1 + volatility) + min_gap, 4)
    tp_price = round(current_price * (1 - volatility) - min_gap, 4)
  else:
    raise ValueError("Invalid side. Must be 'Buy' or 'Sell'.")

  return sl_price, tp_price
                                

def get_balance(asset="USDT"):
  try:
    result = client.get_wallet_balance(accountType="UNIFIED")
    balance = float(result["result"]["list"][0]["totalWalletBalance"])
    print(f"Wallet balance: {balance} {asset}")
    return balance
  except Exception as e:
    print(f"Error fetching balance: {e}")
    return 0.0
            
def calculate_position_size(balance, price, risk_percent=1.0):
  """
  balance: total available balance in quote currency (e.g., USDT)
  price: current market price of the asset
  risk_percent: % of balance to use per trade (default: 1%)
  """
  risk_amount = (5 / 100) * balance
  qty = risk_amount / price
  return round(qty, 3)  # You can adjust precision based on asset rules

# Function for trailing stop logic
def update_trailing_stop(entry_price, current_price, trailing_stop_percent, side):
  # Calculate the trailing stop distance
  trailing_stop_distance = entry_price * trailing_stop_percent

  if side == "Buy":
    # If the price increases, adjust the stop-loss higher
    if current_price > entry_price + trailing_stop_distance:
      trailing_stop_price = current_price - trailing_stop_distance
      return trailing_stop_price
  elif side == "Sell":
    # If the price decreases, adjust the stop-loss lower
    if current_price < entry_price - trailing_stop_distance:
      trailing_stop_price = current_price + trailing_stop_distance
      return trailing_stop_price
    
  return None  # No change to the trailing stop

def scalping_bot(symbol):
  # Fetch market data (already implemented)
  market_data = get_market_data()
  df = pd.DataFrame(market_data)
  # print(df.head())
  # print(df.index.name)  # should be 'timestamp'
  # print(df.columns)
  
  # df.set_index('timestamp', inplace=True)  # if 'timestamp' is a column

  closes = df['closes'].tolist()
  # print(closes.tail())

  # Get closing prices (filter out invalid entries)
  # closes = get_closes(market_data)

  if len(closes) == 0:
    print("No valid closing prices found.")
    return
  
  # Check for buy/sell signal
  side = check_entry_signals(closes)

  sl_price = None
  tp_price = None
  
  if side:
    # Get current price (latest closing price)
    # current_price, highest_price, current_stop_order_id = get_latest_price(symbol)
    current_price = get_latest_price(symbol)
    print(f"Current price before setting SL/TP: {current_price}")
    if current_price is None:
      print("Error: current_price is None, skipping SL/TP setup.")
      return  # prevent execution past this point      
    
    # Calculate the average volatility (price movement) over the last 10 candles
    volatility = calculate_volatility(closes, period=10)

    print(f"Current price before setting SL/TP: {current_price}")
    # Set SL and TP
    # sl_price, tp_price = set_sl_tp(current_price, volatility, side)
    # Typecast current_price to float if it's valid
    if current_price is not None:
      print(current_price)
      current_price = float(current_price)
      # Now you can pass current_price to set_sl_tp
      sl_price, tp_price = set_sl_tp(current_price, volatility, side)
    else:
      print("Error: current_price is None, skipping SL/TP setup.")

    # Print entry, SL, and TP details
    print(f"Entry Price: {current_price}")
    # print(f"Stop-Loss (SL) Price: {sl_price}")
    # print(f"Take-Profit (TP) Price: {tp_price}")
    if sl_price is not None and tp_price is not None:
      print(f"Stop-Loss (SL) Price: {sl_price}")
      print(f"Take-Profit (TP) Price: {tp_price}")
    else:
      print("SL/TP not set due to missing current price or API issue.")
          
    
    # Simulate order placement (for example purposes)
    print(f"Placing {side} order at {current_price} with SL at {sl_price} and TP at {tp_price}")
    # place_order(side, current_price, tp_price, sl_price)
    balance = get_balance()
    if current_price is None:
      print("Error: current_price is None, cannot calculate position size.")
      return  # or skip the order placement
            
    qty = calculate_position_size(balance, current_price, risk_percent=5.0)

    print(f"Calculated Position Size: {qty} {symbol.replace('USDT', '')}")
    order_response = place_order(side, current_price, tp_price, sl_price, qty)
    order_id = order_response.get("result", {}).get("orderId")

    if not order_id:
      print("Failed to place the order. Exiting.")
      return
    print(f"Order placed with order ID: {order_id}")

    # Start trailing stop logic
    entry_price = current_price
    while True:
      current_price = get_latest_price(symbol)

      # Update trailing stop if price has moved in favor
      trailing_stop = update_trailing_stop(entry_price, current_price, trailing_stop_percent, side)

      if trailing_stop:
        print(f"New trailing stop price: {trailing_stop}")
        # Modify the existing order's stop-loss to the new trailing stop
        modify_order(order_id, trailing_stop)
        
        time.sleep(5)  # Check price every 5 seconds

  else:
    print("No action taken (no clear signal)")
  # display_status(current_price, tp_price, sl_price)

def modify_order(order_id, new_stop_loss):
  try:
    response = client.replace_order(
      category="linear",
      symbol=symbol,
      order_id=order_id,
      stop_loss=new_stop_loss
    )
    if response["retCode"] == 0:
      print(f"Successfully modified the order. New stop-loss: {new_stop_loss}")
    else:
      print(f"Failed to modify order: {response}")
  except Exception as e:
    print(f"Error modifying order: {e}")

def display_status(entry, tp, sl, timeout=600):
  start_time = time.time()
  while True:
    current_price = get_latest_price()
    os.system("clear")  # Clear the screen each loop

    sys.stdout.write("=== Scalping Bot Active ===\r\n")
    sys.stdout.write(f"Entry Price   : {entry}\r\n")
    sys.stdout.write(f"Take Profit   : {tp}\r\n")
    sys.stdout.write(f"Stop Loss     : {sl}\r\n")
    sys.stdout.write(f"Current Price : {current_price}\r\n")
    elapsed = int(time.time() - start_time)
    sys.stdout.write(f"Elapsed Time  : {elapsed}s / {timeout}s\r\n")

    sys.stdout.flush()  # Force immediate output

    if elapsed >= timeout:
      sys.stdout.write("\r\n[!] Timeout reached.\r\n")
      sys.stdout.flush()
      break

    time.sleep(1)

def main():
  symbol = 'XRPUSDT'
  scalping_bot(symbol)
  # data = get_market_data(symbol, interval)
  
  # Calculate indicators
  # rsi = calculate_rsi(data['closes'])
  # macd, signal = calculate_macd(data['closes'])
  # upper_band, middle_band, lower_band = calculate_bollinger_bands(data['closes'])
  # stoch = calculate_stochastic(data['highs'], data['lows'], data['closes'])
  
  # Print the results
  # print("RSI:", rsi[-5:])
  # print("MACD:", macd[-5:])
  # print("Signal:", signal[-5:])
  # print("Bollinger Bands:", upper_band, middle_band, lower_band)
  # print("Stochastic:", stoch[-5:])
  
# if __name__ == "__main__":
#     main()
    
#=== START ===
# if __name__ == "__main__":
#     scalping_bot()
main()
