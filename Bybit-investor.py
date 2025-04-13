import time
from pybit.unified_trading import HTTP
import os
import sys
import urllib.request
import json
import pandas as pd
from datetime import datetime

# === CONFIG ===
api_key = "pFjYU5BrYyM6yzWkgx"
api_secret = "CfExgMdMqzZ8CcpmWozrhOlyBrvpBaNKGaiQ"
symbol = "BTCUSDT"
order_side = "Sell"  # "Buy" or "Sell"
qty = 0.01
entry_price = 786.0  # Example entry price
tp_percent = 0.002  # 0.2%
sl_percent = 0.002
timeout_seconds = 180  # 3 minutes

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
def get_latest_price(symbol="XRPUSDT"):
  try:
    # Use Bybit V5 market ticker endpoint
    url = "https://api.bybit.com/v5/market/tickers"
    
    # Send request for linear (USDT Perpetual) market data
    params = {
      "category": "linear",
      "symbol": symbol  
    }

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
      data = response.json()
      if data["retCode"] == 0 and "result" in data and "list" in data["result"]:
        ticker = data["result"]["list"][0]
        last_price = ticker.get("lastPrice")
        if last_price:
          return float(last_price)
        else:
          print("Error: lastPrice not found.")
          return None
      else:
        print("Error: Invalid response or data not found.")
        return None
    else:
      print(f"Error: Received a {response.status_code} status code from Bybit.")
      return None
  except Exception as e:
    print(f"Error fetching price: {e}")
    return None                  

def place_order(side, entry_price, tp, sl):
  try:
    resp = client.place_order(
      category="linear",
      symbol=symbol,
      side=side,
      order_type="Limit",
      qty=qty,
      price=entry_price,
      take_profit=tp,
      stop_loss=sl,
      time_in_force="GTC"
    )
    return resp
  except Exception as e:
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

def calculate_rsi(closes, period=14):
  if len(closes) < period + 1:
    print("Not enough data to calculate RSI.")
    return []
  
  gains = []
  losses = []
  
  # Calculate gains and losses for the price changes
  for i in range(1, len(closes)):  # Start from 1, since we're comparing i and i-1
    change = closes[i] - closes[i - 1]
    if change > 0:
      gains.append(change)
      losses.append(0)
    else:
      gains.append(0)
      losses.append(abs(change))
      
    # Calculate the average gain and average loss over the period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Calculate RSI using the smoothing method
    rsi = []
    for i in range(period, len(closes)):  # Start calculating RSI after 'period' number of candles
      if avg_loss == 0:  # To avoid division by zero
        rsi.append(100)
      else:
        rs = avg_gain / avg_loss
        rsi_value = 100 - (100 / (1 + rs))
        rsi.append(rsi_value)
        
      # Update average gain and loss for the next period
      if i < len(closes) - 1:  # Update the moving averages until the last data point
        change = closes[i + 1] - closes[i]
        gain = change if change > 0 else 0
        loss = abs(change) if change < 0 else 0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
  return rsi

def calculate_ema(prices, period):
  ema_values = []
  multiplier = 2 / (period + 1)
  ema_values.append(sum(prices[:period]) / period)  # Initial EMA value (SMA)
  for i in range(period, len(prices)):
    ema = (prices[i] - ema_values[-1]) * multiplier + ema_values[-1]
    ema_values.append(ema)
  return ema_values

def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):
  fast_ema = calculate_ema(closes, fast_period)
  slow_ema = calculate_ema(closes, slow_period)

  # Calculate MACD line
  macd_line = [fast_ema[i] - slow_ema[i] for i in range(len(slow_ema))]

  # Calculate Signal line
  signal_line = calculate_ema(macd_line, signal_period)
  
  return macd_line, signal_line

def calculate_bollinger_bands(closes, period=20, nbdevup=2, nbdevdn=2):
  sma = sum(closes[-period:]) / period
  variance = sum([(x - sma) ** 2 for x in closes[-period:]]) / period
  stdev = variance ** 0.5

  upper_band = sma + (stdev * nbdevup)
  lower_band = sma - (stdev * nbdevdn)
  
  return upper_band, sma, lower_band

def calculate_stochastic(highs, lows, closes, period=14):
  stoch_values = []
  for i in range(period, len(closes)):
    highest_high = max(highs[i - period:i])
    lowest_low = min(lows[i - period:i])
    stoch = (closes[i] - lowest_low) / (highest_high - lowest_low) * 100
    stoch_values.append(stoch)
  return stoch_values

def find_swing_points(candles):
  highs = [float(c['high']) for c in candles]
  lows = [float(c['low']) for c in candles]
  swing_highs, swing_lows = [], []

  for i in range(2, len(highs) - 2):
    if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
      swing_highs.append(highs[i])
    if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
      swing_lows.append(lows[i])

  return swing_highs[-3:], swing_lows[-3:] if swing_highs and swing_lows else ([], [])

# Function to calculate Moving Average (MA)
def calculate_moving_average(data, period):
    return sum(data[-period:]) / period

def check_entry_signals(prices):
  # Calculate moving averages
  ma7 = calculate_moving_average(prices, 7)
  ma14 = calculate_moving_average(prices, 14)
  ma28 = calculate_moving_average(prices, 28)
    
  # Logic for buy/sell signals
  if ma7 > ma14:
    print("Buy signal: 7-min MA crossed above 14-min MA")
    return "Buy"
  elif ma7 < ma14:
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

def set_sl_tp(current_price, volatility, side="Buy"):
  # SL: 2 times the volatility below current price for Buy
  # SL: 2 times the volatility above current price for Sell
  if side == "Buy":
    sl_price = current_price - (2 * volatility)
    tp_price = current_price + (2 * volatility)  # Example TP logic
  else:
    sl_price = current_price + (2 * volatility)
    tp_price = current_price - (2 * volatility)  # Example TP logic
    
  return sl_price, tp_price

def scalping_bot(symbol):
  # Fetch market data (already implemented)
  market_data = get_market_data()
  df = pd.DataFrame(market_data)
  print(df.head())
  print(df.index.name)  # should be 'timestamp'
  print(df.columns)
  
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
  
  if side:
    # Get current price (latest closing price)
    current_price = closes[-1]
    
    # Calculate the average volatility (price movement) over the last 10 candles
    volatility = calculate_volatility(closes, period=10)
    
    # Set SL and TP
    sl_price, tp_price = set_sl_tp(current_price, volatility, side)

    # Print entry, SL, and TP details
    print(f"Entry Price: {current_price}")
    print(f"Stop-Loss (SL) Price: {sl_price}")
    print(f"Take-Profit (TP) Price: {tp_price}")
    
    # Simulate order placement (for example purposes)
    print(f"Placing {side} order at {current_price} with SL at {sl_price} and TP at {tp_price}")
    place_order(side, current_price, tp_price, sl_price)
  else:
    print("No action taken (no clear signal)")
    
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
