import time
import math
import requests
from pybit.unified_trading import HTTP
import pandas as pd

# Constants
symbol = "XRPUSDT"
interval = 1  # in minutes
risk_amount = 1.0  # risk per trade as a fraction of balance
ema_fast_period = 7
ema_slow_period = 14
atr_period = 14
atr_threshold = 0.005  # 0.5% volatility filter threshold
candlestick_patterns = ['Bullish_Engulfing', 'Bearish_Engulfing', 'Hammer', 'Inverted_Hammer', 'Doji', 
                        'Morning_Star', 'Evening_Star', 'Shooting_Star', 'Piercing_Line']

# Auth
session = HTTP(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET", testnet=False)

# Utility functions
def get_klines(symbol, interval, limit=100):
  res = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
  return res['result']

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
  
def close_position():
  positions = session.my_position(category="linear", symbol=symbol)["result"]
  for position in positions:
    if position["size"] > 0:
      side = "Sell" if position["side"] == "Buy" else "Buy"
      qty = position["size"]
      session.place_order(
        category="linear",
        symbol=symbol,
        side=side,
        order_type="Market",
        qty=round(qty),
        time_in_force="GoodTillCancel",
        reduce_only=True,
        close_on_trigger=False
      )
      print(f"Closed position: {side} for qty {qty}")
      return True
  return False

# Set leverage to 10x
def set_leverage(symbol, leverage):
  try:
    response = session.set_leverage(
      category="linear",  # required: 'linear' for USDT Perpetual, 'inverse' for Inverse Perpetual
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

# Candlestick pattern detection functions
def is_bullish_engulfing(df):
  return df['close'].iloc[-1] > df['open'].iloc[-1] and df['open'].iloc[-2] > df['close'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-2] and df['open'].iloc[-1] < df['close'].iloc[-2]

def is_bearish_engulfing(df):
  return df['open'].iloc[-1] < df['close'].iloc[-1] and df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-2] and df['open'].iloc[-1] > df['close'].iloc[-2]

def is_hammer(df):
  return (df['high'].iloc[-1] - df['close'].iloc[-1]) > (df['close'].iloc[-1] - df['low'].iloc[-1]) and (df['high'].iloc[-1] - df['open'].iloc[-1]) > (df['close'].iloc[-1] - df['low'].iloc[-1])

def is_inverted_hammer(df):
  return (df['high'].iloc[-1] - df['open'].iloc[-1]) > (df['close'].iloc[-1] - df['low'].iloc[-1]) and (df['close'].iloc[-1] - df['low'].iloc[-1]) > (df['high'].iloc[-1] - df['close'].iloc[-1])

def is_doji(df):
  return abs(df['close'].iloc[-1] - df['open'].iloc[-1]) <= (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.1

def is_morning_star(df):
  return df['close'].iloc[-1] > df['open'].iloc[-1] and df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-3] < df['open'].iloc[-3] and df['open'].iloc[-1] > df['close'].iloc[-2]

def is_evening_star(df):
  return df['close'].iloc[-1] < df['open'].iloc[-1] and df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-3] > df['open'].iloc[-3] and df['open'].iloc[-1] < df['close'].iloc[-2]

def is_shooting_star(df):
  return (df['high'].iloc[-1] - df['open'].iloc[-1]) > (df['close'].iloc[-1] - df['low'].iloc[-1]) and (df['high'].iloc[-1] - df['close'].iloc[-1]) > (df['close'].iloc[-1] - df['low'].iloc[-1]) and df['close'].iloc[-1] < df['open'].iloc[-1]

def is_piercing_line(df):
  return df['close'].iloc[-1] > df['open'].iloc[-1] and df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] > (df['open'].iloc[-2] + df['close'].iloc[-2]) / 2

def is_three_white_soldiers(df):
    return (df['close'].iloc[-3] > df['open'].iloc[-3] and
            df['close'].iloc[-2] > df['open'].iloc[-2] and
            df['close'].iloc[-1] > df['open'].iloc[-1] and
            df['close'].iloc[-2] > df['close'].iloc[-3] and
            df['close'].iloc[-1] > df['close'].iloc[-2])

def is_three_black_crows(df):
    return (df['close'].iloc[-3] < df['open'].iloc[-3] and
            df['close'].iloc[-2] < df['open'].iloc[-2] and
            df['close'].iloc[-1] < df['open'].iloc[-1] and
            df['close'].iloc[-2] < df['close'].iloc[-3] and
            df['close'].iloc[-1] < df['close'].iloc[-2])

def is_marubozu(df):
    body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    candle_range = df['high'].iloc[-1] - df['low'].iloc[-1]
    return body_size >= 0.9 * candle_range  # 90% body, very small wicks

def is_hanging_man(df):
    body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    lower_shadow = df['open'].iloc[-1] - df['low'].iloc[-1] if df['open'].iloc[-1] > df['close'].iloc[-1] else df['close'].iloc[-1] - df['low'].iloc[-1]
    return lower_shadow > 2 * body and (df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])) < body

def is_dark_cloud_cover(df):
    return (df['close'].iloc[-2] > df['open'].iloc[-2] and  # previous candle bullish
            df['open'].iloc[-1] > df['close'].iloc[-2] and   # current opens above previous close
            df['close'].iloc[-1] < (df['open'].iloc[-1] + df['close'].iloc[-2]) / 2 and  # closes below midpoint
            df['close'].iloc[-1] < df['open'].iloc[-1])  # current candle bearish

def is_spinning_top(df):
    body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    upper_shadow = df['high'].iloc[-1] - max(df['close'].iloc[-1], df['open'].iloc[-1])
    lower_shadow = min(df['close'].iloc[-1], df['open'].iloc[-1]) - df['low'].iloc[-1]
    total_range = df['high'].iloc[-1] - df['low'].iloc[-1]
    return body <= 0.3 * total_range and upper_shadow >= 0.2 * total_range and lower_shadow >= 0.2 * total_range


# Function to check for all candlestick patterns
def check_candlestick_patterns(df):
    patterns = []
    if is_bullish_engulfing(df):
        patterns.append('Bullish_Engulfing')
    if is_bearish_engulfing(df):
        patterns.append('Bearish_Engulfing')
    if is_hammer(df):
        patterns.append('Hammer')
    if is_inverted_hammer(df):
        patterns.append('Inverted_Hammer')
    if is_doji(df):
        patterns.append('Doji')
    if is_morning_star(df):
        patterns.append('Morning_Star')
    if is_evening_star(df):
        patterns.append('Evening_Star')
    if is_shooting_star(df):
        patterns.append('Shooting_Star')
    if is_piercing_line(df):
        patterns.append('Piercing_Line')
    if is_three_white_soldiers(df):
        patterns.append('Three_White_Soldiers')
    if is_three_black_crows(df):
        patterns.append('Three_Black_Crows')
    if is_marubozu(df):
        patterns.append('Marubozu')
    if is_hanging_man(df):
        patterns.append('Hanging_Man')
    if is_dark_cloud_cover(df):
        patterns.append('Dark_Cloud_Cover')
    if is_spinning_top(df):
        patterns.append('Spinning_Top')
    return patterns


# Main strategy loop
def run_bot():
  current_position = None  # None, "Buy", or "Sell"
    
  while True:
    try:
      # Fetch recent 1-minute candlesticks
      klines = get_klines(symbol=symbol, interval=str(interval), limit=100)["result"]
      closes = [float(c[4]) for c in klines]
      highs = [float(c[2]) for c in klines]
      lows = [float(c[3]) for c in klines]
      df = pd.DataFrame(klines)
            
      # Calculate the indicators
      ema_fast = ema(closes[-ema_fast_period:], ema_fast_period)
      ema_slow = ema(closes[-ema_slow_period:], ema_slow_period)
      atr = calculate_atr(highs, lows, closes, atr_period)
      last_price = closes[-1]
      volatility_ratio = atr / last_price

      # If volatility is too low, skip trade
      if volatility_ratio < atr_threshold:
        print("Volatility too low. Skipping trade.")
        time.sleep(60)
        continue

      # Check for candlestick patterns
      detected_patterns = check_candlestick_patterns(df)
      print(f"Detected Patterns: {detected_patterns}")

      # Determine whether to Buy or Sell based on EMA crossover
      side = "Buy" if ema_fast > ema_slow else "Sell"
            
      # Ensure that any existing position is closed before opening a new one
      if current_position != side:
        if close_position():  # Close the previous position
          print("Previous position closed, proceeding with new trade.")
          balance = get_balance()
          step, min_qty = get_qty_step(symbol)
          qty = balance * risk_amount / last_price
          
          # Round down to the nearest allowed quantity step
          qty = math.floor(qty / step) * step
          
          if qty < min_qty:
            print(f"Qty {qty} too small, skipping.")
            return

          open_trade(side, qty)
          current_position = side
          print(f"Opened {side} trade for qty {qty}")
        else:
          print("No open position to close. Skipping trade.")
      else:
        print(f"No position change (still in {current_position}). Skipping trade.")
        
    except Exception as e:
      print(f"Error: {e}")
      
    time.sleep(60)  # Sleep for 1 minute

# Example usage: Set leverage before opening a position
set_leverage(symbol=symbol, leverage=10)

# Run the bot
run_bot()
