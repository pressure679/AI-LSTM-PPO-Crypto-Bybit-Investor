import time
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import threading
import requests
import math

# API setup
api_key = "wLqYZxlM27F01smJFS"
api_secret = "tuu38d7Z37cvuoYWJBNiRkmpqTU6KGv9uKv7"
session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
# SYMBOLS = ["BNBUSDT", "XRPUSDT", "SHIB1000USDT", "BROCCOLIUSDT"]
SYMBOLS = ["BNBUSDT", "SOLUSDT", "XRPUSDT", "FARTCOINUSDT", "DOGEUSDT", "SHIB1000USDT"]
# SYMBOLS = ["XRPUSDT", "FARTCOINUSDT", "DOGEUSDT", "SHIB1000USDT"]
risk = 0.1  # Example risk per trade
leverage=75
current_trade_info = []
def keep_session_alive(symbol):
    for attempt in range(30):
        try:
            # Example: Get latest position
            result = session.get_positions(category="linear", symbol=symbol)
            break  # If success, break out of loop
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on attempt {attempt+1}, retrying...")
            time.sleep(2)  # wait before retry
        finally:
            threading.Timer(1500, keep_session_alive, args(symbol,)).start()  # Schedule next call
def get_klines_df(symbol, interval="1", limit=100):
    response = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
    data = response['result']['list']
    df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Turnover"])
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.int64), unit='ms')
    return df

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def RSI(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def ADX(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = ATR(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di

def BullsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['High'] - ema

def BearsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['Low'] - ema

def Momentum(series, period=10):
    return series - series.shift(period)

def calculate_indicators(df):
    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_21'] = df['Close'].ewm(span=21).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['Macd_line'], df['Macd_signal'], df['Macd_histogram'] = MACD(df['Close'])
    # === MACD Crossovers ===
    df['Macd_cross_up'] = (df['Macd_line'] > df['Macd_signal']) & (df['Macd_line'].shift(1) <= df['Macd_signal'].shift(1))
    df['Macd_cross_down'] = (df['Macd_line'] < df['Macd_signal']) & (df['Macd_line'].shift(1) >= df['Macd_signal'].shift(1))
    # === MACD Trend Status ===
    df['Macd_trending_up'] = df['Macd_line'] > df['Macd_signal']
    df['Macd_trending_down'] = df['Macd_line'] < df['Macd_signal']

    # df['Momentum'] = df['Close'] - df['Close'].shift(10)
    # Custom momentum (% of recent high-low range)
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    price_range = high_14 - low_14
    df['Momentum'] = 100 * (df['Close'] - df['Close'].shift(14)) / price_range
    # Momentum trend signals: compare current Momentum with previous
    df['Momentum_increasing'] = df['Momentum'] > df['Momentum'].shift(1)
    df['Momentum_decreasing'] = df['Momentum'] < df['Momentum'].shift(1)

    df['RSI'] = RSI(df['Close'], 14)
    adx, plus_di, minus_di = ADX(df)
    df['ADX'] = adx
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['Bulls'] = df['High'] - df['Close'].shift(1)
    df['Bears'] = df['Close'].shift(1) - df['Low']
    df.dropna(inplace=True)
    return df
def generate_signal(df):
    signals = []

    for i in range(len(df)):
        signal = ""
        latest = df.iloc[i]
            
        if latest['ADX'] > 20:
            # Buy Signal
            # if (latest['Momentum_increasing'] and latest['RSI'] > 50 and 
            #     (latest['+DI'] > latest['-DI'] or latest['Bulls'] > latest['Bears']) and
            #     latest['EMA_7'] > latest['EMA_14'] > latest['EMA_21'] > latest['EMA_50'] and
            #     latest['Macd_trending_up']):
            # if (latest['Momentum'] > 0 and latest['RSI'] > 50 and 
            #     (latest['+DI'] > latest['-DI'] or latest['Bulls'] > latest['Bears']) and
            #     latest['EMA_7'] > latest['EMA_14'] > latest['EMA_21'] > latest['EMA_50'] and
            #     latest['Macd_trending_up']):
            if latest['EMA_21'] > latest['EMA_50'] and latest['Macd_trending_up']:
                signal = "Buy"
                # if (latest['Momentum_decreasing'] and latest['RSI'] > 50 and 
            #     (latest['+DI'] < latest['-DI'] or latest['Bulls'] < latest['Bears']) and
            #     latest['EMA_7'] > latest['EMA_14'] > latest['EMA_21'] > latest['EMA_50'] and
            #     latest['Macd_trending_up']):

            # elif (latest['Momentum'] < 0 and latest['RSI'] < 50 and 
            #       (latest['-DI'] > latest['+DI'] or latest['Bulls'] < latest['Bears']) and
            #       latest['EMA_7'] < latest['EMA_14'] < latest['EMA_21'] < latest['EMA_50'] and
            #       latest['Macd_trending_down']):
            if latest['EMA_21'] < latest['EMA_50'] and latest['Macd_trending_down']:
                signal = "Sell"
            else:
                signal = ""
        signals.append(signal)
    df['signal'] = signals
    return df['signal']
def get_balance():
    balance_data = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"]
    return float(balance_data[0]["totalEquity"])

def get_mark_price(symbol):
    price_data = session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]
    return float(price_data["lastPrice"])

def get_qty_step(symbol):
    info = session.get_instruments_info(category="linear", symbol=symbol)
    data = info["result"]["list"][0]
    step = float(data["lotSizeFilter"]["qtyStep"])
    min_qty = float(data["lotSizeFilter"]["minOrderQty"])
    return step, min_qty

def calculate_dynamic_qty(symbol, risk_amount, atr):
    price = get_mark_price(symbol)
    stop_distance = atr * 1.5
    qty = risk_amount / stop_distance
    return round(qty, 6)
def place_sl_and_tp(symbol, side, entry_price, atr, qty, levels=[1.5, 2.5, 3.5, 4.5]):
    """
    Places initial SL and multiple TP orders based on ATR multiples.
    
    Args:
        symbol (str): trading symbol, e.g. "XRPUSDT"
        side (str): "Buy" or "Sell"
        entry_price (float): price at entry
        atr (float): current ATR value
        qty (float): total trade quantity
        levels (list): list of ATR multiples for TP levels
        
    Returns:
        dict with order IDs of SL and TP orders
    """
    orders = {}

    # SL price calculation
    if side == "Buy":
        sl_price = entry_price - 1.5 * atr
    else:  # Sell
        sl_price = entry_price + 1.5 * atr

    # Place stop loss order
    sl_response = session.set_trading_stop(
        category="linear",
        symbol=symbol,
        side=side,
        stop_loss=str(round(sl_price, 6))
        # trailing_stop=None,  # we'll handle trailing later
        # tp_trigger_by="LastPrice"
    )
    orders['sl'] = sl_response

    # Calculate and place TP limit orders
    qty_per_tp = [
        # qty_per_tp[0] = qty * 0.4,
        # qty_per_tp[1] = qty * 0.2,
        # qty_per_tp[2] = qty * 0.2,
        # qty_per_tp[3] = qty * 0.2
        qty * 0.4,
        qty * 0.2,
        qty * 0.2,
        qty * 0.2
    ]
    tp_orders = []
    for i, mult in enumerate(levels):
        if side == "Buy":
            tp_price = entry_price + mult * atr
        else:  # Sell
            tp_price = entry_price - mult * atr
            
        tp_response = session.place_order(
            category="linear",
            symbol=symbol,
            side="Sell" if side == "Buy" else "Buy",
            order_type="Limit",
            # TODO
            qty=round(qty_per_tp[i], 6),
            price=round(tp_price, 6),
            time_in_force="GTC",
            reduce_only=True
        )
        tp_orders.append(tp_response)
    orders['tp'] = tp_orders
    return orders

def enter_trade(signal, df, symbol, risk_pct=0.1):
    global leverage
    global current_trade_info

    if signal not in ["Buy", "Sell"]:
        print("[WARN] Invalid signal for enter_trade.")
        return None

    entry_price = get_mark_price(symbol)  # or df['Close'].iloc[-1] — pick one

    balance = get_balance()
    risk_amount = max(balance * risk_pct, 6)  # minimum risk amount 7

    step, min_qty = get_qty_step(symbol)
    atr = df['ATR'].iloc[-1]
    # total_qty = calculate_dynamic_qty(symbol, risk_amount, atr)
    
    # Round total_qty down to nearest step
    total_qty = math.floor(risk_amount / entry_price) * step

    print(f"risk amount: {risk_amount}")
    print(f"step: {step}")
    print(f"total_qty: {total_qty}")
    
    if total_qty < min_qty:
        print(f"[WARN] Quantity {total_qty} below minimum {min_qty}, skipping trade.")
        return None

    side = "Buy" if signal == "Buy" else "Sell"

    sl_price = round(entry_price - atr * 1.5, 6) if side == "Buy" else round(entry_price + atr * 1.5, 6)

    tp_price = round(entry_price + atr * 4.5, 6) if side == "Buy" else round(entry_price - atr * 4.5, 6)

    try:
        print(f"Placing order for {symbol} side={side} qty={total_qty}")
        response = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=total_qty,
            leverage=leverage,
            take_profit=str(tp_price),
            stop_loss=str(sl_price)
        )
        order_id = response.get("result", {}).get("orderId")
        print(f"[INFO] Placed {side} order with TP={tp_price}, SL={sl_price}")
    except Exception as e:
        print(f"[ERROR] Failed to place order: {e}")
        return None

    # Optionally place additional SL/TP if your function returns more orders/info
    orders = place_sl_and_tp(symbol, side, entry_price, atr, total_qty)
    
    # You need to adapt this depending on what place_sl_and_tp returns
    # For example:
    if orders:
        # Assuming orders contains keys like 'sl', 'tps', 'order_id'
        current_trade_info = {
            "symbol": symbol,
            "entry_price": entry_price,
            "signal": side,
            "qty": total_qty,
            "remaining_qty": total_qty,
            "sl": sl_price,
            "tps": [tp_price],  # or orders['tps'] if multiple levels
            "atr": atr,
            "active_tp_index": 0,
            "order_id": order_id or orders.get('order_id')
        }
    else:
        # If no additional orders placed
        current_trade_info = {
            "symbol": symbol,
            "entry_price": entry_price,
            "signal": side,
            "qty": total_qty,
            "remaining_qty": total_qty,
            "sl": sl_price,
            "tps": [tp_price],
            "atr": atr,
            "active_tp_index": 0,
            "order_id": order_id
        }

# def get_open_orders(symbol):
    
def cancel_old_orders(symbol):
    try:
        # Fetch active orders for the symbol (category may be needed depending on API)
        # open_orders = session.get_active_order(category="linear", symbol=symbol)['result']['list']
        open_orders = session.get_open_orders(category="linear", symbol=symbol)['result']['list']
        for order in open_orders:
            order_id = order['orderId']
            session.cancel_active_orders(category="linear", symbol=symbol, orderId=order_id)
        print(f"Cancelled old orders for {symbol}")
    except Exception as e:
        print(f"Failed to cancel orders for {symbol}: {e}")
def manage_trailing_sl(current_price):
    global current_trade_info
    atr = trade_info['atr']
    side = trade_info['signal']
    entry_price = trade_info['entry_price']
    sl = trade_info['sl']

    if side == "Buy":
        new_sl = max(sl, current_price - atr * 1.5)
    else:
        new_sl = min(sl, current_price + atr * 1.5)

    trade_info['sl'] = new_sl

def take_partial_profit(symbol, side, qty):
    close_side = "Sell" if side == "Buy" else "Buy"
    try:
        response = session.place_active_order(
            symbol=symbol,
            side=close_side,  # opposite side to close position
            order_type="Market",
            qty=qty,
            reduce_only=True,
            time_in_force="ImmediateOrCancel",
            leverage=leverage
        )
        # response = session.place_order(
        #     category="linear",
        #     symbol=symbol,
        #     side=close_side,
        #     order_type="Market",
        #     qty=qty,
        #     reduce_only=True
        # )
        print(f"[INFO] Partially closed {qty} at TP.")
        return response
    except Exception as e:
        print(f"[ERROR] Failed to close partial position: {e}")
        return None
def check_exit_conditions(current_price, atr):
    side = trade_info['side']
    tps = trade_info['tps']
    active_index = trade_info['active_tp_index']
    sl = trade_info['sl']

    # Check SL hit
    if side == "Buy" and current_price <= sl:
        print("[INFO] Stop loss hit (Buy)")
        cancel_old_orders(current_trade_info['symbol'])
        return 'stop'
    elif side == "Sell" and current_price >= sl:
        print("[INFO] Stop loss hit (Sell)")
        cancel_old_orders(current_trade_info['symbol'])
        return 'stop'

    portion = [0.4, 0.2, 0.2, 0.2]

    # Check TP hit
    if active_index < len(tps):
        target = tps[active_index]
        if (side == "Buy" and current_price >= target) or (side == "Sell" and current_price <= target):
            print(f"[INFO] Take profit level {active_index + 1} hit")
            trade_info['active_tp_index'] += 1

            # exit_condition = manage_trailing_sl(current_price)
            update_trailing_sl(symbol, current_price, atr)

            if active_index < len(portion):  # avoid index error
                qty_to_close = portion[active_index] * current_trade_info['qty']
                take_partial_profit(current_trade_info['symbol'], current_trade_info['side'], qty_to_close)
        return exit_condition

    if trade_info['active_tp_index'] >= len(tps):
        print("[INFO] All TP levels hit.")
        cancel_old_orders(current_trade_info['symbol'])
        return 'complete'

    return None
def get_position(symbol):
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        positions = response.get("result", {}).get("list", [])
        
        if not positions:
            print(f"[INFO] No positions found for {symbol}.")
            return None

        # Debug print full position to inspect structure
        # print(f"[DEBUG] Raw position data for {symbol}: {positions[0]}")

        pos = positions[0]
        size = float(pos.get("size", 0))
    # try:
    #     response = session.get_positions(category="linear", symbol=symbol)
    #     print("[DEBUG] get_positions response:", response)
    #     return response
    except Exception as e:
        print(f"[get_position] Error fetching position for {symbol}: {e}")
        return None
def update_trailing_sl(symbol, close, atr):
    pos = get_position(symbol)
    new_sl = close - atr * 1.5 if pos['side'] == "Buy" else close + atr * 1.5
    if (pos['side'] == "Buy" and new_sl > pos['sl']) or (pos['side'] == "Sell" and new_sl < pos['sl']):
        print(f"Updating trailing SL on {symbol} to {new_sl}")
        session.set_trading_stop(
            category="linear",
            symbol=symbol,
            stop_loss=new_sl
        )
def calculate_avg_pnl(df, symbol):
    global current_trade_info
    df['PnL'] = df['Close'] - current_trade_info['entry_price']
    df['PnL'] = df['PnL'] * np.where(df['Close'] > df['EMA_14'], 1, -1)
    df['adj_PnL'] = df['PnL'] / df['ATR']
    # avg_pnl = df['adj_PnL'].tail(360).mean()

    # df[symbol].append(avg_pnl)
    # if len(df[symbol]) > 20:
        # df[symbol] = df[symbol][-20:]

    # df['avg_adj_PnL'] = df['adj_PnL'].rolling(window=360).mean()
    # df['avg_PnL'] = df['adj_PnL'].rolling(window=360).mean()
    avg_pnl = df['adj_PnL'].rolling(window=360).mean()
    # return avg_pnl
    return avg_pnl
def wait_until_next_candle(interval_minutes):
    now = time.time()
    seconds_per_candle = interval_minutes * 60
    sleep_seconds = seconds_per_candle - (now % seconds_per_candle)
    # print(f"Waiting {round(sleep_seconds, 2)} seconds until next candle...")
    time.sleep(sleep_seconds)
def run_bot():
    global current_trade_info
    previous_signals = {symbol: "" for symbol in SYMBOLS}  # track last signal per symbol
    trade_info = []
    while True:
        for symbol in SYMBOLS:
            try:
                df = get_klines_df(symbol)
                df = calculate_indicators(df)
                df['signal'] = generate_signal(df)
                risk_amount = max(get_balance() * 0.1, 6) 
                atr = df['ATR'].iloc[-1]
                step, min_qty = get_qty_step(symbol)
                total_qty = math.floor(risk_amount / df['Close'].iloc[-1]) * step
                df['symbol'] = symbol
                latest = df.iloc[-1]
                balance = get_balance()
                print(f"=== {symbol} Stats ===")
                print(f"Signal: {latest['signal']}")
                print(f"Close: {latest['Close']:.6f}")
                print(f"EMA7: {latest['EMA_7']:.6f}")
                print(f"EMA14: {latest['EMA_14']:.6f}")
                print(f"EMA21: {latest['EMA_21']:.6f}")
                print(f"EMA50: {latest['EMA_50']:.6f}")
                print(f"MacD trend up/down: {latest['Macd_trending_up']}/{latest['Macd_trending_down']}")
                print(f"RSI: {latest['RSI']:.2f}")
                print(f"ADX: {latest['ADX']:.2f}")
                print(f"+DI: {latest['+DI']:.2f}")
                print(f"-DI: {latest['-DI']:.2f}")
                print(f"BullsPower: {latest['Bulls']:.6f}")
                print(f"BearsPower: {latest['Bears']:.6f}")
                print(f"Momentum: {latest['Momentum']:.2f}")
                print(f"Momentum increasing/decreasing: {latest['Momentum_increasing']}/{latest['Momentum_decreasing']}")
                print(f"ATR: {atr:.6f}")
                print(f"Trade Qty (calculated): {total_qty}")
                print(f"Balance: {balance:.2f}")
                position = get_position(symbol)
                open_orders = session.get_open_orders(category="linear", symbol=symbol)['result']['list']
                if open_orders:
                    print(f"Open Position: Side={position['side']} Entry={position['entry_price']:.6f} Qty={position['qty']} PnL={position['unrealizedPnl']:.2f}")
                    update_trailing_sl(symbol, df['Close'].iloc[-1], df['ATR'].iloc[-1])
                    exit_condition = check_exit_condition(symbol, df['ATR'].iloc[-1])
                    # --- AUTO-REVERSE LOGIC ---
                    if latest['signal'] != "" and latest['signal'] != position['side']:
                        print(f"[AUTO-REVERSE] Opposite signal detected: Closing {position['side']} and entering {latest['signal']}")
                        # Exit current position
                        # exit_position(symbol)
                        cancel_old_orders(symbol)
                        time.sleep(2)  # Give time for exit to process
                        # Enter opposite trade
                        enter_trade(latest['signal'], df, symbol)
                        print(f"[TRADE] Reversed position to {latest['signal']}")
                        # Update last known signal
                        previous_signals[symbol] = latest['signal']
                    else:
                        print("[INFO] Holding current position — no reversal needed.")
                        # update_trailing_sl(symbol, df['Close'].iloc[-1], df['ATR'].iloc[-1])
                        # exit_condition = check_exit_condition(symbol, df['ATR'].iloc[-1])
                else:
                    if latest['signal'] != "":
                        print(f"[INFO] No open position — entering new {latest['signal']} trade.")
                        enter_trade(latest['signal'], df, symbol)
                        print(f"current_trade_info: {current_trade_info}")
                        avg_pnl = calculate_avg_pnl(df, symbol)
                        print(f"avg_pnl: {avg_pnl}")
                        previous_signals[symbol] = latest['signal']
                time.sleep(2)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        print("Cycle complete. Waiting for next candle...\n")
        wait_until_next_candle(1)
if __name__ == "__main__":
    run_bot()
# set trailing stop
# session.set_trading_stop(
#     category="linear",
#     symbol=symbol,
#     stop_loss=new_stop_loss
# )

# partial close order
# response = session.place_active_order(
#     symbol=coin,
#     side="Sell",  # opposite side to close position
#     order_type="Market",
#     qty=qty_to_close,
#     reduce_only=True,
#     time_in_force="ImmediateOrCancel",
#     leverage=leverage
# )
