import time
import math
import logging
from pybit.unified_trading import HTTP
from datetime import datetime

# CONFIG
symbol = "BTCUSDT"
timeframe = "1m"
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"

risk_amount = 0.01  # 1% of balance per position
grids = 3
max_active_grid_orders = 3
min_qty = 0.001

# INIT
session = HTTP(api_key=api_key, api_secret=api_secret)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

order_map = {
    "grid_orders": []
}

def get_klines(symbol, interval, limit=50):
    return session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)['result']['list']

def get_balance():
    balance_info = session.get_wallet_balance(accountType="UNIFIED")
    return float(balance_info['result']['list'][0]['totalEquity'])

def get_position():
    position_info = session.get_positions(category="linear", symbol=symbol)
    pos = position_info['result']['list'][0]
    return {
        "size": float(pos['size']),
        "side": pos['side'],
        "entry_price": float(pos['avgPrice']) if pos['avgPrice'] else 0
    }

def place_grid_orders(direction, price, grid_size_pct, qty):
    order_map['grid_orders'].clear()
    for i in range(1, grids + 1):
        if direction == "Buy":
            order_price = price * (1 - grid_size_pct * i)
        else:
            order_price = price * (1 + grid_size_pct * i)

        order = session.place_order(
            category="linear",
            symbol=symbol,
            side=direction,
            order_type="Limit",
            qty=round(qty, 4),
            price=round(order_price, 2),
            time_in_force="GTC",
            reduce_only=False
        )
        order_map['grid_orders'].append(order['result']['orderId'])
        logging.info(f"Placed {direction} grid order at {order_price}")

def cancel_grid_orders():
    for order_id in order_map['grid_orders']:
        session.cancel_order(category="linear", symbol=symbol, orderId=order_id)
        logging.info(f"Canceled grid order: {order_id}")
    order_map['grid_orders'].clear()

def close_position(position):
    if position['size'] > 0:
        close_side = "Sell" if position['side'] == "Buy" else "Buy"
        session.place_order(
            category="linear",
            symbol=symbol,
            side=close_side,
            order_type="Market",
            qty=round(position['size'], 4),
            reduce_only=True
        )
        logging.info(f"Closed {position['side']} position using market order")

def detect_trend(close):
    ma7 = sum(close[-7:]) / 7
    ma14 = sum(close[-14:]) / 14
    last_price = close[-1]

    price_range = max(close[-10:]) - min(close[-10:])
    range_pct = price_range / last_price
    is_trending = range_pct > 0.01

    if not is_trending:
        return None, ma7, ma14  # Consolidating market

    return ("Buy" if ma7 > ma14 else "Sell"), ma7, ma14

def main():
    while True:
        try:
            logging.info("Checking market condition...")
            klines = get_klines(symbol, timeframe)
            close = [float(c[4]) for c in klines]  # close prices
            last_price = close[-1]

            trend, ma7, ma14 = detect_trend(close)
            position = get_position()

            if trend:
                grid_size_pct = (max(close[-10:]) - min(close[-10:])) / last_price / grids
                grid_size_pct = max(grid_size_pct, 0.001)

                balance = get_balance()
                qty = balance * risk_amount / last_price
                qty = max(qty, min_qty)

                # Check and handle trend flip
                if position['size'] > 0:
                    if (trend == "Buy" and position['side'] == "Sell") or (trend == "Sell" and position['side'] == "Buy"):
                        cancel_grid_orders()
                        close_position(position)

                if not order_map['grid_orders']:
                    place_grid_orders(trend, last_price, grid_size_pct, qty)

            else:
                cancel_grid_orders()
                logging.info("Market is consolidating. Waiting for trend...")

            time.sleep(60)

        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
