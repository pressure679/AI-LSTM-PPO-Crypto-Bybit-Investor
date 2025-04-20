import time
import datetime
import logging
from pybit.unified_trading import HTTP
from config import API_KEY, API_SECRET

# Set up logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

symbol = "BTCUSDT"
interval = 1  # minute
qty_pct = 0.1  # Risk amount of balance
leverage = 50
order_map = {"grid_orders": []}


def get_klines(symbol, interval, limit=100):
    res = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
    return [float(candle['close']) for candle in res['result']['list'][::-1]]


def get_balance():
    res = session.get_wallet_balance(accountType="UNIFIED")
    return float(res["result"]["list"][0]["totalEquity"])


def get_position():
    res = session.get_positions(category="linear", symbol=symbol)
    pos = res["result"]["list"][0]
    return {
        "size": float(pos["size"]),
        "side": pos["side"],
        "entryPrice": float(pos["entryPrice"])
    }


def cancel_orders(order_ids):
    for order_id in order_ids:
        try:
            session.cancel_order(category="linear", symbol=symbol, orderId=order_id)
        except Exception as e:
            logging.warning(f"Failed to cancel order {order_id}: {e}")


def close_position():
    pos = get_position()
    if pos['size'] > 0:
        side = 'Sell' if pos['side'] == 'Buy' else 'Buy'
        session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=pos['size'],
            reduceOnly=True
        )


def calculate_ema(prices, period):
    ema = prices[0]
    multiplier = 2 / (period + 1)
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema


def place_grid_orders(direction, price, balance, grid_size_pct=0.005, levels=3):
    order_map["grid_orders"] = []
    for i in range(1, levels + 1):
        level_price = price * (1 + grid_size_pct * i) if direction == "Buy" else price * (1 - grid_size_pct * i)
        qty = round(balance * qty_pct / price, 3)
        order = session.place_order(
            category="linear",
            symbol=symbol,
            side=direction,
            orderType="Limit",
            qty=qty,
            price=round(level_price, 2),
            timeInForce="GTC"
        )
        order_map["grid_orders"].append(order["result"]["orderId"])


def main():
    logging.info("Bot started")
    while True:
        try:
            close = get_klines(symbol, interval)
            last_price = close[-1]

            # Detect trend using EMA crossover
            ema7 = calculate_ema(close[-14:], 7)
            ema14 = calculate_ema(close[-14:], 14)
            trend = "Buy" if ema7 > ema14 else "Sell"

            # Volatility-based grid size
            price_range = max(close[-10:]) - min(close[-10:])
            range_pct = price_range / last_price
            grid_size_pct = max(0.002, min(0.01, range_pct))

            position = get_position()
            balance = get_balance()

            logging.info(f"Price: {last_price}, Trend: {trend}, Position: {position['side']} {position['size']}")

            # If we detect a trend and it's not in our favor
            if position['size'] > 0 and position['side'] != trend:
                cancel_orders(order_map["grid_orders"])
                close_position()
                place_grid_orders(trend, last_price, balance, grid_size_pct)

            # If no position and no orders, place grid
            elif position['size'] == 0 and not order_map["grid_orders"]:
                place_grid_orders(trend, last_price, balance, grid_size_pct)

        except Exception as e:
            logging.error(f"Error: {e}")

        time.sleep(60)


if __name__ == "__main__":
    main()
