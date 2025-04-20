
# Bybit Hybrid Scalping Bot

This is a hybrid scalping bot for Bybit, written in Python using the Pybit V5 API. It combines two trading strategies in one:

- **Trend Trading:** Uses moving average crossovers (MA7/MA14) to detect bullish/bearish trends and executes market orders accordingly.
- **Grid Trading:** When no strong trend is detected, the bot switches to grid trading within a price range. It places layered limit buy/sell orders based on volatility.

## Features

- **Mainnet Support:** Fully functional on Bybit mainnet.
- **MA Crossover Trend Strategy:** Trades long or short when a fast MA (7) crosses above or below a slower MA (14).
- **Grid Trading for Sideways Markets:** Dynamically places grid orders during low-volatility periods.
- **Volatility-Adaptive Grid Spacing:** Grid size percentage adapts based on recent 10-candle price range.
- **Automatic Trend/Consolidation Detection:** Uses a 10-candle price range check to switch strategies.
- **Dynamic Order Size:** Calculates position size based on account balance, risk amount, and entry price.
- **Grid Order Tracking:** Tracks and manages grid orders separately to prevent accidental cancellation.
- **Clean Logging:** Logs actions such as strategy mode, order placement, trend detection, and balance.

## Strategy Logic

**Trend Mode:**
- Triggered when price range over last 10 candles > 1% of current price.
- Checks MA7 and MA14 crossover to decide long (buy) or short (sell).
- Cancels grid orders before placing market orders.

**Grid Mode:**
- Triggered when price range over last 10 candles ≤ 1%.
- Places staggered buy/sell limit orders above and below current price.
- Grid size is based on current volatility range.
- Cancels previous trend orders to avoid overlap.

## Requirements

- Python 3.7+
- `pybit`
- `pandas`

Install dependencies:

```bash
pip install pybit pandas
```

## Usage

1. Clone this repo.
2. Add your Bybit mainnet API keys in the script.
3. Set your risk amount (`risk_amount = 0.01` for 1% of your balance).
4. Run:

```bash
python scalping_bot.py
```

## Notes

- This bot is aggressive and optimized for short timeframes (like 1m).
- Works best in high volatility or choppy market conditions.
- Does **not** use martingale.
- Not financial advice — use at your own risk.
