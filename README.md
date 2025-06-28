# Python Investor

A Python-based crypto futures trading bot designed for Bybit that supports multiple trading strategies such as MACD and ADX trend detection. The bot can trade on Bybit’s testnet or mainnet, with features for automated position management, logging, and balance-aware trade sizing.

## Features

- **Multiple strategies:** Currently supports MACD and ADX indicators for trade signals.
- **Bybit API integration:** Place market orders, get wallet balances, and track positions via the official Bybit V5 API.
- **Testnet & mainnet support:** Easily switch between Bybit testnet and mainnet environments.
- **Automatic trade sizing:** Calculates trade quantity based on available USDT balance, using 5% or minimum $5 per trade.
- **Trade management:** Avoids multiple simultaneous positions; switches orders on signal changes.
- **Basic logging:** Prints trade entries, warnings, and errors to console with clear messages.
- **Configurable symbol and parameters:** Customize trading symbol, API keys, and strategy settings in the main script.

## Requirements

- Python 3.8+
- `pybit` library (Bybit API wrapper)
- Other dependencies as per `requirements.txt` (if applicable)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pressure679/Python-investor.git
cd Python-investor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

*If you don’t have a `requirements.txt`, install pybit manually:*

```bash
pip install pybit
```

## Setup

1. **Get Bybit API keys:**

- For testing, create API keys on [Bybit Testnet](https://testnet.bybit.com).
- For live trading, create API keys on [Bybit Mainnet](https://bybit.com).

2. **Configure API keys and settings:**

Edit the main script (e.g., `crypto-bot.py`) and set your API key, secret, and choose the testnet or mainnet endpoint.

Example snippet:

```python
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
is_testnet = True  # Set to False for mainnet

endpoint = "https://api-testnet.bybit.com" if is_testnet else "https://api.bybit.com"
session = HTTP(endpoint, api_key=api_key, api_secret=api_secret)
```

3. **Customize trading parameters:**

- Symbol (e.g., `"XRPUSDT"`)
- Trade size settings
- Strategy selection

## Usage

Run the bot from the command line:

```bash
python crypto-bot.py
```

The bot will:

- Monitor the market using your selected strategy
- Enter trades based on signals
- Manage active positions and update orders when signals change
- Print trade activity and profit information to the console

## Notes

- This bot is for educational and testing purposes only.
- Trading crypto futures carries significant risk.
- Use testnet API keys and funds to familiarize yourself before trading live.
- Adjust strategy parameters and risk management carefully.

## License

MIT License — see `LICENSE` file for details.
