# 🧠 LSTM-PPO AI Trading Bot

A deep reinforcement learning bot that uses a custom **LSTM + PPO** (Long Short-Term Memory + Proximal Policy Optimization) agent to simulate and learn profitable trading behavior on historical OHLC price data.
.
This project is designed to run efficiently on **low-memory machines** (like older laptops or Chromebooks). It features a **custom PPO agent** with an LSTM network for temporal learning, allowing it to detect profitable patterns over time.

If you want to use this yourself you should edit api_key and api_secret to your bybit api key and secret, further instructions are below.

It trains on 3 months and data and should be updated every once in a while with Kaggle data, I aim to update it every 3 months, the checkpoint files for testing/live deployment are uploaded here unto GitHub, and are available so you don't need to train the LSTM + PPO bot. The checkpoint and win_rate_knn files will be updated every 2-3 months, so you should run install.sh every now and then (command 'sh install.sh', or just copy-paste the content into your Linux terminal, further instructions are below).

---

## 🚀 Features

- ✅ Custom LSTM-PPO implementation
- ✅ Rule-based entry and exit with **MACD signal line** and profit/loss evaluation
- ✅ Uses **pandas**-based technical indicators for state encoding
- ✅ Compatible with multi-symbol training (BTC, ETH, BNB, XRP, XAU)
- ✅ Supports save/load checkpoint functionality
- ✅ Threaded session management for training stability
- ✅ KNN-based experience filtering

---

## 📁 Data Setup

The bot is trained on **1-minute historical OHLCV data aggregated into 15m timeframe**. You can download the data used from the following Kaggle datasets:

| Symbol  | Source |
|---------|--------|
| **XAUUSD (Gold)** | [XAU_1m_data.csv](https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024?utm_source=chatgpt.com&select=XAU_1m_data.csv) |
| **BTCUSD** | [BTC 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data) |
| **BNBUSD** | [BNB 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-bnbusd-1m-data) |
| **ETHUSD** | [ETH 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-ethusd-1m-data) |
| **XRPUSD** | [XRP 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-xrpusd-1m-data) |

If you want to train the bot yourself make sure to place the `.csv` files in the correct folder and edit the `load_last_mb()` and `load_last_mb_xauusd()` file paths in your script to match that path. And set train = True for training in the main function (At the end of the file).

---

## 🧠 Agent Details

This bot is based on a **custom LSTM-based PPO agent**, making it suitable for small machines with limited RAM.

- **Inputs**: 19 technical indicators + OHLC-derived features
- **Memory**: LSTM for short/long-term temporal pattern tracking
- **Actions**: Buy, Sell, Hold, or Close
- **Rewards**:  
  - Profit percentage

> ✨ Future additions may include a GUI for Android and Windows and ChromeBook.

---

## 🛠️ How to Use (If you find it too difficult to install the program you can copytrade me on ByBit, my username is Naamik)

1. **Install install.sh**
   copy and paste content of install.sh into your linux terminal, whether on Window Subsystem for Linux, Linux, or Linux for ChromeBook.

2. **Create an API key and secret (and Bybit account if not already created, and verify yourself with passport and proof of address)**
   API key and secret creation: https://www.bybit.com/future-activity/en/developer

3. **Paste the API key and secret into line 38 and 39 in lstm-ppo-bot.py**

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run training or test**
   ```bash
   cd AI-LSTM-PPO-Crypto-Bybit-Investor; python lstm-ppo-bot.py
   ```
   or if on Android:
   ```
   cd AI-LSTM-PPO-Crypto-Bybit-Investor; python lstm-ppo-bot.py
   ```
8. **Stop program**
   Press Ctrl-C 2-3 times or close the terminal
---

## 🧪 Optional: KNN Reward Filter

A `WinRateKNN` module is included (commented out by default). It filters actions based on expected reward quality using a K-Nearest Neighbor regressor. You can experiment with this for advanced selective behavior based on historical context.

---

## ✅ Example Output
```
[XRPUSD] Day 2025-05-16 - Trades: 2 - Avg Profit: 1868.28, 10.98% - PnL: 1.09% - Balance: 344045.75 - Sharpe: 0.88 - Sortino: 11.27
[ETHUSD] Day 2025-05-27 - Trades: 6 - Avg Profit: 480.47, 2.68% - PnL: 0.80% - Balance: 360953.31 - Sharpe: 0.83 - Sortino: 8.24
[XAUUSD] Day 2025-06-12 - Trades: 0 - Avg Profit: 0.00, 0.00% - PnL: 0.00% - Balance: 375799.80 - Sharpe: 1.11 - Sortino: 36.00
```

```
=== [BNBUSDT] stats at 2025-08-03 20:45:00 ===
price: 750.0
adx: 0
rsi: 3.00
atr: 1.728571
ema7/14/28 crossover above/below: True/False (bullish)
macd zone: 1.00 (bullish)
OSMA zone: 1.00 - (bullish)
+DI_val/-DI_val: 3.00/3.00 (neutral)

=== [BTCUSDT] stats at 2025-08-03 20:45:01 ===
price: 114273.5
adx: 0
rsi: 4.00
atr: 169.671429
ema7/14/28 crossover above/below: True/False (bullish)
macd zone: 1.00 (bullish)
OSMA zone: 1.00 - (bullish)
+DI_val/-DI_val: 4.00/3.00 (bullish)
...
```

---

## ⚙️ Notes

- This bot **does not require GPUs** and is optimized for low-resource environments.
- All indicators and reward logic are implemented in **pure Python (NumPy, pandas)**.
- Training is **thread-safe** and supports multi-symbol backtesting.
- You can skip training by setting train = False in the main function (at the end of lstm-ppo-bot.py, False by default) if you already trained the bot.

---

## 🧑‍💻 Author

This project was built with AI assistance and custom development by [@pressure679](https://github.com/pressure679), with a focus on hands-on experimentation, performance on minimal hardware, and building real-world profitable trading agents.
