# 🧠 LSTM-PPO AI Trading Bot

A deep reinforcement learning bot that uses a custom **LSTM + PPO** (Proximal Policy Optimization) agent to simulate and learn profitable trading behavior on historical OHLC price data.

This project is designed to run efficiently on **low-memory machines** (like older laptops or Chromebooks). It features a **custom PPO agent** with an LSTM network for temporal learning, allowing it to detect profitable patterns over time.

If you want to use this yourself you should edit api_key and api_secret to your bybit api key and secret.

It trains on 3 months and data and should be updated every once in a while with Kaggle data, I aim to update it every 3 months.

---

## 🚀 Features

- ✅ Custom LSTM-PPO implementation
- ✅ Reward shaping with **MACD signal line** and profit/loss evaluation
- ✅ Uses **pandas**-based technical indicators for state encoding
- ✅ Compatible with multi-symbol training (BTC, ETH, BNB, XRP, XAU)
- ✅ Supports save/load checkpoint functionality
- ✅ Threaded session management for training stability
- 🧪 *(Optional)* KNN-based experience filtering (commented out for now)

---

## 📁 Data Setup

The bot is trained on **1-minute historical OHLCV data**. You can download the data used from the following Kaggle datasets:

| Symbol  | Source |
|---------|--------|
| **XAUUSD (Gold)** | [XAU_1m_data.csv](https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024?utm_source=chatgpt.com&select=XAU_1m_data.csv) |
| **BTCUSD** | [BTC 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data) |
| **BNBUSD** | [BNB 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-bnbusd-1m-data) |
| **ETHUSD** | [ETH 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-ethusd-1m-data) |
| **XRPUSD** | [XRP 1m data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-xrpusd-1m-data) |

Make sure to place the `.csv` files in the correct folder and edit the `load_last_mb()` function in your script to match that path.

---

## 🧠 Agent Details

This bot is based on a **custom LSTM-based PPO agent**, making it suitable for small machines with limited RAM.

- **Inputs**: 15 technical indicators + OHLC-derived features
- **Memory**: LSTM for short/long-term temporal pattern tracking
- **Actions**: Buy, Sell, Hold, or Close
- **Rewards**:  
  - Profit percentage

> ✨ Future additions may include DI+/DI-, Bulls/Bears Power, and KNN-based action filtering and GUI for Android.

---

## 🛠️ How to Use

1. **Install install.sh**
   copy and paste content of install.sh into your linux terminal, whether on Window Subsystem for Linux, Linux, or Linux for ChromeBook.

2. **Create an API key and secret (and Bybit account if not already created, and verify yourself with passport and proof of address)**
   API key and secret creation: https://www.bybit.com/future-activity/en/developer

3. **Paste the API key and secret into line 38 and 39 in lstm-ppo-bot.py**

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Edit file paths**  
   Open the script and modify the following functions to point to your desired directories:
   - `load_last_mb()` — Where to load the kaggle csv files
   - `load_last_mb_xauusd` — Where to load the kaggle XAUUSD csv file (rename the file from XAU_1m_Binance.csv to XAUUSD_1m_Binance.csv)
   - `save_checkpoint()` — where to save the model checkpoints
   - `load_checkpoint()` — where to load previous models from
   - `load_last_mb()` — path to your downloaded OHLCV `.csv` files

7. **Run training or test**
   ```bash
   python lstm-ppo.py
   ```

---

## 🧪 Optional: KNN Reward Filter

A `WinRateKNN` module is included (commented out by default). It filters actions based on expected reward quality using a K-Nearest Neighbor regressor. You can experiment with this for advanced selective behavior based on historical context.

---

## ✅ Example Output

```
[BNBUSD] Day 2025-04-30 - Trades: 4 - Avg Profit: 30.71, 9.69% - PnL: 1.90% - Balance: 6460.76 - Sharpe: 0.55 - Sortino: 20.12
[XAUUSD] Day 2025-05-20 - Trades: 0 - Avg Profit: 0.00, 0.00% - PnL: 0.00% - Balance: 6460.76 - Sharpe: 0.74 - Sortino: 91.68
...
```

---

## ⚙️ Notes

- This bot **does not require GPUs** and is optimized for low-resource environments.
- All indicators and reward logic are implemented in **pure Python (NumPy, pandas)**.
- Training is **thread-safe** and supports multi-symbol backtesting.
- In training it went from $1000 to $500 the first week, and after training after a week it started getting profitable and went to $2500 in 3 months.
- You can skip training by setting train = False in the main function if you already trained the bot.
- I am currently testing it in ByBit demo mode, so if you want to disable that you comment out demo=True in the session call just above the main function.

---

## 🧑‍💻 Author

This project was built with AI assistance and custom development by [@pressure679](https://github.com/pressure679), with a focus on hands-on experimentation, performance on minimal hardware, and building real-world profitable trading agents.
