cd ~/
sudo apt install wget git python3
python3 -m pip install pybit scikit-learn pandas numpy
git clone https://github.com/pressure679/AI-LSTM-PPO-Crypto-Bybit-Investor
cd AI-LSTM-PPO-Crypto-Bybit-Investor
mkdir LSTM-PPO-saves
mv 2025-08-04-XAUUSD.win_rate_knn.pkl LSTM-PPO-saves
mv 2025-08-04-XAUUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv 2025-08-04-XRPUSD.win_rate_knn.pkl LSTM-PPO-saves
mv 2025-08-04-XRPUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv 2025-08-04-BNBUSD.win_rate_knn.pkl LSTM-PPO-saves
mv 2025-08-04-BNBUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv 2025-08-04-ETHUSD.win_rate_knn.pkl LSTM-PPO-saves
mv 2025-08-04-ETHUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv 2025-08-04-BTCUSD.win_rate_knn.pkl LSTM-PPO-saves
mv 2025-08-04-BTCUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
