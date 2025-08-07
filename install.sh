cd ~/
sudo apt install wget git python3
python3 -m pip install pybit scikit-learn pandas numpy
git clone https://github.com/pressure679/AI-LSTM-PPO-Crypto-Bybit-Investor
cd AI-LSTM-PPO-Crypto-Bybit-Investor
mkdir LSTM-PPO-saves
mv *XAUUSD.win_rate_knn.pkl LSTM-PPO-saves
mv *XAUUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv *XRPUSD.win_rate_knn.pkl LSTM-PPO-saves
mv *XRPUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv *BNBUSD.win_rate_knn.pkl LSTM-PPO-saves
mv *BNBUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv *ETHUSD.win_rate_knn.pkl LSTM-PPO-saves
mv *ETHUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
mv *BTCUSD.win_rate_knn.pkl LSTM-PPO-saves
mv *BTCUSD.checkpoint.lstm-ppo.pkl LSTM-PPO-saves
