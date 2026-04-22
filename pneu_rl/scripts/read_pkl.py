import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# 현재 스크립트의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# (1) pkl 파일 불러오기
file_path = os.path.join(current_dir, '..', 'models', '0310', 'infos.pkl')
data = read_pkl(file_path)

print(data[1])

# 각 value['reward'] 추출
rewards = np.array([value['reward'] for value in data.values()])

# 이동평균과 표준편차 계산 (convolution 방식)
window_size = 50
weights = np.ones(window_size) / window_size

# 이동평균 계산 (원본 길이 유지)
smoothed_rewards = np.convolve(rewards, weights, mode='same')

# 표준편차 계산: sqrt(E[x^2] - (E[x])^2)
smoothed_rewards_sq = np.convolve(rewards**2, weights, mode='same')
stds = np.sqrt(smoothed_rewards_sq - smoothed_rewards**2)

# retrain point (예시: 인덱스 950)
retrain_point = 1000

episodes = np.arange(len(rewards))

plt.figure(figsize=(8, 5))
plt.plot(episodes, smoothed_rewards, label='Mean Reward', linewidth=2)
plt.fill_between(episodes,
                 smoothed_rewards - stds,
                 smoothed_rewards + stds,
                 alpha=0.3)
plt.axvline(x=retrain_point, color='red', linestyle='--', label='Retrain Point', linewidth=2)

plt.xlabel('Episode', fontsize=15)
plt.ylabel('Average Reward', fontsize=15)
plt.xlim(0, len(rewards))
plt.legend()
plt.show()

alpha_values = np.array([value['alpha'].item() for value in data.values()])

# alpha 시각화
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, label='Alpha' )
plt.axvline(x=retrain_point, color='red', linestyle='--', label='Retrain Point', linewidth=2)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Alpha', fontsize=15)
plt.legend()
plt.show()
