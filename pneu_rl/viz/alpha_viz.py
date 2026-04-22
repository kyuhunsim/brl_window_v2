import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import deque
import pickle

from pneu_utils.utils import checker

# model_name = input('\033[94m' + '[INPUT] pneu_rl <==  model name: ' + '\033[0m')
# model_name = 'ref15obs15step1_rndsin_hid32buf25E4_alpha'
# checker(model_name, 'pneu_rl V> model_name: ')
base_model_name = 'alpha_test/ref15obs15hid32gamma090rndsin_alpha'
datas = dict()
for i in range(5):
    infos_file = f'../models/{base_model_name}{i}/infos.pkl'

    with open(infos_file, 'rb') as f:
        data_info = pickle.load(f)

    data_deque = dict(
        epis = deque(),
        steps = deque(),
        rewards = deque(),
        alphas = deque()
    )

    for k, v in data_info.items():
        data_deque['epis'].append(k)
        data_deque['steps'].append(v['steps'])
        data_deque['rewards'].append(v['reward'])
        data_deque['alphas'].append(v['alpha'].item())

    data = dict(
        epis = np.array(data_deque['epis']),
        steps = np.array(data_deque['steps']),
        rewards = np.array(data_deque['rewards']),
        alphas = np.array(data_deque['alphas'])
    )
    datas[i] = data

steps = datas[0]['steps']
stacked_rewards = np.vstack((
    datas[0]['rewards'],
    datas[1]['rewards'],
    datas[2]['rewards'],
    datas[3]['rewards'],
    datas[4]['rewards'],
))
mean_rewards = np.mean(stacked_rewards, axis=0)
stacked_alphas = np.vstack((
    datas[0]['alphas'],
    datas[1]['alphas'],
    datas[2]['alphas'],
    datas[3]['alphas'],
    datas[4]['alphas'],
))
mean_alphas = np.mean(stacked_alphas, axis=0)


fontname = 'Times New Roman'
fontsize = 18
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

for i in range(4):
    ax1.plot(
        datas[i]['steps'],
        datas[i]['rewards'],
        linewidth=2, color='silver'
    )
ax1.plot(
    steps,
    mean_rewards,
    linewidth=2, color='black', label='Average'
)
ax1.set_xlabel('Steps', fontname=fontname, fontsize=fontsize)
ax1.set_title('Total Reward', fontname=fontname, fontsize=fontsize)
# ax1.grid(which='major', color='silver', linewidth=1)
# ax1.grid(which='minor', color='lightgray', linewidth=0.5)
ax1.legend(loc='upper right')
ax1.minorticks_on()

for i in range(4):
    ax2.plot(
        datas[i]['steps'],
        datas[i]['alphas'],
        linewidth=2, color='silver'
    )
ax2.plot(
    steps,
    mean_alphas,
    linewidth=2, color='black', label='Average'
)
ax2.set_xlabel('Steps', fontname=fontname, fontsize=fontsize)
ax2.set_title('Temperature Parameter', fontname=fontname, fontsize=fontsize)
# ax2.grid(which='major', color='silver', linewidth=1)
# ax2.grid(which='minor', color='lightgray', linewidth=0.5)
ax2.legend(loc='upper right')
ax2.minorticks_on()
ax2.set(xlim=(None, None), ylim=(None, None))

plt.tight_layout()
plt.savefig(f'alpha.png')
plt.show()


