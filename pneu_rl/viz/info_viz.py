import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import deque
import pickle

from pneu_utils.utils import checker

model_name = input('\033[94m' + '[INPUT] pneu_rl <==  model name: ' + '\033[0m')
checker(model_name, 'pneu_rl V> model_name: ')
infos_file = f'../models/{model_name}/infos.pkl'
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
    # data_deque['alphas'].append(v['alpha'].item())

data = dict(
    epis = np.array(data_deque['epis']),
    steps = np.array(data_deque['steps']),
    rewards = np.array(data_deque['rewards']),
    # alphas = np.array(data_deque['alphas'])
)


fontname = 'Times New Roman'
fontsize = 18
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax1.plot(
    data['steps'],
    data['rewards'],
    linewidth=2, color='black'
)
ax1.set_xlabel('Steps', fontname=fontname, fontsize=fontsize)
ax1.set_title('Total Reward', fontname=fontname, fontsize=fontsize)
# ax1.grid(which='major', color='silver', linewidth=1)
# ax1.grid(which='minor', color='lightgray', linewidth=0.5)
ax1.legend(loc='upper right')
ax1.minorticks_on()

ax2.plot(
    data['steps'],
    data['rewards'],
    linewidth=2, color='black'
)
ax2.set_xlabel('Steps', fontname=fontname, fontsize=fontsize)
ax2.set_title('Temperature Parameter', fontname=fontname, fontsize=fontsize)
# ax2.grid(which='major', color='silver', linewidth=1)
# ax2.grid(which='minor', color='lightgray', linewidth=0.5)
ax2.legend(loc='upper right')
ax2.minorticks_on()
ax2.set(xlim=(None, None), ylim=(0, 1.7))

plt.tight_layout()
plt.savefig(f'fig.png')
plt.show()


