import numpy as np
import pandas as pd
import pickle

from pneu_utils.utils import get_pkg_path

import matplotlib.pyplot as plt

model_name = 'BASE_v010'
model_folder_path = f'{get_pkg_path("pneu_rl")}/models/{model_name}'

with open(f'{model_folder_path}/infos.pkl', 'rb') as f:
    info = pickle.load(f)

data = dict(
    epi = [],
    step = [],
    reward = [],
    alpha = []
)

for key, value in info.items():
    data['epi'].append(key)
    data['step'].append(value['steps'])
    data['reward'].append(value['reward'])
    data['alpha'].append(value['alpha'].detach().numpy()[0])

for key, value in data.items():
    data[key] = np.array(value)

df = pd.DataFrame(data)
df.to_csv(f'info_{model_name}.csv', index=False)

plt.figure()
plt.plot(data['epi'], data['alpha'])
plt.show()


