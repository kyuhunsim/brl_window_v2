import os
import shutil

from pneu_rl.sac import SAC

from pneu_ref.random_ref import RandomRef
from pneu_env.env import PneuEnv
from pneu_env.sim import PneuSim
from pneu_env.real import PneuReal
from pneu_env.pred import PneuPred
from pneu_utils.utils import (
    delete_lines, 
    color, 
    get_pkg_path,
    save_yaml,
    load_yaml
)

# get retrain model
print(color('[INPUT] Retrain Model:', 'blue'))
models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
for i, model in enumerate(models):
    print(color(f'\t{i+1}. {model}', 'yellow'))
print(color('\t---', 'blue'))
model_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
model_name = models[model_idx]
delete_lines(len(models) + 3)
print(f'[ INFO] Model name: {model_name}')

# retrain model name
retrain_model_name = input(color('[INPUT] Retrain Model Name: ', 'blue')) 
delete_lines(1)
if retrain_model_name == '':
    print(f'[ INFO] Retrain model name is set automatically.')
else:
    print(f'[ INFO] Retrain model name: {retrain_model_name}')


print(color('[INPUT] Observation Mode:', 'blue'))
print(color('\t1. Sim', 'yellow'))
print(color('\t2. Real', 'yellow'))
print(color('\t---', 'blue'))
obs_mode = input(color('\tOBS: ', 'blue')) 
delete_lines(5)

kwargs = load_yaml(model_name)

# Change Parameters
kwargs['parent_model'] = model_name
# kwargs['retrain_epi'] = 200
kwargs['retrain_epi'] = 50
kwargs['type'] = 'simulation' if obs_mode == '1' else 'real'
# kwargs['pred']['noise'] = False
# kwargs['pred']['noise_std'] = 0
# kwargs['model']['spatial_weight'] = 0
# kwargs['model']['temporal_weight'] = 5


kwargs['rnd_ref']['pos_max_off'] = 220
kwargs['rnd_ref']['pos_min_off'] = 170
kwargs['rnd_ref']['neg_max_off'] = 35
kwargs['rnd_ref']['neg_min_off'] = 15
kwargs['rnd_ref']['pos_max_amp'] = 20 
kwargs['rnd_ref']['neg_max_amp'] = 10

# kwargs['obs']['delay'] = 0.1
# kwargs['obs']['offset_pos'] = 0
# kwargs['obs']['offset_neg'] = 0
kwargs['rnd_ref']['pos_max_ts'] = 10 
kwargs['rnd_ref']['neg_max_ts'] = 10

# 아마도 여기가 주석처리 되어있어서 그런 것으로 추정
# kwargs['rnd_ref']['pos_max_amp'] = 0
# kwargs['rnd_ref']['neg_max_amp'] = 0
# kwargs['alpha'] = dict(
#     alpha = 0.35,
#     # automatic_entropy_tunning = True
#     automatic_entropy_tunning = False
# )
kwargs['temporal_weight_hardening'] = dict(
    initial_weight = 1.25,
    max_weight = 1.25 
)
# kwargs['model']['spatial_weight'] = 0.5
total_steps = 100*kwargs['model']['horizon']
kwargs['temporal_weight_hardening']['rate'] = \
    (kwargs['temporal_weight_hardening']['max_weight'] - kwargs['temporal_weight_hardening']['initial_weight'])/total_steps

if obs_mode == '1':    
    print('[ INFO] Observation Mode: Simulation')
    obs = PneuSim(**kwargs['obs'])
    pred = PneuPred(**kwargs['pred']) if kwargs['pred'] is not None else None
elif obs_mode == '2':
    print('[ INFO] Observation Mode: Real')
    obs = PneuReal(
        freq = kwargs['obs']['freq'],
        scale = kwargs['obs']['scale']
    )
    pred = PneuPred(**kwargs['pred']) if kwargs['pred'] is not None else None

ref = RandomRef(**kwargs['rnd_ref'])
env = PneuEnv(
    obs = obs,
    ref = ref,
    pred = pred,
    **kwargs['env']
)
# if kwargs["pid"] is not None:
#     env.set_pid(**kwargs['pid'])

model = SAC(env, **kwargs['model'])
model.load(name=model_name, train=True)

#여기서 모델 불러왔을 때, alpha값 업데이트 하고 있는지 확인해야함.

retrain_model_name = model.set_retrain(retrain_model_name)
if 'alpha' in kwargs.keys():
    model.set_alpha(**kwargs['alpha'])
if 'temporal_weight_hardening' in kwargs.keys():
    model.set_temporal_weight_hardening(**kwargs['temporal_weight_hardening'])
model.clear_buffer()

save_yaml(retrain_model_name, kwargs, 'retrain_cfg.yaml')

model.train(episode = kwargs['retrain_epi'])
