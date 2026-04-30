import inspect
import sys
from pathlib import Path

from pneu_rl.sac_with_loss import SAC

from pneu_ref.random_ref import RandomRef
from pneu_env.env import PneuEnv
from pneu_env.sim2 import PneuSim
from pneu_env.real.real import PneuReal
from pneu_env.pred2 import PneuPred
from pneu_utils.utils import checker, save_yaml, load_yaml, color, delete_lines


model_name = input('\033[94m' + '[INPUT] pneu_rl <==  model name: ' + '\033[0m')
checker(model_name, 'pneu_rl V> model_name: ')

print(color('[INPUT] Train mode?', 'blue'))
print(color('\t1. Ours', 'yellow'))
print(color('\t2. SAC', 'yellow'))
print(color('\t3. SAC + MPObs', 'yellow'))
print(color('\t4. SAC + CAPS', 'yellow'))
print(color('\t---', 'blue'))
train_mode = input(color('\tTM: ', 'blue')) 
delete_lines(7)

# ==> My model <==
if train_mode == '1':
    kwargs = dict(
        obs = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            scale = True
        ),
        pred = dict(
            freq = 50,
            delay = 0,
            noise = False,
            noise_std = 0,
            scale = True,
        ),
        # pred = None,
        rnd_ref = dict(
            pos_min_off = 145,
            pos_max_off = 210,
            neg_min_off = 15,
            neg_max_off = 35,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 20,
            neg_max_amp = 5,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 10,
            num_pred = 15,
            num_act = 5,
            rwd_kwargs = dict(
                pos_prev_rwd_coeff = 0.0,
                neg_prev_rwd_coeff = 0.0,
                pos_curr_rwd_coeff = 0.3*1,
                neg_curr_rwd_coeff = 0.3*1,
                pos_fut_rwd_coeff = 0.01*1,
                neg_fut_rwd_coeff = 0.01*1,
                pos_pred_rwd_coeff = 0.25*1,
                neg_pred_rwd_coeff = 0.25*1,
                pos_diff_rwd_coeff = 0.0,
                neg_diff_rwd_coeff = 0.0,
            ),
            pos_pred_rnd_offset_range = 0, 
            neg_pred_rnd_offset_range = 0, 
        ),
        model = dict(
            learning_rate = 3e-4,
            gamma = 0.9,
            tau = 0.005,
            alpha = 0.5,
            automatic_entropy_tunning = True,
            hidden_dim = 256,
            buffer_size = 50e4,
            batch_size = 128,
            epoch = 1,
            horizon = 512,
            start_epi = 10,
            max_grad_norm = 0.5,
            log_std_min = -10,
            log_std_max = 1,
            temporal_weight = 1,
            spatial_weight = 0.4,
            noise_std = 1.5
        ),
        epi = 1500,
        pid_cfg = dict(
            Kp_pos = 0.0,
            Ki_pos = 0.01,
            Kd_pos = 0.0,
            Kp_neg = 0.0,
            Ki_neg = 0.01,
            Kd_neg = 0.0,
            Ka = 1
        ),
        pid = None
    )
    print(f'[ INFO] Train mode: Ours')

# ==> Comparison <==
elif train_mode == '2':
    kwargs = dict(
        obs = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            scale = True
        ),
        pred = None,
        rnd_ref = dict(
            pos_min_off = 145,
            pos_max_off = 240,
            neg_min_off = 15,
            neg_max_off = 35,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 20,
            neg_max_amp = 5,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 0,
            num_pred = 1,
            num_act = 1,
            
            rwd_kwargs = dict(
                pos_curr_rwd_coeff = 0.3*1,
                neg_curr_rwd_coeff = 0.3*1,
                pos_fut_rwd_coeff = 0.0,
                neg_fut_rwd_coeff = 0.0,
                pos_pred_rwd_coeff = 0.0,
                neg_pred_rwd_coeff = 0.0,
                
                pos_prev_rwd_coeff = 0.0,
                neg_prev_rwd_coeff = 0.0,
                pos_diff_rwd_coeff = 0.0,
                neg_diff_rwd_coeff = 0.0,
            ),
            pos_pred_rnd_offset_range = 0, 
            neg_pred_rnd_offset_range = 0, 
        ),
        model = dict(
            learning_rate = 3e-4,
            gamma = 0.9,
            tau = 0.005,
            alpha = 0.5,
            automatic_entropy_tunning = True,
            hidden_dim = 256,
            buffer_size = 50e4,
            batch_size = 128,
            epoch = 1,
            horizon = 512,
            start_epi = 10,
            max_grad_norm = 0.5,
            log_std_min = -10,
            log_std_max = 1,

            temporal_weight = 0.0,
            spatial_weight = 0.0,
            noise_std = 0.0
        ),
        epi = 1500,
        pid = None
    )
    print(f'[ INFO] Train mode: SAC')

elif train_mode == '3':
    kwargs = dict(
        obs = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            scale = True
        ),
        pred = dict(
            freq = 50,
            delay = 0,
            noise = False,
            noise_std = 0,
            scale = True,
        ),
        rnd_ref = dict(
            pos_min_off = 145,
            pos_max_off = 240,
            neg_min_off = 15,
            neg_max_off = 35,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 20,
            neg_max_amp = 5,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 10,
            num_pred = 15,
            num_act = 5,
            
            rwd_kwargs = dict(
                pos_curr_rwd_coeff = 0.3*1,
                neg_curr_rwd_coeff = 0.3*1,
                pos_fut_rwd_coeff = 0.01*1,
                neg_fut_rwd_coeff = 0.01*1,
                pos_pred_rwd_coeff = 0.25*1,
                neg_pred_rwd_coeff = 0.25*1,
                
                pos_prev_rwd_coeff = 0.0,
                neg_prev_rwd_coeff = 0.0,
                pos_diff_rwd_coeff = 0.0,
                neg_diff_rwd_coeff = 0.0,
            ),
            pos_pred_rnd_offset_range = 0, 
            neg_pred_rnd_offset_range = 0, 
        ),
        model = dict(
            learning_rate = 3e-4,
            gamma = 0.9,
            tau = 0.005,
            alpha = 0.5,
            automatic_entropy_tunning = True,
            hidden_dim = 256,
            buffer_size = 50e4,
            batch_size = 128,
            epoch = 1,
            horizon = 512,
            start_epi = 10,
            max_grad_norm = 0.5,
            log_std_min = -10,
            log_std_max = 1,

            temporal_weight = 0,
            spatial_weight = 0,
            noise_std = 0
        ),
        epi = 1200,
        pid = None
    )
    print(f'[ INFO] Train mode: SAC + MPObs')

elif train_mode == '4':
    kwargs = dict(
        obs = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            scale = True
        ),
        pred = None,
        rnd_ref = dict(
            pos_min_off = 145,
            pos_max_off = 240,
            neg_min_off = 15,
            neg_max_off = 35,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 20,
            neg_max_amp = 5,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 0,
            num_pred = 1,
            num_act = 1,
            
            rwd_kwargs = dict(
                pos_curr_rwd_coeff = 0.3*1,
                neg_curr_rwd_coeff = 0.3*1,
                pos_fut_rwd_coeff = 0.0,
                neg_fut_rwd_coeff = 0.0,
                pos_pred_rwd_coeff = 0.0,
                neg_pred_rwd_coeff = 0.0,
                
                pos_prev_rwd_coeff = 0.0,
                neg_prev_rwd_coeff = 0.0,
                pos_diff_rwd_coeff = 0.0,
                neg_diff_rwd_coeff = 0.0,
            ),
            pos_pred_rnd_offset_range = 0, 
            neg_pred_rnd_offset_range = 0, 
        ),
        model = dict(
            learning_rate = 3e-4,
            gamma = 0.9,
            tau = 0.005,
            alpha = 0.5,
            automatic_entropy_tunning = True,
            hidden_dim = 256,
            buffer_size = 50e4,
            batch_size = 128,
            epoch = 1,
            horizon = 512,
            start_epi = 10,
            max_grad_norm = 0.5,
            log_std_min = -10,
            log_std_max = 1,

            temporal_weight = 0.5,
            spatial_weight = 0.4,
            noise_std = 1.0
        ),
        epi = 200,
        pid = None
    )
    print(f'[ INFO] Train mode: SAC + CAPS')


obs = PneuSim(**kwargs['obs'])
print(f"[ DBG] Loaded obs simulator library: {obs.lib._name}")

pred = PneuPred(**kwargs['pred']) if kwargs['pred'] is not None else None
if pred is not None:
    print(f"[ DBG] Loaded pred simulator library: {pred.lib._name}")

ref = RandomRef(**kwargs['rnd_ref'])
env = PneuEnv(
    obs = obs,
    pred = pred,
    ref = ref,
    **kwargs['env']
)
if kwargs["pid"] is not None:
    env.set_pid(**kwargs['pid'])

model = SAC(
    env = env, 
    **kwargs['model']
)
model.set_logger(model_name)

save_yaml(model_name, kwargs)

model.train(episode = kwargs['epi'])
