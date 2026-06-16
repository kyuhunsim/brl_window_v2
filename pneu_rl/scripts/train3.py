from pneu_rl.sac_with_loss import SAC

from pneu_ref.random_ref import RandomRef
from pneu_env.env3 import PneuEnv3
from pneu_env.sim3 import PneuSim
from pneu_env.pred3 import PneuPred
from pneu_utils.utils import checker, save_yaml, color, delete_lines


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

print(color('[INPUT] Episode state carry?', 'blue'))
print(color('\t1. On (carry previous episode pressure state)', 'yellow'))
print(color('\t2. Off (reset to initial pressure state)', 'yellow'))
print(color('\t---', 'blue'))
episode_carry_mode = input(color('\tCarry: ', 'blue'))
delete_lines(5)
episode_carry_state = episode_carry_mode != '2'
print(f'[ INFO] Episode carry state: {"On" if episode_carry_state else "Off"}')

print(color('[INPUT] Chamber control mode?', 'blue'))
print(color('\t1. RL control (6-control policy)', 'yellow'))
print(color('\t2. Steady pressure regulator (4-control policy)', 'yellow'))
print(color('\t---', 'blue'))
chamber_ctrl_mode = input(color('\tCH: ', 'blue'))
delete_lines(5)
if chamber_ctrl_mode == '1':
    fixed_chamber_ctrl = None
    steady_chamber_ctrl = None
    print('[ INFO] Chamber control: RL policy output')
elif chamber_ctrl_mode == '2':
    fixed_chamber_ctrl = None
    steady_chamber_ctrl = dict(
        pos_target=180.0,
        neg_target=25.0,
        kp=0.04,
        ki=0.01,
        deadband=1.0,
        integral_limit=100.0,
        min_open=0.15,
        max_open=0.85,
    )
    print('[ INFO] Chamber control: steady pressure regulator (pos 180.0 kPa, neg 25.0 kPa)')
else:
    raise ValueError(color(f'[ERROR] Unknown chamber control mode: {chamber_ctrl_mode}', 'red'))

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
            pos_min_off = 115,
            pos_max_off = 140,
            neg_min_off = 60,
            neg_max_off = 80,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 10,
            neg_max_amp = 5,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 10,
            num_pred = 15,
            num_act = 5,
            verbose = True,
            episode_carry_state = episode_carry_state,
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
                # 2026-05-28:
                # lib3에서 RL+PID의 tracking은 유지하면서도 제어 진동과 chamber drift를 줄이기 위한 약한 보상 항
                action_delta_rwd_coeff = 0.08,
                conflict_rwd_coeff = 0.15,
                chamber_reserve_rwd_coeff = 0.002,
                chamber_margin_kpa = 15.0,
                chamber_deadband_kpa = 5.0,
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
            noise_std = 1.5,
            train_diag_interval = 1,
            train_diag_atm_band = 8.0
        ),
        epi = 1000,
        # sim3 actuator-pressure PID:
        pid = dict(
            Kp_act_pos_in = 0.0,
            Ki_act_pos_in = 0.002,
            Kd_act_pos_in = 0.0,
            Kp_act_pos_out = 0.0,
            Ki_act_pos_out = 0.002,
            Kd_act_pos_out = 0.0,
            Kp_act_neg_in = 0.0,
            Ki_act_neg_in = 0.002,
            Kd_act_neg_in = 0.0,
            Kp_act_neg_out = 0.0,
            Ki_act_neg_out = 0.002,
            Kd_act_neg_out = 0.0,
            Ka = 1
        )
        # pid = None
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
            pos_min_off = 115,
            pos_max_off = 140,
            neg_min_off = 60,
            neg_max_off = 80,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 10,
            neg_max_amp = 5,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 0,
            num_pred = 1,
            num_act = 1,
            verbose = True,
            episode_carry_state = episode_carry_state,

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
                action_delta_rwd_coeff = 0.0,
                conflict_rwd_coeff = 0.0,
                chamber_reserve_rwd_coeff = 0.0,
                chamber_margin_kpa = 15.0,
                chamber_deadband_kpa = 5.0,
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
            noise_std = 0.0,
            train_diag_interval = 1,
            train_diag_atm_band = 8.0
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
            pos_min_off = 110,
            pos_max_off = 180,
            neg_min_off = 40,
            neg_max_off = 120,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 20,
            neg_max_amp = 20,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 10,
            num_pred = 15,
            num_act = 5,
            verbose = False,
            episode_carry_state = episode_carry_state,

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
                action_delta_rwd_coeff = 0.0,
                conflict_rwd_coeff = 0.0,
                chamber_reserve_rwd_coeff = 0.0,
                chamber_margin_kpa = 15.0,
                chamber_deadband_kpa = 5.0,
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
            noise_std = 0,
            train_diag_interval = 1,
            train_diag_atm_band = 8.0
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
            pos_min_off = 110,
            pos_max_off = 180,
            neg_min_off = 40,
            neg_max_off = 120,
            pos_max_ts = 5,
            neg_max_ts = 5,
            pos_max_amp = 20,
            neg_max_amp = 20,
            pos_max_per = 10,
            neg_max_per = 10
        ),
        env = dict(
            num_prev = 0,
            num_pred = 1,
            num_act = 1,
            verbose = False,
            episode_carry_state = episode_carry_state,

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
                action_delta_rwd_coeff = 0.0,
                conflict_rwd_coeff = 0.0,
                chamber_reserve_rwd_coeff = 0.0,
                chamber_margin_kpa = 15.0,
                chamber_deadband_kpa = 5.0,
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
            noise_std = 1.0,
            train_diag_interval = 1,
            train_diag_atm_band = 8.0
        ),
        epi = 200,
        pid = None
    )
    print(f'[ INFO] Train mode: SAC + CAPS')

else:
    raise ValueError(color(f'[ERROR] Unknown train mode: {train_mode}', 'red'))

kwargs["env"]["fixed_chamber_ctrl"] = fixed_chamber_ctrl
kwargs["env"]["steady_chamber_ctrl"] = steady_chamber_ctrl

obs = PneuSim(**kwargs['obs'])
print(f"[ DBG] Loaded obs simulator library: {obs.lib._name}")

pred = PneuPred(**kwargs['pred']) if kwargs['pred'] is not None else None
if pred is not None:
    print(f"[ DBG] Loaded pred simulator library: {pred.lib._name}")

ref = RandomRef(**kwargs['rnd_ref'])
env = PneuEnv3(
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
