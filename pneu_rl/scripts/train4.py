from pneu_rl.sac_with_loss import SAC

from pneu_ref.random_ref import RandomRef
from pneu_ref.disp_ref import PressureDiffDisplacementRef, PressureDisplacementRef, RandomDispRef
from pneu_env.env4 import PneuEnv4
from pneu_env.sim4 import PneuSim
from pneu_env.pred4 import PneuPred
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

DEFAULT_DISP_REF = dict(
    max_ts=5,
    max_amp=3.0,
    max_per=10.0,
    min_off=10.0,
    max_off=18.0,
    seed=61099,
)

DEFAULT_PRESSURE_DIFF_RND_REF = dict(
    pos_min_off=112,
    pos_max_off=126,
    neg_min_off=84,
    neg_max_off=88,
    pos_max_ts=5,
    neg_max_ts=5,
    pos_max_amp=4,
    neg_max_amp=2,
    pos_max_per=10,
    neg_max_per=10,
)

DEFAULT_PRESSURE_DIFF_DISP_REF = dict(
    dp_to_disp_slope=0.6011927721229401,
    dp_to_disp_intercept=-5.325254604545463,
    min_disp=7.0,
    max_disp=21.0,
)

DISP_REWARD_DEFAULTS = dict(
    disp_prev_rwd_coeff=0.0,
    disp_curr_rwd_coeff=0.2,
    disp_fut_rwd_coeff=0.0,
    disp_pred_rwd_coeff=0.0,
    disp_vel_rwd_coeff=0.001,
)

DISP_REWARD_WITH_PRED = dict(
    disp_prev_rwd_coeff=0.0,
    disp_curr_rwd_coeff=0.2,
    disp_fut_rwd_coeff=0.005,
    disp_pred_rwd_coeff=0.08,
    disp_vel_rwd_coeff=0.001,
)

TUNED_L_INIT_TOTAL_M = 0.03192691119048332
TUNED_INIT_CELL_LENGTH_M = TUNED_L_INIT_TOTAL_M / 2.0
TUNED_INIT_POS_PRESS_KPA = 135.98092651367188
TUNED_INIT_NEG_PRESS_KPA = 42.16064834594727
TUNED_INIT_ACT_POS_PRESS_KPA = 110.0009307861328
TUNED_INIT_ACT_NEG_PRESS_KPA = 87.99779510498047
SCALE_LOW = 0.8
SCALE_HIGH = 1.0

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
            # pos_min_off = 115,
            pos_min_off=108,
            pos_max_off = 140,
            # neg_min_off = 60,
            # neg_max_off = 80,
            neg_min_off=70,
            neg_max_off=90,
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
                # lib4에서 RL+PID의 tracking은 유지하면서도 제어 진동과 chamber drift를 줄이기 위한 약한 보상 항
                action_delta_rwd_coeff = 0.08,
                # conflict_rwd_coeff = 0.15,
                conflict_rwd_coeff = 1.5,
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
            # horizon = 512,
            horizon = 2048,
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
        # sim4 actuator-pressure PID:
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

kwargs["disp_ref"] = DEFAULT_DISP_REF.copy()
kwargs["ref_mode"] = "pressure_diff_displacement"
kwargs["rnd_ref"].update(DEFAULT_PRESSURE_DIFF_RND_REF)
kwargs["pressure_diff_disp_ref"] = DEFAULT_PRESSURE_DIFF_DISP_REF.copy()
disp_reward = DISP_REWARD_WITH_PRED if kwargs["pred"] is not None else DISP_REWARD_DEFAULTS
kwargs["obs"].setdefault("init_length", TUNED_INIT_CELL_LENGTH_M)
kwargs["obs"].setdefault("init_pos_press", TUNED_INIT_POS_PRESS_KPA)
kwargs["obs"].setdefault("init_neg_press", TUNED_INIT_NEG_PRESS_KPA)
kwargs["obs"].setdefault("init_act_pos_press", TUNED_INIT_ACT_POS_PRESS_KPA)
kwargs["obs"].setdefault("init_act_neg_press", TUNED_INIT_ACT_NEG_PRESS_KPA)
kwargs["obs"].setdefault("scale_low", SCALE_LOW)
kwargs["obs"].setdefault("scale_high", SCALE_HIGH)
if kwargs["pred"] is not None:
    kwargs["pred"].setdefault("init_length", TUNED_INIT_CELL_LENGTH_M)
    kwargs["pred"].setdefault("init_pos_press", TUNED_INIT_POS_PRESS_KPA)
    kwargs["pred"].setdefault("init_neg_press", TUNED_INIT_NEG_PRESS_KPA)
    kwargs["pred"].setdefault("init_act_pos_press", TUNED_INIT_ACT_POS_PRESS_KPA)
    kwargs["pred"].setdefault("init_act_neg_press", TUNED_INIT_ACT_NEG_PRESS_KPA)
    kwargs["pred"].setdefault("scale_low", SCALE_LOW)
    kwargs["pred"].setdefault("scale_high", SCALE_HIGH)
kwargs["env"]["rwd_kwargs"].update(disp_reward)
kwargs["env"].setdefault("displacement_obs_key", "sen_total_contraction_from_rest")
kwargs["env"].setdefault("displacement_scale", 1000.0)
kwargs["env"].setdefault("velocity_scale", 1000.0)
kwargs["env"]["fixed_chamber_ctrl"] = fixed_chamber_ctrl
kwargs["env"]["steady_chamber_ctrl"] = steady_chamber_ctrl

obs = PneuSim(**kwargs['obs'])
print(f"[ DBG] Loaded obs simulator library: {obs.lib._name}")

pred = PneuPred(**kwargs['pred']) if kwargs['pred'] is not None else None
if pred is not None:
    print(f"[ DBG] Loaded pred simulator library: {pred.lib._name}")

pressure_ref = RandomRef(**kwargs['rnd_ref'])
if kwargs.get("ref_mode") == "pressure_diff_displacement":
    ref = PressureDiffDisplacementRef(
        pressure_ref,
        **kwargs["pressure_diff_disp_ref"],
    )
else:
    displacement_ref = RandomDispRef(**kwargs["disp_ref"])
    ref = PressureDisplacementRef(pressure_ref, displacement_ref)
env = PneuEnv4(
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
