import copy
import os

from collections import deque
import numpy as np
import pandas as pd
import time
from datetime import datetime
import pickle
import yaml
import threading

from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_ref.random_ref import RandomRef
from pneu_ref.step_ref import StepCasesRef, StepRef
from pneu_ref.sine_ref import SineRef, DynamicOscillatorRef, CenterStepOscillationRef
from pneu_ref.traj_ref import TrajRef
from pneu_env.env3 import PneuEnv3
from pneu_env.env3_all_open import PneuEnv3AllOpen
from pneu_env.sim3 import PneuSim
from pneu_env.pred3 import PneuPred
from pneu_env.real.real9 import PneuReal
from pneu_rl.sac_with_loss import SAC
from pneu_rl.exp_paths import ensure_model_exp_dir
from pneu_rl.exp_paths import find_experiment_file
from pneu_rl.plotting import plot_lib3_pressure_and_controls
from pneu_utils.utils import (
    delete_lines, 
    color, 
    get_pkg_path,
    load_yaml
)
from quick_viz_utils import (
    compute_tracking_metrics,
    datas_to_df,
    print_tracking_metrics,
)

ATM = 101.325

viz_kwargs = dict(
    env = dict(
        sim = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            noise_std = 0,
            offset_pos = 0,
            offset_neg = 0,
            offset_act_pos = 0,
            offset_act_neg = 0,
            scale = True
        ),
        real = dict(
            freq = 50,
            scale = True
        ),
        pred = dict(
            freq = 50,
            delay = 0,
            noise = False,
            noise_std = 0,
            offset_pos = 0,
            offset_neg = 0,
            offset_act_pos = 0,
            offset_act_neg = 0,
            scale = True,
        ),
        init_press = dict(
            init_pos_press = ATM,
            init_neg_press = ATM,
            init_act_pos_press = ATM,
            init_act_neg_press = ATM
        ),
        pid = dict(
            Kp_act_pos_in = 0.001,
            Ki_act_pos_in = 0.0,
            Kd_act_pos_in = 0.0,
            Kp_act_pos_out = 0.001,
            Ki_act_pos_out = 0.0,
            Kd_act_pos_out = 0.0,
            Kp_act_neg_in = 0.001,
            Ki_act_neg_in = 0.001,
            Kd_act_neg_in = 0.001,
            Kp_act_neg_out = 0.001,
            Ki_act_neg_out = 0.001,
            Kd_act_neg_out = 0.001,
            Ka = 1
        )
    ),
    ref = dict(
        stepcases = dict(
            time_step = 5,
            ref_pos_max = 140,
            ref_pos_min = 115,
            ref_neg_max = 80,
            ref_neg_min = 60
        ),
        traj = dict(
            # file = "Pos_Neg_MPC_w_SH_v12_24_07_02",
            # file = "Pos_Neg_MPC_w_SH_v12_24_07_02",
            # file = "241104_14_53_47_A000_v04_Real"
            # file = "241104_14_55_26_A000_v04_Real"
            # file = "241105_11_20_54_B000_v01_Real"
            # file = "241112_00_14_47_PID_Real"
            file = "241113_16_09_05_PID_Real"
        ),
        random = dict(
            # pos_max_off = 140,
            pos_max_off = 125,
            pos_min_off = 115,
            neg_max_off = 80,
            neg_min_off = 65,
            # neg_min_off = 60,
            pos_max_ts = 10,
            neg_max_ts = 10,
            pos_max_amp = 10,
            neg_max_amp = 5,
            seed = 61098
        ),
        sine = dict(
            pos_amp = 10,
            pos_per = 10,
            pos_off = 130,
            neg_amp = 5,
            neg_per = 10,
            neg_off = 70,
            iter = 2
        ),
        dynamic = dict(
            # trans_time = 30,
            # pos_init_press = 60 + ATM,
            # pos_final_press = 120 + ATM,
            # pos_amp = 10,
            # pos_per = 7,
            # neg_init_press = - 30 + ATM,
            # neg_final_press = - 70 + ATM,
            # neg_amp = 7,
            # neg_per = 5,
            trans_time = 60,
            pos_init_press = 80 + ATM,
            pos_final_press = 110 + ATM,
            pos_amp = 5,
            pos_per = 8,
            neg_init_press = - 70 + ATM,
            neg_final_press = - 80 + ATM,
            neg_amp = 3, 
            neg_per = 10,
        ),
        center_step = dict(
            trans_time = 40,
            pos_time_step = 10,
            pos_center_step = 15,
            pos_amp = 5,
            pos_per = 5,
            pos_init_press = 50 + ATM,
            neg_time_step = 10,
            neg_center_step = - 5,
            neg_amp = 2,
            neg_per = 6,
            neg_init_press = - 50 + ATM,
        )
    )
)

def infos_to_datas(infos: deque):
    datas = dict(
        curr_time = deque(),
        sensor_time = deque(),
        reference_time = deque(),
        press_pos = deque(),
        press_neg = deque(),
        act_pos_press = deque(),
        act_neg_press = deque(),
        ref_act_pos = deque(),
        ref_act_neg = deque(),
        ctrl_ch_pos = deque(),
        ctrl_ch_neg = deque(),
        ctrl_act_pos_in = deque(),
        ctrl_act_pos_out = deque(),
        ctrl_act_neg_in = deque(),
        ctrl_act_neg_out = deque(),
        flowrate1 = deque(),
        flowrate2 = deque(),
        flowrate3 = deque(),
        flowrate4 = deque(),
        flowrate5 = deque(),
        flowrate6 = deque(),
    )
    for entry in infos:
        obs = entry["obs"] if "obs" in entry else entry
        datas['curr_time'].append(entry.get('policy_time', obs['curr_time']))
        datas['sensor_time'].append(obs['curr_time'])
        datas['reference_time'].append(entry.get('reference_time', np.nan))
        datas['press_pos'].append(obs['sen_pos'])
        datas['press_neg'].append(obs['sen_neg'])
        datas['act_pos_press'].append(obs['sen_act_pos'])
        datas['act_neg_press'].append(obs['sen_act_neg'])
        datas['ref_act_pos'].append(obs['ref_act_pos'])
        datas['ref_act_neg'].append(obs['ref_act_neg'])
        datas['ctrl_ch_pos'].append(obs['ctrl_pos'])
        datas['ctrl_ch_neg'].append(obs['ctrl_neg'])
        datas['ctrl_act_pos_in'].append(obs['ctrl_act_pos_in'])
        datas['ctrl_act_pos_out'].append(obs['ctrl_act_pos_out'])
        datas['ctrl_act_neg_in'].append(obs['ctrl_act_neg_in'])
        datas['ctrl_act_neg_out'].append(obs['ctrl_act_neg_out'])
        for key in ('flowrate1', 'flowrate2', 'flowrate3', 'flowrate4', 'flowrate5', 'flowrate6'):
            datas[key].append(obs.get(key, np.nan))
    return datas

def calc_metrics(datas):
    return compute_tracking_metrics(datas)

def save_datas(
    datas,
    model_name,
    obs_mode,
    ref_mode,
    save_name=None,
    kwargs=None,
    pid_enabled=False,
):
    if save_name is not None:
        print('[ INFO] Saving data starts ...')
    
    obs_mode = "real" if obs_mode.lower() == "real" else "simulation"

    if save_name is not None:
        output_dir = ensure_model_exp_dir(model_name, save_name)
        df = datas_to_df(datas)
        # df.to_csv(f'/Users/greenlandshark/MATLAB/main/datas/{save_name}.csv', index=False)
        df.to_csv(output_dir / f'{save_name}.csv', index=False)

    metrics = calc_metrics(datas)
    print_tracking_metrics(metrics)

    save_cfg = copy.deepcopy(kwargs) if kwargs is not None else {}
    save_cfg["model_name"] = model_name
    save_cfg["obs_mode"] = obs_mode
    save_cfg["ref_mode"] = ref_mode
    save_cfg["metrics"] = metrics
    save_cfg.setdefault("runtime", {}).update(
        pid_enabled=bool(pid_enabled),
        pid_label="On" if pid_enabled else "Off",
    )
    if "env" not in save_cfg:
        save_cfg["env"] = {}
    save_cfg["env"]["pid_enabled"] = bool(pid_enabled)

    if save_name is not None:
        with open(output_dir / 'cfg.yaml', 'w') as f:
            yaml.dump(save_cfg, f)
    
    if save_name is not None:
        print('[ INFO] Saving data Done!')

def plot_datas(datas, model_name, save_name=None):
    output_prefix = None
    if save_name is not None:
        output_dir = ensure_model_exp_dir(model_name, save_name)
        output_prefix = output_dir / save_name
    plot_lib3_pressure_and_controls(
        datas_to_df(datas),
        output_prefix,
        show=True,
    )

def model_bank_reference(config, trajectory_index=0):
    spec = copy.deepcopy(reference_spec_from_model_config(config))
    if spec.get("kind") not in ("optimized_trajectory_bank", "simulator_trajectory_bank", "trajectory_bank"):
        raise ValueError("Selected model does not contain a trajectory bank; choose another reference mode")
    spec["kwargs"].update(repeat_after_end=False, repeat_same=False,
                          trajectory_index=trajectory_index)
    ref = make_pressure_reference(spec)
    ref.max_time = 40.0
    preview = (config['env']['num_pred'] + 1) / config['obs']['freq']
    if ref.max_time + preview > float(ref.trajectories[trajectory_index]['duration']):
        raise ValueError('Reference bank is too short for 40s evaluation plus policy preview')
    return ref, spec


def build_viz_env(config, obs, ref, pred=None):
    """Honor saved simulation initialization; never precondition real hardware here."""
    initialization = config.get('initialization') or {}
    mode = initialization.get('mode')
    if mode not in (None, 'pump_on_all_open'):
        raise ValueError(f'Unsupported initialization mode: {mode}')
    if isinstance(obs, PneuSim) and mode == 'pump_on_all_open':
        return PneuEnv3AllOpen(obs=obs, ref=ref, pred=pred,
            precondition_seconds=initialization.get('precondition_seconds', 30),
            **config['env'])
    return PneuEnv3(obs=obs, ref=ref, pred=pred, **config['env'])


def collect_rollout(env, model, ref):
    """Close exactly once before saving/plotting, including Ctrl+C/errors."""
    infos = deque()
    try:
        state, _ = env.reset()
        curr_time = 0.0
        duration = 100.0 if ref.max_time == float('inf') else ref.max_time
        while curr_time < duration:
            reference_time = env.t if isinstance(env, PneuEnv3AllOpen) else None
            action = model.predict(state)
            state, _, _, _, info = env.step(action)
            if isinstance(env, PneuEnv3AllOpen):
                curr_time = env.t
                info = copy.deepcopy(info)
                info['policy_time'] = curr_time
                info['reference_time'] = reference_time
            else:
                curr_time = info['obs']['curr_time']
            infos.append(info)
    except KeyboardInterrupt:
        print('\n[ INFO] Interrupted; publishing shutdown command before saving.')
    finally:
        env.close()
    return infos


if __name__ == '__main__':
    print('[ INFO] Control Mode: RL')
    print(color('[INPUT] Control Model:', 'blue'))
    
    models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
    for i, model in enumerate(models):
        print(color(f'\t{i+1}. {model}', 'yellow'))
    print(color('\t---', 'blue'))
    model_idx = int(input(color('\tMODEL: ', 'blue'))) - 1 
    model_name = models[model_idx]
    delete_lines(len(models) + 3)
    print(f'[ INFO] Control Model: {model_name}')

    kwargs = load_yaml(model_name)
        
    print(color('[INPUT] Reference Mode:', 'blue'))
    print(color('\t1. Step case', 'yellow'))
    print(color('\t2. Random', 'yellow'))
    print(color('\t3. Trajectory', 'yellow'))
    print(color('\t4. Sinusoidal', 'yellow'))
    print(color('\t5. Paper(dynamic oscillator)', 'yellow'))
    print(color('\t6. Model bank: trajectory 0 for 40s (no repeat) (default)', 'yellow'))
    print(color('\t---', 'blue'))
    ref_mode = input(color('\tREF: ', 'blue')).strip() or '6'
    delete_lines(9)

    if ref_mode == '6':
        ref, reference_spec = model_bank_reference(kwargs)
        viz_kwargs['reference'] = reference_spec
        viz_kwargs['duration_sec'] = 40.0
        ref_type = 'model_bank_0_40s'
    elif ref_mode == '1':
        print(f'[ INFO] Reference Mode: Step case')
        ref = StepCasesRef(**viz_kwargs['ref']['stepcases'])
        ref_type = "stepcases"
    elif ref_mode == '2': 
        print(f'[ INFO] Reference Mode: Random')
        ref = RandomRef(**viz_kwargs['ref']['random'])
        ref_type = "random"
    elif ref_mode =='3':
        print(f'[ INFO] Reference Mode: Trajectory')
        csv_file_name = viz_kwargs["ref"]["traj"]["file"]
        csv_file_name = find_experiment_file(csv_file_name, f"{csv_file_name}.csv")
        csv_data = pd.read_csv(csv_file_name).to_dict(orient="list")
        keys = [
            "curr_time",
        ]
        dict_data = {k: np.array(csv_data[k]) for k in keys}
        if "ref_act_pos" in csv_data and "ref_act_neg" in csv_data:
            dict_data["ref_pos"] = np.array(csv_data["ref_act_pos"])
            dict_data["ref_neg"] = np.array(csv_data["ref_act_neg"])
        else:
            dict_data["ref_pos"] = np.array(csv_data["ref_pos"])
            dict_data["ref_neg"] = np.array(csv_data["ref_neg"])
        dict_data["curr_time"] -= dict_data["curr_time"][0]

        ref = TrajRef(
            traj_time = dict_data["curr_time"],
            traj_pos = dict_data["ref_pos"],
            traj_neg = dict_data["ref_neg"]
        )
        ref_type = "trajectory"
    elif ref_mode == '4':
        print(f'[ INFO] Reference Mode: Sinusoidal')
        ref = SineRef(**viz_kwargs['ref']['sine'])
        ref_type = "sine"
    
    elif ref_mode == '5':
        print(f'[ INFO] Reference Mode: dynamic oscillator')
        ref = DynamicOscillatorRef(**viz_kwargs['ref']['dynamic'])
        ref_type = "dynamic_oscillator"
        
        # ref = CenterStepOscillationRef(**viz_kwargs['ref']['center_step'])
        # ref_type = "center_step"

    print(color('[INPUT] Observation Mode:', 'blue'))
    print(color('\t1. Sim', 'yellow'))
    print(color('\t2. Real', 'yellow'))
    print(color('\t---', 'blue'))
    obs_mode = input(color('\tOBS: ', 'blue')) 
    delete_lines(5)
    if obs_mode == '1':
        obs_type = 'Simulation'
    elif obs_mode == '2':
        obs_type = 'Real'
    else:
        raise ValueError(color(f'[ERROR] Unknown observation mode: {obs_mode}', 'red'))
    print(f'[ INFO] Observation Mode: {obs_type}')

    print(color('[INPUT] PID on?', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    pid_mode = input(color('\tPID: ', 'blue')) 
    delete_lines(5)

    pid_enabled = pid_mode == '1'

    if pid_enabled:
        print(f'[ INFO] PID: On')
    else:
        print(f'[ INFO] PID: Off')

    print(color('[INPUT] Save data?', 'blue'))
    print(color('\t1. Yes', 'yellow'))
    print(color('\t2. No', 'yellow'))
    print(color('\t---', 'blue'))
    data_log = input(color('\tLogging: ', 'blue')) 
    delete_lines(5)
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    if data_log == '1':
        save_name = f'{formatted_time}_{model_name}_{obs_type}'
    else:
        save_name = None
    print(f'[ INFO] Data logging: {"False" if data_log == "2" else f"{save_name}.csv"}')

    # Match the selected model's timing, scaling and initial state.
    obs_config = copy.deepcopy(kwargs['obs'])
    viz_kwargs['env']['sim'] = copy.deepcopy(obs_config)
    viz_kwargs['env']['real'] = copy.deepcopy(obs_config)
    obs = PneuSim(**obs_config) if obs_mode == '1' else PneuReal(**obs_config)
    pred_config = copy.deepcopy(kwargs.get('pred'))
    viz_kwargs['env']['pred'] = pred_config
    pred = PneuPred(**pred_config) if pred_config is not None else None

    env = build_viz_env(kwargs, obs, ref, pred)
    viz_kwargs['initialization'] = copy.deepcopy(kwargs.get('initialization'))
    viz_kwargs['policy_env'] = copy.deepcopy(kwargs['env'])
    prepared = isinstance(env, PneuEnv3AllOpen)
    viz_kwargs['runtime'] = dict(
        start_protocol=('pump_on_all_open_sim' if prepared else
                        'continuous_pump_on_no_start_wait' if obs_mode == '2' else 'standard_sim_reset'),
        reference_clock='policy_step/freq' if prepared else 'sensor_time',
        preparation_seconds=env.precondition_seconds if prepared else None,
        history_initialization='prepared_sim_history' if prepared else 'standard_env_reset',
    )
    
    if pid_enabled:
        env.set_pid(**viz_kwargs['env']['pid'])
    

    model = SAC(
        env = env,
        **kwargs['model']
    )
    model.set_logger(model_name)
    model.load_model(path=model.logger.model_path, evaluate=True)

    infos = collect_rollout(env, model, ref)
    if obs_mode == '2':
        print('[ END] All-open command published. The pump is not controlled by this script.')
    if infos:
        datas = infos_to_datas(infos)
        save_datas(
            datas, model_name, obs_type, ref_type, save_name, viz_kwargs,
            pid_enabled=pid_enabled,
        )
        plot_datas(datas, model_name, save_name)
