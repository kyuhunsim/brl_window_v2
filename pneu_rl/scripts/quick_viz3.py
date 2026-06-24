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

from pneu_ref.random_ref import RandomRef
from pneu_ref.step_ref import StepCasesRef, StepRef
from pneu_ref.sine_ref import SineRef, DynamicOscillatorRef, CenterStepOscillationRef
from pneu_ref.traj_ref import TrajRef
from pneu_env.env3 import PneuEnv3
from pneu_env.sim3 import PneuSim
from pneu_env.pred3 import PneuPred
from pneu_env.real.real9 import PneuReal
from pneu_rl.sac_with_loss import SAC
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
Times_New_Roman = fm.FontProperties(fname=font_path)


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
        datas['curr_time'].append(obs['curr_time'])
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
        os.makedirs(f'{get_pkg_path("pneu_rl")}/exp/{save_name}')
        df = datas_to_df(datas)
        # df.to_csv(f'/Users/greenlandshark/MATLAB/main/datas/{save_name}.csv', index=False)
        df.to_csv(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/{save_name}.csv', index=False)

    metrics = calc_metrics(datas)
    print_tracking_metrics(metrics)

    save_cfg = copy.deepcopy(kwargs) if kwargs is not None else {}
    save_cfg["model_name"] = model_name
    save_cfg["obs_mode"] = obs_mode
    save_cfg["ref_mode"] = ref_mode
    save_cfg["metrics"] = metrics
    save_cfg["runtime"] = dict(
        pid_enabled=bool(pid_enabled),
        pid_label="On" if pid_enabled else "Off",
    )
    if "env" not in save_cfg:
        save_cfg["env"] = {}
    save_cfg["env"]["pid_enabled"] = bool(pid_enabled)

    if save_name is not None:
        with open(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/cfg.yaml', 'w') as f:
            yaml.dump(save_cfg, f)
    
    if save_name is not None:
        print('[ INFO] Saving data Done!')

def plot_datas(datas, save_name=None):
    fontname = Times_New_Roman
    label_font_size = 18
    
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(6, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[2:4,0])
    ax3 = fig.add_subplot(gs[4:6,0])

    ax1.plot(
        np.array(datas['curr_time']),
        np.array(datas['press_pos']),
        linewidth=2, color='red', label='CH_POS'
    )
    ax1.plot(
        np.array(datas['curr_time']),
        np.array(datas['press_neg']),
        linewidth=2, color='blue', label='CH_NEG'
    )
    ax1.set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ax1.set_ylabel('Chamber [kPa]', fontproperties=fontname, fontsize=label_font_size)
    ax1.grid(which='major', color='silver', linewidth=1)
    ax1.grid(which='minor', color='lightgray', linewidth=0.5)
    ax1.minorticks_on()
    ax1.legend(loc='upper right')
    ax1.set(xlim=(0, None), ylim=(None, None))
    
    ax2.plot(
        np.array(datas['curr_time']),
        np.array(datas['ref_act_pos']),
        linewidth=2, color='black', label='REF'
    )
    ax2.plot(
        np.array(datas['curr_time']),
        np.array(datas['act_pos_press']),
        linewidth=2, color='red', label='ACT_POS'
    )
    ax2.set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ax2.set_ylabel('Act Pos [kPa]', fontproperties=fontname, fontsize=label_font_size)
    ax2.grid(True)
    ax2.grid(which='major', color='silver', linewidth=1)
    ax2.grid(which='minor', color='lightgray', linewidth=0.5)
    ax2.minorticks_on()
    ax2.legend(loc='upper right')
    ax2.sharex(ax1)
    ax2.set(xlim=(0, None), ylim=(None, None))

    ax3.plot(
        np.array(datas['curr_time']),
        np.array(datas['ref_act_neg']),
        linewidth=2, color='black', label='REF'
    )
    ax3.plot(
        np.array(datas['curr_time']),
        np.array(datas['act_neg_press']),
        linewidth=2, color='blue', label='ACT_NEG'
    )
    ax3.set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ax3.set_ylabel('Act Neg [kPa]', fontproperties=fontname, fontsize=label_font_size)
    ax3.grid(True)
    ax3.grid(which='major', color='silver', linewidth=1)
    ax3.grid(which='minor', color='lightgray', linewidth=0.5)
    ax3.minorticks_on()
    ax3.legend(loc='upper right')
    ax3.sharex(ax1)
    ax3.set(xlim=(0, None), ylim=(None, None))

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/{save_name}_main.png')

    ctrl_fig, ctrl_axes = plt.subplots(6, 1, figsize=(9, 10), sharex=True)
    ctrl_series = [
        ("ctrl_ch_pos", "Positive Chamber", "#d62728"),
        ("ctrl_ch_neg", "Negative Chamber", "#1f77b4"),
        ("ctrl_act_pos_in", "Actuator Positive In", "#2ca02c"),
        ("ctrl_act_pos_out", "Actuator Positive Out", "#ff7f0e"),
        ("ctrl_act_neg_in", "Actuator Negative In", "#9467bd"),
        ("ctrl_act_neg_out", "Actuator Negative Out", "#8c564b"),
    ]
    time_arr = np.array(datas["curr_time"])
    for ax, (key, title, color_code) in zip(ctrl_axes, ctrl_series):
        ax.plot(time_arr, np.array(datas[key]), linewidth=1.8, color=color_code)
        ax.set_ylabel(title, fontproperties=fontname, fontsize=12)
        ax.grid(which='major', color='silver', linewidth=1)
        ax.grid(which='minor', color='lightgray', linewidth=0.5)
        ax.minorticks_on()
        ax.set(xlim=(0, None), ylim=(None, None))
    ctrl_axes[-1].set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ctrl_fig.tight_layout()
    if save_name is not None:
        ctrl_fig.savefig(f'{get_pkg_path("pneu_rl")}/exp/{save_name}/{save_name}_ctrl.png')
    plt.show()

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
    print(color('\t---', 'blue'))
    ref_mode = input(color('\tREF: ', 'blue')) 
    delete_lines(8)

    if ref_mode == '1':
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
        csv_file_name = f"{get_pkg_path('pneu_rl')}/exp/{csv_file_name}/{csv_file_name}"
        csv_file_name = f"{csv_file_name}.csv"
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

    if obs_mode == '1':
        obs = PneuSim(**viz_kwargs["env"]["sim"])
    else:
        obs = PneuReal(**viz_kwargs["env"]["real"])
    pred = PneuPred(**viz_kwargs["env"]["pred"])

    obs.set_init_press(**viz_kwargs["env"]["init_press"] )
    env = PneuEnv3(
        obs = obs,
        ref = ref,
        pred = pred if kwargs['pred'] is not None else None,
        **kwargs['env']
    )
    
    if pid_enabled:
        env.set_pid(**viz_kwargs['env']['pid'])
    

    model = SAC(
        env = env,
        **kwargs['model']
    )
    model.set_logger(model_name)
    model.load_model(path=model.logger.model_path, evaluate=True)

    try:
        state, info = env.reset()
        curr_time = 0
        if ref.max_time == float('inf'):
            # ref.max_time = 100
            ref.max_time = 100
        
        infos = deque()
        time_flag = 0
        while curr_time < ref.max_time:
            action = model.predict(state)
            state, _, _, _, info = env.step(action)
            curr_time = info['obs']['curr_time']
            elapsed_time_flag = curr_time - time_flag
            infos.append(info)

        env.close()

        datas = infos_to_datas(infos)
        save_datas(
            datas,
            model_name,
            obs_type,
            ref_type,
            save_name,
            viz_kwargs,
            pid_enabled=pid_enabled,
        )
        plot_datas(datas, save_name)

    except KeyboardInterrupt:
        print()
        print(color('[ INFO] Keyboard interrupt received.', 'red'))
        datas = infos_to_datas(infos)
        save_datas(
            datas,
            model_name,
            obs_type,
            ref_type,
            save_name,
            viz_kwargs,
            pid_enabled=pid_enabled,
        )
        plot_datas(datas, save_name)
    
    finally:
        env.close()


    

    

        
