import os
import copy
import hashlib

from collections import deque
import numpy as np
import pandas as pd
import time
from datetime import datetime
import pickle
import yaml
import threading

from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_ref.step_ref import StepCasesRef, StepRef
from pneu_ref.sine_ref import SineRef, DynamicOscillatorRef, CenterStepOscillationRef
from pneu_ref.traj_ref import TrajRef
from pneu_env.env import PneuEnv
from pneu_env.sim2 import PneuSim
from pneu_env.real.real import PneuReal
from pneu_env.pred2 import PneuPred
# Use the diagnostic-compatible SAC implementation.  New screening configs
# contain train_diag_* options that the legacy SAC constructor does not accept;
# both implementations use the same policy/checkpoint format, while this one
# also remains compatible with the older model folders.
from pneu_rl.sac_with_loss import SAC
from pneu_rl.exp_paths import ensure_model_exp_dir, find_experiment_file
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
EVAL_STEPS = 2048
EVAL_DURATION_SEC = EVAL_STEPS / 50.0


def reference_sha256(reference_spec):
    """Hash a canonical reference spec for matched ablation evaluation."""
    canonical = yaml.safe_dump(reference_spec or {}, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

viz_kwargs = dict(
    env = dict(
        sim = dict(
            freq = 50,
            delay = 0.1,
            noise = False,
            noise_std = 0,
            offset_pos = 0,
            offset_neg = 0,
            scale = True,
            scale_low = 0.8,
            scale_high = 1.0,
        ),
        real = dict(
            freq = 50,
            scale = True
        ),
        pred = dict(
            freq = 50,
            delay = 0,
            noise = False,
            scale = True,
            scale_low = 0.8,
            scale_high = 1.0,
        ),
        init_press = dict(
            init_pos_press = ATM,
            init_neg_press = ATM
        ),
        pid = dict(
            Kp_pos = 0.0,
            Ki_pos = 0.01,
            Kd_pos = 0.0,
            Kp_neg = 0.0,
            Ki_neg = 0.01,
            Kd_neg = 0.0,
            Ka = 10
        )
    ),
    ref = dict(
        stepcases = dict(
            # 40.96 s quick diagnostic inside the characterized easy region.
            # Longer 10--15 s holds belong in the separate system-ID test.
            time_step = 5,
            ref_pos_max = 180,
            ref_pos_min = 160,
            ref_neg_max = 25,
            ref_neg_min = 15
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
            pos_max_off = 210,
            pos_min_off = 140,
            neg_max_off = 32,
            neg_min_off = 10,
            pos_max_ts = 10,
            neg_max_ts = 10,
            pos_max_amp = 10,
            neg_max_amp = 10,
            seed = 61098
        ),
        sine = dict(
            pos_amp = 10,
            pos_per = 10,
            pos_off = 80 + ATM,
            neg_amp = 10,
            neg_per = 10,
            neg_off = - 80 + ATM,
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
        sen_pos = deque(),
        sen_neg = deque(),
        ref_pos = deque(),
        ref_neg = deque(),
        ctrl_pos = deque(),
        ctrl_neg = deque()
    )
    for info in infos:
        for key, value in info.items():
            datas[key].append(value)
    return datas

def calc_metrics(datas):
    return compute_tracking_metrics(datas)

def save_datas(
    datas,
    model_name,
    obs_mode,
    ref_mode,
    save_name=None,
    model_cfg=None,
    pid_enabled=False,
    pid_config=None,
    ref_cfg=None,
    reference_match=None,
):
    if save_name is not None:
        print('[ INFO] Saving data starts ...')
    
    obs_mode = "Real" if obs_mode == "2" else "simulation" 

    if save_name is not None:
        output_dir = ensure_model_exp_dir(model_name, save_name)
        df = datas_to_df(datas)
        df.to_csv(output_dir / f'{save_name}.csv', index=False)

    metrics = calc_metrics(datas)
    print_tracking_metrics(metrics)

    model_cfg = model_cfg or {}
    record = dict(
        model_name=model_name,
        model_variant=model_cfg.get("variant"),
        obs_mode=obs_mode,
        ref_mode=ref_mode,
        metrics=metrics,
        training=dict(
            pid_enabled=model_cfg.get("pid") is not None,
            iterm_enabled=model_cfg.get("pid") is not None
            or "iterm" in str(model_cfg.get("variant", "")),
            variant=model_cfg.get("variant"),
            reference=reference_spec_from_model_config(model_cfg),
            reference_sha256=(reference_match or {}).get("training_sha256"),
            action_scale=dict(
                low=(model_cfg.get("obs") or {}).get("scale_low", 0.8),
                high=(model_cfg.get("obs") or {}).get("scale_high", 1.0),
            ),
        ),
        evaluation=dict(
            pid_enabled=bool(pid_enabled),
            pid_config=pid_config if pid_enabled else None,
            reference=ref_cfg,
            reference_sha256=(reference_match or {}).get("evaluation_sha256"),
            reference_content_match=(reference_match or {}).get("content_match"),
            action_scale=dict(
                low=(model_cfg.get("obs") or {}).get("scale_low", 0.8),
                high=(model_cfg.get("obs") or {}).get("scale_high", 1.0),
            ),
        ),
        model_config=model_cfg,
    )

    if save_name is not None:
        with open(output_dir / 'cfg.yaml', 'w') as f:
            yaml.safe_dump(record, f, sort_keys=False)
    
    if save_name is not None:
        print('[ INFO] Saving data Done!')

def plot_datas(datas, model_name, save_name=None):
    fontname = Times_New_Roman
    label_font_size = 18
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(5, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[2:4,0])
    ax3 = fig.add_subplot(gs[4,0])    

    ax1.plot(
        np.array(datas['curr_time']),
        np.array(datas['ref_pos']),
        linewidth=2, color='black', label='REF'
    )
    ax1.plot(
        np.array(datas['curr_time']),
        np.array(datas['sen_pos']),
        linewidth=2, color='red', label='POS'
    )
    ax1.set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ax1.set_ylabel('Pressure [kPa]', fontproperties=fontname, fontsize=label_font_size)
    ax1.grid(which='major', color='silver', linewidth=1)
    ax1.grid(which='minor', color='lightgray', linewidth=0.5)
    ax1.minorticks_on()
    ax1.legend(loc='upper right')
    ax1.set(xlim=(0, None), ylim=(None, None))
    
    ax2.plot(
        np.array(datas['curr_time']),
        np.array(datas['ref_neg']),
        linewidth=2, color='black', label='REF'
    )
    ax2.plot(
        np.array(datas['curr_time']),
        np.array(datas['sen_neg']),
        linewidth=2, color='blue', label='NEG'
    )
    ax2.set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ax2.set_ylabel('Pressure [kPa]', fontproperties=fontname, fontsize=label_font_size)
    ax2.grid(True)
    ax2.grid(which='major', color='silver', linewidth=1)
    ax2.grid(which='minor', color='lightgray', linewidth=0.5)
    ax2.minorticks_on()
    ax2.legend(loc='upper right')
    ax2.sharex(ax1)
    ax2.set(xlim=(0, None), ylim=(None, None))

    ax3.plot(
        datas['curr_time'],
        datas['ctrl_pos'],
        linewidth=2, color='red', label='POS'
    )
    ax3.plot(
        datas['curr_time'],
        datas['ctrl_neg'],
        linewidth=2, color='blue', label='NEG'

    )
    ax3.set_xlabel('Time [sec]', fontproperties=fontname, fontsize=label_font_size)
    ax3.set_ylabel('Control', fontproperties=fontname, fontsize=label_font_size)
    ax3.legend(loc='upper right')
    ax3.sharex(ax1)
    ax3.set(xlim=(0, None), ylim=(0.8, 1.0))
    ax3.set_yticks(np.arange(0.8, 1.01, 0.05))

    for axis in (ax1, ax2, ax3):
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontproperties(fontname)
        for legend_text in axis.get_legend().get_texts():
            legend_text.set_fontproperties(fontname)

    plt.tight_layout(pad=1.2)
    if save_name is not None:
        output_dir = ensure_model_exp_dir(model_name, save_name)
        plt.savefig(
            output_dir / f'{save_name}.png',
            dpi=180,
            bbox_inches='tight',
        )
    plt.show()

if __name__ == '__main__':
    model_override = os.environ.get("QUICK_VIZ_MODEL")
    reference_profile_override = os.environ.get("QUICK_VIZ_REFERENCE_PROFILE")
    obs_mode_override = os.environ.get("QUICK_VIZ_OBS_MODE")
    pid_mode_override = os.environ.get("QUICK_VIZ_PID_MODE")
    save_override = os.environ.get("QUICK_VIZ_SAVE")
    save_name_override = os.environ.get("QUICK_VIZ_SAVE_NAME")
    allow_reference_mismatch = os.environ.get("QUICK_VIZ_ALLOW_REFERENCE_MISMATCH", "0") == "1"
    print('[ INFO] Control Mode: RL')
    print(color('[INPUT] Control Model:', 'blue'))
    
    models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
    for i, model in enumerate(models):
        print(color(f'\t{i+1}. {model}', 'yellow'))
    print(color('\t---', 'blue'))
    if model_override:
        if model_override not in models:
            raise ValueError(f"Unknown QUICK_VIZ_MODEL: {model_override}")
        model_name = model_override
    else:
        model_idx = int(input(color('\tMODEL: ', 'blue'))) - 1
        model_name = models[model_idx]
        delete_lines(len(models) + 3)
    print(f'[ INFO] Control Model: {model_name}')

    kwargs = load_yaml(model_name)
    training_reference = reference_spec_from_model_config(kwargs)
    training_reference_hash = reference_sha256(training_reference)
    reference_match = dict(
        training_sha256=training_reference_hash,
        evaluation_sha256=None,
        content_match=None,
    )
        
    if reference_profile_override:
        with open(reference_profile_override, encoding="utf-8") as handle:
            reference_profile = yaml.safe_load(handle)
        evaluation_reference_hash = reference_sha256(reference_profile)
        reference_match.update(
            evaluation_sha256=evaluation_reference_hash,
            content_match=training_reference_hash == evaluation_reference_hash,
        )
        if not reference_match["content_match"] and not allow_reference_mismatch:
            raise ValueError(
                "QUICK_VIZ_REFERENCE_PROFILE differs from the model training reference. "
                "Use the exact training profile for ablation validation, or set "
                "QUICK_VIZ_ALLOW_REFERENCE_MISMATCH=1 only for an explicit transfer test."
            )
        ref = make_pressure_reference(reference_profile)
        random_cfg = reference_profile
        ref_mode = 'profile'
        ref_type = 'profile'
        print(f'[ INFO] Reference Profile: {reference_profile_override}')
        print(f'[ INFO] Reference content match: {reference_match["content_match"]}')
    else:
        print(color('[INPUT] Reference Mode:', 'blue'))
        print(color('\t1. Step case', 'yellow'))
        print(color('\t2. Random', 'yellow'))
        print(color('\t3. Trajectory', 'yellow'))
        print(color('\t4. Sinusoidal', 'yellow'))
        print(color('\t5. Paper(dynamic oscillator)', 'yellow'))
        print(color('\t---', 'blue'))
        ref_mode = input(color('\tREF: ', 'blue'))
        delete_lines(8)

    if reference_profile_override:
        pass
    elif ref_mode == '1':
        print(f'[ INFO] Reference Mode: Step case')
        ref = StepCasesRef(**viz_kwargs['ref']['stepcases'])
        ref_type = "stepcases"
    elif ref_mode == '2': 
        print(f'[ INFO] Reference Mode: Random')
        # Use the model's training random-reference distribution by default.
        # The legacy quick-viz reference is only a fallback for old folders
        # that do not contain rnd_ref in their model config.
        random_cfg = (
            reference_spec_from_model_config(kwargs)
            if kwargs.get("reference") is not None or kwargs.get("rnd_ref") is not None
            else viz_kwargs["ref"]["random"]
        )
        ref = make_pressure_reference(random_cfg)
        evaluation_reference_hash = reference_sha256(random_cfg)
        reference_match.update(
            evaluation_sha256=evaluation_reference_hash,
            content_match=training_reference_hash == evaluation_reference_hash,
        )
        ref_type = "random"
    elif ref_mode =='3':
        print(f'[ INFO] Reference Mode: Trajectory')
        csv_file_name = viz_kwargs["ref"]["traj"]["file"]
        csv_file_name = find_experiment_file(csv_file_name, f"{csv_file_name}.csv")
        csv_data = pd.read_csv(csv_file_name).to_dict(orient="list")
        keys = [
            "curr_time",
            "sen_pos",
            "sen_neg",
            "ref_pos",
            "ref_neg",
        ]
        dict_data = {}
        for k, v in zip(keys, csv_data.values()):
            dict_data[k] = np.array(v)
        dict_data["curr_time"] -= dict_data["curr_time"][0]
        # dict_data["ref_pos"] += ATM
        # dict_data["ref_neg"] += ATM

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
    if obs_mode_override:
        obs_mode = '1' if obs_mode_override.lower() in ('1', 'sim', 'simulation') else '2'
    else:
        obs_mode = input(color('\tOBS: ', 'blue'))
    obs_type = 'Simulation' if obs_mode == "1" else "Real"
    delete_lines(5)
    print(f'[ INFO] Observation Mode: {"Simulation" if obs_mode == "1" else "Real"}')

    if kwargs.get('pid') is not None:
        # Matched evaluation must preserve the closed loop used in training.
        # A separate PID-off ablation can be added explicitly later, but it
        # should not happen accidentally from the normal quick-viz flow.
        pid_mode = '1'
        print(color('[ INFO] I-term model detected: PID/I-term forced ON for matched validation', 'yellow'))
    elif pid_mode_override:
        pid_mode = '1' if pid_mode_override.lower() in ('1', 'on', 'true', 'yes') else '2'
    else:
        print(color('[INPUT] PID on?', 'blue'))
        print(color('\t1. Yes', 'yellow'))
        print(color('\t2. No', 'yellow'))
        print(color('\t---', 'blue'))
        pid_mode = input(color('\tPID: ', 'blue'))
        delete_lines(5)

    if pid_mode == '1':
        print(f'[ INFO] PID: On')
    else:
        print(f'[ INFO] PID: Off')

    if save_override:
        data_log = '1' if save_override.lower() in ('1', 'yes', 'true', 'on') else '2'
    else:
        print(color('[INPUT] Save data?', 'blue'))
        print(color('\t1. Yes', 'yellow'))
        print(color('\t2. No', 'yellow'))
        print(color('\t---', 'blue'))
        data_log = input(color('\tLogging: ', 'blue'))
        delete_lines(5)
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    if data_log == '1':
        save_name = save_name_override or f'{formatted_time}_{model_name}_{obs_type}'
    else:
        save_name = None
    print(f'[ INFO] Data logging: {"False" if data_log == "2" else f"{save_name}.csv"}')

    # Matched validation must use the model's simulator/initial-state config.
    # The legacy quick-viz defaults are retained only for old model folders
    # that do not contain the corresponding fields.
    model_obs_cfg = copy.deepcopy(kwargs.get("obs") or {})
    model_pred_cfg = copy.deepcopy(kwargs.get("pred") or {})
    sim_cfg = copy.deepcopy(viz_kwargs["env"]["sim"])
    sim_cfg.update(model_obs_cfg)
    pred_cfg = copy.deepcopy(viz_kwargs["env"]["pred"])
    pred_cfg.update(model_pred_cfg)
    init_cfg = dict(
        init_pos_press=float(model_obs_cfg.get("init_pos_press", ATM)),
        init_neg_press=float(model_obs_cfg.get("init_neg_press", ATM)),
    )

    if obs_mode == '1':
        obs = PneuSim(**sim_cfg)
        pred = PneuPred(**pred_cfg)
    elif obs_mode == '2':
        obs = PneuReal(**viz_kwargs["env"]["real"])
        pred = PneuPred(**pred_cfg)

    if obs_mode == '1':
        obs.set_init_press(**init_cfg)
    env = PneuEnv(
        obs = obs,
        ref = ref,
        pred = pred if kwargs['pred'] is not None else None,
        **kwargs['env']
    )
    
    # For an I-term model, use the PID gains saved with that model.  For a
    # pure-RL model with manually enabled PID, fall back to the legacy viz
    # gains so that the evaluation choice remains explicit.
    active_pid_cfg = kwargs.get('pid') or viz_kwargs['env']['pid']
    if pid_mode == '1':
        env.set_pid(**active_pid_cfg)
    

    model = SAC(
        env = env,
        **kwargs['model']
    )
    # Evaluation only needs policy/critic weights.  Do not load the replay
    # buffer: an interrupted training run can leave buffer_sac.pkl truncated,
    # which must not prevent screen/quick-viz evaluation.
    model.load(name=model_name, load_buffer=False)

    try:
        state, info = env.reset()
        curr_time = 0
        # Keep every model and reference mode comparable, including models
        # trained with a 512-step horizon.  Long real-system tests can use the
        # same fixed duration through this setting as well.
        ref.max_time = EVAL_DURATION_SEC
        
        infos = deque()
        time_flag = 0
        for _ in range(EVAL_STEPS):
            action = model.predict(state)
            state, _, _, _, info = env.step(action)
            curr_time = info['obs']['curr_time']
            elapsed_time_flag = curr_time - time_flag
            infos.append(info['obs'])

        env.close()

        datas = infos_to_datas(infos)
        save_datas(
            datas,
            model_name,
            obs_type,
            ref_type,
            save_name,
            model_cfg=kwargs,
            pid_enabled=pid_mode == '1',
            pid_config=active_pid_cfg if pid_mode == '1' else None,
            ref_cfg=random_cfg if ref_type in ('random', 'legacy_easy', 'profile') else viz_kwargs['ref'].get(ref_type),
            reference_match=reference_match,
        )
        plot_datas(datas, model_name, save_name)

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
            model_cfg=kwargs,
            pid_enabled=pid_mode == '1',
            pid_config=active_pid_cfg if pid_mode == '1' else None,
            ref_cfg=random_cfg if ref_type in ('random', 'legacy_easy', 'profile') else viz_kwargs['ref'].get(ref_type),
            reference_match=reference_match,
        )
        plot_datas(datas, model_name, save_name)
    
    finally:
        env.close()


    

    

        
