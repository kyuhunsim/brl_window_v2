import os
import argparse
import copy
import json
import shutil
from pathlib import Path
import torch

from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_env.env3 import PneuEnv3
from pneu_env.sim3 import PneuSim
from pneu_env.pred3 import PneuPred
from pneu_env.real.real9 import PneuReal
from pneu_rl.sac_with_loss import SAC
from pneu_rl.deployment import load_deployment_config, build_deployment_env, continuous_finetune_config
from pneu_utils.utils import (
    delete_lines,
    color,
    get_pkg_path,
    save_yaml,
    load_yaml,
)


REAL_RETRAIN_PID = dict(
    Kp_act_pos_in=0.0,
    Ki_act_pos_in=0.0,
    Kd_act_pos_in=0.0,
    Kp_act_pos_out=0.0,
    Ki_act_pos_out=0.0,
    Kd_act_pos_out=0.0,
    Kp_act_neg_in=0.001,
    Ki_act_neg_in=0.001,
    Kd_act_neg_in=0.001,
    Kp_act_neg_out=0.001,
    Ki_act_neg_out=0.001,
    Kd_act_neg_out=0.001,
    Ka=1,
)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Fine-tune lib3 in a new model directory.')
    parser.add_argument('--model', help='Source model folder; omitted values are prompted.')
    parser.add_argument('--name', help='New output model folder; the source is preserved.')
    parser.add_argument('--obs', choices=('real', 'sim'))
    parser.add_argument('--pid', choices=('off', 'on'))
    parser.add_argument('--episodes', type=positive_int, default=200)
    parser.add_argument('--update-every', type=positive_int,
                        help='Transitions per update (Real default: 5; Sim: saved setting).')
    parser.add_argument('--load-buffer', action='store_true',
                        help='Continue with the source replay snapshot in the new folder.')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    model_name = args.model
    if model_name is None:
        print(color("[INPUT] Retrain Model:", "blue"))
        models = sorted(os.listdir(f"{get_pkg_path('pneu_rl')}/models"))
        for i, model in enumerate(models):
            print(color(f"\t{i+1}. {model}", "yellow"))
        print(color("\t---", "blue"))
        model_idx = int(input(color("\tMODEL: ", "blue"))) - 1
        model_name = models[model_idx]
        delete_lines(len(models) + 3)
    print(f"[ INFO] Model name: {model_name}")

    retrain_model_name = args.name
    if retrain_model_name is None:
        retrain_model_name = input(color("[INPUT] Retrain Model Name: ", "blue"))
        delete_lines(1)
    if retrain_model_name == "":
        print("[ INFO] Retrain model name is set automatically.")
    else:
        print(f"[ INFO] Retrain model name: {retrain_model_name}")

    if args.obs is None:
        print(color("[INPUT] Observation Mode:", "blue"))
        print(color("\t1. Sim", "yellow"))
        print(color("\t2. Real", "yellow"))
        print(color("\t---", "blue"))
        obs_mode = input(color("\tOBS: ", "blue"))
        delete_lines(5)
    else:
        obs_mode = '2' if args.obs == 'real' else '1'

    kwargs = load_deployment_config(model_name)
    if obs_mode not in ('1', '2'):
        raise ValueError(f'Unknown observation mode: {obs_mode}')
    kwargs = continuous_finetune_config(kwargs)
    if args.update_every is not None:
        kwargs['model']['update_every'] = args.update_every
    elif obs_mode == '2':
        kwargs['model']['update_every'] = 5
    torch.set_num_threads(int(kwargs.get('torch_threads', 2)))

    kwargs["parent_model"] = model_name
    kwargs["retrain_epi"] = args.episodes
    kwargs["type"] = "simulation" if obs_mode == "1" else "real"

    if obs_mode == "2":
        if args.pid is None:
            print(color("[INPUT] PID on for this real retrain?", "blue"))
            print(color("\t1. Yes", "yellow"))
            print(color("\t2. No", "yellow"))
            print(color("\t---", "blue"))
            pid_mode = input(color("\tPID: ", "blue"))
            delete_lines(5)
        else:
            pid_mode = '2' if args.pid == 'off' else '1'
        if pid_mode == "1":
            if (kwargs.get('initialization') or {}).get('mode') == 'pump_on_all_open':
                raise ValueError('Prepared H/HR fine-tuning requires PID OFF')
            kwargs["pid"] = REAL_RETRAIN_PID.copy()
            print("[ INFO] PID: enabled for this real retrain")
        elif pid_mode == "2":
            kwargs["pid"] = None
            print("[ INFO] PID: disabled for this real retrain")
        else:
            raise ValueError(color(f"[ERROR] Unknown PID mode: {pid_mode}", "red"))

    # Preserve reward/CAPS; concatenate the selected model's bank for continuous adaptation.

    # Resolve and validate the reference before constructing any real backend.
    ref = make_pressure_reference(reference_spec_from_model_config(kwargs))
    if obs_mode == "1":
        print("[ INFO] Observation Mode: Simulation")
        obs = PneuSim(**kwargs["obs"])
    elif obs_mode == "2":
        print("[ INFO] Observation Mode: Real")
        obs = PneuReal(**kwargs["obs"])
    else:
        raise ValueError(color(f"[ERROR] Unknown observation mode: {obs_mode}", "red"))

    pred = PneuPred(**kwargs["pred"]) if kwargs["pred"] is not None else None
    env = build_deployment_env(kwargs, obs, ref, pred)

    if kwargs.get("pid") is not None:
        env.set_pid(**kwargs["pid"])
    else:
        print("[ INFO] PID: disabled")

    model = SAC(env, **kwargs["model"])
    model.load(name=model_name, train=True, load_buffer=args.load_buffer)

    retrain_model_name = model.set_retrain(retrain_model_name, load_buffer=args.load_buffer,
                                         copy_buffer=args.load_buffer)
    state_path = Path(model.logger.folder_path) / 'async_state.json'
    if args.load_buffer and state_path.is_file():
        model.total_steps = max(model.total_steps, int(json.loads(state_path.read_text())['steps']))
    if "alpha" in kwargs:
        model.set_alpha(**kwargs["alpha"])
    if "temporal_weight_hardening" in kwargs:
        model.set_temporal_weight_hardening(**kwargs["temporal_weight_hardening"])
    if not args.load_buffer:
        model.clear_buffer()
    print(f'[ INFO] Fine-tuning: episodes={args.episodes}, update_every={model.update_every}, '
          f'batch_size={model.batch_size}, replay={len(model.buffer)}, '
          f'start_episode={model.last_epi + 1}', flush=True)

    # Make the fine-tuned result self-contained for future quick-viz/transfer too.
    saved_cfg = copy.deepcopy(kwargs)
    ref_kwargs = saved_cfg.get('reference', {}).get('kwargs', {})
    if 'trajectory_npz' in ref_kwargs:
        target = Path(model.logger.folder_path) / 'reference.npz'
        source = Path(ref_kwargs['trajectory_npz'])
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        ref_kwargs['trajectory_npz'] = 'reference.npz'
    saved_cfg['finetuning_protocol'] = dict(episodes=args.episodes, fresh_replay=not args.load_buffer,
                                          preparation='session_start_only',
                                          reference='continuous_with_rate_limited_bridges')

    if obs_mode == "2":
        from pneu_rl.async_training import train_async

        kwargs["async_training"] = dict(queue_capacity=4096, sync_every=5, shutdown_timeout=180)
        saved_cfg['async_training'] = kwargs['async_training']
        save_yaml(retrain_model_name, saved_cfg)
        save_yaml(retrain_model_name, saved_cfg, "retrain_cfg.yaml")
        train_async(model, episode=kwargs["retrain_epi"], **kwargs["async_training"])
    else:
        save_yaml(retrain_model_name, saved_cfg)
        save_yaml(retrain_model_name, saved_cfg, "retrain_cfg.yaml")
        try:
            model.train(episode=kwargs["retrain_epi"])
        finally:
            env.close()


if __name__ == "__main__":
    main()
