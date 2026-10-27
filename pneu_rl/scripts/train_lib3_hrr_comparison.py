"""Matched HRR baseline / training-only dynamics randomization, from fresh SAC.

The saved obs/env fields always describe nominal evaluation/deployment. Only this
runner reads training_dynamics_randomization; quick-viz/Real never sample it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for package in ('pneu_env', 'pneu_ref', 'pneu_rl', 'pneu_utils'):
    sys.path.insert(0, str(ROOT / package / 'src'))

import numpy as np
import torch
import yaml

from train_screening import make_components, seed_everything
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import save_yaml

SOURCE = ROOT / 'pneu_rl/models/261026_lib3_pressurecal_type1_HR_smooth_seed0/cfg.yaml'
EXPERIMENT = ROOT / 'pneu_rl/exp/lib3_hrr_dr_20261026'
DR_PROFILE = ROOT / 'pneu_rl/config/domain_randomization/lib3_type1_v34_v1.yaml'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def build_config(variant, *, episodes=2000, threads=2, library=None):
    cfg = yaml.safe_load(SOURCE.read_text())
    if cfg['lib'] != 3 or cfg['env']['physics_feature_mode'] != 'rate':
        raise ValueError('Expected the saved recalibrated Type1 HR source')
    if cfg['initialization']['mode'] != 'pump_on_all_open':
        raise ValueError('Expected prepared resets')
    provenance = cfg['plant_provenance']
    for path, digest in ((cfg['obs']['library_path'], provenance['simulator_library_sha256']),
                         (ROOT / cfg['reference']['kwargs']['trajectory_npz'],
                          provenance['reference_bank_sha256'])):
        if sha256(path) != digest:
            raise ValueError(f'Source library/reference hash changed: {path}')
    cfg['env']['physics_feature_mode'] = 'rate_reserve'
    cfg['env']['verbose'] = False
    cfg['variant'] = 'history_action_rate_reserve'
    cfg['episodes'] = int(episodes)
    cfg['torch_threads'] = int(threads)
    cfg['source_config'] = str(SOURCE.relative_to(ROOT))
    cfg['model']['checkpoint_interval'] = 500
    cfg['comparison'] = {
        'variant': variant,
        'source_config_sha256': sha256(SOURCE),
        'fresh_actor_critics_optimizers_replay': True,
        'expected_state_dim': 162,
        'deployment': 'Standard HRR, nominal library/reference; no Real randomization.',
        'reference_handling': 'Same nominal reference bank, no per-scenario rebasing. Handover pressure differences are logged.',
    }
    if variant == 'dr':
        if library is None:
            raise ValueError('DR requires an explicitly built isolated library')
        library = Path(library).resolve()
        build_manifest = json.loads((library.parent / 'build.json').read_text())
        if (sha256(library) != build_manifest['library_sha256'] or
                build_manifest['original_library_sha256'] != provenance['simulator_library_sha256']):
            raise ValueError('Randomized plant does not match the nominal source')
        cfg['training_dynamics_randomization'] = yaml.safe_load(DR_PROFILE.read_text())
        cfg['training_dynamics_randomization'].update(
            library_path=str(library), library_sha256=sha256(library),
            profile_sha256=sha256(DR_PROFILE),
            build_manifest_sha256=sha256(library.parent / 'build.json'),
        )
    elif variant == 'baseline':
        cfg.pop('training_dynamics_randomization', None)
    else:
        raise ValueError(f'Unknown variant: {variant}')
    return cfg


def make_training_components(cfg, *, scenario_log=None):
    """Use the training-only subclass without changing the shared factory."""
    dr = copy.deepcopy(cfg.get('training_dynamics_randomization'))
    if not dr or not dr.get('enabled'):
        return make_components(cfg)
    from pneu_env.env3_randomized import PneuEnv3Randomized
    from pneu_env.sim3 import PneuSim
    from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
    if cfg.get('pred') is not None or cfg.get('pid') is not None:
        raise ValueError('This comparison uses pure SAC without predictor or PID')
    obs_cfg = dict(cfg['obs'], library_path=dr['library_path'])
    if sha256(obs_cfg['library_path']) != dr['library_sha256']:
        raise ValueError('Randomized library changed')
    if scenario_log is not None:
        dr['log_path'] = str(scenario_log)
    obs = PneuSim(**obs_cfg)
    env = PneuEnv3Randomized(
        obs=obs,
        ref=make_pressure_reference(reference_spec_from_model_config(cfg)),
        pred=None,
        precondition_seconds=cfg['initialization']['precondition_seconds'],
        dynamics_randomization=dr,
        **cfg['env'],
    )
    env.action_space.seed(cfg['train_seed'])
    return env


def policy_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_gate(path, cfg):
    report = json.loads(Path(path).read_text())
    if not report.get('passed'):
        raise ValueError('Preflight report did not pass')
    if report['source_config_sha256'] != sha256(SOURCE):
        raise ValueError('Source config changed after preflight')
    if report['dr_profile_sha256'] != sha256(DR_PROFILE):
        raise ValueError('DR profile changed after preflight')
    for path, expected in report['implementation_hashes'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Implementation changed after preflight: {path}')
    if cfg.get('training_dynamics_randomization'):
        if report['randomized_library_sha256'] != cfg['training_dynamics_randomization']['library_sha256']:
            raise ValueError('Randomized library changed after preflight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'dr'], required=True)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--name')
    parser.add_argument('--library', type=Path, default=EXPERIMENT / 'build/lib3_type1_randomized.so')
    parser.add_argument('--validation-report', type=Path, default=EXPERIMENT / 'preflight.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.threads < 1:
        parser.error('episodes and threads must be positive')
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    name = args.name or f'261026_lib3_pressurecal_type1_HRR{"_DR" if args.variant == "dr" else ""}_smooth_seed0'
    if Path(name).name != name or name in ('.', '..'):
        parser.error('name must be one model folder name')
    destination = ROOT / 'pneu_rl/models' / name
    if destination.exists() and not args.dry_run:
        parser.error(f'Refusing to overwrite existing model: {destination}')
    cfg = build_config(args.variant, episodes=args.episodes, threads=args.threads, library=args.library)
    if not args.dry_run:
        validate_gate(args.validation_report, cfg)
    seed_everything(cfg['train_seed'])
    env = make_training_components(
        cfg, scenario_log=None if args.dry_run else destination / 'dynamics_randomization.jsonl')
    model = SAC(env=env, **cfg['model'])
    if env.observation_space.shape != (162,):
        raise RuntimeError(f'Unexpected HRR shape: {env.observation_space.shape}')
    initial_hash = policy_hash(model)
    if args.dry_run:
        try:
            state, info = env.reset()
            if not np.isfinite(state).all():
                raise RuntimeError('Invalid prepared state')
            print(json.dumps(dict(variant=args.variant, state_dim=len(state),
                initial_actor_sha256=initial_hash, pressure_kpa=env.curr_full_press.tolist(),
                reference_kpa=env._current_actuator_reference().tolist(),
                policy_time=env.t, native_time=env.obs.lib.get_time_c(),
                carry_blocks=env.episode_carry_blocks,
                randomization=info.get('dynamics_randomization'),
                episodes=args.episodes, threads=args.threads), indent=2))
        finally:
            env.close()
        return
    model.set_logger(name)
    save_yaml(name, cfg)
    write_json(destination / 'comparison_manifest.json', {
        **cfg['comparison'], 'initial_actor_sha256': initial_hash,
        'validation_report': str(args.validation_report),
        'validation_report_sha256': sha256(args.validation_report),
        'training_dynamics_randomization': cfg.get('training_dynamics_randomization'),
    })
    started = time.time()
    timing = dict(started_unix=started, episodes_requested=args.episodes, completed=False,
                  torch_threads=args.threads, variant=args.variant, initialization=cfg['initialization'])
    write_json(destination / 'wall_time.json', timing)
    print(json.dumps(dict(started_model=name, state_dim=162, episodes=args.episodes,
                         initial_actor_sha256=initial_hash, threads=args.threads)), flush=True)
    try:
        model.train(episode=args.episodes)
        timing['completed'] = True
    finally:
        timing.update(finished_unix=time.time(), elapsed_seconds=time.time() - started,
                      policy_steps=model.total_steps)
        write_json(destination / 'wall_time.json', timing)
        env.close()


if __name__ == '__main__':
    main()
