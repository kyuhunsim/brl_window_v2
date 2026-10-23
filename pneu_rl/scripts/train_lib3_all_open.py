"""Train a fresh H/HR policy from its original config with all-open sim resets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for package in ('pneu_env','pneu_ref','pneu_rl','pneu_utils'):
    sys.path.insert(0,str(ROOT/package/'src'))
import torch
import numpy as np
import yaml
from train_screening import make_components, seed_everything
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import save_yaml


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_plant_reference(parser, args, cfg):
    """Require matching simulator/reference provenance for a new plant run."""
    if (args.library_path is None) != (args.reference_profile is None):
        parser.error('--library-path and --reference-profile must be supplied together')
    if args.library_path is None:
        return
    library = args.library_path.resolve()
    profile_path = args.reference_profile.resolve()
    if not library.is_file() or not profile_path.is_file():
        parser.error('simulator library and reference profile must both exist')
    profile = yaml.safe_load(profile_path.read_text())
    if not isinstance(profile, dict):
        parser.error('reference profile must be a YAML mapping')
    design = profile.get('design', {})
    if design.get('library') != 3:
        parser.error('reference profile must target lib3')
    library_hash = sha256(library)
    if design.get('simulator_library_sha256') != library_hash:
        parser.error('reference was not certified with the selected simulator library')
    if not np.allclose(design.get('action_scale', []),
                       [cfg['obs']['scale_low'], cfg['obs']['scale_high']]):
        parser.error('reference and training action scales differ')
    initialization = design.get('initialization', {})
    if initialization.get('mode') != 'pump_on_all_open' or not np.isclose(
            initialization.get('precondition_seconds', -1), args.precondition_seconds):
        parser.error('reference and training preparation conditions differ')
    if (design.get('block_steps') != cfg['model']['horizon'] or
            design.get('blocks_per_hard_reset') != cfg['env']['episode_carry_blocks']):
        parser.error('reference duration and training carry/horizon settings differ')
    trajectory = (ROOT / profile['kwargs']['trajectory_npz']).resolve()
    if not trajectory.is_file():
        parser.error(f'reference bank does not exist: {trajectory}')
    trajectory_hash = sha256(trajectory)
    if design.get('training_npz_sha256') != trajectory_hash:
        parser.error('reference bank differs from its certified profile')
    with np.load(trajectory) as bank:
        pressures = np.asarray(bank['pressure'])
        if set(bank.files) != {'time', 'pressure'} or not np.isfinite(pressures).all():
            parser.error('reference bank must contain only finite time/pressure targets')
        if not np.allclose(pressures[:, 0, :],
                           initialization['policy_start_observation_kpa'][2:4], atol=1e-3):
            parser.error('reference bank does not start at its certified handover pressure')
    # Keep one reference definition, and save the selected physical model explicitly.
    cfg['obs']['library_path'] = str(library)
    cfg['reference'] = profile
    cfg.pop('rnd_ref', None)
    cfg['reference_kind'] = 'profile'
    cfg['reference_profile_source'] = str(profile_path)
    cfg['plant_provenance'] = dict(
        calibration_type=design.get('calibration_type'),
        simulator_library=str(library), simulator_library_sha256=library_hash,
        reference_profile_sha256=sha256(profile_path),
        reference_bank=str(trajectory), reference_bank_sha256=trajectory_hash,
        augmented_state_schema=2,
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=['H','HR'],required=True)
    parser.add_argument('--episodes',type=int,default=2000)
    parser.add_argument('--precondition-seconds',type=float,default=30)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--name')
    parser.add_argument('--library-path',type=Path)
    parser.add_argument('--reference-profile',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.episodes<1 or args.threads<1: parser.error('episodes and threads must be positive')
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    source=ROOT/f'pneu_rl/models/260831_lib3_histreward2x2_r025_{args.policy}_smooth_seed0/cfg.yaml'
    cfg=yaml.safe_load(source.read_text())
    cfg['initialization']=dict(mode='pump_on_all_open',precondition_seconds=args.precondition_seconds,
        reference_clock='policy_step/freq; preserved across carried blocks')
    cfg['source_config']=str(source.relative_to(ROOT))
    cfg['episodes']=args.episodes
    cfg['torch_threads']=args.threads
    cfg['model']['checkpoint_interval']=500
    cfg['env']['verbose']=False
    apply_plant_reference(parser,args,cfg)
    name=args.name or f'261015_lib3_allopen30_{args.policy}_smooth_seed0'
    if '/' in name or name in ('','.','..'): parser.error('name must be a model folder name')
    destination=ROOT/'pneu_rl/models'/name
    if destination.exists() and not args.dry_run:
        parser.error(f'Refusing to overwrite existing model folder: {destination}')
    seed_everything(cfg['train_seed'])
    env=make_components(cfg)
    if args.library_path is not None and env.obs.runtime_state_schema != 2:
        env.close()
        parser.error('physical baseline requires opening-dynamics simulator schema 2')
    model=SAC(env=env,**cfg['model'])  # Fresh actor/critics/optimizers/replay; no checkpoint loaded.
    if args.dry_run:
        state,info=env.reset()
        if not np.isfinite(state).all() or np.any(env.curr_full_press <= 0):
            raise RuntimeError('Invalid initial policy state or absolute pressure')
        if args.reference_profile is not None:
            expected=cfg['reference']['design']['initialization']['policy_start_observation_kpa']
            if not np.allclose(env.curr_full_press,expected,atol=0.02):
                raise RuntimeError('Training reset does not reproduce reference preparation')
        print(json.dumps(dict(dry_run=True,name=name,state_dim=len(state),
            pressure_kpa=env.curr_full_press.tolist(),policy_time=env.t,
            physical_time=env.obs.lib.get_time_c(),episodes=args.episodes,
            horizon=cfg['model']['horizon'],carry_blocks=env.episode_carry_blocks,
            runtime_state_schema=env.obs.runtime_state_schema,
            plant_provenance=cfg.get('plant_provenance'),
            initial_reference_kpa=env._current_actuator_reference().tolist(),
            checkpoint_interval=500),indent=2))
        env.close()
        return
    model.set_logger(name)
    save_yaml(name,cfg)
    started=time.time()
    start_clock=time.perf_counter()
    timing=dict(started_unix=started,episodes_requested=args.episodes,completed=False,
                torch_threads=args.threads,initialization=cfg['initialization'])
    (destination/'wall_time.json').write_text(json.dumps(timing,indent=2)+'\n')
    try:
        model.train(episode=args.episodes)
        timing['completed']=True
    finally:
        env.close()
        timing.update(finished_unix=time.time(),elapsed_seconds=time.perf_counter()-start_clock,
                      policy_steps=model.total_steps)
        (destination/'wall_time.json').write_text(json.dumps(timing,indent=2)+'\n')
    print(f'Completed {name} in {timing["elapsed_seconds"]/3600:.2f} hours',flush=True)

if __name__=='__main__':
    main()
