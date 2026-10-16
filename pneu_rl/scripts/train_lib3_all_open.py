"""Train a fresh H/HR policy from its original config with all-open sim resets."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for package in ('pneu_env','pneu_ref','pneu_rl','pneu_utils'):
    sys.path.insert(0,str(ROOT/package/'src'))
import torch
import yaml
from train_screening import make_components, seed_everything
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import save_yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=['H','HR'],required=True)
    parser.add_argument('--episodes',type=int,default=2000)
    parser.add_argument('--precondition-seconds',type=float,default=30)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--name')
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
    name=args.name or f'261015_lib3_allopen30_{args.policy}_smooth_seed0'
    if '/' in name or name in ('','.','..'): parser.error('name must be a model folder name')
    destination=ROOT/'pneu_rl/models'/name
    if destination.exists() and not args.dry_run:
        parser.error(f'Refusing to overwrite existing model folder: {destination}')
    seed_everything(cfg['train_seed'])
    env=make_components(cfg)
    model=SAC(env=env,**cfg['model'])  # Fresh actor/critics/optimizers/replay; no checkpoint loaded.
    if args.dry_run:
        state,info=env.reset()
        print(json.dumps(dict(dry_run=True,name=name,state_dim=len(state),
            pressure_kpa=env.curr_full_press.tolist(),policy_time=env.t,
            physical_time=env.obs.lib.get_time_c(),episodes=args.episodes,
            horizon=cfg['model']['horizon'],carry_blocks=env.episode_carry_blocks,
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
