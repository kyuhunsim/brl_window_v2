"""SIM ONLY: precondition with all six valves open, then evaluate frozen H/HR.

No plant/time reset occurs at policy handover. Reference time is explicitly
policy-step/50; simulator and delayed sensor timestamps continue uninterrupted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from ctypes import POINTER, c_double, c_int
import numpy as np
import pandas as pd
import yaml
from analyze_lib3_initial_pressure_counterfactual import MODELS, ROOT, digest, metrics, resolve_checkpoint

OUTPUT = ROOT / 'pneu_rl/exp/lib3_all_open_start_20261015'


def physical_state(obs):
    """Read all pump/pressure/valve states without advancing or resetting them."""
    if not hasattr(obs.lib, 'get_augmented_state_dim_c'):
        return None  # Older binary supports continuation but not full-state export.
    obs.lib.get_augmented_state_dim_c.restype = c_int
    obs.lib.get_augmented_state_c.argtypes = [POINTER(c_double)]
    obs.lib.get_augmented_state_c.restype = None
    state = (c_double * obs.lib.get_augmented_state_dim_c())()
    obs.lib.get_augmented_state_c(state)
    return np.asarray(state).copy()


def prepare_all_open(env, seconds):
    """Advance only the simulator; preserve its delay buffer and hidden state."""
    steps = round(seconds * env.obs.freq)
    if steps < env.num_obs + round(env.obs.delay * env.obs.freq):
        raise ValueError('Preparation must fill pressure history and sensor delay buffer')
    if env.dim_act != 6 or env.is_pid or not env.obs.scale or env.obs.scale_high != 1:
        raise ValueError('Requires six direct actions, PID off, scale_high=1')
    action = np.ones(6)
    history = []
    rows = []
    for i in range(steps):
        observed, info = env.obs.observe(action.copy(), np.full(2, 101.325))
        applied = [info['Observation'][key] for key in ('ctrl_pos','ctrl_neg','ctrl_act_pos_in','ctrl_act_pos_out','ctrl_act_neg_in','ctrl_act_neg_out')]
        np.testing.assert_array_equal(applied, np.ones(6))
        history.append(observed[1:5].copy())
        rows.append(dict(physical_elapsed_sec=(i+1)/env.obs.freq,
                         sensor_time=float(observed[0]),
                         **dict(zip(['chamber_pos','chamber_neg','act_pos','act_neg'], history[-1]))))
    history = np.asarray(history)
    debug_before = env.obs.get_valve_debug().copy()
    physical_before = physical_state(env.obs)
    time_before = env.obs.lib.get_time_c()
    buffer_before = np.asarray(env.obs.obs_buf).copy()
    env.obs_traj = history[-env.num_obs:].copy()
    env.curr_full_press = env.obs_traj[-1].copy()
    env.prev_full_press = env.obs_traj[-2].copy()
    env.action_history = np.ones((env.action_history_len, 6))
    env.prev_action = action.copy()
    env.prev_prev_action = action.copy()
    env.last_policy_action = action.copy()
    env.has_episode_state = True
    env.t = 0.0
    env.prev_ref_current = env.goal.buf.reshape(-1, env.dim_ref)[env.num_obs-1].copy()
    state, _ = env._state_from_current(action)
    assert debug_before == env.obs.get_valve_debug()
    if physical_before is not None:
        np.testing.assert_array_equal(physical_before, physical_state(env.obs))
    assert time_before == env.obs.lib.get_time_c()
    np.testing.assert_array_equal(buffer_before, np.asarray(env.obs.obs_buf))
    np.testing.assert_array_equal(env.obs_traj, history[-env.num_obs:])
    return state, pd.DataFrame(rows), debug_before


def worker(args):
    import torch
    from pneu_env.env3 import PneuEnv3
    from pneu_env.sim3 import PneuSim
    from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
    from pneu_rl.sac_with_loss import SAC
    from evaluate_lib3_screening import reset_to_reference_member, _row_from_info
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    model_dir = ROOT/'pneu_rl/models'/MODELS[args.model]
    checkpoint = resolve_checkpoint(model_dir, args.model)
    config = model_dir/'cfg.yaml'
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in (config,checkpoint)}
    cfg = yaml.full_load(config.read_text())
    assert cfg['lib']==3 and cfg['obs']['freq']==50
    assert cfg['env'].get('action_chunk_steps',1)==1 and cfg.get('pid') is None and cfg.get('pred') is None
    obs = PneuSim(**cfg['obs'])
    env_cfg = dict(cfg['env'], verbose=False)
    env = PneuEnv3(obs=obs, ref=make_pressure_reference(reference_spec_from_model_config(cfg)), pred=None, **env_cfg)
    model = SAC(env=env, **cfg['model'])
    model.load_model(path=str(checkpoint), evaluate=True)
    state, _ = reset_to_reference_member(env, 0)
    folder = args.output_root/args.model/f'prepare_{args.seconds:g}s'
    folder.mkdir(parents=True,exist_ok=True)
    prep = None
    if args.seconds:
        state, prep, _ = prepare_all_open(env,args.seconds)
        prep.to_csv(folder/'preparation.csv',index=False)
    exported_state = physical_state(obs)
    initial = dict(pressure_kpa=env.curr_full_press.tolist(),
                   augmented_physical_state=None if exported_state is None else exported_state.tolist(),
                   simulator_physical_time_sec=float(obs.lib.get_time_c()),
                   pressure_history=env.obs_traj.tolist(), action_history=env.action_history.tolist(),
                   valve_debug={k:float(v) for k,v in obs.get_valve_debug().items()},
                   delayed_sensor_buffer=np.asarray(obs.obs_buf).tolist(), policy_state=np.asarray(state).tolist())
    rows=[]
    try:
        for i in range(2000):
            # Shared reference clock across cold and preconditioned cases.
            # Do not call simulator time_reset: it would reset valve states.
            env.t=i/50.0
            action=model.predict(state)
            state,reward,_,_,info=env.step(action)
            row=_row_from_info(env,info,float(reward),i)
            row['control_elapsed_sec']=(i+1)/50.0
            row['reference_query_sec']=i/50.0
            rows.append(row)
    finally:
        env.close()
    frame=pd.DataFrame(rows)
    assert np.isfinite(frame[['act_pos','act_neg','chamber_pos','chamber_neg','ref_pos','ref_neg']].to_numpy()).all()
    frame.to_csv(folder/'rollout.csv',index=False)
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in (config,checkpoint)}
    result=dict(model=args.model, preparation_seconds=args.seconds, initial=initial,
                input_sha256=hashes, reference_member=0, control_steps=2000,
                simulator_sha256=digest(Path(obs.lib._name)),
                frequency_hz=50, metrics={
                    'whole':metrics(frame),'first_5_sec':metrics(frame[frame.control_elapsed_sec<=5]),
                    'after_5_sec':metrics(frame[frame.control_elapsed_sec>5])},
                clock='Reference queried at policy step/50, separately from uninterrupted delayed sensor time. Same for every case.',
                scope='Frozen selected 2000-episode checkpoint, nominal lib3, bank0 only. No real hardware or learning. Full physical state retained at handover, no snapshot restore.',
                preparation_drift=None)
    if prep is not None:
        cols=['chamber_pos','chamber_neg','act_pos','act_neg']
        last=prep[prep.physical_elapsed_sec>args.seconds-5][cols]
        previous=prep[prep.physical_elapsed_sec.between(args.seconds-10,args.seconds-5,inclusive='right')][cols]
        result['preparation_drift']=dict(last5_mean_kpa=last.mean().tolist(),last5_std_kpa=last.std(ddof=0).tolist(),
                                          last5_minus_previous5_mean_kpa=(last.mean()-previous.mean()).tolist())
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(args.model,args.seconds,result['metrics']['whole'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=OUTPUT)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--model',choices=list(MODELS))
    p.add_argument('--seconds',type=float,default=30)
    args=p.parse_args()
    os.chdir(ROOT)
    args.output_root=args.output_root.resolve()
    if args.worker:
        worker(args)
        return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/brl_all_open_mpl')
    env['PYTHONPATH']=':'.join(str(ROOT/x/'src') for x in ['pneu_env','pneu_ref','pneu_rl','pneu_utils'])+':'+env.get('PYTHONPATH','')
    results=[]
    for model in MODELS:
        for seconds in (0,30,60):
            subprocess.run([sys.executable,__file__,'--worker','--model',model,'--seconds',str(seconds),'--output-root',str(args.output_root)],check=True,env=env)
            folder=args.output_root/model/f'prepare_{seconds}s'
            results.append(json.loads((folder/'summary.json').read_text()))
    frames=[pd.read_csv(args.output_root/r['model']/f"prepare_{r['preparation_seconds']:g}s"/'rollout.csv') for r in results]
    for frame in frames[1:]:
        np.testing.assert_array_equal(frame[['ref_pos','ref_neg']],frames[0][['ref_pos','ref_neg']])
    for duration in (30,60):
        pair=[r for r in results if r['preparation_seconds']==duration]
        assert pair[0]['initial']['pressure_kpa']==pair[1]['initial']['pressure_kpa']
        assert pair[0]['initial']['valve_debug']==pair[1]['initial']['valve_debug']
    (args.output_root/'summary.json').write_text(json.dumps(dict(results=results,identical_reference_verified=True),indent=2)+'\n')
    write_artifacts(args.output_root, results, frames)
    print('Completed six frozen-policy simulations; reference equality and matched preparation verified.')


def write_artifacts(output_root, results, frames):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/brl_all_open_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ROOT/'docs/img/lib3_all_open_start_20261015'
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(results=results,identical_reference_verified=True),indent=2)+'\n')
    flat=[]
    for result in results:
        for window, values in result['metrics'].items():
            flat.append(dict(model=result['model'], preparation_seconds=result['preparation_seconds'],window=window,**values))
    pd.DataFrame(flat).to_csv(out/'metrics.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True,constrained_layout=True)
    for column,model in enumerate(MODELS):
        for result,frame in zip(results,frames):
            if result['model']!=model or result['preparation_seconds']==60:
                continue
            for row,side in enumerate(('pos','neg')):
                ax=axes[row,column]
                if result['preparation_seconds']==0:
                    ax.plot(frame.control_elapsed_sec,frame[f'ref_{side}'],'k--',label='Reference')
                ax.plot(frame.control_elapsed_sec,frame[f'act_{side}'],label=f"{result['preparation_seconds']:g}s all-open preparation")
                ax.set_ylabel(f'{side.upper()} pressure (kPa abs)')
                ax.grid(alpha=.2)
        axes[0,column].set_title(model)
        axes[0,column].legend(fontsize=8)
        axes[1,column].set_xlabel('Policy elapsed time (s)')
    fig.savefig(out/'tracking.png',dpi=160)
    plt.close(fig)
    prep=pd.read_csv(output_root/'H_smooth/prepare_60s/preparation.csv')
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for col in ('chamber_pos','chamber_neg','act_pos','act_neg'):
        ax.plot(prep.physical_elapsed_sec,prep[col],label=col)
    ax.axvline(30,color='black',linestyle='--',linewidth=.8)
    ax.set_xlabel('All-open preparation time (s)')
    ax.set_ylabel('Delayed measured pressure (kPa abs)')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(out/'preparation.png',dpi=160)
    plt.close(fig)

if __name__=='__main__':
    main()
