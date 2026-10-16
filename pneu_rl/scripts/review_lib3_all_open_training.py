"""Reproduce frozen-checkpoint evaluation; no hardware or training updates."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
for package in ('pneu_env','pneu_ref','pneu_rl','pneu_utils'):
    sys.path.insert(0,str(ROOT/package/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from evaluate_lib3_screening import build_model_env, reset_to_reference_member, run_episode
OUT=ROOT/'docs/img/lib3_all_open_training_review_20261016'
OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'pneu_rl/exp/lib3_all_open_training_review_20261016'
RAW.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
np.random.seed(0)
rows=[]
audit={}
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
for policy,color in [('H','tab:blue'),('HR','tab:orange')]:
    name=f'261015_lib3_allopen30_{policy}_smooth_seed0'
    folder=ROOT/'pneu_rl/models'/name
    data=pd.read_csv(folder/'train_diagnostics.csv')
    assert len(data)==2000 and data.epi.tolist()==list(range(1,2001))
    assert np.isfinite(data[['reward','pos_rmse','neg_rmse']]).all().all()
    audit[policy]={'wall_time':json.loads((folder/'wall_time.json').read_text()),'checkpoints':{},'training_windows':{}}
    for start in (901,1401,1901):
        w=data[(data.epi>=start)&(data.epi<start+100)]
        audit[policy]['training_windows'][str(start)]={k:float(w[k].mean()) for k in ['reward','pos_rmse','neg_rmse']}
    for ax,col in zip(axes,['reward','pos_rmse','neg_rmse']):
        ax.plot(data.epi,data[col],color=color,alpha=.12,lw=.5)
        ax.plot(data.epi,data[col].rolling(100,min_periods=100).mean(),color=color,label=policy+' 100-episode mean')
        ax.set_ylabel(col);ax.grid(alpha=.2);ax.legend()
    for episode in (1000,1500,2000):
        path=folder/f'model_epi_{episode:04d}.pth'
        audit[policy]['checkpoints'][str(episode)]=hashlib.sha256(path.read_bytes()).hexdigest()
        cfg,env,model=build_model_env(name,checkpoint_episode=episode)
        try:
            for member in range(16):
                state,_=reset_to_reference_member(env,member)
                df=run_episode(env,model,2000,state=state)
                assert len(df)==2000 and abs(env.t-40)<1e-9
                assert np.isfinite(df[['act_pos','act_neg','ref_pos','ref_neg','chamber_pos','chamber_neg']]).all().all()
                df['sensor_time']=df.time
                df['time']=(np.arange(len(df))+1)/50
                df['reference_time']=np.arange(len(df))/50
                df.to_csv(RAW/f'{policy}_{episode}_bank{member:02d}.csv',index=False)
                for window,part in [('whole',df),('first5',df.iloc[:250]),('after5',df.iloc[250:])]:
                    errors=part[['act_pos','act_neg']].to_numpy()-part[['ref_pos','ref_neg']].to_numpy()
                    a=part[[f'ctrl{i}_a' for i in range(1,7)]].to_numpy()
                    r=cfg['env']['rwd_kwargs']
                    row=dict(policy=policy,episode=episode,member=member,window=window,
                        rmse=float(np.sqrt(np.mean(errors**2))),pos_rmse=float(np.sqrt(np.mean(errors[:,0]**2))),neg_rmse=float(np.sqrt(np.mean(errors[:,1]**2))),
                        pos_bias=float(errors[:,0].mean()),neg_bias=float(errors[:,1].mean()),
                        pos_ch_max=float(part.chamber_pos.max()),neg_ch_min=float(part.chamber_neg.min()),
                        boundary_fraction=float(np.mean((part.chamber_pos>r.get('chamber_pos_max_kpa',220))|(part.chamber_pos<r.get('chamber_pos_min_kpa',101.325))|(part.chamber_neg<r.get('chamber_neg_min_kpa',10))|(part.chamber_neg>r.get('chamber_neg_max_kpa',101.325)))),
                        near_action_bound_fraction=float(np.mean((a<=.01)|(a>=.99))),
                        physical_action_delta_mean=float(np.abs(np.diff(part[[f'ctrl{i}' for i in range(1,7)]].to_numpy(),axis=0)).mean()))
                    rows.append(row)
                print(policy,episode,'bank',member,'RMSE',round(rows[-3]['rmse'],3),flush=True)
                pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        finally: env.close()
axes[-1].set_xlabel('Episode');fig.suptitle('Training diagnostics (stochastic policy; 512-step blocks)');fig.tight_layout();fig.savefig(OUT/'training.png',dpi=150);plt.close(fig)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
metrics=pd.DataFrame(rows)
summary=metrics.groupby(['policy','episode','window']).agg(pooled_rmse=('rmse',lambda s:np.sqrt(np.mean(s**2))),worst_member_rmse=('rmse','max'),boundary_fraction=('boundary_fraction','mean'),near_action_bound_fraction=('near_action_bound_fraction','mean'),physical_action_delta_mean=('physical_action_delta_mean','mean')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True)
for row,policy in enumerate(('H','HR')):
    for episode in (1000,1500,2000):
        df=pd.read_csv(RAW/f'{policy}_{episode}_bank00.csv')
        for col,channel in enumerate(('pos','neg')):
            axes[row,col].plot(df.time,df[f'act_{channel}'],label=str(episode),lw=.9)
    for col,channel in enumerate(('pos','neg')):
        axes[row,col].plot(df.time,df[f'ref_{channel}'],'k--',label='reference',lw=1.2)
        axes[row,col].set_title(f'{policy} bank0 {channel}');axes[row,col].set_ylabel('kPa absolute');axes[row,col].legend();axes[row,col].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'bank0_tracking.png',dpi=150);plt.close(fig)
print(summary.to_string(index=False),flush=True)
