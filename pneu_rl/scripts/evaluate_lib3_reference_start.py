"""Frozen-policy reference-start ablation, simulation only; never edits models."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
for p in ('pneu_env','pneu_ref','pneu_rl','pneu_utils'):sys.path.insert(0,str(ROOT/p/'src'))
import numpy as np,pandas as pd,yaml,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_screening import make_components
from evaluate_lib3_screening import reset_to_reference_member,run_episode
from pneu_rl.sac_with_loss import SAC
OUT=ROOT/'docs/img/lib3_reference_start_20261016';OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'pneu_rl/exp/lib3_reference_start_20261016';RAW.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
cases=[('old_HR2000','260831_lib3_histreward2x2_r025_HR_smooth_seed0',2000),('new_H2000','261015_lib3_allopen30_H_smooth_seed0',2000),('new_HR1500','261015_lib3_allopen30_HR_smooth_seed0',1500),('new_HR2000','261015_lib3_allopen30_HR_smooth_seed0',2000)]
rows=[];audit={}
for label,name,ep in cases:
 folder=ROOT/'pneu_rl/models'/name;cfg=yaml.full_load((folder/'cfg.yaml').read_text())
 cfg['initialization']={'mode':'pump_on_all_open','precondition_seconds':30};cfg['env']['verbose']=False
 env=make_components(cfg);model=SAC(env,**cfg['model']);checkpoint=folder/f'model_epi_{ep:04d}.pth'
 model.load_model(str(checkpoint),evaluate=True)
 bank=env.goal.ref;original=bank.get_goal
 audit[label]={'checkpoint':str(checkpoint.relative_to(ROOT)),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
 try:
  for member in (0,5):
   initial=None;baseline=None
   for duration in (0,2,5):
    bank.get_goal=original
    state,_=reset_to_reference_member(env,member)
    physical=env.curr_full_press.copy();hist=env.obs_traj.copy();ahist=env.action_history.copy();buf=np.asarray(env.obs.obs_buf).copy()
    if initial is None:initial=(physical,hist,ahist,buf)
    else:
     for a,b in zip(initial,(physical,hist,ahist,buf)):np.testing.assert_array_equal(a,b)
    start=physical[2:4].copy()
    if duration:
     def blended(t,T=duration,p0=start):
      r=np.asarray(original(t));z=np.clip(float(t)/T,0,1);w=10*z**3-15*z**4+6*z**5
      return tuple((1-w)*p0+w*r)
     bank.get_goal=blended
     # Keep PneuRef's standard constant initial history/preview padding.
     # Only its value changes; subsequent future samples use the blended curve.
     env.goal.buf=np.tile(start,env.goal.num_ref)
     env.prev_ref_current=start.copy()
     state,_=env._state_from_current(env.last_policy_action)
     np.testing.assert_allclose(bank.get_goal(0),start,atol=1e-10)
     for t in (duration,duration+.3,39):np.testing.assert_allclose(bank.get_goal(t),original(t),atol=1e-10)
    df=run_episode(env,model,2000,state=state);df['sensor_time']=df.time;df['time']=(np.arange(len(df))+1)/50
    assert abs(env.t-40)<1e-9
    if baseline is None:baseline=df.copy()
    else:np.testing.assert_allclose(df[['ref_pos','ref_neg']].iloc[275:],baseline[['ref_pos','ref_neg']].iloc[275:],atol=1e-10)
    df.to_csv(RAW/f'{label}_bank{member}_blend{duration}.csv',index=False)
    rate=np.gradient(df.ref_neg.to_numpy(),.02)
    settled=(pd.Series(np.abs(rate)<.2).rolling(25,min_periods=25).sum()==25).to_numpy() & (df.time.to_numpy()>6)
    for window,mask in [('whole',np.ones(len(df),bool)),('first5',df.time.to_numpy()<=5),('after6',df.time.to_numpy()>6),('neg_plateau',settled)]:
     d=df.loc[mask];e=d[['act_pos','act_neg']].to_numpy()-d[['ref_pos','ref_neg']].to_numpy()
     rows.append(dict(model=label,member=member,blend_seconds=duration,window=window,samples=len(d),combined_rmse=float(np.sqrt(np.mean(e**2))),neg_rmse=float(np.sqrt(np.mean(e[:,1]**2))),neg_bias=float(e[:,1].mean()),pos_rmse=float(np.sqrt(np.mean(e[:,0]**2))),neg_reserve_mean=float((d.act_neg-d.chamber_neg).mean()),ch_neg_min=float(d.chamber_neg.min()),physical_action_delta=float(np.abs(np.diff(d[[f'ctrl{i}' for i in range(1,7)]],axis=0)).mean()) if window!='neg_plateau' else np.nan))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    print(label,member,duration,'after6',rows[-2]['combined_rmse'],'neg bias',rows[-2]['neg_bias'],flush=True)
 finally:env.close()
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
for member in (0,5):
 fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True)
 for row,(label,_,_) in enumerate(cases):
  for dur in (0,2,5):
   d=pd.read_csv(RAW/f'{label}_bank{member}_blend{dur}.csv')
   for col,ch in enumerate(('pos','neg')):
    axes[row,col].plot(d.time,d[f'act_{ch}'],label=f'blend {dur}s',lw=.9)
    axes[row,col].plot(d.time,d[f'ref_{ch}'],ls='--',lw=.7,alpha=.5)
  for col,ch in enumerate(('pos','neg')):
   axes[row,col].set_title(label+' '+ch);axes[row,col].set_ylabel('kPa abs');axes[row,col].legend();axes[row,col].grid(alpha=.2)
 fig.suptitle(f'Bank {member}: solid=pressure, dashed=corresponding reference');fig.tight_layout();fig.savefig(OUT/f'bank{member}_comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
for label,_,_ in cases:
 if label=='new_H2000':continue
 d=pd.read_csv(RAW/f'{label}_bank0_blend0.csv')
 axes[0].plot(d.time,d.act_neg,label=label);axes[1].plot(d.time,d.act_neg-d.ref_neg,label=label)
axes[0].plot(d.time,d.ref_neg,'k--',label='reference')
for ax in axes:ax.legend();ax.grid(alpha=.2)
axes[0].set_ylabel('Negative pressure, kPa abs');axes[1].set_ylabel('Observed - ref, kPa');axes[1].set_xlabel('Policy elapsed seconds');fig.tight_layout();fig.savefig(OUT/'old_new_hr.png',dpi=150)
