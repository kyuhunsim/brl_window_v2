"""Bounded, isolated SAC continuation from old HR; simulation only."""
from pathlib import Path
import sys,time,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
for p in ('pneu_env','pneu_ref','pneu_rl','pneu_utils'):sys.path.insert(0,str(ROOT/p/'src'))
import numpy as np,pandas as pd,yaml,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_screening import make_components,seed_everything
from evaluate_lib3_screening import reset_to_reference_member,run_episode
from pneu_rl.sac_with_loss import SAC
from pneu_utils.utils import save_yaml
NAME='261016_lib3_oldHR_allopen_short20_seed0'
DEST=ROOT/'pneu_rl/models'/NAME
OUT=ROOT/'docs/img/lib3_hr_short_finetune_20261016'
RAW=ROOT/'pneu_rl/exp/lib3_hr_short_finetune_20261016'
SOURCE=ROOT/'pneu_rl/models/260831_lib3_histreward2x2_r025_HR_smooth_seed0'
checkpoint=SOURCE/'model_epi_2000.pth'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if DEST.exists():raise FileExistsError(f'Refusing overwrite: {DEST}')
OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);seed_everything(0)
cfg=yaml.full_load((SOURCE/'cfg.yaml').read_text())
cfg['initialization']={'mode':'pump_on_all_open','precondition_seconds':30}
cfg['env']['verbose']=False;cfg['model']['start_epi']=0;cfg['model']['checkpoint_interval']=5;cfg['episodes']=20;cfg['torch_threads']=2
cfg['continuation']={'source_checkpoint':str(checkpoint.relative_to(ROOT)),'source_sha256':sha(checkpoint),'replay':'fresh; 2048 stochastic source-policy steps before updates','restored':'actor, critic, target critic, actor/critic optimizers, log_alpha','entropy_optimizer':'fresh; absent from source checkpoint','warmup_fixed_action':'disabled','additional_episodes':20}
env=make_components(cfg);model=SAC(env,**cfg['model']);model.load_model(str(checkpoint),evaluate=False)
initial={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
assert len(model.buffer)==0
model.set_logger(NAME);save_yaml(NAME,cfg);model.save_model(str(DEST/'model_epi_0000.pth'))
updates=[0];update=model.update_parameters
def counted():
 update();updates[0]+=1
model.update_parameters=counted
record={'source':cfg['continuation'],'started_unix':time.time(),'completed':False}
(DEST/'wall_time.json').write_text(json.dumps(record,indent=2))
try:
 model.warmup_buffer(2048,deterministic=False)
 assert updates[0]==0 and len(model.buffer)==2048
 env.request_hard_reset()
 model.train(episode=20)
 model.buffer.save_buffer(model.logger.buffer_path)
 record.update(completed=True,finished_unix=time.time(),updates=updates[0],policy_steps=model.total_steps,replay_size=len(model.buffer),actor_parameter_delta_l2=float(np.sqrt(sum(float(((v-initial[k])**2).sum()) for k,v in model.policy.state_dict().items()))))
 assert updates[0]>0 and record['actor_parameter_delta_l2']>0
 assert all(bool(torch.isfinite(v).all()) for v in model.policy.state_dict().values())
finally:
 env.close();record['elapsed_seconds']=time.time()-record['started_unix'];(DEST/'wall_time.json').write_text(json.dumps(record,indent=2))
assert sha(checkpoint)==cfg['continuation']['source_sha256']
rows=[]
for ep in (0,5,10,20):
 env=make_components(cfg);policy=SAC(env,**cfg['model']);policy.load_model(str(DEST/f'model_epi_{ep:04d}.pth'),evaluate=True)
 try:
  for bank_id in (range(16) if ep in (0,20) else (0,5)):
   state,_=reset_to_reference_member(env,bank_id);d=run_episode(env,policy,2000,state)
   d['sensor_time']=d.time;d['time']=(np.arange(len(d))+1)/50
   assert abs(env.t-40)<1e-9
   d.to_csv(RAW/f'ep{ep}_bank{bank_id}.csv',index=False)
   for window,part in [('whole',d),('first5',d.iloc[:250]),('after6',d.iloc[300:])]:
    err=part[['act_pos','act_neg']].to_numpy()-part[['ref_pos','ref_neg']].to_numpy()
    rows.append(dict(ep=ep,bank=bank_id,window=window,rmse=float(np.sqrt(np.mean(err**2))),neg_bias=float(err[:,1].mean()),neg_rmse=float(np.sqrt(np.mean(err[:,1]**2))),pos_rmse=float(np.sqrt(np.mean(err[:,0]**2))),boundary_fraction=float(((part.chamber_pos<101.325)|(part.chamber_pos>220)|(part.chamber_neg<10)|(part.chamber_neg>101.325)).mean()),command_delta=float(np.abs(np.diff(part[[f'ctrl{i}' for i in range(1,7)]],axis=0)).mean())))
   pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
   print('EVAL',ep,bank_id,rows[-3]['rmse'],flush=True)
 finally:env.close()
# Same 60s constant-target diagnostic used in preceding analysis.
holds=[]
for ep in (0,20):
 env=make_components(cfg);policy=SAC(env,**cfg['model']);policy.load_model(str(DEST/f'model_epi_{ep:04d}.pth'),evaluate=True)
 try:
  state,_=reset_to_reference_member(env,0);original=env.goal.ref.get_goal
  env.goal.ref.get_goal=lambda t:original(min(t,20))
  d=run_episode(env,policy,3000,state);d['sensor_time']=d.time;d['time']=(np.arange(len(d))+1)/50
  d.to_csv(RAW/f'ep{ep}_hold60.csv',index=False)
  tail=d.iloc[-500:];e=tail.act_neg-tail.ref_neg
  holds.append(dict(ep=ep,neg_bias=float(e.mean()),neg_rmse=float(np.sqrt(np.mean(e**2))),neg_std=float(tail.act_neg.std()),target_neg=float(tail.ref_neg.iloc[-1])))
 finally:env.close()
pd.DataFrame(holds).to_csv(OUT/'hold_metrics.csv',index=False)
metrics=pd.DataFrame(rows)
summary=metrics.groupby(['ep','window']).agg(banks=('bank','count'),pooled_rmse=('rmse',lambda x:np.sqrt(np.mean(x*x))),worst_rmse=('rmse','max'),neg_bias=('neg_bias','mean'),boundary_fraction=('boundary_fraction','mean'),command_delta=('command_delta','mean')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for row,bank_id in enumerate((0,5)):
 for ep in (0,5,10,20):
  d=pd.read_csv(RAW/f'ep{ep}_bank{bank_id}.csv')
  for col,ch in enumerate(('pos','neg')):axes[row,col].plot(d.time,d[f'act_{ch}'],label=f'+{ep} epi',lw=.8)
 for col,ch in enumerate(('pos','neg')):
  axes[row,col].plot(d.time,d[f'ref_{ch}'],'k--',label='reference');axes[row,col].set_title(f'Bank{bank_id} {ch}');axes[row,col].set_ylabel('kPa abs');axes[row,col].legend();axes[row,col].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'tracking.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
for ep in (0,20):
 d=pd.read_csv(RAW/f'ep{ep}_hold60.csv');axes[0].plot(d.time,d.act_neg,label=f'+{ep} epi');axes[1].plot(d.time,d.act_neg-d.ref_neg,label=f'+{ep} epi')
axes[0].plot(d.time,d.ref_neg,'k--',label='reference')
for ax in axes:ax.legend();ax.grid(alpha=.2)
axes[0].set_ylabel('kPa absolute');axes[1].set_ylabel('Negative error (kPa)');axes[1].set_xlabel('Policy seconds');fig.tight_layout();fig.savefig(OUT/'hold.png',dpi=150);plt.close(fig)
print(summary.to_string(index=False),flush=True);print(holds,flush=True)
(OUT/'run.json').write_text(json.dumps(record,indent=2));assert sha(checkpoint)==cfg['continuation']['source_sha256']
