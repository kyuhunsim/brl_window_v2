"""Hold a bank reference at its 20s value to distinguish lag from persistent bias."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
for p in ('pneu_env','pneu_ref','pneu_rl','pneu_utils'):sys.path.insert(0,str(ROOT/p/'src'))
import numpy as np,pandas as pd,yaml,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_screening import make_components
from evaluate_lib3_screening import reset_to_reference_member,run_episode
from pneu_rl.sac_with_loss import SAC
OUT=ROOT/'docs/img/lib3_reference_start_20261016';RAW=ROOT/'pneu_rl/exp/lib3_reference_start_20261016'
torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
rows=[];fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
for label,name,ep in [('old_HR2000','260831_lib3_histreward2x2_r025_HR_smooth_seed0',2000),('new_HR1500','261015_lib3_allopen30_HR_smooth_seed0',1500),('new_HR2000','261015_lib3_allopen30_HR_smooth_seed0',2000)]:
 cfg=yaml.full_load((ROOT/'pneu_rl/models'/name/'cfg.yaml').read_text());cfg['initialization']={'mode':'pump_on_all_open','precondition_seconds':30};cfg['env']['verbose']=False
 env=make_components(cfg);model=SAC(env,**cfg['model']);model.load_model(str(ROOT/'pneu_rl/models'/name/f'model_epi_{ep:04d}.pth'),evaluate=True)
 try:
  state,_=reset_to_reference_member(env,0);bank=env.goal.ref;original=bank.get_goal;target=original(20)
  bank.get_goal=lambda t:original(min(t,20))
  d=run_episode(env,model,3000,state);d['sensor_time']=d.time;d['time']=(np.arange(len(d))+1)/50
  d.to_csv(RAW/f'{label}_hold60.csv',index=False)
  tail=d[d.time>50]
  np.testing.assert_allclose(tail.ref_neg,target[1]);np.testing.assert_allclose(tail.ref_pos,target[0])
  rows.append(dict(model=label,target_pos=target[0],target_neg=target[1],neg_bias=float((tail.act_neg-tail.ref_neg).mean()),neg_rmse=float(np.sqrt(np.mean((tail.act_neg-tail.ref_neg)**2))),neg_std=float(tail.act_neg.std()),pos_bias=float((tail.act_pos-tail.ref_pos).mean()),negative_reserve=float((tail.act_neg-tail.chamber_neg).mean()),chamber_neg_mean=float(tail.chamber_neg.mean())))
  axes[0].plot(d.time,d.act_neg,label=label);axes[1].plot(d.time,d.act_neg-d.ref_neg,label=label);axes[2].plot(d.time,d.chamber_neg,label=label)
  print(rows[-1],flush=True)
 finally:env.close()
axes[0].plot(d.time,d.ref_neg,'k--',label='reference')
for ax in axes:ax.axvline(20,c='gray',ls=':');ax.grid(alpha=.2);ax.legend()
for ax,title in zip(axes,['Actuator negative pressure (kPa abs)','Negative error (observed - reference, kPa)','Supply negative pressure (kPa abs)']):ax.set_ylabel(title)
axes[-1].set_xlabel('Policy time (s)');fig.tight_layout();fig.savefig(OUT/'hr_constant_hold.png',dpi=150)
pd.DataFrame(rows).to_csv(OUT/'hold_metrics.csv',index=False)
