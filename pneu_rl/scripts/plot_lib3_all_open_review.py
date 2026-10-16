from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'docs/img/lib3_all_open_training_review_20261016'
RAW=ROOT/'pneu_rl/exp/lib3_all_open_training_review_20261016'
m=pd.read_csv(P/'metrics.csv')
fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
for row,pol in enumerate(['H','HR']):
 d=pd.read_csv(RAW/f'{pol}_2000_bank00.csv')
 ax=axs[row,0];ax.plot(d.time,d.chamber_pos,label='Supply +');ax.plot(d.time,d.chamber_neg,label='Supply -');ax.plot(d.time,d.act_pos,label='Actuator +');ax.plot(d.time,d.act_neg,label='Actuator -');ax.set_title(pol+' 2000 bank0 pressures');ax.set_ylabel('kPa absolute');ax.legend();ax.grid(alpha=.2)
 ax=axs[row,1]
 for i in range(1,7):ax.plot(d.time,d[f'ctrl{i}'],label=f'Valve {i}',lw=.7)
 ax.set_title(pol+' 2000 physical commands');ax.set_ylim(.79,1.01);ax.legend(ncol=3);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(P/'bank0_supply_commands.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,window in zip(axs,['whole','first5','after5']):
 for pol in ['H','HR']:
  d=m[(m.window==window)&(m.policy==pol)]
  values=d.groupby('episode').rmse.apply(lambda s:np.sqrt(np.mean(s**2)))
  ax.plot(values.index,values.values,'o-',label=pol)
 ax.set_title(window);ax.set_ylabel('Pooled RMSE (kPa)');ax.set_xlabel('Checkpoint episode');ax.set_xticks([1000,1500,2000]);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(P/'checkpoint_comparison.png',dpi=150);plt.close(fig)
# Plot the worst bank of each final policy, selected by combined RMSE.
fig,axs=plt.subplots(2,2,figsize=(13,7),sharex=True)
for row,pol in enumerate(['H','HR']):
 sub=m[(m.policy==pol)&(m.episode==2000)&(m.window=='whole')]
 member=int(sub.loc[sub.rmse.idxmax(),'member']);d=pd.read_csv(RAW/f'{pol}_2000_bank{member:02d}.csv')
 for col,ch in enumerate(['pos','neg']):
  ax=axs[row,col];ax.plot(d.time,d[f'ref_{ch}'],'k--',label='reference');ax.plot(d.time,d[f'act_{ch}'],label=pol);ax.set_title(f'{pol} 2000 worst bank {member}, {ch}');ax.set_ylabel('kPa absolute');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(P/'worst_member_tracking.png',dpi=150)
