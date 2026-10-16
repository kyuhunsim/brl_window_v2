"""Offline command replay and apparent lag scan; never opens hardware files for writing."""
from pathlib import Path
import sys,json,hashlib
from ctypes import c_double,POINTER
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
for pkg in ('pneu_env','pneu_utils'):sys.path.insert(0,str(ROOT/pkg/'src'))
from pneu_env.sim3 import PneuSim


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/img/lib3_command_replay_20261016')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.csv);t=d.curr_time.to_numpy();t-=t[0]
    ps=['press_pos','press_neg','act_pos_press','act_neg_press']
    cs=['ctrl_ch_pos','ctrl_ch_neg','ctrl_act_pos_in','ctrl_act_pos_out','ctrl_act_neg_in','ctrl_act_neg_out']
    u=d[cs].to_numpy();real=d[ps].to_numpy()
    assert (np.diff(t)>0).all() and np.isfinite(real).all() and np.isfinite(u).all()
    assert u.min()>=.8-1e-6 and u.max()<=1+1e-6
    # A 1ms grid preserves wall duration; commands held until the next logged command.
    grid=np.arange(0,t[-1]+.001,.001)
    records=[];curves=[];replays={}
    for mode in ['warm_sim','measured_pressure_reset']:
        kwargs=dict(freq=50,delay=0,noise=False,scale=False)
        if mode=='measured_pressure_reset':
            kwargs.update(dict(zip(['init_pos_press','init_neg_press','init_act_pos_press','init_act_neg_press'],real[0])))
        sim=PneuSim(**kwargs)
        initial=real[0].copy()
        if mode=='warm_sim':
            for _ in range(1500):
                obs,_=sim.observe(np.ones(6))
            initial=obs[1:5].copy()
        lib=sim.lib
        replay=np.empty((len(grid),4));replay[0]=initial
        indices=np.searchsorted(t,grid[:-1],side='right')-1
        for j,idx in enumerate(indices):
            replay[j+1]=list(lib.step_c((c_double*6)(*u[idx]),.001+1e-12)[1:5])
        assert np.isfinite(replay).all()
        replays[mode]=replay
        frame=pd.DataFrame(replay,columns=['sim_'+p for p in ps]);frame.insert(0,'time',grid);frame.to_csv(args.output/(mode+'.csv'),index=False)
        # Fixed common support for all lags. Slow trends removed only for correlation.
        sample=np.arange(1,39,.02)
        y=np.column_stack([np.interp(sample,t,real[:,k]) for k in range(4)])
        yp=y-pd.DataFrame(y).rolling(51,center=True,min_periods=1).mean().to_numpy()
        for k,p in enumerate(ps):
            for lo,hi in [(6,20),(20,38),(6,38)]:
                mask=(sample>=lo)&(sample<hi)
                candidates=[]
                for lag in np.arange(-.3,.30001,.002):
                    pred=np.interp(sample-lag,grid,replay[:,k]);hp=pred-pd.Series(pred).rolling(51,center=True,min_periods=1).mean().to_numpy()
                    corr=float(np.corrcoef(yp[mask,k],hp[mask])[0,1]);rmse=float(np.sqrt(np.mean((y[mask,k]-pred[mask])**2)))
                    row=dict(mode=mode,channel=p,start=lo,end=hi,lag_s=float(lag),correlation=corr,rmse=rmse)
                    curves.append(row);candidates.append(row)
                best=max(candidates,key=lambda r:r['correlation']);zero=min(candidates,key=lambda r:abs(r['lag_s']));delay100=min(candidates,key=lambda r:abs(r['lag_s']-.1))
                records.append(dict(best,zero_corr=zero['correlation'],zero_rmse=zero['rmse'],delay100_rmse=delay100['rmse']))
        print(mode,'completed',flush=True)
    pd.DataFrame(records).to_csv(args.output/'lag_summary.csv',index=False)
    pd.DataFrame(curves).to_csv(args.output/'lag_curves.csv',index=False)
    fig,axes=plt.subplots(4,2,figsize=(14,11))
    for k,p in enumerate(ps):
        for col,(lo,hi) in enumerate([(0,40),(20,22)]):
            ax=axes[k,col];mask=(t>=lo)&(t<=hi);ax.plot(t[mask],real[mask,k],label='Real',color='black',lw=1)
            for mode,replay in replays.items():
                m=(grid>=lo)&(grid<=hi);ax.plot(grid[m],replay[m,k],label=mode,alpha=.8,lw=.8)
            ax.set_title(p);ax.set_ylabel('kPa');ax.grid(alpha=.2)
    axes[0,0].legend();axes[-1,0].set_xlabel('Command timestamp (s)');axes[-1,1].set_xlabel('Command timestamp (s)');fig.tight_layout();fig.savefig(args.output/'replay.png',dpi=140)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    frame=pd.DataFrame(curves)
    for ax,p in zip(axes.flat,ps):
        for lo,hi in [(6,20),(20,38)]:
            z=frame[(frame.channel==p)&(frame['start']==lo)&(frame['end']==hi)&(frame['mode']=='measured_pressure_reset')];ax.plot(z.lag_s*1000,z.correlation,label=f'{lo}-{hi}s')
        ax.axvline(100,color='grey',ls='--');ax.set_title(p);ax.set_xlabel('Apparent real lag relative to Sim (ms)');ax.set_ylabel('High-pass correlation');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'lag_scan.png',dpi=140)
    (args.output/'audit.json').write_text(json.dumps(dict(source=str(args.csv),sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),command_mapping=cs,sim_library_sha256=hashlib.sha256((ROOT/'pneu_env/src/pneu_env/lib3/libpneumatic_simulator.so').read_bytes()).hexdigest(),grid_dt=.001,positive_lag='Real(t) matches Sim(t-lag)',limitations=['EXP uses command timestamps, not physical sensor timestamps','Returned controls are assumed to represent applied commands; RT implementation unavailable','Closed-loop periodic excitation makes correlation lag ambiguous','Measured pressure initialization does not identify hidden pump/valve state']),indent=2))
    print(pd.DataFrame(records).query('start == 6 and end == 38').to_string(index=False))

if __name__=='__main__':main()
