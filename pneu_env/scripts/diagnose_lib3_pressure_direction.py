"""Isolated applied-pressure-direction experiment; no parameter fitting.

Copy C++ source to /tmp and change only applied flow at nonpositive pressure
head. Normal model sources, existing libraries, and fitted results are preserved.
The prototype intentionally retains lag states, and reports gate switching at
the native integration scale. This is a diagnostic, not a deployment command.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
TUNE_EXP=ROOT.parent/'pneu_tune/exp'
sys.path.insert(0,str(ROOT/'pneu_env/src'))
from pneu_env.pump_replay import NativeLib3, load_trajectory, replay, channel_metrics, CHANNEL_NAMES, PRESSURE_NAMES, OBJECTIVE_WEIGHTS

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(kind):
    started=time.perf_counter()
    baseline_dir=TUNE_EXP/'lib3_pump_feasible_20261022'/kind
    report_path=baseline_dir/'report.json'
    report=json.loads(report_path.read_text())
    coeff=np.array(report['coeff_micro'])
    raw_path=Path(report['datasets'][0]['path'])
    baseline_csv=baseline_dir/'replay_1.csv'
    original=ROOT/'pneu_env/src/pneu_env/lib3'
    header=TUNE_EXP/f'lib3_feasible_integration_20261022/{kind}/valve_parameters.h'
    lib0=Path(json.loads((baseline_dir/'config.json').read_text())['library_path'])
    protected=[*original.glob('*.cpp'),original/'pneumatic_simulator.h',report_path,baseline_csv,header,lib0,raw_path]
    before={str(p):sha(p) for p in protected}
    temporary=Path(tempfile.mkdtemp(prefix=f'lib3_direction_gate_{kind}_'))
    for name in ('pneumatic_simulator.h','pneumatic_simulator.cpp','pneumatic_CT.cpp'):
        shutil.copy2(original/name,temporary/name)
    copied=temporary/'pneumatic_CT.cpp'
    text=copied.read_text()
    anchor='double PneumaticCT::solenoid_valve('
    assert text.count(anchor)==1
    instrumentation='''
namespace {
unsigned long long diagnostic_switches[6] = {};
unsigned long long diagnostic_blocked[6] = {};
bool diagnostic_last_gate[6] = {};
bool diagnostic_seen[6] = {};
}
extern "C" void diagnostic_get_counters(unsigned long long* out) {
    for(int i=0;i<6;++i) { out[i]=diagnostic_switches[i]; out[i+6]=diagnostic_blocked[i]; }
}
'''
    text=text.replace(anchor,instrumentation+'\n'+anchor)
    old='const double q_pred_lpm = std::max(state->x1, 0.0);'
    assert text.count(old)==1
    gate='''const bool flow_allowed = P_inlet > P_outlet;
    const int diagnostic_index = static_cast<int>(type) - 1;
    if (diagnostic_index >= 0 && diagnostic_index < 6) {
        if (diagnostic_seen[diagnostic_index] && flow_allowed != diagnostic_last_gate[diagnostic_index])
            ++diagnostic_switches[diagnostic_index];
        diagnostic_seen[diagnostic_index] = true;
        diagnostic_last_gate[diagnostic_index] = flow_allowed;
        if (!flow_allowed && state->x1 > 0.1) ++diagnostic_blocked[diagnostic_index];
    }
    const double q_pred_lpm = flow_allowed ? std::max(state->x1, 0.0) : 0.0;'''
    text=text.replace(old,gate)
    copied.write_text(text)
    library=temporary/'lib3_direction_gate.so'
    command=['g++','-O3','-std=c++17','-shared','-fPIC','-DPNEU_EXACT_VALVE_HYSTERESIS','-DPNEU_FAST_VALVE_FLOW',
             f'-DPNEU_VALVE_PARAMS_HEADER="{header}"',str(temporary/'pneumatic_simulator.cpp'),str(copied),'-o',str(library)]
    subprocess.run(command,check=True)
    class CountingNative(NativeLib3):
        def __init__(self,path):
            super().__init__(path)
            self.lib.diagnostic_get_counters.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
            self.lib.diagnostic_get_counters.restype=None
            self.last=np.zeros(12,dtype=np.uint64)
            self.counts=[]
        def step(self,control,dt):
            result=super().step(control,dt)
            now=np.empty(12,dtype=np.uint64)
            self.lib.diagnostic_get_counters(now.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)))
            self.counts.append(now-self.last)
            self.last=now
            return result
    sim=CountingNative(library)
    trajectory=load_trajectory(raw_path)
    times,prediction,truth=replay(sim,trajectory,coeff)
    baseline=pd.read_csv(baseline_csv)
    assert np.max(abs(times-baseline.time.to_numpy()))<1e-8
    for i,name in enumerate(CHANNEL_NAMES):
        assert np.max(abs(truth[:,i]-baseline['real_'+name].to_numpy()))<1e-8
    previous=np.column_stack([baseline['sim_'+n] for n in CHANNEL_NAMES])
    dt=.02
    def domain(values):
        ans={}
        for j,n in enumerate(PRESSURE_NAMES):
            p=values[:,j]
            ans[n]=dict(min=float(p.min()),max=float(p.max()),negative_samples=int(sum(p<0)),
                        negative_duration_sec=float(sum(p<0)*dt),min_time=float(times[np.argmin(p)]))
        return ans
    def wrong_direction(values):
        p=values[:,:4]
        heads=np.column_stack((p[:,0]-101.325,101.325-p[:,1],p[:,0]-p[:,2],
                               p[:,2]-101.325,101.325-p[:,3],p[:,3]-p[:,1]))
        ans={}
        for i in range(6):
            mask=(heads[:,i]<=0)&(values[:,i+4]>.1)
            ans[f'flow{i+1}']=dict(duration_sec=float(sum(mask)*dt),
                        max_lpm=float(values[mask,i+4].max()) if mask.any() else 0.,
                        max_nonpositive_head_kpa=float(-heads[mask,i].min()) if mask.any() else 0.)
        return ans,heads
    count=np.array(sim.counts,dtype=np.uint64)
    switches=count[:,:6]
    switching={f'v{i+1}':dict(total=int(switches[:,i].sum()),
                 maximum_per_20ms=int(switches[:,i].max()),
                 frames_over_10_switches=int(sum(switches[:,i]>10)),
                 blocked_effective_sec=float(count[:,i+6].sum()*.000025)) for i in range(6)}
    output=TUNE_EXP/'lib3_pump_feasible_20261022/pressure_direction_diagnostic'
    output.mkdir(exist_ok=True,parents=True)
    raw_output=output/'traces'
    raw_output.mkdir(exist_ok=True,parents=True)
    fields={'time':times,'source_time':times+trajectory.source_time[0]}
    for j,name in enumerate(CHANNEL_NAMES):
        fields['real_'+name]=truth[:,j];fields['baseline_'+name]=previous[:,j];fields['gated_'+name]=prediction[:,j]
    for i in range(6):fields[f'gate_switches{i+1}']=switches[:,i]
    csv_path=raw_output/f'{kind}.csv'
    pd.DataFrame(fields).to_csv(csv_path,index=False)
    before_metrics=channel_metrics(previous,truth);after_metrics=channel_metrics(prediction,truth)
    before_wrong,_=wrong_direction(previous);after_wrong,heads=wrong_direction(prediction)
    result=dict(kind=kind,experiment='Applied one-way flow only: Pin<=Pout -> mdot=0; internal lag state retained.',
                no_pressure_clip=True,no_new_fitted_parameter=True,coefficients_fixed_micro=coeff.tolist(),
                baseline_path=str(baseline_csv),baseline_library=str(lib0),temporary_source=str(copied),
                diagnostic_script=str(Path(__file__).resolve()),diagnostic_script_sha256=sha(Path(__file__).resolve()),
                patched_source_sha256=sha(copied),compile_command=command,
                raw_csv=str(csv_path),all_results_finite=bool(np.isfinite(prediction).all()),
                before_pressure_domain=domain(previous),after_pressure_domain=domain(prediction),
                before_metrics=before_metrics,after_metrics=after_metrics,
                before_objective=sum(w*before_metrics[n]['mae'] for n,w in OBJECTIVE_WEIGHTS.items()),
                after_objective=sum(w*after_metrics[n]['mae'] for n,w in OBJECTIVE_WEIGHTS.items()),
                before_endpoint_wrong_direction_flow=before_wrong,after_endpoint_wrong_direction_flow=after_wrong,
                native_gate_switches=switching,
                cautions=['No refit. Diagnostic prototype only; not adopted into production.',
                          'Flow is sampled at end of interval but pressure includes RK weighted state; tiny endpoint head mismatch may remain.',
                          'Wrong-direction endpoint statistics use flow>0.1 L/min for diagnostic visibility; this is not an acceptance threshold.',
                          'Gate toggle counts span RK stages, not independent hardware oscillations.',
                          'Valve lag parameters were fitted without this direction guard.',
                          'All data used for fitting; not held-out transfer validation.'],
                protected_hashes=before,protected_files_unchanged=all(sha(Path(p))==h for p,h in before.items()),
                elapsed_sec=time.perf_counter()-started)
    (output/f'{kind}.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(5,2,figsize=(16,15),sharex=True)
    for j,(name,ax) in enumerate(zip(CHANNEL_NAMES,axes.flat)):
        ax.plot(times,truth[:,j],lw=.7,color='black',label='Real')
        ax.plot(times,previous[:,j],lw=.65,alpha=.65,label='Current fit')
        ax.plot(times,prediction[:,j],lw=.7,alpha=.8,label='Direction gate, same parameters')
        ax.set_title(f"{name}: RMSE {before_metrics[name]['rmse']:.2f} -> {after_metrics[name]['rmse']:.2f}")
        if j<4:ax.axhline(0,c='red',lw=.7,ls=':')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[-1,0].set_xlabel('Elapsed time (s)');axes[-1,1].set_xlabel('Elapsed time (s)')
    fig.suptitle(f'{kind}: physical flow-direction diagnostic, frozen coefficients')
    fig.tight_layout();fig.savefig(output/f'{kind}.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(13,8),sharex=True)
    for i,(j,name) in enumerate(((2,'act_pos_press'),(6,'flow3'),(7,'flow4'))):
        axs[i].plot(times,truth[:,j],color='black',label='Real',lw=.8)
        axs[i].plot(times,previous[:,j],label='Current fit',lw=.8)
        axs[i].plot(times,prediction[:,j],label='Gate, same parameters',lw=.8)
        axs[i].set_ylabel(name);axs[i].legend();axs[i].grid(alpha=.2)
    axs[0].axhline(101.325,c='gray',lw=.6,ls='--');axs[0].axhline(0,c='red',lw=.6,ls=':')
    axs[-1].set(xlim=(100,132),xlabel='Elapsed time (s)')
    fig.tight_layout();fig.savefig(output/f'{kind}_closure.png',dpi=140);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('kind','before_pressure_domain','after_pressure_domain','native_gate_switches','before_objective','after_objective','protected_files_unchanged','elapsed_sec')},indent=2))


def summarize_baseline_feasibility():
    results={}
    for kind in ('type1','type2'):
        directory=TUNE_EXP/'lib3_pump_feasible_20261022'/kind
        path=directory/'replay_1.csv'
        if not path.exists() or not (directory/'report.json').exists():
            continue
        frame=pd.read_csv(path)
        dt=float(np.median(np.diff(frame.time)))
        pressure=np.column_stack([frame['sim_'+n] for n in PRESSURE_NAMES])
        heads=np.column_stack((pressure[:,0]-101.325,101.325-pressure[:,1],
                               pressure[:,0]-pressure[:,2],pressure[:,2]-101.325,
                               101.325-pressure[:,3],pressure[:,3]-pressure[:,1]))
        domains={}
        for i,name in enumerate(PRESSURE_NAMES):
            values=pressure[:,i]
            ids=np.flatnonzero(values<0)
            splits=np.r_[0,np.flatnonzero(np.diff(ids)>1)+1,len(ids)]
            intervals=[]
            for lo,hi in zip(splits[:-1],splits[1:]):
                subset=ids[lo:hi]
                if len(subset):
                    intervals.append(dict(start_sec=float(frame.time.iloc[subset[0]]),
                                          end_sec=float(frame.time.iloc[subset[-1]]),
                                          duration_sec=float(len(subset)*dt),
                                          min_kpa=float(values[subset].min())))
            domains[name]=dict(min_kpa=float(values.min()),max_kpa=float(values.max()),
                              min_time_sec=float(frame.time.iloc[np.argmin(values)]),
                              negative_samples=len(ids),negative_duration_sec=float(len(ids)*dt),
                              negative_intervals=intervals)
        wrong={}
        for i in range(6):
            flow=frame[f'sim_flow{i+1}'].to_numpy()
            mask=(heads[:,i]<=0)&(flow>.1)
            wrong[f'v{i+1}']=dict(samples=int(mask.sum()),duration_sec=float(mask.sum()*dt),
                                 max_lpm=float(flow[mask].max()) if mask.any() else 0.,
                                 max_nonpositive_head_kpa=float(-heads[mask,i].min()) if mask.any() else 0.)
        all_values=frame[[f'sim_{n}' for n in CHANNEL_NAMES]].to_numpy()
        results[kind]=dict(replay_path=str(path),replay_sha256=sha(path),
                           fit_coeff_micro=json.loads((directory/'report.json').read_text())['coeff_micro'],
                           all_finite=bool(np.isfinite(all_values).all()),
                           positive_absolute_pressure_pass=bool((pressure>0).all()),
                           pressure_domain=domains,endpoint_wrong_direction_flow=wrong)
    result=dict(datasets=results,
                interpretation='Whole-model pressure-domain feasibility is separate from optimizer convergence and isolated valve parity.',
                endpoint_wrong_direction_definition='Pin<=Pout and output flow>0.1 L/min; sample visibility threshold only.',
                numerical_caution='Output flow is from the final RK stage; endpoint pressure uses the weighted RK update.',
                pressure_units='absolute kPa',flow_units='L/min')
    destination=TUNE_EXP/'lib3_pump_feasible_20261022/whole_model_feasibility.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type',choices=('type1','type2','all'),default='all')
    parser.add_argument('--baseline-summary-only',action='store_true')
    args=parser.parse_args()
    if not args.baseline_summary_only:
        for kind in (('type1','type2') if args.type=='all' else (args.type,)):
            main(kind)
    summarize_baseline_feasibility()
