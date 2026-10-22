"""Inspect archived calibration data; no fitting, simulation, or hardware I/O."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd


def main():
    archive = Path('/home/rise/test/pneu_tune')
    out = Path('docs/img/lib3_calibration_archive_20261018')
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(p):
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    durations = []
    for p in sorted((archive / 'exp').glob('*.csv')):
        if not any(x in p.name for x in ('260520_', '260521_', '260527_', '10min', '260617_13_14')):
            continue
        d = pd.read_csv(p)
        tc = next((x for x in ('curr_time', 'time') if x in d), None)
        if tc:
            record(p)
            durations.append(dict(file=p.name, rows=len(d), start=d[tc].iloc[0],
                                  end=d[tc].iloc[-1], minutes=(d[tc].iloc[-1]-d[tc].iloc[0])/60))
    pd.DataFrame(durations).to_csv(out / 'durations.csv', index=False)
    metrics = []
    for stamp in ('260520_14_44_14', '260521_17_21_45'):
        rp = archive / f'exp/{stamp}_Flowrate_RND6_suite_joint_tuning.csv'
        sp = archive / f'{stamp}_Flowrate_RND6_suite_joint_tuning_sim3.csv'
        record(rp)
        record(sp)
        real, sim = pd.read_csv(rp), pd.read_csv(sp)
        tr, ts = real['time'], sim.curr_time
        mask = (ts >= tr.min()) & (ts <= tr.max())
        for c in ('press_pos', 'press_neg', 'act_pos_press', 'act_neg_press'):
            # Unprefixed pressure columns in replay CSV are ALSO SIM, not measured data.
            assert np.array_equal(sim[c], sim['sim_' + c])
            y = np.interp(ts[mask], tr, real[c])
            pred = sim.loc[mask, 'sim_' + c].to_numpy()
            e = pred-y
            metrics.append(dict(dataset=stamp, channel=c, n=len(e),
                                rmse_kpa=np.sqrt(np.mean(e**2)),
                                r2=1-np.sum(e**2)/np.sum((y-y.mean())**2),
                                one_minus_mape_pct=100*(1-np.mean(abs(e/y)))))
    pd.DataFrame(metrics).to_csv(out / 'archived_replay_metrics.csv', index=False)
    cpp = Path('pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp')
    fit = archive / 'tune_result/output_12.txt'
    record(cpp)
    record(fit)
    matches = {}
    for name, block in re.findall(r'const ValveModelParams\s+(\w+)\s*=\s*\{([^}]+)', cpp.read_text()):
        old = re.search(r'const ValveModelParams\s+'+name+r'\s*=\s*\{([^}]+)', fit.read_text())
        matches[name] = bool(old and block.split() == old[1].split())
    (out / 'audit.json').write_text(json.dumps(dict(
        sources=sources, source_valves_match_output12=matches,
        limitations=['Source parameter equality does not prove historical .so equality.',
                     'Archived replay binary and parameters are not version-locked.',
                     'No fresh simulator execution; interpolation uses overlapping recorded timestamps.',
                     'Recorded duration is not independently verified wall-clock interaction time.',
                     'No held-out status is inferred from file names.']), indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == '__main__':
    main()
