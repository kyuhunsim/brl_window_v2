"""Diagnostic only: measured-flow balance under current lib3 assumptions.

No identification, time-lag correction, flow clipping, or model changes.
This cannot distinguish sensor units/offsets/timing, volume, temperature or leaks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-root", type=Path, default=Path("/home/rise/test/pneu_tune"))
    parser.add_argument("--output-dir", type=Path, default=Path("../pneu_tune/exp/lib3_all_valves_feasible_20261022"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, examples, sources = [], [], {}
    rho, gas_r, temperature = 1.20411831637462, .287, 323.15
    for kind in ("type1", "type2"):
        source = next((args.tune_root/"exp").glob(f"261022*{kind}*.csv"))
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        d = pd.read_csv(source)
        t = d.time.to_numpy()
        for side, inlet, outlet, volume in (("pos", 3, 4, .156), ("neg", 5, 6, .1)):
            pressure = d[f"act_{side}_press"].to_numpy()
            q = d[f"flow{inlet}"].to_numpy()-d[f"flow{outlet}"].to_numpy()
            factor = 1000*gas_r*temperature/volume*rho/60000
            integrated = cumulative_trapezoid(q, t, initial=0)*factor
            for window in (1., 2., 5.):
                starts = np.arange(5, t[-1]-window, window)
                ends = starts+window
                predicted = np.interp(ends, t, integrated)-np.interp(starts, t, integrated)
                observed = np.interp(ends, t, pressure)-np.interp(starts, t, pressure)
                error = predicted-observed
                rows.append(dict(type=kind, side=side, window_sec=window, n=len(error),
                                 deltaP_rmse_kpa=np.sqrt(np.mean(error**2)),
                                 deltaP_bias_kpa=error.mean(),
                                 real_deltaP_rms_kpa=np.sqrt(np.mean(observed**2))))
            selection = (t >= 10) & (t < 15)
            examples.append(dict(type=kind, side=side, start=10, end=15,
                                 mean_flow_in=float(d.loc[selection, f"flow{inlet}"].mean()),
                                 mean_flow_out=float(d.loc[selection, f"flow{outlet}"].mean()),
                                 pressure_first=float(pressure[selection][0]),
                                 pressure_last=float(pressure[selection][-1])))
    pd.DataFrame(rows).to_csv(args.output_dir/"measured_flow_balance_diagnostic.csv", index=False)
    pd.DataFrame(examples).to_csv(args.output_dir/"measured_flow_balance_examples.csv", index=False)
    (args.output_dir/"measured_flow_balance_audit.json").write_text(json.dumps(dict(
        sources=sources, assumptions=dict(R_kpa_l_per_kg_k=gas_r, T_k=temperature,
            density_kg_m3=rho, volumes_l=dict(pos=.156, neg=.1),
            flows="Assumed common standard L/min; not verified against sensor specifications",
            timing="CSV recorded time; no sensor-lag alignment"),
        limitations=["Residual is not proof of a valve-model error or a physical conservation violation.",
                     "Unmetered leaks and other mass paths are omitted, as in current lib3.",
                     "Sensor units, channel mapping, sampling age and effective volumes require verification."]
    ), indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
