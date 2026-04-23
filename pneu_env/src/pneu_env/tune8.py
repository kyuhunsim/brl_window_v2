import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from tuner8 import PneuSimTuner8


from utils.utils import get_pkg_path


DEFAULT_INITIAL_GUESS = [1.1256394620423595, 5.401279325612009]
DEFAULT_MAXITER = 10000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="exp csv base names or csv paths")
    ap.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER, help="optimizer max iterations")
    ap.add_argument("--maxfev", type=int, default=None, help="optimizer max function evaluations (optional)")
    ap.add_argument("--init", nargs=2, type=float, default=DEFAULT_INITIAL_GUESS, help="initial guess: C_IN C_OUT")
    ap.add_argument("--tail-sec", type=float, default=None, help="use only last N seconds of each exp csv (rebased to t=0)")
    ap.add_argument("--clip-start-sec", type=float, default=None, help="drop data before this time (sec)")
    ap.add_argument("--clip-end-sec", type=float, default=None, help="drop data after this time (sec)")
    ap.add_argument("--quiet", action="store_true", help="reduce per-iteration prints")
    ap.add_argument("--print-every", type=int, default=1, help="print every N iters (only when not --quiet)")
    ap.add_argument("--no-verify", action="store_true", help="skip verificate() (faster)")
    ap.add_argument("--fast", action="store_true", help="quick sanity run (maxiter=200, tail-sec=60, no-verify, quiet)")
    ap.add_argument("--ctrl-domain", choices=["unit", "bipolar"], default="unit", help="control domain stored in real csv")
    ap.add_argument("--sim-scale", action="store_true", help="apply sim8 scale mode during replay")
    ap.add_argument("--freq", type=float, default=50.0, help="sim8 replay frequency [Hz]")
    ap.add_argument(
        "--real-flow-cols",
        nargs=2,
        default=["flowrate2", "flowrate5"],
        help="real csv flow columns to match (default: flowrate2 flowrate5)",
    )
    ap.add_argument(
        "--sim-flow-keys",
        nargs=2,
        default=["pump_out", "pump_in"],
        help="sim8 flow keys to match (default: pump_out pump_in)",
    )
    args = ap.parse_args()

    maxiter = int(args.maxiter)
    initial_guess = [float(args.init[0]), float(args.init[1])]
    tail_sec = args.tail_sec
    verbose = not bool(args.quiet)
    print_every = int(args.print_every) if int(args.print_every) > 0 else 1
    no_verify = bool(args.no_verify)

    if args.fast:
        maxiter = 200
        tail_sec = 60.0
        verbose = False
        no_verify = True
        print_every = 1
        if args.maxfev is None:
            args.maxfev = 600

    kwargs = dict(
        data_names=list(args.data),
        tune=dict(
            initial_guess=initial_guess,
            options={k: v for k, v in dict(maxiter=maxiter, maxfev=args.maxfev).items() if v is not None},
        ),
        tuner=dict(
            clip_start_sec=args.clip_start_sec,
            clip_end_sec=args.clip_end_sec,
            clip_tail_sec=tail_sec,
            verbose=verbose,
            print_every=print_every,
            ctrl_domain=args.ctrl_domain,
            sim_scale=bool(args.sim_scale),
            sim_freq=float(args.freq),
            real_flow_cols=tuple(args.real_flow_cols),
            sim_flow_keys=tuple(args.sim_flow_keys),
        ),
    )

    now = datetime.now()
    save_name = now.strftime("%y%m%d_%H_%M_%S") + "_discharge_coeff_lib8"
    folder_path = f"{get_pkg_path('pneu_env')}/data/discharge_coeff_result/{save_name}"
    os.makedirs(folder_path, exist_ok=True)

    print(f"[ INFO] Tuner8 ==> Save folder: {save_name}")
    with open(f"{folder_path}/cfg.yaml", "w", encoding="utf-8") as f:
        yaml.dump(kwargs, f)

    tuner = PneuSimTuner8(
        data_names=kwargs["data_names"],
        **kwargs["tuner"],
    )

    tune_info = dict()
    try:
        result = tuner.tune(**kwargs["tune"])
        print(result)
        coeff = list(result.x)
        tune_info["inlet_pump_coeff"] = float(coeff[0])
        tune_info["outlet_pump_coeff"] = float(coeff[1])
        with open(f"{folder_path}/result.pkl", "wb") as f:
            pickle.dump(result, f)
    except Exception:
        coeff = tuner.get_coeff()
        tune_info["inlet_pump_coeff"] = float(coeff[0])
        tune_info["outlet_pump_coeff"] = float(coeff[1])
        raise
    finally:
        with open(f"{folder_path}/coeff.yaml", "w", encoding="utf-8") as f:
            yaml.dump(tune_info, f)

    if not no_verify:
        tuner.verificate(np.array(coeff, dtype=np.float64), save_name)


if __name__ == "__main__":
    main()
