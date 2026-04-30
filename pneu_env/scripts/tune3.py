from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import yaml

from pneu_env.tuner3 import INITIAL_GUESS, OPTIMIZER_OPTIONS, PneuSimTuner3
from pneu_utils.utils import get_pkg_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="exp csv base names or csv paths")
    ap.add_argument("--start", type=float, default=None, help="drop data before this time (sec)")
    ap.add_argument("--end", type=float, default=None, help="drop data after this time (sec)")
    ap.add_argument("--tail-sec", type=float, default=None, help="use only last N seconds")
    ap.add_argument("--tag", default=None, help="optional suffix for save folder")
    ap.add_argument("--quiet", action="store_true", help="reduce per-iteration prints")
    ap.add_argument("--no-verify", action="store_true", help="skip verificate()")
    ap.add_argument("--fast", action="store_true", help="quick sanity run (tail-sec=60, maxiter=200, no-verify, quiet)")
    args = ap.parse_args()

    tail_sec = args.tail_sec
    no_verify = bool(args.no_verify)
    verbose = not bool(args.quiet)
    tune_options = dict(OPTIMIZER_OPTIONS)

    if args.fast:
        tail_sec = 60.0 if tail_sec is None else tail_sec
        no_verify = True
        verbose = False
        tune_options["maxiter"] = 200

    kwargs = dict(
        data_names=list(args.data),
        tune=dict(
            initial_guess=list(INITIAL_GUESS),
            options=tune_options,
        ),
        tuner=dict(
            clip_start_sec=args.start,
            clip_end_sec=args.end,
            clip_tail_sec=tail_sec,
            verbose=verbose,
        ),
    )

    now = datetime.now()
    suffix = f"_{args.tag}" if args.tag else ""
    save_name = now.strftime("%y%m%d_%H_%M_%S") + f"_discharge_coeff_lib3{suffix}"
    folder_path = f"{get_pkg_path('pneu_env')}/data/discharge_coeff_result/{save_name}"
    os.makedirs(folder_path, exist_ok=True)

    print(f"[ INFO] Tuner3 ==> Save folder: {save_name}")
    with open(f"{folder_path}/cfg.yaml", "w", encoding="utf-8") as f:
        yaml.dump(kwargs, f)

    tuner = PneuSimTuner3(
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
        tuner.verificate(np.asarray(coeff, dtype=np.float64), save_name)


if __name__ == "__main__":
    main()
