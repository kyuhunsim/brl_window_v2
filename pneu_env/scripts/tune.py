import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import yaml

from pneu_env.tuner import PneuSimTuner
from pneu_utils.utils import get_pkg_path


DEFAULT_DATA_NAMES = [
    "241030_16_03_46_Flowrate_RND_10min",
]

DEFAULT_INITIAL_GUESS = [
    1.4141074922667403,
    33.21979373399334,
]


def main():
    parser = argparse.ArgumentParser(description="Tune lib1 pump discharge coefficients.")
    parser.add_argument("data", nargs="*", default=DEFAULT_DATA_NAMES, help="Data name or CSV path under pneu_env. Example: exp/foo.csv")
    parser.add_argument("--start", type=float, default=None, help="Start time [s] for each CSV")
    parser.add_argument("--end", type=float, default=None, help="End time [s] for each CSV")
    parser.add_argument("--initial-guess", type=float, nargs=2, default=DEFAULT_INITIAL_GUESS)
    parser.add_argument("--maxiter", type=int, default=10000)
    parser.add_argument("--tag", default="discharge_coeff", help="Suffix for save folder")
    args = parser.parse_args()

    kwargs = dict(
        data_names=args.data,
        start=args.start,
        end=args.end,
        tune=dict(
            initial_guess=args.initial_guess,
            options=dict(maxiter=args.maxiter),
        ),
    )

    now = datetime.now()
    formatted_time = now.strftime("%y%m%d_%H_%M_%S")
    save_name = f"{formatted_time}_{args.tag}"
    folder_path = f"{get_pkg_path('pneu_env')}/data/discharge_coeff_result/{save_name}"
    os.mkdir(folder_path)

    print(f"[ INFO] Tuner ==> Save folder: {save_name}")

    with open(f"{folder_path}/cfg.yaml", "w") as f:
        yaml.dump(kwargs, f)

    tuner = PneuSimTuner(
        data_names=kwargs["data_names"],
        start=kwargs["start"],
        end=kwargs["end"],
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
        with open(f"{folder_path}/coeff.yaml", "w") as f:
            yaml.dump(tune_info, f)

    tuner.verificate(np.array(coeff), save_name)


if __name__ == "__main__":
    main()
