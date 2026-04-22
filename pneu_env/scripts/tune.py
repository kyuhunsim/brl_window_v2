import os

import numpy as np
import pandas as pd

from datetime import datetime

import yaml
import pickle

from pneu_env.tuner import PneuSimTuner
from pneu_env.sim import PneuSim
from pneu_utils.utils import get_pkg_path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

kwargs = dict(
    data_names = [
        # # "241002_14_54_53_Flowrate_pump_pos",
        # "241002_14_56_56_Flowrate_pump_neg",
        # # "241002_14_59_07_Flowrate_095_pump_pos",
        # "241002_15_01_12_Flowrate_095_pump_neg",
        # # "241002_15_03_41_Flowrate_090_pump_pos",
        # "241002_15_05_44_Flowrate_090_pump_neg",
        # # "241002_15_08_10_Flowrate_095_ct_pump_pos",
        # # "241002_15_09_17_Flowrate_095_ct_pump_neg",
        # "241028_17_33_23_Flowrate_random",
        # "241028_17_34_37_Flowrate_random",
        # "241028_17_35_48_Flowrate_random",
        # "241029_16_26_23_Flowrate_085_NEG",
        # "241029_16_29_08_Flowrate_085_POS",
        # "241029_16_32_41_Flowrate_RND"
        # "241029_16_32_41_Flowrate_RND",
        # "241029_17_29_40_Flowrate_RND",
        # "241029_17_33_25_Flowrate_RND",
        # "241029_17_35_17_Flowrate_RND"
        # "241030_13_50_44_Flowrate_RND_5min",
        # "241030_13_57_56_Flowrate_RND_5min",
        # "241030_14_04_02_Flowrate_RND_5min"
        "241030_16_03_46_Flowrate_RND_10min"
    ],
    tune = dict(
        # initial_guess = [1, 1],
        # initial_guess = np.array([
        #     1.9946870163604586,    
        #     106.05290675081713
        # ]),
        initial_guess = [
            # 1.7802329591058923,
            # 5.14708328492747
            # 1.2195075767450598,
            # 30.427941426400608
            1.4141074922667403,
            33.21979373399334
        ],
        options = dict(
            maxiter = 10000
        )
    )
)

now = datetime.now()
formatted_time = now.strftime("%y%m%d_%H_%M_%S")
save_name = f"{formatted_time}_discharge_coeff"
folder_path = f"{get_pkg_path('pneu_env')}/data/discharge_coeff_result/{save_name}"
os.mkdir(folder_path)

print(f"[ INFO] Tuner ==> Save folder: {save_name}")

with open(f"{folder_path}/cfg.yaml", "w") as f:
    yaml.dump(kwargs, f)

tuner = PneuSimTuner(
    data_names = kwargs["data_names"],
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
    
except:
    coeff = tuner.get_coeff()
    tune_info["inlet_pump_coeff"] = float(coeff[0])
    tune_info["outlet_pump_coeff"] = float(coeff[1])

finally:
    with open(f"{folder_path}/coeff.yaml", "w") as f:
        yaml.dump(tune_info, f)

# Simulate
# coeff = np.array([
#     1,
#     1
# ])
# tuner.verificate(np.array(coeff))
tuner.verificate(np.array(coeff), save_name)

    
