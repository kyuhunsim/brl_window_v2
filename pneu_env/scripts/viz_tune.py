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
        # "241002_15_09_17_Flowrate_095_ct_pump_neg",
        # "241028_17_33_23_Flowrate_random",
        # "241028_17_34_37_Flowrate_random",
        # "241028_17_35_48_Flowrate_random",
        # "241029_16_26_23_Flowrate_085_NEG",
        # "241029_16_29_08_Flowrate_085_POS",
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
        initial_guess = [1, 1],
        # initial_guess = np.array([
        #     1.9946870163604586,    
        #     106.05290675081713
        # ]),
        options = dict(
            maxiter = 10000
        )
    )
)


tuner = PneuSimTuner(
    data_names = kwargs["data_names"],
)

# Keep!
coeff = np.array([
    # 1.7802329591058923,
    # 5.14708328492747
    # 1.2195075767450598,
    # 30.427941426400608
    # 1.154404699788912,
    # 28.554316440453242
    1.4141074922667403,
    33.21979373399334
])
tuner.verificate(np.array(coeff))

    
