from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from collections import deque

from scipy import optimize
# from scipy.spatial.distance import cdist

from pneu_env.sim import PneuSim
from pneu_utils.utils import get_pkg_path
from pneu_ref.ctrl_ref import CtrlTraj
from pneu_env.pred import PneuPred

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

STD_RHO = 1.20411831637462

class PneuSimTuner():
    def __init__(
        self,
        data_names: List[str]
    ):
        super().__init__()
        self.datas = self.load_datas(data_names)
        self.iter_num = 0
    
    def objective_function(
        self, 
        params: np.ndarray
    ):
        self.iter_num += 1
        self.params = params
        total_error = 0
        for data_name, data in self.datas.items():
            print()
            print(f"[ INFO] Tuner ==> Data name: {data_name}")
            sim = PneuSim(
                freq = 50,
                delay = 0.1,
                noise = False,
                scale = False
            )
            
            sim.set_discharge_coeff(
                inlet_pump_coeff = 1e-6*float(params[0]),
                outlet_pump_coeff = 1e-6*float(params[1])
            )
            # total_error += self.get_error(sim,data)
            total_error += self.get_mass_flowrate_error(sim, data)
        
        print()
        print(f"[ INFO] Tuner (iter: {self.iter_num}) ==> Coeff: {params} err: {total_error}")
        print()
        
        return total_error

    def tune(
        self,
        initial_guess: np.ndarray,
        options: Dict[str, Any]
    ):
        result = optimize.minimize(
            self.objective_function,
            np.array(initial_guess),
            method = "Nelder-Mead",
            options = options
        )
        return result
    
    def get_mass_flowrate_error(
        self,
        sim: PneuSim,
        data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        sim.set_init_press(
            init_pos_press = data["press_pos"][0],
            init_neg_press = data["press_neg"][0]
        )

        ctrl = CtrlTraj(
            traj_time = data["curr_time"],
            traj_pos = data["ctrl_pos"],
            traj_neg = data["ctrl_neg"]
        )

        sim_time = deque()
        sim_pump_out = deque()
        sim_pump_in = deque()
        sim_press_pos = deque()
        sim_press_neg = deque()

        real_pump_in = data["flowrate2"]
        real_pump_out = data["flowrate1"]
        real_press_pos = data["press_pos"]
        real_press_neg = data["press_neg"]

        curr_time = 0
        while curr_time < data["curr_time"][-1]:
            act = ctrl.get_ctrl(curr_time)
            curr_obs, info = sim.observe(act)

            mean_mass_flowrate = sim.get_mean_mass_flowrate()
            sim_time.append(curr_obs[0])
            sim_press_pos.append(curr_obs[1])
            sim_press_neg.append(curr_obs[2])
            sim_pump_in.append(mean_mass_flowrate["pump_in"])
            sim_pump_out.append(mean_mass_flowrate["pump_out"])
            
            curr_time = curr_obs[0]
        
        sim_pump_in = np.array(sim_pump_in)*60000/STD_RHO
        sim_pump_out = np.array(sim_pump_out)*60000/STD_RHO
        sim_press_pos = np.array(sim_press_pos)
        sim_press_neg = np.array(sim_press_neg)
        
        sim_idx, real_idx = self.match_size(data["curr_time"], np.array(sim_time))

        mass_flowrate_in_error = 0.1*np.mean(np.abs(sim_pump_in[sim_idx] - real_pump_in[real_idx]))
        mass_flowrate_out_error = 0.1*np.mean(np.abs(sim_pump_out[sim_idx] - real_pump_out[real_idx]))
        press_pos_error = 1.5*np.mean(np.abs(sim_press_pos[sim_idx] - real_press_pos[real_idx]))
        press_neg_error = 1*np.mean(np.abs(sim_press_neg[sim_idx] - real_press_neg[real_idx]))
        error = mass_flowrate_in_error
        error += mass_flowrate_out_error
        error += press_pos_error
        error += press_neg_error

        print(f"[ INFO] Tuner ==> Mass flowrate in error: {mass_flowrate_in_error}")
        print(f"[ INFO] Tuner ==> Mass flowrate out error: {mass_flowrate_out_error}")
        print(f"[ INFO] Tuner ==> Pressure pos error: {press_pos_error}")
        print(f"[ INFO] Tuner ==> Pressure neg error: {press_neg_error}")
        print(f"[ INFO] Tuner ==> Total error: {error}")

        return error

    def get_error(
        self,
        sim: PneuSim,
        data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        sim.set_init_press(
            init_pos_press = data["sen_pos"][0],
            init_neg_press = data["sen_neg"][0]
        )

        ctrl = CtrlTraj(
            traj_time = data["curr_time"],
            traj_pos = data["ctrl_pos"],
            traj_neg = data["ctrl_neg"],
        )

        sim_time = deque()
        sen_pos = deque()
        sen_neg = deque()
        curr_time = 0
        while curr_time < data["curr_time"][-1]:
            act = ctrl.get_ctrl(curr_time)
            curr_obs, info = sim.observe(
                ctrl = act,
                goal = np.array([101.325, 101.325])
            )
            curr_time = curr_obs[0]
            sim_time.append(curr_obs[0])
            sen_pos.append(curr_obs[1])
            sen_neg.append(curr_obs[2])

        sen_pos = np.array(sen_pos)
        sen_neg = np.array(sen_neg)
        
        sim_idx, real_idx = self.match_size(data["curr_time"], np.array(sim_time))

        err = 0.05*np.sum(np.abs(sen_pos[sim_idx] - data["sen_pos"][real_idx]))
        err += 0.01*np.sum(np.abs(sen_neg[sim_idx] - data["sen_neg"][real_idx]))

        return err
    
    def match_size(self, real_data, sim_data):
        if len(real_data) > len(sim_data):
            long_arr = real_data
            short_arr = sim_data
            long_arr_name = "real"
        else:
            short_arr = real_data
            long_arr = sim_data
            long_arr_name = "sim"
        
        long_idx = deque()
        short_idx = deque()
        for i, val in enumerate(short_arr):
            closest_idx = np.argmin(np.abs(long_arr - val))
            long_idx.append(closest_idx)
            short_idx.append(i)

        if long_arr_name == "real":
            real_idx = np.array(long_idx)
            sim_idx = np.array(short_idx)
        elif long_arr_name == "sim":
            sim_idx = np.array(long_idx)
            real_idx = np.array(short_idx)
        
        return sim_idx, real_idx
    
    def get_coeff(self):
        return list(self.params)
    
    def load_datas(
        self, 
        data_names: List[str]
    ) -> None:
        datas = dict()
        
        for data_name in data_names:
            df = pd.read_csv(f"{get_pkg_path('pneu_env')}/exp/{data_name}.csv")
            data = {col: df[col].to_numpy() for col in df.columns}
            datas[data_name] = data
        
        return datas
    
    def verificate(
        self,
        params: np.ndarray,
        save_name: Optional[str] = None
    ):
        for data_name, data in self.datas.items():
            print()
            print(f"[ INFO] Tuner ==> Data name: {data_name}")
            sim = PneuSim(
                freq = 50,
                delay = 0.1,
                noise = False,
                scale = False
            )
            # sim = PneuPred(
            #     freq = 50,
            #     delay = 0.1,
            #     noise = False,
            #     scale = False
            # )
            
            # sim.set_discharge_coeff(
            #     inlet_pump_coeff = 1e-6*float(params[0]),
            #     outlet_pump_coeff = 1e-6*float(params[1])
            # )    
            sim.set_init_press(
                init_pos_press = data["press_pos"][0],
                init_neg_press = data["press_neg"][0]
            )

            ctrl = CtrlTraj(
                traj_time = data["curr_time"],
                traj_pos = data["ctrl_pos"],
                traj_neg = data["ctrl_neg"]
            )

            sim_time = deque()
            sim_pump_out = deque()
            sim_pump_in = deque()
            sim_press_pos = deque()
            sim_press_neg = deque()
            sim_valve_pos = deque()
            sim_valve_neg = deque()

            real_time = data["curr_time"]
            real_pump_in = data["flowrate2"]
            real_pump_out = data["flowrate1"]
            real_press_pos = data["press_pos"]
            real_press_neg = data["press_neg"]

            curr_time = 0
            while curr_time < data["curr_time"][-1]:
                act = ctrl.get_ctrl(curr_time)
                curr_obs, info = sim.observe(act)

                mean_mass_flowrate = sim.get_mean_mass_flowrate()

                sim_time.append(curr_obs[0])
                sim_press_pos.append(curr_obs[1])
                sim_press_neg.append(curr_obs[2])
                sim_pump_in.append(mean_mass_flowrate["pump_in"]*60000/STD_RHO)
                sim_pump_out.append(mean_mass_flowrate["pump_out"]*60000/STD_RHO)
                sim_valve_pos.append(mean_mass_flowrate["valve_pos"]*60000/STD_RHO)
                sim_valve_neg.append(mean_mass_flowrate["valve_neg"]*60000/STD_RHO)
                
                curr_time = curr_obs[0]
            
            sim_time = np.array(sim_time)
            sim_pump_in = np.array(sim_pump_in)
            sim_pump_out = np.array(sim_pump_out)
            sim_press_pos = np.array(sim_press_pos)
            sim_press_neg = np.array(sim_press_neg)
            sim_valve_pos = np.array(sim_valve_pos)
            sim_valve_neg = np.array(sim_valve_neg)
            
            df = pd.DataFrame(dict(
                curr_time = sim_time,
                press_pos = sim_press_pos,
                press_neg = sim_press_neg,
                mf_in = sim_pump_in,
                mf_out = sim_pump_out,
                mf_pos = sim_valve_pos,
                mf_neg = sim_valve_neg
            ))
            df.to_csv(f"{data_name}_simulation.csv")
            
            # Visualize
            fontname = 'Times New Roman'
            label_font_size = 15
            
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(5, 1, figure=fig)

            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[2,0])    
            ax4 = fig.add_subplot(gs[3,0])    
            ax5 = fig.add_subplot(gs[4,0])    

            ax1.plot(sim_time, sim_press_pos, linewidth=2, color='red', label='sim')
            ax1.plot(real_time, real_press_pos, linewidth=2, color='blue', label='real')
            ax1.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
            ax1.set_ylabel('Pressure [kPa]', fontname=fontname, fontsize=label_font_size)
            ax1.grid(which='major', color='silver', linewidth=1)
            ax1.grid(which='minor', color='lightgray', linewidth=0.5)
            ax1.minorticks_on()
            ax1.legend(loc='upper right')
            ax1.set(xlim=(0, None), ylim=(None, None))
            
            ax2.plot(sim_time, sim_press_neg, linewidth=2, color='red', label='sim')
            ax2.plot(real_time, real_press_neg, linewidth=2, color='blue', label='real')
            ax2.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
            ax2.set_ylabel('Pressure [kPa]', fontname=fontname, fontsize=label_font_size)
            ax2.grid(True)
            ax2.grid(which='major', color='silver', linewidth=1)
            ax2.grid(which='minor', color='lightgray', linewidth=0.5)
            ax2.minorticks_on()
            ax2.legend(loc='upper right')
            ax2.sharex(ax1)
            ax2.set(xlim=(0, None), ylim=(None, None))

            ax3.plot(sim_time, sim_pump_in, linewidth=2, color='red', label='sim')
            ax3.plot(real_time, real_pump_in, linewidth=2, color='blue', label='real')
            ax3.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
            ax3.set_ylabel('Mass flowrate [kg/s]', fontname=fontname, fontsize=label_font_size)
            ax3.grid(True)
            ax3.grid(which='major', color='silver', linewidth=1)
            ax3.grid(which='minor', color='lightgray', linewidth=0.5)
            ax3.minorticks_on()
            ax3.legend(loc='upper right')
            ax3.sharex(ax1)
            ax3.set(xlim=(0, None), ylim=(None, None))

            ax4.plot(sim_time, sim_pump_out, linewidth=2, color='red', label='sim')
            ax4.plot(real_time, real_pump_out, linewidth=2, color='blue', label='real')
            ax4.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
            ax4.set_ylabel('Mass flowrate [kg/s]', fontname=fontname, fontsize=label_font_size)
            ax4.grid(True)
            ax4.grid(which='major', color='silver', linewidth=1)
            ax4.grid(which='minor', color='lightgray', linewidth=0.5)
            ax4.minorticks_on()
            ax4.legend(loc='upper right')
            ax4.sharex(ax1)
            ax4.set(xlim=(0, None), ylim=(None, None))

            ax5.plot(sim_time, sim_valve_pos, linewidth=2, color='red', label='pos')
            ax5.plot(sim_time, sim_valve_neg, linewidth=2, color='blue', label='neg')
            ax5.set_xlabel('Time [sec]', fontname=fontname, fontsize=label_font_size)
            ax5.set_ylabel('Mass flowrate [kg/s]', fontname=fontname, fontsize=label_font_size)
            ax5.grid(True)
            ax5.grid(which='major', color='silver', linewidth=1)
            ax5.grid(which='minor', color='lightgray', linewidth=0.5)
            ax5.minorticks_on()
            ax5.legend(loc='upper right')
            ax5.sharex(ax1)
            ax5.set(xlim=(0, None), ylim=(None, None))
            
            plt.tight_layout()

        if save_name is not None:
            plt.savefig(f'{get_pkg_path("pneu_env")}/data/discharge_coeff_result/{save_name}/{save_name}.png') 
        plt.show()
        
    
