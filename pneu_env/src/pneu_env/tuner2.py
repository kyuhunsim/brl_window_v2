from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from pneu_env.sim2 import PneuSim
from pneu_utils.utils import get_pkg_path


STD_RHO = 1.20411831637462
SIM_FREQ = 50.0
SIM_DELAY = 0.1
INITIAL_GUESS = [3.79683467, 7.73269091]

OPTIMIZER_OPTIONS = dict(
    maxiter=10000,
    xatol=1e-2,
    fatol=1e-2,
    disp=True,
)
ERROR_WEIGHTS = dict(
    press_pos=1.5,
    press_neg=1.0,
    flow_out=0.1,
    flow_in=0.1,
)
COEFF_MIN = 1e-12


class PneuSimTuner2:
    """
    Pump discharge-coefficient tuner for lib2/sim2.

    Assumptions:
    - real flow1 measures pump discharge (pump_out)
    - real flow2 measures pump suction   (pump_in)
    - real ctrl1/ctrl2 are physical valve commands in [0, 1]
    """

    def __init__(
        self,
        data_names: List[str],
        *,
        clip_start_sec: Optional[float] = None,
        clip_end_sec: Optional[float] = None,
        clip_tail_sec: Optional[float] = None,
        verbose: bool = True,
    ):
        self.clip_start_sec = clip_start_sec
        self.clip_end_sec = clip_end_sec
        self.clip_tail_sec = clip_tail_sec
        self.verbose = bool(verbose)

        self.datas = self.load_datas(data_names)
        self.iter_num = 0
        self.params = np.asarray(INITIAL_GUESS, dtype=np.float64)
        self._sims = {data_name: self._make_sim(data) for data_name, data in self.datas.items()}

    def _sanitize_coeffs(self, params: np.ndarray) -> np.ndarray:
        arr = np.asarray(params, dtype=np.float64).copy()
        if arr.shape != (2,):
            raise ValueError(f"Pump coeff params must have shape (2,), got {arr.shape}")
        arr[0] = max(float(arr[0]), COEFF_MIN)
        arr[1] = max(float(arr[1]), COEFF_MIN)
        return arr

    def _resolve_data_path(self, data_name: str) -> Path:
        candidate = Path(data_name)
        if candidate.exists():
            return candidate

        exp_dir = Path(get_pkg_path("pneu_env")) / "exp"
        resolved = exp_dir / (candidate.name if candidate.suffix.lower() == ".csv" else f"{candidate.name}.csv")
        if resolved.exists():
            return resolved

        raise FileNotFoundError(f"CSV not found: {data_name} (also tried {resolved})")

    def _time_column(self, df: pd.DataFrame, path: Path) -> str:
        if "curr_time" in df.columns:
            return "curr_time"
        if "time" in df.columns:
            return "time"
        raise ValueError(f"{path}: missing required time column: curr_time/time")

    def _unit_to_sim2_ctrl(self, ctrl_unit: np.ndarray) -> np.ndarray:
        ctrl_unit = np.clip(np.asarray(ctrl_unit, dtype=np.float64), 0.0, 1.0)
        return np.clip(2.0 * ctrl_unit - 1.0, -1.0, 1.0)

    def _ctrl_at_time(
        self,
        traj_time: np.ndarray,
        ctrls_unit: np.ndarray,
        t: float,
        idx: int,
    ) -> tuple[np.ndarray, int]:
        n = int(traj_time.shape[0])
        if n == 0:
            raise ValueError("Empty trajectory")

        while idx + 1 < n and t >= float(traj_time[idx + 1]):
            idx += 1

        return self._unit_to_sim2_ctrl(ctrls_unit[idx]), idx

    def _make_sim(self, data: Dict[str, np.ndarray]) -> PneuSim:
        return PneuSim(
            freq=SIM_FREQ,
            delay=SIM_DELAY,
            noise=False,
            scale=False,
            init_pos_press=float(data["press_pos"][0]),
            init_neg_press=float(data["press_neg"][0]),
        )

    def _reset_sim(self, sim: PneuSim, data: Dict[str, np.ndarray]) -> None:
        sim.set_init_press(
            init_pos_press=float(data["press_pos"][0]),
            init_neg_press=float(data["press_neg"][0]),
        )
        sim.obs_buf.clear()

    def _sim_pump_flows_lpm(self, sim: PneuSim) -> tuple[float, float]:
        mf = sim.get_mean_mass_flowrate()
        flow_out = float(mf["pump_out"] * 60000.0 / STD_RHO)
        flow_in = float(mf["pump_in"] * 60000.0 / STD_RHO)
        return flow_out, flow_in

    def tune(
        self,
        initial_guess: np.ndarray = np.asarray(INITIAL_GUESS, dtype=np.float64),
        options: Optional[Dict[str, Any]] = None,
    ):
        tune_options = dict(OPTIMIZER_OPTIONS)
        if options:
            tune_options.update(options)

        return optimize.minimize(
            self.objective_function,
            np.asarray(initial_guess, dtype=np.float64),
            method="Nelder-Mead",
            tol=1e-3,
            options=tune_options,
        )

    def objective_function(self, params: np.ndarray) -> float:
        self.iter_num += 1
        self.params = self._sanitize_coeffs(params)
        total_error = 0.0

        for data_name, data in self.datas.items():
            if self.verbose:
                print()
                print(f"[ INFO] Tuner2 ==> Data name: {data_name}")

            sim = self._sims[data_name]
            sim.set_discharge_coeff(
                inlet_pump_coeff=1e-6 * float(self.params[0]),
                outlet_pump_coeff=1e-6 * float(self.params[1]),
            )
            self._reset_sim(sim, data)
            total_error += self.get_error(sim, data)

        if self.verbose:
            print()
            print(f"[ INFO] Tuner2 (iter: {self.iter_num}) ==> Coeff: {self.params} err: {total_error}")
            print()

        return float(total_error)

    def get_error(self, sim: PneuSim, data: Dict[str, np.ndarray]) -> float:
        traj_time = data["curr_time"]
        ctrls_unit = data["ctrls_unit"]
        real_press_pos = data["press_pos"]
        real_press_neg = data["press_neg"]
        real_flow_out = data["flow1"]
        real_flow_in = data["flow2"]
        t_end = float(traj_time[-1])

        n_est = max(1, int(np.ceil(t_end * SIM_FREQ)) + 4)
        sim_time = np.empty(n_est, dtype=np.float64)
        sim_press_pos = np.empty(n_est, dtype=np.float64)
        sim_press_neg = np.empty(n_est, dtype=np.float64)
        sim_flow_out = np.empty(n_est, dtype=np.float64)
        sim_flow_in = np.empty(n_est, dtype=np.float64)

        idx = 0
        curr_time = 0.0
        k = 0

        while curr_time < t_end:
            act, idx = self._ctrl_at_time(traj_time, ctrls_unit, curr_time, idx)
            curr_obs, _ = sim.observe(act)
            flow_out, flow_in = self._sim_pump_flows_lpm(sim)

            if k >= n_est:
                sim_time = np.resize(sim_time, n_est * 2)
                sim_press_pos = np.resize(sim_press_pos, n_est * 2)
                sim_press_neg = np.resize(sim_press_neg, n_est * 2)
                sim_flow_out = np.resize(sim_flow_out, n_est * 2)
                sim_flow_in = np.resize(sim_flow_in, n_est * 2)
                n_est *= 2

            sim_time[k] = float(curr_obs[0])
            sim_press_pos[k] = float(curr_obs[1])
            sim_press_neg[k] = float(curr_obs[2])
            sim_flow_out[k] = flow_out
            sim_flow_in[k] = flow_in

            curr_time = sim_time[k]
            k += 1

        sim_idx, real_idx = self.match_size(traj_time, sim_time[:k])

        press_pos_error = ERROR_WEIGHTS["press_pos"] * np.mean(np.abs(sim_press_pos[:k][sim_idx] - real_press_pos[real_idx]))
        press_neg_error = ERROR_WEIGHTS["press_neg"] * np.mean(np.abs(sim_press_neg[:k][sim_idx] - real_press_neg[real_idx]))
        flow_out_error = ERROR_WEIGHTS["flow_out"] * np.mean(np.abs(sim_flow_out[:k][sim_idx] - real_flow_out[real_idx]))
        flow_in_error = ERROR_WEIGHTS["flow_in"] * np.mean(np.abs(sim_flow_in[:k][sim_idx] - real_flow_in[real_idx]))
        error = press_pos_error + press_neg_error + flow_out_error + flow_in_error

        if self.verbose:
            print(f"[ INFO] Tuner2 ==> Pressure pos error: {press_pos_error}")
            print(f"[ INFO] Tuner2 ==> Pressure neg error: {press_neg_error}")
            print(f"[ INFO] Tuner2 ==> Pump flow out error: {flow_out_error}")
            print(f"[ INFO] Tuner2 ==> Pump flow in error: {flow_in_error}")
            print(f"[ INFO] Tuner2 ==> Total error: {error}")

        return float(error)

    def match_size(self, real_data: np.ndarray, sim_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        real_arr = np.asarray(real_data, dtype=np.float64)
        sim_arr = np.asarray(sim_data, dtype=np.float64)
        if real_arr.size == 0 or sim_arr.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        if real_arr.size >= sim_arr.size:
            long_arr = real_arr
            short_arr = sim_arr
            long_is = "real"
        else:
            long_arr = sim_arr
            short_arr = real_arr
            long_is = "sim"

        idx1 = np.searchsorted(long_arr, short_arr, side="left")
        idx0 = np.clip(idx1 - 1, 0, long_arr.size - 1)
        idx1 = np.clip(idx1, 0, long_arr.size - 1)

        d0 = np.abs(long_arr[idx0] - short_arr)
        d1 = np.abs(long_arr[idx1] - short_arr)
        choose = np.where(d1 < d0, idx1, idx0).astype(int)

        short_idx = np.arange(short_arr.size, dtype=int)
        if long_is == "real":
            return short_idx, choose
        return choose, short_idx

    def get_coeff(self) -> list[float]:
        return list(self._sanitize_coeffs(self.params))

    def load_datas(self, data_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        datas: Dict[str, Dict[str, np.ndarray]] = {}

        for data_name in data_names:
            path = self._resolve_data_path(data_name)
            df = pd.read_csv(path)
            time_col = self._time_column(df, path)

            required_cols = [
                time_col,
                "press_pos",
                "press_neg",
                "ctrl1",
                "ctrl2",
                "flow1",
                "flow2",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{path}: missing required columns: {missing}")

            df = df.sort_values(time_col).reset_index(drop=True)
            t0 = float(df[time_col].iloc[0])
            t1 = float(df[time_col].iloc[-1])

            start = self.clip_start_sec
            end = self.clip_end_sec
            if self.clip_tail_sec is not None:
                start = max(t0, t1 - float(self.clip_tail_sec))

            if start is not None:
                df = df[df[time_col] >= float(start)]
            if end is not None:
                df = df[df[time_col] <= float(end)]

            df = df.reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f"{path}: no rows after time clipping")

            curr_time = df[time_col].to_numpy(dtype=np.float64)
            curr_time = curr_time - float(curr_time[0])
            ctrls_unit = np.column_stack([
                np.clip(df["ctrl1"].to_numpy(dtype=np.float64), 0.0, 1.0),
                np.clip(df["ctrl2"].to_numpy(dtype=np.float64), 0.0, 1.0),
            ]).astype(np.float64)

            datas[path.stem] = {
                "curr_time": curr_time,
                "press_pos": df["press_pos"].to_numpy(dtype=np.float64),
                "press_neg": df["press_neg"].to_numpy(dtype=np.float64),
                "ctrls_unit": ctrls_unit,
                "flow1": df["flow1"].to_numpy(dtype=np.float64),
                "flow2": df["flow2"].to_numpy(dtype=np.float64),
            }

        return datas

    def verificate(self, params: np.ndarray, save_name: Optional[str] = None) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "matplotlib is required for verificate(); install matplotlib or run with --no-verify."
            ) from e

        fig_handles = []

        for data_name, data in self.datas.items():
            print()
            print(f"[ INFO] Tuner2 ==> Data name: {data_name}")

            sim = self._sims[data_name]
            sim.set_discharge_coeff(
                inlet_pump_coeff=1e-6 * float(params[0]),
                outlet_pump_coeff=1e-6 * float(params[1]),
            )
            self._reset_sim(sim, data)

            traj_time = data["curr_time"]
            ctrls_unit = data["ctrls_unit"]
            t_end = float(traj_time[-1])
            n_est = max(1, int(np.ceil(t_end * SIM_FREQ)) + 4)

            sim_time = np.empty(n_est, dtype=np.float64)
            sim_press_pos = np.empty(n_est, dtype=np.float64)
            sim_press_neg = np.empty(n_est, dtype=np.float64)
            sim_flow_out = np.empty(n_est, dtype=np.float64)
            sim_flow_in = np.empty(n_est, dtype=np.float64)

            idx = 0
            curr_time = 0.0
            k = 0

            while curr_time < t_end:
                act, idx = self._ctrl_at_time(traj_time, ctrls_unit, curr_time, idx)
                curr_obs, _ = sim.observe(act)
                flow_out, flow_in = self._sim_pump_flows_lpm(sim)

                if k >= n_est:
                    sim_time = np.resize(sim_time, n_est * 2)
                    sim_press_pos = np.resize(sim_press_pos, n_est * 2)
                    sim_press_neg = np.resize(sim_press_neg, n_est * 2)
                    sim_flow_out = np.resize(sim_flow_out, n_est * 2)
                    sim_flow_in = np.resize(sim_flow_in, n_est * 2)
                    n_est *= 2

                sim_time[k] = float(curr_obs[0])
                sim_press_pos[k] = float(curr_obs[1])
                sim_press_neg[k] = float(curr_obs[2])
                sim_flow_out[k] = flow_out
                sim_flow_in[k] = flow_in
                curr_time = sim_time[k]
                k += 1

            sim_time = sim_time[:k]
            sim_press_pos = sim_press_pos[:k]
            sim_press_neg = sim_press_neg[:k]
            sim_flow_out = sim_flow_out[:k]
            sim_flow_in = sim_flow_in[:k]

            fig1 = plt.figure(figsize=(12, 6))
            ax1 = fig1.add_subplot(2, 1, 1)
            ax2 = fig1.add_subplot(2, 1, 2)
            ax1.plot(traj_time, data["press_pos"], label="real_press_pos")
            ax1.plot(sim_time, sim_press_pos, label="sim_press_pos")
            ax1.set_title(f"{data_name} - Pressure Pos")
            ax1.grid(True)
            ax1.legend()
            ax2.plot(traj_time, data["press_neg"], label="real_press_neg")
            ax2.plot(sim_time, sim_press_neg, label="sim_press_neg")
            ax2.set_title(f"{data_name} - Pressure Neg")
            ax2.grid(True)
            ax2.legend()

            fig2 = plt.figure(figsize=(12, 6))
            ax3 = fig2.add_subplot(2, 1, 1)
            ax4 = fig2.add_subplot(2, 1, 2)
            ax3.plot(traj_time, data["flow1"], label="real_pump_flow_out")
            ax3.plot(sim_time, sim_flow_out, label="sim_pump_flow_out")
            ax3.set_title(f"{data_name} - Pump Flow Out")
            ax3.grid(True)
            ax3.legend()
            ax4.plot(traj_time, data["flow2"], label="real_pump_flow_in")
            ax4.plot(sim_time, sim_flow_in, label="sim_pump_flow_in")
            ax4.set_title(f"{data_name} - Pump Flow In")
            ax4.grid(True)
            ax4.legend()

            fig_handles.extend([fig1, fig2])

        if save_name is not None:
            save_dir = Path(get_pkg_path("pneu_env")) / "data" / "discharge_coeff_result" / save_name
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, fig in enumerate(fig_handles, start=1):
                fig.savefig(save_dir / f"verify_{i}.png", dpi=150, bbox_inches="tight")

        plt.show()
