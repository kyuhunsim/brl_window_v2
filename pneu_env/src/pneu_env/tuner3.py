from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from pneu_env.sim3 import PneuSim
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
    flow1=0.1,
    flow2=0.1,
)


class PneuSimTuner3:
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

    def _unit_to_sim3_ctrl(self, ctrl_unit: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * np.asarray(ctrl_unit, dtype=np.float64) - 1.0, -1.0, 1.0)

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

        return self._unit_to_sim3_ctrl(ctrls_unit[idx]), idx

    def _make_sim(self, data: Dict[str, np.ndarray]) -> PneuSim:
        return PneuSim(
            freq=SIM_FREQ,
            delay=SIM_DELAY,
            noise=False,
            scale=False,
            init_pos_press=float(data["press_pos"][0]),
            init_neg_press=float(data["press_neg"][0]),
            init_act_pos_press=float(data["act_pos_press"][0]),
            init_act_neg_press=float(data["act_neg_press"][0]),
        )

    def _reset_sim(self, sim: PneuSim, data: Dict[str, np.ndarray]) -> None:
        sim.set_init_press(
            init_pos_press=float(data["press_pos"][0]),
            init_neg_press=float(data["press_neg"][0]),
            init_act_pos_press=float(data["act_pos_press"][0]),
            init_act_neg_press=float(data["act_neg_press"][0]),
        )
        sim.obs_buf.clear()

    def _sim_flow1_flow2_lpm(self, sim: PneuSim) -> tuple[float, float]:
        mf = sim.get_mass_flowrate()
        if len(mf) < 6:
            raise ValueError(f"sim3 mass-flow vector must have at least 6 values, got {len(mf)}")

        flow1 = float(mf[4] * 60000.0 / STD_RHO)
        flow2 = float(mf[5] * 60000.0 / STD_RHO)
        return flow1, flow2

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
        self.params = np.asarray(params, dtype=np.float64)
        total_error = 0.0

        for data_name, data in self.datas.items():
            if self.verbose:
                print()
                print(f"[ INFO] Tuner3 ==> Data name: {data_name}")

            sim = self._sims[data_name]
            sim.set_discharge_coeff(
                inlet_pump_coeff=1e-6 * float(self.params[0]),
                outlet_pump_coeff=1e-6 * float(self.params[1]),
            )
            self._reset_sim(sim, data)
            total_error += self.get_error(sim, data)

        if self.verbose:
            print()
            print(f"[ INFO] Tuner3 (iter: {self.iter_num}) ==> Coeff: {self.params} err: {total_error}")
            print()

        return float(total_error)

    def get_error(self, sim: PneuSim, data: Dict[str, np.ndarray]) -> float:
        traj_time = data["curr_time"]
        ctrls_unit = data["ctrls_unit"]
        real_press_pos = data["press_pos"]
        real_press_neg = data["press_neg"]
        real_flow1 = data["flow1"]
        real_flow2 = data["flow2"]
        t_end = float(traj_time[-1])

        n_est = max(1, int(np.ceil(t_end * SIM_FREQ)) + 4)
        sim_time = np.empty(n_est, dtype=np.float64)
        sim_press_pos = np.empty(n_est, dtype=np.float64)
        sim_press_neg = np.empty(n_est, dtype=np.float64)
        sim_flow1 = np.empty(n_est, dtype=np.float64)
        sim_flow2 = np.empty(n_est, dtype=np.float64)

        idx = 0
        curr_time = 0.0
        k = 0

        while curr_time < t_end:
            act, idx = self._ctrl_at_time(traj_time, ctrls_unit, curr_time, idx)
            curr_obs, _ = sim.observe(act)
            flow1, flow2 = self._sim_flow1_flow2_lpm(sim)

            if k >= n_est:
                sim_time = np.resize(sim_time, n_est * 2)
                sim_press_pos = np.resize(sim_press_pos, n_est * 2)
                sim_press_neg = np.resize(sim_press_neg, n_est * 2)
                sim_flow1 = np.resize(sim_flow1, n_est * 2)
                sim_flow2 = np.resize(sim_flow2, n_est * 2)
                n_est *= 2

            sim_time[k] = float(curr_obs[0])
            sim_press_pos[k] = float(curr_obs[1])
            sim_press_neg[k] = float(curr_obs[2])
            sim_flow1[k] = flow1
            sim_flow2[k] = flow2

            curr_time = sim_time[k]
            k += 1

        sim_idx, real_idx = self.match_size(traj_time, sim_time[:k])

        press_pos_error = ERROR_WEIGHTS["press_pos"] * np.mean(np.abs(sim_press_pos[:k][sim_idx] - real_press_pos[real_idx]))
        press_neg_error = ERROR_WEIGHTS["press_neg"] * np.mean(np.abs(sim_press_neg[:k][sim_idx] - real_press_neg[real_idx]))
        flow1_error = ERROR_WEIGHTS["flow1"] * np.mean(np.abs(sim_flow1[:k][sim_idx] - real_flow1[real_idx]))
        flow2_error = ERROR_WEIGHTS["flow2"] * np.mean(np.abs(sim_flow2[:k][sim_idx] - real_flow2[real_idx]))
        error = press_pos_error + press_neg_error + flow1_error + flow2_error

        if self.verbose:
            print(f"[ INFO] Tuner3 ==> Pressure pos error: {press_pos_error}")
            print(f"[ INFO] Tuner3 ==> Pressure neg error: {press_neg_error}")
            print(f"[ INFO] Tuner3 ==> flow1 error: {flow1_error}")
            print(f"[ INFO] Tuner3 ==> flow2 error: {flow2_error}")
            print(f"[ INFO] Tuner3 ==> Total error: {error}")

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
        return list(np.asarray(self.params, dtype=np.float64))

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
                "ctrl1", "ctrl2", "ctrl3", "ctrl4", "ctrl5", "ctrl6",
                "flow1", "flow2",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{path}: missing required columns: {missing}")

            if "act_pos_press" not in df.columns:
                df["act_pos_press"] = 101.325
            if "act_neg_press" not in df.columns:
                df["act_neg_press"] = 101.325

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
                np.clip(df["ctrl3"].to_numpy(dtype=np.float64), 0.0, 1.0),
                np.clip(df["ctrl4"].to_numpy(dtype=np.float64), 0.0, 1.0),
                np.clip(df["ctrl5"].to_numpy(dtype=np.float64), 0.0, 1.0),
                np.clip(df["ctrl6"].to_numpy(dtype=np.float64), 0.0, 1.0),
            ]).astype(np.float64)

            datas[path.stem] = {
                "curr_time": curr_time,
                "press_pos": df["press_pos"].to_numpy(dtype=np.float64),
                "press_neg": df["press_neg"].to_numpy(dtype=np.float64),
                "act_pos_press": df["act_pos_press"].to_numpy(dtype=np.float64),
                "act_neg_press": df["act_neg_press"].to_numpy(dtype=np.float64),
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
            print(f"[ INFO] Tuner3 ==> Data name: {data_name}")

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
            sim_flow1 = np.empty(n_est, dtype=np.float64)
            sim_flow2 = np.empty(n_est, dtype=np.float64)

            idx = 0
            curr_time = 0.0
            k = 0

            while curr_time < t_end:
                act, idx = self._ctrl_at_time(traj_time, ctrls_unit, curr_time, idx)
                curr_obs, _ = sim.observe(act)
                flow1, flow2 = self._sim_flow1_flow2_lpm(sim)

                if k >= n_est:
                    sim_time = np.resize(sim_time, n_est * 2)
                    sim_press_pos = np.resize(sim_press_pos, n_est * 2)
                    sim_press_neg = np.resize(sim_press_neg, n_est * 2)
                    sim_flow1 = np.resize(sim_flow1, n_est * 2)
                    sim_flow2 = np.resize(sim_flow2, n_est * 2)
                    n_est *= 2

                sim_time[k] = float(curr_obs[0])
                sim_press_pos[k] = float(curr_obs[1])
                sim_press_neg[k] = float(curr_obs[2])
                sim_flow1[k] = flow1
                sim_flow2[k] = flow2
                curr_time = sim_time[k]
                k += 1

            sim_time = sim_time[:k]
            sim_press_pos = sim_press_pos[:k]
            sim_press_neg = sim_press_neg[:k]
            sim_flow1 = sim_flow1[:k]
            sim_flow2 = sim_flow2[:k]

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
            ax3.plot(traj_time, data["flow1"], label="real_flow1")
            ax3.plot(sim_time, sim_flow1, label="sim_flow1")
            ax3.set_title(f"{data_name} - Flow1")
            ax3.grid(True)
            ax3.legend()
            ax4.plot(traj_time, data["flow2"], label="real_flow2")
            ax4.plot(sim_time, sim_flow2, label="sim_flow2")
            ax4.set_title(f"{data_name} - Flow2")
            ax4.grid(True)
            ax4.legend()

            fig_handles.extend([fig1, fig2])

        if save_name is not None:
            save_dir = Path(get_pkg_path("pneu_env")) / "data" / "discharge_coeff_result" / save_name
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, fig in enumerate(fig_handles, start=1):
                fig.savefig(save_dir / f"verify_{i}.png", dpi=150, bbox_inches="tight")

        plt.show()
