from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from sim8 import PneuSim


from utils.utils import get_pkg_path

try:
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:
    plt = None  # type: ignore


STD_RHO = 1.20411831637462

DEFAULT_REAL_FLOW_COLS = ("flow1", "flow2")
DEFAULT_SIM_FLOW_KEYS = ("flow1", "flow2")


class PneuSimTuner8:
    def __init__(
        self,
        data_names: List[str],
        *,
        clip_start_sec: Optional[float] = None,
        clip_end_sec: Optional[float] = None,
        clip_tail_sec: Optional[float] = None,
        verbose: bool = True,
        print_every: int = 1,
        ctrl_domain: str = "unit",
        sim_scale: bool = False,
        sim_freq: float = 50.0,
        real_flow_cols: Sequence[str] = DEFAULT_REAL_FLOW_COLS,
        sim_flow_keys: Sequence[str] = DEFAULT_SIM_FLOW_KEYS,
    ):
        if tuple(real_flow_cols) != ("flow1", "flow2"):
            raise ValueError(f"0403 fixed format only. real_flow_cols must be ('flow1','flow2'), got {real_flow_cols}")
        if len(sim_flow_keys) != 2:
            raise ValueError("sim_flow_keys must have length 2")

        self.clip_start_sec = clip_start_sec
        self.clip_end_sec = clip_end_sec
        self.clip_tail_sec = clip_tail_sec
        self.verbose = bool(verbose)
        self.print_every = int(print_every) if int(print_every) > 0 else 1
        self.ctrl_domain = str(ctrl_domain)
        self.sim_scale = bool(sim_scale)
        self.sim_freq = float(sim_freq)
        self.real_flow_cols = tuple(real_flow_cols)
        self.sim_flow_keys = tuple(sim_flow_keys)

        self.datas = self.load_datas(data_names)
        self.iter_num = 0
        self.params = np.array([1.1256394620423595, 5.401279325612009], dtype=np.float64)

        # Python 최적화: 데이터셋별 sim 객체 미리 생성 후 재사용
        self._sims: Dict[str, PneuSim] = {}
        for data_name, data in self.datas.items():
            self._sims[data_name] = PneuSim(
                freq=self.sim_freq,
                delay=0.1,
                noise=False,
                scale=self.sim_scale,
                init_pos_press=float(data["press_pos"][0]),
                init_neg_press=float(data["press_neg"][0]),
            )

    def _resolve_data_path(self, data_name: str) -> Path:
        candidate = Path(data_name)
        if candidate.exists():
            return candidate

        exp_dir = Path(get_pkg_path("pneu_env")) / "exp"
        if candidate.suffix.lower() == ".csv":
            resolved = exp_dir / candidate.name
        else:
            resolved = exp_dir / f"{candidate.name}.csv"

        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"CSV not found: {data_name} (also tried {resolved})")

    def _convert_ctrl_domain(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if self.ctrl_domain == "unit":
            return np.clip(arr, 0.0, 1.0)
        if self.ctrl_domain == "bipolar":
            return np.clip(arr, -1.0, 1.0)
        raise ValueError(f"Unsupported ctrl domain: {self.ctrl_domain}")

    def _ctrl_at_time(
        self,
        traj_time: np.ndarray,
        ctrls: np.ndarray,
        t: float,
        idx: int,
    ) -> tuple[np.ndarray, int]:
        n = int(traj_time.shape[0])
        if n == 0:
            raise ValueError("Empty trajectory")

        while idx + 1 < n and t >= traj_time[idx + 1]:
            idx += 1

        return ctrls[idx], idx

    def _get_sim_flow_pair(self, sim: PneuSim) -> tuple[float, float]:
        mf = sim.get_mass_flowrate_dict()
        try:
            return float(mf[self.sim_flow_keys[0]]), float(mf[self.sim_flow_keys[1]])
        except KeyError as e:
            raise KeyError(
                f"sim_flow_keys={self.sim_flow_keys} not found. Available keys: {sorted(mf.keys())}"
            ) from e

    def tune(
            self,
            initial_guess: np.ndarray,
            options: Dict[str, Any],
        ):
            options.update({
                'maxiter': 1000, # 최대 반복 횟수 강제 제한
                'xatol': 1e-2,   # 파라미터 변화가 0.01 이하일 때 종료
                'fatol': 1e-2,   # 오차 변화가 0.01 이하일 때 종료
                'disp': True     # 진행 상황 터미널 출력
            })
            result = optimize.minimize(
                self.objective_function,
                np.asarray(initial_guess, dtype=np.float64),
                method="Nelder-Mead",
                tol=1e-3,        # 허용 오차
                options=options,
            )
            return result

    def objective_function(self, params: np.ndarray) -> float:
        self.iter_num += 1
        self.params = np.asarray(params, dtype=np.float64)
        total_error = 0.0

        for data_name, data in self.datas.items():
            if self.verbose and (self.iter_num % self.print_every == 0):
                print()
                print(f"[ INFO] Tuner8 ==> Data name: {data_name}")

            sim = self._sims[data_name]
            sim.set_discharge_coeff(
                inlet_pump_coeff=1e-6 * float(self.params[0]),
                outlet_pump_coeff=1e-6 * float(self.params[1]),
            )
            total_error += self.get_error(sim, data)

        if self.verbose and (self.iter_num % self.print_every == 0):
            print()
            print(f"[ INFO] Tuner8 (iter: {self.iter_num}) ==> Coeff: {self.params} err: {total_error}")
            print()

        return float(total_error)

    def get_error(self, sim: PneuSim, data: Dict[str, np.ndarray]) -> float:
        sim.set_init_press(
            init_pos_press=float(data["press_pos"][0]),
            init_neg_press=float(data["press_neg"][0]),
        )

        traj_time = data["curr_time"]
        ctrls = data["ctrls"]
        real_press_pos = data["press_pos"]
        real_press_neg = data["press_neg"]
        real_flow_a = data["flow1"]
        real_flow_b = data["flow2"]

        t_end = float(traj_time[-1])

        # 50 Hz 기준 거의 같은 길이로 잡고 약간 여유
        n_est = max(1, int(np.ceil(t_end * self.sim_freq)) + 4)

        sim_time = np.empty(n_est, dtype=np.float64)
        sim_press_pos = np.empty(n_est, dtype=np.float64)
        sim_press_neg = np.empty(n_est, dtype=np.float64)
        sim_flow_a = np.empty(n_est, dtype=np.float64)
        sim_flow_b = np.empty(n_est, dtype=np.float64)

        idx = 0
        curr_time = 0.0
        k = 0

        while curr_time < t_end:
            act, idx = self._ctrl_at_time(traj_time, ctrls, curr_time, idx)
            curr_obs, _ = sim.observe(act)
            flow_a, flow_b = self._get_sim_flow_pair(sim)

            if k >= n_est:
                # 혹시 추정 길이 초과하면 확장
                sim_time = np.resize(sim_time, n_est * 2)
                sim_press_pos = np.resize(sim_press_pos, n_est * 2)
                sim_press_neg = np.resize(sim_press_neg, n_est * 2)
                sim_flow_a = np.resize(sim_flow_a, n_est * 2)
                sim_flow_b = np.resize(sim_flow_b, n_est * 2)
                n_est *= 2

            sim_time[k] = float(curr_obs[0])
            sim_press_pos[k] = float(curr_obs[1])
            sim_press_neg[k] = float(curr_obs[2])
            sim_flow_a[k] = float(flow_a * 60000.0 / STD_RHO)
            sim_flow_b[k] = float(flow_b * 60000.0 / STD_RHO)

            curr_time = sim_time[k]
            k += 1

        sim_time = sim_time[:k]
        sim_press_pos = sim_press_pos[:k]
        sim_press_neg = sim_press_neg[:k]
        sim_flow_a = sim_flow_a[:k]
        sim_flow_b = sim_flow_b[:k]

        sim_idx, real_idx = self.match_size(traj_time, sim_time)

        press_pos_error = 1.5 * np.mean(np.abs(sim_press_pos[sim_idx] - real_press_pos[real_idx]))
        press_neg_error = 1.0 * np.mean(np.abs(sim_press_neg[sim_idx] - real_press_neg[real_idx]))
        flow_a_error = 0.1 * np.mean(np.abs(sim_flow_a[sim_idx] - real_flow_a[real_idx]))
        flow_b_error = 0.1 * np.mean(np.abs(sim_flow_b[sim_idx] - real_flow_b[real_idx]))
        error = press_pos_error + press_neg_error + flow_a_error + flow_b_error

        if self.verbose and (self.iter_num % self.print_every == 0):
            print(f"[ INFO] Tuner8 ==> Pressure pos error: {press_pos_error}")
            print(f"[ INFO] Tuner8 ==> Pressure neg error: {press_neg_error}")
            print(f"[ INFO] Tuner8 ==> flow1 error: {flow_a_error}")
            print(f"[ INFO] Tuner8 ==> flow2 error: {flow_b_error}")
            print(f"[ INFO] Tuner8 ==> Total error: {error}")

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
            real_idx = choose
            sim_idx = short_idx
        else:
            sim_idx = choose
            real_idx = short_idx

        return sim_idx, real_idx

    def get_coeff(self) -> list[float]:
        return list(np.asarray(self.params, dtype=np.float64))

    def load_datas(self, data_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        datas: Dict[str, Dict[str, np.ndarray]] = {}

        for data_name in data_names:
            path = self._resolve_data_path(data_name)
            df = pd.read_csv(path)

            required_cols = [
                "time",
                "press_pos",
                "press_neg",
                "ctrl1", "ctrl2", "ctrl3", "ctrl4", "ctrl5", "ctrl6",
                "flow1", "flow2",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{path}: missing required columns: {missing}")

            df = df.sort_values("time").reset_index(drop=True)

            t0 = float(df["time"].iloc[0])
            t1 = float(df["time"].iloc[-1])

            start = self.clip_start_sec
            end = self.clip_end_sec
            if self.clip_tail_sec is not None:
                start = max(t0, t1 - float(self.clip_tail_sec))

            if start is not None:
                df = df[df["time"] >= float(start)]
            if end is not None:
                df = df[df["time"] <= float(end)]

            df = df.reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(f"{path}: no rows after time clipping")

            curr_time = df["time"].to_numpy(dtype=np.float64)
            curr_time = curr_time - curr_time[0]

            ctrls = np.column_stack([
                self._convert_ctrl_domain(df["ctrl1"].to_numpy(dtype=np.float64)),
                self._convert_ctrl_domain(df["ctrl2"].to_numpy(dtype=np.float64)),
                self._convert_ctrl_domain(df["ctrl3"].to_numpy(dtype=np.float64)),
                self._convert_ctrl_domain(df["ctrl4"].to_numpy(dtype=np.float64)),
                self._convert_ctrl_domain(df["ctrl5"].to_numpy(dtype=np.float64)),
                self._convert_ctrl_domain(df["ctrl6"].to_numpy(dtype=np.float64)),
            ]).astype(np.float64)

            datas[path.stem] = {
                "curr_time": curr_time,
                "press_pos": df["press_pos"].to_numpy(dtype=np.float64),
                "press_neg": df["press_neg"].to_numpy(dtype=np.float64),
                "ctrls": ctrls,
                "flow1": df["flow1"].to_numpy(dtype=np.float64),
                "flow2": df["flow2"].to_numpy(dtype=np.float64),
            }

        return datas

    def verificate(self, params: np.ndarray, save_name: Optional[str] = None) -> None:
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is required for verificate(); install matplotlib or run with --no-verify."
            )

        fig_handles = []

        for data_name, data in self.datas.items():
            print()
            print(f"[ INFO] Tuner8 ==> Data name: {data_name}")

            sim = self._sims[data_name]
            sim.set_discharge_coeff(
                inlet_pump_coeff=1e-6 * float(params[0]),
                outlet_pump_coeff=1e-6 * float(params[1]),
            )
            sim.set_init_press(
                init_pos_press=float(data["press_pos"][0]),
                init_neg_press=float(data["press_neg"][0]),
            )

            traj_time = data["curr_time"]
            ctrls = data["ctrls"]
            t_end = float(traj_time[-1])

            n_est = max(1, int(np.ceil(t_end * self.sim_freq)) + 4)
            sim_time = np.empty(n_est, dtype=np.float64)
            sim_press_pos = np.empty(n_est, dtype=np.float64)
            sim_press_neg = np.empty(n_est, dtype=np.float64)
            sim_flow_a = np.empty(n_est, dtype=np.float64)
            sim_flow_b = np.empty(n_est, dtype=np.float64)

            idx = 0
            curr_time = 0.0
            k = 0

            while curr_time < t_end:
                act, idx = self._ctrl_at_time(traj_time, ctrls, curr_time, idx)
                curr_obs, _ = sim.observe(act)
                flow_a, flow_b = self._get_sim_flow_pair(sim)

                if k >= n_est:
                    sim_time = np.resize(sim_time, n_est * 2)
                    sim_press_pos = np.resize(sim_press_pos, n_est * 2)
                    sim_press_neg = np.resize(sim_press_neg, n_est * 2)
                    sim_flow_a = np.resize(sim_flow_a, n_est * 2)
                    sim_flow_b = np.resize(sim_flow_b, n_est * 2)
                    n_est *= 2

                sim_time[k] = float(curr_obs[0])
                sim_press_pos[k] = float(curr_obs[1])
                sim_press_neg[k] = float(curr_obs[2])
                sim_flow_a[k] = float(flow_a * 60000.0 / STD_RHO)
                sim_flow_b[k] = float(flow_b * 60000.0 / STD_RHO)

                curr_time = sim_time[k]
                k += 1

            sim_time = sim_time[:k]
            sim_press_pos = sim_press_pos[:k]
            sim_press_neg = sim_press_neg[:k]
            sim_flow_a = sim_flow_a[:k]
            sim_flow_b = sim_flow_b[:k]

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
            ax3.plot(sim_time, sim_flow_a, label=f"sim_{self.sim_flow_keys[0]}")
            ax3.set_title(f"{data_name} - Flow1")
            ax3.grid(True)
            ax3.legend()

            ax4.plot(traj_time, data["flow2"], label="real_flow2")
            ax4.plot(sim_time, sim_flow_b, label=f"sim_{self.sim_flow_keys[1]}")
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
