"""Synchronous lib3 replay for pump calibration, without ROS or policy delays.

The C library holds a singleton simulator. Use one adapter per process, and replay
datasets sequentially; use separate processes for independent calibration jobs.
"""
from __future__ import annotations

from ctypes import CDLL, POINTER, c_bool, c_double
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize

STD_RHO = 1.20411831637462
PRESSURE_NAMES = ("press_pos", "press_neg", "act_pos_press", "act_neg_press")
FLOW_NAMES = tuple(f"flow{i}" for i in range(1, 7))
CHANNEL_NAMES = PRESSURE_NAMES + FLOW_NAMES
# Preserve the former pump objective; the remaining channels are reported.
OBJECTIVE_WEIGHTS = {"press_pos": 1.5, "press_neg": 1.0, "flow1": .1, "flow2": .1}
INITIAL_GUESS = np.array([3.6198491, 7.49941545])
COEFF_MIN = 1e-12


@dataclass(frozen=True)
class Trajectory:
    path: Path
    time: np.ndarray
    source_time: np.ndarray
    controls: np.ndarray
    values: np.ndarray
    duplicate_time_rows_dropped: int = 0


def load_trajectory(path: Path, *, start_sec: float | None = None,
                    end_sec: float | None = None,
                    duration_sec: float | None = None) -> Trajectory:
    path = Path(path).resolve()
    df = pd.read_csv(path)
    time_name = "curr_time" if "curr_time" in df else "time"
    required = [time_name, *PRESSURE_NAMES, *FLOW_NAMES,
                *(f"ctrl{i}" for i in range(1, 7))]
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: nonfinite required data")
    full_time = df[time_name].to_numpy(dtype=float)
    if len(full_time) < 2 or np.any(np.diff(full_time) < 0):
        raise ValueError(f"{path}: timestamps must be nondecreasing")
    # A few acquisition rows have identical timestamps. An earlier row at the
    # same time has zero duration; use the last recorded sample consistently
    # for both command hold and all sensor interpolation, without editing CSVs.
    duplicate_count = int(df[time_name].duplicated(keep="last").sum())
    df = df.drop_duplicates(time_name, keep="last")
    full_time = df[time_name].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    if start_sec is not None:
        keep &= full_time >= start_sec
    if end_sec is not None:
        keep &= full_time <= end_sec
    df = df.loc[keep]
    if duration_sec is not None:
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if len(df):
            df = df[df[time_name] <= float(df[time_name].iloc[0]) + duration_sec]
    if len(df) < 2:
        raise ValueError(f"{path}: fewer than two rows after clipping")
    controls = df[[f"ctrl{i}" for i in range(1, 7)]].to_numpy(dtype=float)
    if np.any((controls < -1e-9) | (controls > 1 + 1e-9)):
        raise ValueError(f"{path}: ctrl1..6 must be physical commands in [0, 1]")
    source_time = df[time_name].to_numpy(dtype=float)
    return Trajectory(path, source_time - source_time[0], source_time,
                      np.ascontiguousarray(np.clip(controls, 0., 1.)),
                      df[list(CHANNEL_NAMES)].to_numpy(dtype=float), duplicate_count)


class NativeLib3:
    """Minimal adapter: pressure and flow are taken from the same native step."""
    def __init__(self, library_path: Path, volume_pos: float = .75,
                 volume_neg: float = .4):
        self.library_path = Path(library_path).resolve(strict=True)
        self.lib = CDLL(str(self.library_path))
        self.lib.set_init_env_act_c.argtypes = [c_double] * 4
        self.lib.set_init_env_act_c.restype = None
        self.lib.time_reset_c.argtypes = []
        self.lib.time_reset_c.restype = None
        self.lib.set_volume_c.argtypes = [c_double] * 2
        self.lib.set_volume_c.restype = None
        self.lib.set_discharge_coeff_c.argtypes = [c_double] * 4
        self.lib.set_discharge_coeff_c.restype = None
        self.lib.step_c.argtypes = [POINTER(c_double), c_double]
        self.lib.step_c.restype = POINTER(c_double)
        self.lib.get_mass_flowrate_c.argtypes = []
        self.lib.get_mass_flowrate_c.restype = POINTER(c_double)
        self.lib.set_logging_c.argtypes = [c_bool]
        self.lib.set_logging_c.restype = None
        self.lib.set_logging_c(False)
        self.lib.set_volume_c(volume_pos, volume_neg)

    def reset(self, pressures: np.ndarray, coeff_micro: np.ndarray) -> None:
        self.lib.time_reset_c()
        self.lib.set_init_env_act_c(*map(float, pressures))
        inlet, outlet = np.asarray(coeff_micro, dtype=float) * 1e-6
        self.lib.set_discharge_coeff_c(inlet, outlet, inlet, outlet)

    def step(self, control: np.ndarray, dt: float) -> tuple[float, np.ndarray]:
        ctrl = np.ascontiguousarray(control, dtype=np.float64)
        obs = np.ctypeslib.as_array(
            self.lib.step_c(ctrl.ctypes.data_as(POINTER(c_double)), dt), shape=(5,)).copy()
        # C ABI: four pump flows, followed by six measured valve flows (kg/s).
        flow = np.ctypeslib.as_array(self.lib.get_mass_flowrate_c(), shape=(10,))[4:10].copy()
        return float(obs[0]), np.r_[obs[1:5], flow * (60000. / STD_RHO)]


def replay(sim, trajectory: Trajectory, coeff_micro: np.ndarray,
           frequency: float = 50.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use elapsed physical time for command lookup, never delayed observations.

    Commands are held from the most recent CSV sample at each interval start.
    The final partial interval is excluded. All ten real channels are linearly
    interpolated onto the very same output timestamps; no channel-specific lag
    or learned offset is introduced.
    """
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be positive")
    dt = 1. / frequency
    count = int(np.floor(float(trajectory.time[-1]) * frequency + 1e-8))
    if count < 1:
        raise ValueError("trajectory is shorter than one simulation step")
    times = np.arange(1, count + 1, dtype=float) * dt
    # The source clock has decimal sample times: roundoff at an exact boundary
    # should not defer the next command by an entire frame.
    indices = np.searchsorted(trajectory.time, times - dt + 1e-10, side="right") - 1
    indices = np.clip(indices, 0, len(trajectory.time) - 1)
    prediction = np.empty((count, len(CHANNEL_NAMES)))
    sim.reset(trajectory.values[0, :4], coeff_micro)
    for k, index in enumerate(indices):
        native_time, prediction[k] = sim.step(trajectory.controls[index], dt)
        if not np.isfinite(native_time) or abs(native_time - times[k]) > 1e-7 * max(1., times[k]):
            raise RuntimeError(f"native time {native_time} differs from replay time {times[k]}; "
                               "frequency must match the library integration step")
        if not np.isfinite(prediction[k]).all():
            raise FloatingPointError(f"nonfinite native result at {times[k]:.6f}s")
    truth = np.column_stack([np.interp(times, trajectory.time, trajectory.values[:, j])
                             for j in range(len(CHANNEL_NAMES))])
    return times, prediction, truth


def channel_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    result = {}
    for j, name in enumerate(CHANNEL_NAMES):
        error = prediction[:, j] - truth[:, j]
        sst = float(np.sum((truth[:, j] - np.mean(truth[:, j])) ** 2))
        result[name] = {"mae": float(np.mean(np.abs(error))),
                        "rmse": float(np.sqrt(np.mean(error ** 2))),
                        "bias": float(np.mean(error)),
                        "r2": float(1. - np.sum(error ** 2) / sst) if sst > 0 else None}
    return result


class PumpReplayTuner:
    def __init__(self, sim, trajectories: list[Trajectory], frequency: float = 50.,
                 on_evaluation: Callable[[dict], None] | None = None):
        self.sim = sim
        self.trajectories = trajectories
        self.frequency = frequency
        self.on_evaluation = on_evaluation
        self.evaluations = 0
        self.params = INITIAL_GUESS.copy()
        self.result = None

    def objective(self, params: np.ndarray) -> float:
        self.evaluations += 1
        coeff = np.asarray(params, dtype=float)
        record = {"evaluation": self.evaluations, "coeff_micro": coeff.tolist()}
        try:
            if coeff.shape != (2,) or not np.isfinite(coeff).all() or np.any(coeff < COEFF_MIN):
                raise ValueError("pump coefficients must be finite and positive")
            total = 0.
            per_data = []
            for data in self.trajectories:
                times, prediction, truth = replay(self.sim, data, coeff, self.frequency)
                metrics = channel_metrics(prediction, truth)
                value = sum(weight * metrics[name]["mae"] for name, weight in OBJECTIVE_WEIGHTS.items())
                total += value
                per_data.append({"path": str(data.path), "objective": value,
                                 "samples": len(times), "metrics": metrics})
            record.update(objective=total, datasets=per_data)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            total = 1e12
            record.update(objective=total, failure=str(exc))
        if self.on_evaluation:
            self.on_evaluation(record)
        return float(total)

    def tune(self, initial_guess: np.ndarray = INITIAL_GUESS,
             options: dict | None = None):
        self.result = optimize.minimize(
            self.objective, np.asarray(initial_guess, dtype=float), method="Nelder-Mead",
            bounds=optimize.Bounds([COEFF_MIN] * 2, [np.inf] * 2),
            options={"maxiter": 1000, "xatol": 1e-2, "fatol": 1e-2, **(options or {})})
        # The last objective evaluation need not be the optimizer's best point.
        self.params = np.asarray(self.result.x, dtype=float).copy()
        return self.result
