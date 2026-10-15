#!/usr/bin/env python3
"""Numerical local-observability audit of the lib3 pneumatic simulator.

The production dynamics are compiled into a separate analysis shared library.
The added C ABI only snapshots/restores the simulator's internal state; it does
not change the equations used by normal lib3 training.

The audit fixes the six valve branch modes and the six last-current memory
values, then differentiates the finite-horizon output map with respect to the
25-dimensional smooth-core state slice:

    4 measured pressures + crank phase + 2 pump pressures
    + 6 * (valve hysteresis z, flow q, flow-rate q_dot).

The full C++ snapshot also contains 6 continuous I_prev values and 6 binary
branch values. They are held fixed here so the finite-difference Jacobian stays
inside one smooth hybrid mode; this is not a claim that the full 37-scalar
execution state is observable.

It compares actuator-only vs all-pressure sensing, constant vs exciting known
inputs, zero vs five-sample measurement delay, and H1/H2/H5/H10/H20 output
windows.  It also verifies that [p, finite-difference p_dot] and H2 have the
same local row space: p_dot exposes the first dynamic output coordinate but
does not add information beyond the two pressure samples used to compute it.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIB3_DIR = ROOT / "pneu_env/src/pneu_env/lib3"
DEFAULT_OUTPUT = ROOT / "pneu_rl/exp/lib3_nonlinear_observability"

FREQ_HZ = 50.0
DT = 1.0 / FREQ_HZ
STATE_DIM = 37
ODE_STATE_DIM_NO_CLOCK = 7
VALVE_COUNT = 6
VALVE_RUNTIME_DIM = 5
PRESSURE_STATE_INDICES = np.asarray([0, 1, 5, 6], dtype=np.int64)
ACTUATOR_OUTPUT_INDICES = np.asarray([2, 3], dtype=np.int64)
VALVE_NAMES = (
    "ch_pos",
    "ch_neg",
    "act_pos_in",
    "act_pos_out",
    "act_neg_in",
    "act_neg_out",
)


def state_names() -> list[str]:
    names = [
        "P_ch_pos",
        "P_ch_neg",
        "theta",
        "P_pump_1",
        "P_pump_2",
        "P_act_pos",
        "P_act_neg",
    ]
    for valve in VALVE_NAMES:
        names.extend(
            [
                f"{valve}.z",
                f"{valve}.q",
                f"{valve}.q_dot",
                f"{valve}.I_prev",
                f"{valve}.branch",
            ]
        )
    return names


STATE_NAMES = state_names()
CONTINUOUS_INDICES = np.asarray(
    list(range(ODE_STATE_DIM_NO_CLOCK))
    + [
        ODE_STATE_DIM_NO_CLOCK + valve * VALVE_RUNTIME_DIM + component
        for valve in range(VALVE_COUNT)
        for component in range(3)
    ],
    dtype=np.int64,
)
CONTINUOUS_NAMES = [STATE_NAMES[index] for index in CONTINUOUS_INDICES]


@dataclass(frozen=True)
class RankResult:
    rank: int
    sigma_max: float
    sigma_min_retained: float
    condition_retained: float
    singular_values: np.ndarray


class Lib3Probe:
    def __init__(self, library: Path) -> None:
        self.lib = ctypes.CDLL(str(library))
        double_p = ctypes.POINTER(ctypes.c_double)
        self.lib.step_c.argtypes = [double_p, ctypes.c_double]
        self.lib.step_c.restype = double_p
        self.lib.set_init_env_act_c.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.lib.set_volume_c.argtypes = [ctypes.c_double, ctypes.c_double]
        self.lib.get_augmented_state_dim_c.restype = ctypes.c_int
        self.lib.get_augmented_state_c.argtypes = [double_p]
        self.lib.set_augmented_state_c.argtypes = [double_p]
        actual_dim = int(self.lib.get_augmented_state_dim_c())
        if actual_dim != STATE_DIM:
            raise RuntimeError(f"augmented state dimension {actual_dim} != {STATE_DIM}")
        self.lib.set_volume_c(0.75, 0.4)

    @staticmethod
    def _c_array(values: np.ndarray) -> ctypes.Array:
        values = np.asarray(values, dtype=np.float64)
        return (ctypes.c_double * len(values))(*values.tolist())

    def reset(self) -> None:
        atm = 101.325
        self.lib.set_init_env_act_c(atm, atm, atm, atm)

    def snapshot(self) -> np.ndarray:
        result = np.empty(STATE_DIM, dtype=np.float64)
        buffer = (ctypes.c_double * STATE_DIM)()
        self.lib.get_augmented_state_c(buffer)
        result[:] = list(buffer)
        return result

    def restore(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state shape {state.shape} != {(STATE_DIM,)}")
        self.lib.set_augmented_state_c(self._c_array(state))

    def output(self) -> np.ndarray:
        state = self.snapshot()
        return state[PRESSURE_STATE_INDICES].copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,):
            raise ValueError(f"action shape {action.shape} != (6,)")
        ptr = self.lib.step_c(self._c_array(action), DT)
        return np.asarray(list(ptr[1:5]), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--delay-steps", type=int, nargs="+", default=[0, 5])
    # Ten multisine steps produce a non-saturated local operating point.  A
    # long preparation can drive the calibrated ch_neg hysteresis state into
    # its +/-1e6 clamp, where a smooth Jacobian rank test is not valid.
    parser.add_argument("--prepare-steps", type=int, default=10)
    parser.add_argument("--fd-fraction", type=float, default=2e-4)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def compile_probe(output_library: Path) -> None:
    output_library.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-o",
        str(output_library),
        str(LIB3_DIR / "pneumatic_simulator.cpp"),
        str(LIB3_DIR / "pneumatic_CT.cpp"),
    ]
    subprocess.run(command, check=True)


def multisine_actions(length: int, phase_offset: float = 0.0) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    actions = np.empty((length, 6), dtype=np.float64)
    for channel in range(6):
        phase = phase_offset + 0.71 * channel
        actions[:, channel] = (
            0.885
            + 0.055 * np.sin(2.0 * np.pi * t / (17.0 + 2.0 * channel) + phase)
            + 0.020 * np.sin(2.0 * np.pi * t / (7.0 + channel) + 0.5 * phase)
        )
    return np.clip(actions, 0.805, 0.975)


def plateau_actions(length: int, final_prepare_action: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(final_prepare_action, dtype=np.float64), (length, 1))


def prepare_operating_state(probe: Lib3Probe, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probe.reset()
    actions = multisine_actions(steps, phase_offset=0.31)
    snapshots = [probe.snapshot()]
    for action in actions:
        probe.step(action)
        snapshots.append(probe.snapshot())
    return probe.snapshot(), np.asarray(snapshots), actions[-1].copy()


def continuous_scales(snapshot: np.ndarray, preparation: np.ndarray) -> np.ndarray:
    scales = np.empty(len(CONTINUOUS_INDICES), dtype=np.float64)
    for column, state_index in enumerate(CONTINUOUS_INDICES):
        name = STATE_NAMES[state_index]
        values = preparation[:, state_index]
        dynamic = max(float(np.ptp(values)), 3.0 * float(np.std(values)))
        magnitude = 0.1 * abs(float(snapshot[state_index]))
        if name.startswith("P_"):
            floor = 5.0
        elif name == "theta":
            floor = 0.5
        elif name.endswith(".z"):
            floor = 1.0
        elif name.endswith(".q"):
            floor = 0.2
        else:
            floor = 2.0
        scales[column] = max(dynamic, magnitude, floor)
    return scales


def rollout_outputs(probe: Lib3Probe, snapshot: np.ndarray, actions: np.ndarray) -> np.ndarray:
    probe.restore(snapshot)
    rows = [probe.output()]
    for action in actions:
        rows.append(probe.step(action))
    return np.asarray(rows, dtype=np.float64)


def finite_difference_jacobian(
    probe: Lib3Probe,
    snapshot: np.ndarray,
    actions: np.ndarray,
    scales: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = rollout_outputs(probe, snapshot, actions)
    jacobian = np.empty((baseline.size, len(CONTINUOUS_INDICES)), dtype=np.float64)
    for column, (state_index, scale) in enumerate(zip(CONTINUOUS_INDICES, scales)):
        delta = max(float(scale) * fraction, 1e-8)
        plus = snapshot.copy()
        minus = snapshot.copy()
        plus[state_index] += delta
        minus[state_index] -= delta
        y_plus = rollout_outputs(probe, plus, actions)
        y_minus = rollout_outputs(probe, minus, actions)
        jacobian[:, column] = ((y_plus - y_minus) / (2.0 * delta)).reshape(-1)
    probe.restore(snapshot)
    return baseline, jacobian


def select_output_rows(
    full_jacobian: np.ndarray,
    total_samples: int,
    delay: int,
    window: int,
    sensor_indices: np.ndarray,
) -> np.ndarray:
    tensor = full_jacobian.reshape(total_samples, 4, -1)
    selected = tensor[delay : delay + window, sensor_indices, :]
    return selected.reshape(-1, tensor.shape[-1])


def scaled_rank(jacobian: np.ndarray, state_scales: np.ndarray, rtol: float) -> RankResult:
    # A 10 kPa output perturbation is one dimensionless pressure unit.
    scaled = jacobian * state_scales[None, :] / 10.0
    singular = np.linalg.svd(scaled, compute_uv=False)
    threshold = rtol * singular[0] if singular.size and singular[0] > 0.0 else np.inf
    retained = singular[singular > threshold]
    rank = int(retained.size)
    sigma_max = float(singular[0]) if singular.size else 0.0
    sigma_min = float(retained[-1]) if retained.size else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    return RankResult(rank, sigma_max, sigma_min, condition, singular)


def row_space_projector(matrix: np.ndarray, rtol: float) -> np.ndarray:
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    rank = int(np.sum(singular > rtol * singular[0]))
    basis = vh[:rank].T
    return basis @ basis.T


def configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    if min(args.windows) < 1 or min(args.delay_steps) < 0:
        raise ValueError("windows must be positive and delay steps non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    library = args.output_dir / "libpneumatic_simulator_observability.so"
    if not args.no_build:
        compile_probe(library)
    if not library.is_file():
        raise FileNotFoundError(library)

    probe = Lib3Probe(library)
    operating_state, preparation, final_action = prepare_operating_state(
        probe, args.prepare_steps
    )
    scales = continuous_scales(operating_state, preparation)
    max_window = max(args.windows)
    max_delay = max(args.delay_steps)
    action_count = max_delay + max_window - 1
    scenario_actions = {
        "plateau": plateau_actions(action_count, final_action),
        "exciting": multisine_actions(action_count, phase_offset=1.17),
    }
    sensors = {
        "actuator_2": ACTUATOR_OUTPUT_INDICES,
        "all_pressure_4": np.arange(4, dtype=np.int64),
    }

    summary_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    equivalence_rows: list[dict[str, object]] = []
    spectra: dict[tuple[str, str, int, int], np.ndarray] = {}

    for scenario, actions in scenario_actions.items():
        baseline, full_jacobian = finite_difference_jacobian(
            probe,
            operating_state,
            actions,
            scales,
            args.fd_fraction,
        )
        total_samples = len(baseline)
        for sensor_name, sensor_indices in sensors.items():
            for delay in args.delay_steps:
                for window in args.windows:
                    jacobian = select_output_rows(
                        full_jacobian,
                        total_samples,
                        delay,
                        window,
                        sensor_indices,
                    )
                    result = scaled_rank(jacobian, scales, args.rank_rtol)
                    spectra[(scenario, sensor_name, delay, window)] = result.singular_values
                    summary_rows.append(
                        {
                            "scenario": scenario,
                            "sensors": sensor_name,
                            "delay_steps": delay,
                            "window_samples": window,
                            "history_seconds": (window - 1) * DT,
                            "rows": jacobian.shape[0],
                            "continuous_states": jacobian.shape[1],
                            "numerical_rank": result.rank,
                            "sigma_max_scaled": result.sigma_max,
                            "sigma_min_retained_scaled": result.sigma_min_retained,
                            "condition_retained": result.condition_retained,
                        }
                    )

                longest = select_output_rows(
                    full_jacobian,
                    total_samples,
                    delay,
                    max_window,
                    sensor_indices,
                )
                column_norms = np.linalg.norm(longest * scales[None, :] / 10.0, axis=0)
                for state_name, norm in zip(CONTINUOUS_NAMES, column_norms):
                    sensitivity_rows.append(
                        {
                            "scenario": scenario,
                            "sensors": sensor_name,
                            "delay_steps": delay,
                            "window_samples": max_window,
                            "state": state_name,
                            "scaled_column_norm": float(norm),
                        }
                    )

                if max_window >= 2:
                    h2 = select_output_rows(
                        full_jacobian, total_samples, delay, 2, sensor_indices
                    )
                    output_dim = len(sensor_indices)
                    p0 = h2[:output_dim]
                    p1 = h2[output_dim:]
                    p_rate = np.vstack([p0, (p1 - p0) / DT])
                    h2_scaled = h2 * scales[None, :]
                    p_rate_scaled = p_rate * scales[None, :]
                    projector_gap = np.linalg.norm(
                        row_space_projector(h2_scaled, args.rank_rtol)
                        - row_space_projector(p_rate_scaled, args.rank_rtol),
                        ord="fro",
                    )
                    equivalence_rows.append(
                        {
                            "scenario": scenario,
                            "sensors": sensor_name,
                            "delay_steps": delay,
                            "rank_H2": scaled_rank(h2, scales, args.rank_rtol).rank,
                            "rank_p_plus_pdot": scaled_rank(
                                p_rate, scales, args.rank_rtol
                            ).rank,
                            "row_space_projector_gap": float(projector_gap),
                        }
                    )

    summary_path = args.output_dir / "observability_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    sensitivity_path = args.output_dir / "state_sensitivity_h20.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    equivalence_path = args.output_dir / "rate_history_equivalence.csv"
    with equivalence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(equivalence_rows[0]))
        writer.writeheader()
        writer.writerows(equivalence_rows)

    state_path = args.output_dir / "operating_state_and_scales.csv"
    with state_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "value", "continuous_audit", "characteristic_scale"])
        scale_by_index = dict(zip(CONTINUOUS_INDICES.tolist(), scales.tolist()))
        for index, (name, value) in enumerate(zip(STATE_NAMES, operating_state)):
            writer.writerow(
                [name, float(value), index in scale_by_index, scale_by_index.get(index, "")]
            )

    configure_plotting()
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, sharey=True)
    for row, scenario in enumerate(("plateau", "exciting")):
        for column, sensor_name in enumerate(("actuator_2", "all_pressure_4")):
            axis = axes[row, column]
            for delay, color, marker in ((0, "#3B6EA5", "o"), (5, "#E24A33", "s")):
                selected = [
                    item
                    for item in summary_rows
                    if item["scenario"] == scenario
                    and item["sensors"] == sensor_name
                    and item["delay_steps"] == delay
                ]
                selected.sort(key=lambda item: int(item["window_samples"]))
                axis.plot(
                    [int(item["window_samples"]) for item in selected],
                    [int(item["numerical_rank"]) for item in selected],
                    marker=marker,
                    color=color,
                    linewidth=2.0,
                    label=f"delay={delay} samples",
                )
            axis.axhline(len(CONTINUOUS_INDICES), color="black", linestyle=":", linewidth=1.0)
            scenario_label = "Constant input" if scenario == "plateau" else "Time-varying input"
            sensor_label = (
                "actuator pressures (2)"
                if sensor_name == "actuator_2"
                else "all pressures (4)"
            )
            axis.set_title(f"{scenario_label}; {sensor_label}")
            axis.set_xlabel("Output-history samples")
            axis.set_ylabel("Numerical local rank")
            axis.set_xticks(args.windows)
            axis.legend(loc="lower right")
    figure.suptitle("lib3 finite-horizon output-map rank", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    rank_plot = args.output_dir / "rank_vs_history.png"
    figure.savefig(rank_plot, dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for axis, scenario in zip(axes, ("plateau", "exciting")):
        for delay, color in ((0, "#3B6EA5"), (5, "#E24A33")):
            singular = spectra[(scenario, "all_pressure_4", delay, max_window)]
            axis.semilogy(
                np.arange(1, len(singular) + 1),
                singular,
                marker="o",
                markersize=3.5,
                color=color,
                linewidth=1.8,
                label=f"delay={delay} samples",
            )
        axis.set_title(f"{scenario}; all 4 pressures; H{max_window}")
        axis.set_xlabel("Singular-value index")
        axis.set_ylabel("Scaled singular value")
        axis.legend()
    figure.suptitle("lib3 observability spectrum", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    spectrum_plot = args.output_dir / "singular_spectrum_h20.png"
    figure.savefig(spectrum_plot, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "continuous_state_dimension": len(CONTINUOUS_INDICES),
        "full_execution_state_dimension_excluding_delay_buffer": STATE_DIM,
        "hybrid_memory_fixed": [
            name
            for name in STATE_NAMES
            if name.endswith("I_prev") or name.endswith("branch")
        ],
        "frequency_hz": FREQ_HZ,
        "windows": args.windows,
        "delay_steps": args.delay_steps,
        "finite_difference_fraction": args.fd_fraction,
        "rank_relative_tolerance": args.rank_rtol,
        "common_operating_state": operating_state.tolist(),
        "final_preparation_action_physical": final_action.tolist(),
        "analysis_library": str(library),
    }
    with (args.output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"saved: {summary_path}")
    print(f"saved: {equivalence_path}")
    print(f"saved: {rank_plot}")
    print(f"saved: {spectrum_plot}")


if __name__ == "__main__":
    main()
