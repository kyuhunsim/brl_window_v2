"""Independent feasibility checks for the area-state valve implementation.

This script does not fit parameters, correct sensors, or clip pressure outputs.
Recorded commands are replayed separately for each fitted type.  Invariants and
integration-step sensitivity are reported separately from tracking accuracy.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TUNE_EXP = ROOT.parent / "pneu_tune/exp"
sys.path.insert(0, str(ROOT / "pneu_env/src"))
from pneu_env.pump_replay import (CHANNEL_NAMES, PRESSURE_NAMES, NativeLib3,
                                  channel_metrics, load_trajectory, replay)

PTR = ctypes.POINTER(ctypes.c_double)
ATM = 101.325


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_library(directory, header, step, macros):
    source = ROOT / "pneu_env/src/pneu_env/lib3"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("pneumatic_simulator.h", "pneumatic_simulator.cpp", "pneumatic_CT.cpp"):
        shutil.copy2(source / name, directory / name)
    hp = directory / "pneumatic_simulator.h"
    text = hp.read_text()
    if "#ifndef TS" not in text:
        assert text.count("#define TS 0.0001") == 1
        hp.write_text(text.replace("#define TS 0.0001", f"#define TS {step:.17g}"))
        timestep_flag = []
    else:
        timestep_flag = [f"-DTS={step:.17g}"]
    library = directory / "lib3_physical.so"
    command = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
               *timestep_flag, *[f"-D{name}" for name in macros],
               f'-DPNEU_VALVE_PARAMS_HEADER="{Path(header).resolve()}"',
               str(directory / "pneumatic_simulator.cpp"),
               str(directory / "pneumatic_CT.cpp"), "-o", str(library)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return library, command


class InspectableSim(NativeLib3):
    def __init__(self, path):
        super().__init__(path)
        self.lib.get_augmented_state_dim_c.restype = ctypes.c_int
        self.state_dimension = self.lib.get_augmented_state_dim_c()
        self.lib.get_augmented_state_c.argtypes = [PTR]
        if hasattr(self.lib, "set_augmented_state_checked_c"):
            self.lib.set_augmented_state_checked_c.argtypes = [PTR, ctypes.c_int]
            self.lib.set_augmented_state_checked_c.restype = ctypes.c_int
        else:
            self.lib.set_augmented_state_c.argtypes = [PTR]
        self.lib.get_valve_debug_c.restype = PTR
        self.state_minimum = np.full((6, 2), np.inf)
        self.state_maximum = np.full((6, 2), -np.inf)
        self.gas_minimum = np.full(6, np.inf)
        self.gas_maximum = np.full(6, -np.inf)

    def reset(self, pressures, coeff):
        super().reset(pressures, coeff)
        self.state_minimum[:] = np.inf
        self.state_maximum[:] = -np.inf
        self.gas_minimum[:] = np.inf
        self.gas_maximum[:] = -np.inf

    def step(self, control, dt):
        result = super().step(control, dt)
        snapshot = self.snapshot()
        state = snapshot[7:].reshape(6, 5)[:, :2]
        self.state_minimum = np.minimum(self.state_minimum, state)
        self.state_maximum = np.maximum(self.state_maximum, state)
        gas = snapshot[[0, 1, 3, 4, 5, 6]]
        self.gas_minimum = np.minimum(self.gas_minimum, gas)
        self.gas_maximum = np.maximum(self.gas_maximum, gas)
        return result

    def state_domain(self):
        return dict(minimum_h=float(self.state_minimum[:, 0].min()),
                    maximum_h=float(self.state_maximum[:, 0].max()),
                    minimum_opening=float(self.state_minimum[:, 1].min()),
                    maximum_opening=float(self.state_maximum[:, 1].max()),
                    minimum_all_six_gas_pressures_kpa=self.gas_minimum.tolist(),
                    maximum_all_six_gas_pressures_kpa=self.gas_maximum.tolist(),
                    all_six_gas_pressures_positive=bool((self.gas_minimum > 0.).all()),
                    h_bounded=bool((self.state_minimum[:, 0] >= -1.-1e-12).all()
                                   and (self.state_maximum[:, 0] <= 1.+1e-12).all()),
                    opening_bounded=bool((self.state_minimum[:, 1] >= 0.).all()
                                         and (self.state_maximum[:, 1] <= 1.).all()))

    def snapshot(self):
        state = np.empty(self.state_dimension)
        self.lib.get_augmented_state_c(state.ctypes.data_as(PTR))
        return state

    def valve_debug(self):
        return np.ctypeslib.as_array(self.lib.get_valve_debug_c(), shape=(54,)).copy().reshape(6, 9)


def heads(pressure):
    pressure = np.asarray(pressure)
    return np.stack((pressure[..., 0]-ATM, ATM-pressure[..., 1],
                     pressure[..., 0]-pressure[..., 2], pressure[..., 2]-ATM,
                     ATM-pressure[..., 3], pressure[..., 3]-pressure[..., 1]), axis=-1)


def domain_summary(prediction):
    pressure, flow = prediction[:, :4], prediction[:, 4:]
    bad = (heads(pressure) < -1e-7) & (flow > 1e-6)
    return dict(finite=bool(np.isfinite(prediction).all()),
                strictly_positive_absolute_pressure=bool(np.all(pressure > 0)),
                nonnegative_valve_flow=bool(np.all(flow >= -1e-12)),
                minimum_pressure_kpa={name: float(pressure[:, i].min())
                                      for i, name in enumerate(PRESSURE_NAMES)},
                maximum_pressure_kpa={name: float(pressure[:, i].max())
                                      for i, name in enumerate(PRESSURE_NAMES)},
                wrong_head_nonzero_flow_samples=int(bad.sum()),
                max_wrong_head_flow_lpm=float(flow[bad].max()) if bad.any() else 0.)


def synthetic_runs(sim, coeff, duration):
    scenarios = {
        "full_open": np.ones(6),
        "all_at_low_command": np.full(6, .8),
        "act_pos_exhaust_only": np.array([1., 1., .8, 1., .8, .8]),
        "act_neg_supply_only": np.array([1., 1., .8, .8, 1., .8]),
        "actuator_valves_at_low_command": np.array([1., 1., .8, .8, .8, .8]),
    }
    results = {}
    for name, control in scenarios.items():
        initial = np.array([180., 45., 150., 70.])
        sim.reset(initial, coeff)
        values, debug, state_bounds = [], [], []
        for k in range(round(duration*50)):
            _, row = sim.step(control, .02)
            values.append(row)
            debug.append(sim.valve_debug())
            state_bounds.append(sim.snapshot()[7:].reshape(6, 5)[:, :2])
        values, debug, state_bounds = map(np.asarray, (values, debug, state_bounds))
        # Pump ripple is physical in this model. Compare slow-window ranges
        # after settling; do not label every nonzero variance an instability.
        width = max(1, min(250, len(values)//4))
        spans = [np.ptp(chunk[:, :4], axis=0) for chunk in
                 (values[-2*width:-width], values[-width:]) if len(chunk)]
        result = domain_summary(values)
        result.update(control=control.tolist(), duration_sec=duration,
                      maximum_abs_h=float(np.abs(state_bounds[:, :, 0]).max()),
                      minimum_area_state=float(state_bounds[:, :, 1].min()),
                      maximum_area_state=float(state_bounds[:, :, 1].max()),
                      late_pressure_ranges_kpa=np.asarray(spans).tolist(),
                      pressure_drift_last_5s_kpa=(values[-1, :4]-values[-width, :4]).tolist(),
                      debug_finite=bool(np.isfinite(debug).all()))
        result["state_domain"] = sim.state_domain()
        if name == "act_pos_exhaust_only":
            result["act_pos_below_atmosphere_kpa"] = max(0., ATM-float(values[:, 2].min()))
        if name == "act_neg_supply_only":
            result["act_neg_above_atmosphere_kpa"] = max(0., float(values[:, 3].max())-ATM)
        results[name] = result
    return results


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(purpose="Whole-model feasibility and numerical refinement; no fitting.",
                  no_pressure_clipping_in_validation=True, kinds={},
                  native_steps_sec=args.steps, macros=args.macros,
                  limitations=["Frozen pump coefficients were fitted with the previous valve dynamics.",
                               "Type1 and type2 source recordings were used during fitting, not reserved validation.",
                               "Endpoint flow-direction audit requires flows refreshed at the final pressure state.",
                               "Finite stress tests establish tested cases only, not a global positivity proof."])
    with tempfile.TemporaryDirectory(prefix="lib3_physical_validation_") as temporary:
        for kind in (("type1", "type2") if args.type == "all" else (args.type,)):
            report_path = args.pump_root / kind / "report.json"
            old_report = json.loads(report_path.read_text())
            coeff = np.asarray(old_report["coeff_micro"])
            path = Path(old_report["datasets"][0]["path"])
            trajectory = load_trajectory(path, duration_sec=args.duration)
            header = args.build_root / kind / "valve_parameters.h"
            kind_result = dict(coefficients_micro=coeff.tolist(), source=str(path),
                               source_sha256=digest(path), header_sha256=digest(header), steps=[])
            reference = None
            previous = None
            for step in args.steps:
                library, command = compile_library(Path(temporary)/kind/format(step, ".9g"),
                                                   header, step, args.macros)
                sim = InspectableSim(library)
                before = time.perf_counter()
                times, prediction, truth = replay(sim, trajectory, coeff)
                record = dict(native_step_sec=step, compile_command=command,
                              library_sha256=digest(library), replay_seconds=time.perf_counter()-before,
                              samples=len(times), domain=domain_summary(prediction),
                              state_domain=sim.state_domain(),
                              metrics=channel_metrics(prediction, truth))
                if reference is None:
                    reference = prediction.copy()
                else:
                    delta = prediction-reference
                    record["difference_from_coarsest"] = {
                        name: dict(rmse=float(np.sqrt(np.mean(delta[:, j]**2))),
                                   max_absolute=float(np.abs(delta[:, j]).max()))
                        for j, name in enumerate(CHANNEL_NAMES)}
                if previous is not None:
                    delta = prediction - previous
                    record["difference_from_previous_step"] = {
                        name: dict(rmse=float(np.sqrt(np.mean(delta[:, j]**2))),
                                   max_absolute=float(np.abs(delta[:, j]).max()))
                        for j, name in enumerate(CHANNEL_NAMES)}
                previous = prediction.copy()
                record["synthetic"] = synthetic_runs(sim, coeff, args.synthetic_duration)
                kind_result["steps"].append(record)
            result["kinds"][kind] = kind_result
            print(json.dumps({"type": kind, "steps": [{"native_dt": s["native_step_sec"],
                  "domain": s["domain"], "replay_seconds": s["replay_seconds"]}
                  for s in kind_result["steps"]]}), flush=True)
    result["elapsed_sec"] = time.perf_counter()-started
    result["source_hashes"] = {str(path): digest(path) for path in
                               (ROOT/"pneu_env/src/pneu_env/lib3").glob("*.cpp")}
    (args.output/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("all", "type1", "type2"), default="all")
    parser.add_argument("--build-root", type=Path,
                        default=TUNE_EXP/"lib3_feasible_integration_20261022")
    parser.add_argument("--pump-root", type=Path,
                        default=TUNE_EXP/"lib3_pump_feasible_20261022")
    parser.add_argument("--output", type=Path,
                        default=TUNE_EXP/"lib3_physical_model_20261023")
    parser.add_argument("--duration", type=float, help="Replay source prefix (seconds); default all.")
    parser.add_argument("--synthetic-duration", type=float, default=30.)
    parser.add_argument("--steps", nargs="+", type=float, default=[.0001, .00005, .000025])
    parser.add_argument("--macros", nargs="+",
                        default=["PNEU_EXACT_VALVE_HYSTERESIS", "PNEU_OPENING_DYNAMICS"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
