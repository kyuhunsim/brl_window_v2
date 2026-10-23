"""Check the compiled lib3 valve path against the independent calibration core.

This checks numerical implementation parity, not simulator accuracy. Each type
keeps its own six fitted parameter vectors. No fit, sensor correction, or new
full-opening threshold is introduced. Libraries are compiled in /tmp; the
production shared libraries are never overwritten.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TUNE_ROOT = ROOT.parent / "pneu_tune"
PTR = ctypes.POINTER(ctypes.c_double)
NATIVE_DT = .0001 / 4
PRESSURE_COLUMNS = (
    ("press_pos", None), (None, "press_neg"),
    ("press_pos", "act_pos_press"), ("act_pos_press", None),
    (None, "act_neg_press"), ("act_neg_press", "press_neg"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phi(pin, pout):
    ratio = np.clip(pout / np.maximum(pin, 1e-9), 0, 1)
    critical = (2 / 2.4) ** (1.4 / .4)
    return np.where(ratio <= critical, np.sqrt(1.4 * (2 / 2.4) ** 6),
                    np.sqrt(np.maximum(0, 7 * (ratio ** (2 / 1.4)
                                               - ratio ** (2.4 / 1.4)))))


def direction_for(current, initial=0.):
    direction = np.empty(len(current), dtype=np.float64)
    direction[0] = initial
    for k in range(1, len(current)):
        direction[k] = (1. if current[k] > current[k - 1] + 1e-4 else
                        0. if current[k] < current[k - 1] - 1e-4 else
                        direction[k - 1])
    if len(current) > 1:
        direction[0] = direction[1]
    return direction


def compile_wrapper(source, header, directory, fast_flow=False):
    """Include the actual production translation unit, adding only a test ABI."""
    wrapper = directory / "parity_wrapper.cpp"
    wrapper.write_text(r'''
#include "pneumatic_CT.cpp"
extern "C" int production_replay(int n, int valve, const double* command,
    const double* pin, const double* pout, double dt, int substeps, int repeats,
    double initial_q, double initial_h, double initial_current,
    double initial_direction, double* q, double* h, double* area,
    double* force, double* raw_q) {
    if (valve < 1 || valve > 6 || n < 1) return 1;
    const auto previous_stream_state = std::cout.rdstate();
    std::cout.setstate(std::ios_base::failbit);
    PneumaticCT model;
    std::cout.clear(previous_stream_state);
    double runtime[VALVE_COUNT * VALVE_RUNTIME_DIM] = {};
    const int base = (valve-1) * VALVE_RUNTIME_DIM;
    runtime[base] = initial_h;
    runtime[base+1] = initial_q;
    runtime[base+3] = initial_current;
    runtime[base+4] = initial_direction;
    model.set_valve_runtime_states(runtime);
    for (int k = 0; k < n; ++k) {
        for (int repeat = 0; repeat < repeats; ++repeat)
            model.solenoid_valve(pin[k], pout[k], command[k], valve, 1., dt/repeats, substeps);
        const double* debug = model.get_valve_debug() + (valve-1)*9;
        q[k] = debug[7]; h[k] = debug[3];
        area[k] = debug[5]; force[k] = debug[4];
        model.get_valve_runtime_states(runtime);
        raw_q[k] = runtime[base+1];
    }
    return 0;
}
''')
    library = directory / "lib3_valve_parity.so"
    command = ["g++", "-std=c++11", "-O3", "-shared", "-fPIC",
               "-DPNEU_EXACT_VALVE_HYSTERESIS",
               '-DPNEU_VALVE_PARAMS_HEADER="' + str(header.resolve()) + '"',
               "-I", str(source.parent), str(wrapper), "-o", str(library)]
    if fast_flow:
        command.insert(1, "-DPNEU_FAST_VALVE_FLOW")
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = ctypes.CDLL(str(library))
    result.production_replay.argtypes = ([ctypes.c_int, ctypes.c_int]
        + [PTR] * 3 + [ctypes.c_double, ctypes.c_int, ctypes.c_int]
        + [ctypes.c_double] * 4 + [PTR] * 5)
    result.production_replay.restype = ctypes.c_int
    return result, command


def run_case(core, production, valve, command, pin, pout, measured, parameters,
             dt, label, native_reset=False):
    arrays = [np.ascontiguousarray(a, dtype=np.float64)
              for a in (command, pin, pout, measured)]
    command, pin, pout, measured = arrays
    current = np.ascontiguousarray(.3 * np.clip((command - .5) * 2, 0, 1))
    direction = direction_for(current)
    if native_reset:
        # Prepend a genuine zero-current, zero-flow call to both implementations.
        # The core's first-sample rule then matches a native runtime reset.
        command = np.r_[.5, command]
        current = np.r_[0., current]
        pin, pout = np.r_[pin[0], pin], np.r_[pout[0], pout]
        measured = np.zeros(len(command))
        direction = direction_for(current)
        direction[0] = 0.
    n = len(command)
    capacity = np.ascontiguousarray(phi(pin, pout))
    qreal = np.ascontiguousarray(measured)
    reference = [np.full(n, np.nan) for _ in range(4)]
    status_core = core.simulate_feasible(n,
        *[a.ctypes.data_as(PTR) for a in (current, direction, pin, capacity, qreal)],
        dt, parameters.ctypes.data_as(PTR),
        *[a.ctypes.data_as(PTR) for a in reference])
    actual = [np.full(n, np.nan) for _ in range(5)]
    status_lib3 = production.production_replay(n, valve,
        *[a.ctypes.data_as(PTR) for a in (command, pin, pout)], dt, 20,
        1, float(qreal[0]), 0., float(current[0]), float(direction[0]),
        *[a.ctypes.data_as(PTR) for a in actual])
    # Calibration reports normalized area, while deployed debug gives Amax*area.
    actual[2] /= parameters[0]
    metrics = {}
    for name, expected, observed in zip(("flow_lpm", "h", "area_ratio", "force"),
                                         reference, actual):
        valid = np.isfinite(expected) & np.isfinite(observed)
        delta = abs(observed[valid] - expected[valid])
        metrics[name] = dict(
            compared=int(valid.sum()),
            max_absolute_error=float(delta.max()) if len(delta) else None,
            max_scaled_error=float(np.max(delta / (1 + abs(expected[valid]))))
                if len(delta) else None)
    finite = all(np.isfinite(a).all() for a in actual)
    parity = (status_core == 0 and status_lib3 == 0 and finite
              and all(m["max_scaled_error"] <= 1e-10 for m in metrics.values()))
    return dict(case=label, valve=valve, samples=n, dt=dt, substeps=20,
                core_status=int(status_core), lib3_status=int(status_lib3),
                parity_pass=bool(parity), metrics=metrics,
                finite=bool(finite), max_abs_h=float(np.max(abs(actual[1]))),
                min_area_ratio=float(actual[2].min()),
                max_area_ratio=float(actual[2].max()),
                max_abs_force=float(np.max(abs(actual[3]))),
                min_raw_flow_lpm=float(actual[4].min()),
                negative_raw_flow_samples=int(np.count_nonzero(actual[4] < -1e-9)),
                note=("Native reset emulated by an initial zero-current call."
                      if native_reset else "Initial flow/current/direction match calibration."))


def check_native_sweep(production, valve, inputs, parameters):
    """Repeat actual 25us lib3 calls for each 20ms commanded sweep sample."""
    command, pin, pout, measured = [np.ascontiguousarray(a) for a in inputs]
    n = len(command)
    outputs = [np.full(n, np.nan) for _ in range(5)]
    status = production.production_replay(n, valve,
        *[a.ctypes.data_as(PTR) for a in (command, pin, pout)], .02, 20, 800,
        float(measured[0]), 0., float(.3*np.clip(2*command[0]-1, 0, 1)), 0.,
        *[a.ctypes.data_as(PTR) for a in outputs])
    q, h, area, force, raw_q = outputs
    return dict(case="pressure_and_history_sweep_native_updates", valve=valve,
                samples=n, native_calls=n*800, valve_dt=NATIVE_DT,
                finite=bool(all(np.isfinite(a).all() for a in outputs)),
                lib3_status=int(status), max_abs_h=float(max(abs(h))),
                min_area_ratio=float(area.min()/parameters[0]),
                max_area_ratio=float(area.max()/parameters[0]),
                max_abs_force=float(max(abs(force))),
                min_raw_flow_lpm=float(raw_q.min()),
                negative_raw_flow_samples=int(np.count_nonzero(raw_q < -1e-9)),
                note="Flow output retains existing max(raw second-order state,0); state is recorded before that guard.")


def synthetic_case(pin, pout, parameters, n=2501):
    """Slow reversals and a pressure sweep inside recorded pressure ranges."""
    t = np.arange(n) * .02
    command = np.interp(t, [0, 2, 8, 12, 18, 22, 28, 32, 38, 42, 50],
                       [.8, .8, 1, 1, .82, .82, .98, .98, .8, .8, 1])
    pin_sweep = np.median(pin) + .4 * np.ptp(pin) * np.sin(t / 7)
    pout_sweep = np.median(pout) + .4 * np.ptp(pout) * np.sin(t / 9 + 1)
    # Keep a positive head here; equal/reverse head is checked separately.
    pout_sweep = np.minimum(pout_sweep, pin_sweep - .5)
    force0 = .3 * (2 * command[0] - 1) + parameters[3] * pin_sweep[0] - parameters[2]
    area0 = np.exp(-parameters[8] * np.logaddexp(0, -parameters[1] * force0))
    initial_q = parameters[0] * area0 * pin_sweep[0] * phi(pin_sweep[:1], pout_sweep[:1])[0]
    return command, pin_sweep, pout_sweep, np.full(n, initial_q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", choices=("all", "type1", "type2"), default="all")
    parser.add_argument("--tune-root", type=Path, default=TUNE_ROOT)
    parser.add_argument("--parameter-root", type=Path,
                        default=TUNE_ROOT / "exp/lib3_all_valves_feasible_20261022")
    parser.add_argument("--build-root", type=Path,
                        default=TUNE_ROOT / "exp/lib3_feasible_integration_20261022")
    parser.add_argument("--output", type=Path,
                        help="Defaults to build-root/parity[_fast].json")
    parser.add_argument("--fast-flow", action="store_true",
                        help="Validate the optional affine 20-substep flow update.")
    parser.add_argument("--skip-native-sweeps", action="store_true",
                        help="Skip the expensive native-timestep stress sweeps already checked separately.")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.build_root / ("parity_fast.json" if args.fast_flow else "parity.json")
    core_path = ROOT / "pneu_env/scripts/retry_lib3_valve_feasible.py"
    spec = importlib.util.spec_from_file_location("calibration_core", core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    core, core_source, _ = module.build_core()
    source = ROOT / "pneu_env/src/pneu_env/lib3/pneumatic_CT.cpp"
    kinds = ("type1", "type2") if args.types == "all" else (args.types,)
    cases, native_sweeps, hashes, compile_commands = [], [], {}, {}
    hashes[str(source)] = digest(source)
    hashes[str(core_source)] = digest(core_source)
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="lib3_valve_parity_") as temporary:
        for kind in kinds:
            directory = Path(temporary) / kind
            directory.mkdir()
            header = args.build_root / kind / "valve_parameters.h"
            if not header.exists():
                raise FileNotFoundError(f"Build candidate parameters first: {header}")
            production, compile_commands[kind] = compile_wrapper(source, header, directory,
                                                                 args.fast_flow)
            hashes[str(header)] = digest(header)
            raw_path = next((args.tune_root / "exp").glob(f"261022*{kind}*.csv"))
            hashes[str(raw_path)] = digest(raw_path)
            raw = pd.read_csv(raw_path)
            dt = float(np.mean(np.diff(raw.time)))
            for valve in range(1, 7):
                parameters_path = args.parameter_root / kind / f"v{valve}/parameters.npz"
                hashes[str(parameters_path)] = digest(parameters_path)
                parameters = np.ascontiguousarray(np.load(parameters_path)["parameters"])
                columns = PRESSURE_COLUMNS[valve - 1]
                pin, pout = [np.full(len(raw), 101.325) if column is None
                             else raw[column].to_numpy(dtype=np.float64)
                             for column in columns]
                command = raw[f"ctrl{valve}"].to_numpy(dtype=np.float64)
                measured = raw[f"flow{valve}"].to_numpy(dtype=np.float64)
                for name, step, native in (("measured_trace_calibration_dt", dt, False),
                                           ("measured_sequence_native_dt", NATIVE_DT, False),
                                           ("native_reset", NATIVE_DT, True)):
                    result = run_case(core, production, valve, command, pin, pout,
                                      measured, parameters, step, name, native)
                    result["type"] = kind
                    cases.append(result)
                synthetic = synthetic_case(pin, pout, parameters)
                result = run_case(core, production, valve, *synthetic, parameters,
                                  .02, "pressure_and_history_sweep")
                result["type"] = kind
                cases.append(result)
                if not args.skip_native_sweeps:
                    native = check_native_sweep(production, valve, synthetic, parameters)
                    native["type"] = kind
                    native_sweeps.append(native)
                # Fresh zero-flow states at zero and reverse pressure head.
                for label, p_out in (("equal_pressure", 101.325),
                                     ("reverse_pressure", 120.)):
                    n = 501
                    result = run_case(core, production, valve, np.ones(n),
                                      np.full(n, 101.325), np.full(n, p_out),
                                      np.zeros(n), parameters, NATIVE_DT, label)
                    result["type"] = kind
                    cases.append(result)
    summary = dict(
        fast_flow=args.fast_flow,
        parity_pass=all(c["parity_pass"] for c in cases),
        recorded_sequence_parity_pass=all(c["parity_pass"] for c in cases
            if c["case"] in ("measured_trace_calibration_dt",
                              "measured_sequence_native_dt", "native_reset")),
        native_numeric_stability_pass=(all(c["finite"] and c["max_abs_h"] <= 1+1e-10
            and c["min_area_ratio"] >= 0 and c["max_area_ratio"] <= 1+1e-10
            for c in native_sweeps) if native_sweeps else None),
        native_flow_guard_exercised=(any(c["negative_raw_flow_samples"]
                                        for c in native_sweeps) if native_sweeps else None),
        cases=cases, native_sweeps=native_sweeps,
        hashes=hashes, compile_commands=compile_commands,
        purpose="Numerical valve parity; not pressure-model accuracy or real transfer.",
        limitations=[
            "The calibration-dt case deliberately supplies .02s to the deployed valve method.",
            "Normal lib3 still advances each valve at TS/4 = 25us; native-dt cases check that path.",
            "Measured sequence at native dt preserves sample ordering, not its physical wall time.",
            "Calibration matched initialization uses first measured flow, unlike a normal simulator reset.",
            "No 95% area or measured-flow fit threshold is used as a feasibility acceptance criterion.",
            "Each fitted type is checked independently; these data are not held-out validation.",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    failed = [f"{c['type']}/v{c['valve']}/{c['case']}" for c in cases if not c["parity_pass"]]
    print(json.dumps(dict(parity_pass=summary["parity_pass"], cases=len(cases),
                          failed=failed, output=str(args.output)), indent=2))
    raise SystemExit(0 if summary["parity_pass"] else 1)


if __name__ == "__main__":
    main()
