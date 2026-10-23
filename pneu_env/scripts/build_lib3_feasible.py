"""Build separate whole-lib3 libraries for the two independently fitted valve sets.

Default deployed libraries and legacy coefficients are left in place. Candidate
builds use the same exact normalized hysteresis update as valve calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TUNE_EXP = ROOT.parent / "pneu_tune/exp"
SOURCE = ROOT / "pneu_env/src/pneu_env/lib3"
NAMES = ("CHAMBER_POS_PARAMS", "CHAMBER_NEG_PARAMS", "ACT_POS_IN_PARAMS",
         "ACT_POS_OUT_PARAMS", "ACT_NEG_IN_PARAMS", "ACT_NEG_OUT_PARAMS")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_header(fit_root: Path, kind: str, destination: Path):
    blocks, inputs = [], []
    for number, name in enumerate(NAMES, 1):
        path = fit_root / kind / f"v{number}/parameters.npz"
        with np.load(path) as data:
            p = np.asarray(data["parameters"], dtype=float)
        if p.shape != (13,) or not np.isfinite(p).all():
            raise ValueError(f"Invalid coefficient vector: {path}")
        if np.any(p[[0, 1, 8, 9, 10, 11, 12]] <= 0):
            raise ValueError(f"Capacity, opening shape, frequencies and damping must be positive: {path}")
        rate = p[6] + p[7]
        if not (rate > 0 and p[6] >= p[7] >= 0
                and np.isclose(p[5], rate, rtol=1e-12, atol=1e-12)):
            raise ValueError(f"Not a normalized feasible hysteresis fit: {path}")
        blocks.append(f"const ValveModelParams {name} = {{\n    "
                      + ", ".join(format(float(value), ".17g") for value in p) + "\n};\n")
        inputs.append(dict(path=str(path), sha256=sha256(path), valve=number,
                           parameters=p.tolist()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("// Generated independently fitted " + kind + " coefficients.\n"
                           + "// Original fit acceptance flags remain in the source reports.\n\n"
                           + "\n".join(blocks))
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("all", "type1", "type2"), default="all")
    parser.add_argument("--fit-root", type=Path,
                        default=TUNE_EXP / "lib3_all_valves_feasible_20261022")
    parser.add_argument("--output-dir", type=Path,
                        default=TUNE_EXP / "lib3_feasible_integration_20261022")
    parser.add_argument("--library-dir", type=Path,
                        default=TUNE_EXP / "lib3_feasible_integration_20261022/libraries")
    parser.add_argument("--optimization", choices=("O0", "O3"), default="O3")
    parser.add_argument("--fast-flow", action="store_true",
                        help="precompute the same 20-Euler-substep affine transition")
    parser.add_argument("--opening-dynamics", action="store_true",
                        help="first-order bounded opening and instantaneous pressure-dependent flow")
    parser.add_argument("--time-step", type=float, default=.0001,
                        help="native pressure integration interval, seconds")
    parser.add_argument("--pump-coeff", nargs=2, type=float, metavar=("INLET", "OUTLET"),
                        help="embed tuned default pump coefficients in micro units")
    args = parser.parse_args()
    if not np.isfinite(args.time_step) or args.time_step <= 0:
        parser.error("time-step must be finite and positive")
    if args.pump_coeff is not None and (not np.isfinite(args.pump_coeff).all()
                                        or np.any(np.asarray(args.pump_coeff) <= 0)):
        parser.error("pump coefficients must be finite and positive")
    args.library_dir.mkdir(parents=True, exist_ok=True)
    for kind in (("type1", "type2") if args.type == "all" else (args.type,)):
        output = args.output_dir / kind
        header = (output / "valve_parameters.h").resolve()
        inputs = make_header(args.fit_root.resolve(), kind, header)
        suffix = ("_opening" if args.opening_dynamics else "") + ("_fast" if args.fast_flow else "") + ("_pumpfit" if args.pump_coeff else "")
        if args.time_step != .0001:
            suffix += "_dt" + format(args.time_step, ".8g")
        library = (args.library_dir / f"libpneumatic_simulator_{kind}_{args.optimization}{suffix}.so").resolve()
        command = ["g++", f"-{args.optimization}", "-std=c++17", "-shared", "-fPIC",
                   "-DPNEU_EXACT_VALVE_HYSTERESIS", f'-DPNEU_VALVE_PARAMS_HEADER="{header}"',
                   str(SOURCE / "pneumatic_simulator.cpp"), str(SOURCE / "pneumatic_CT.cpp"),
                   "-o", str(library)]
        if args.fast_flow:
            command.insert(1, "-DPNEU_FAST_VALVE_FLOW")
        if args.opening_dynamics:
            command.insert(1, "-DPNEU_OPENING_DYNAMICS")
        command.insert(1, f"-DTS={args.time_step:.17g}")
        if args.pump_coeff:
            command[1:1] = [f"-DCIN_={args.pump_coeff[0]:.17g}",
                            f"-DCOUT_={args.pump_coeff[1]:.17g}"]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        manifest = dict(type=kind, optimization=args.optimization, fast_flow=args.fast_flow,
                        opening_dynamics=args.opening_dynamics, time_step=args.time_step,
                        pump_coeff_micro=args.pump_coeff,
                        command=command,
                        library=str(library), library_sha256=sha256(library),
                        build_seconds=time.perf_counter()-started,
                        source_hashes={str(path): sha256(path) for path in (
                            SOURCE / "pneumatic_simulator.h", SOURCE / "pneumatic_simulator.cpp",
                            SOURCE / "pneumatic_CT.cpp", header)},
                        valve_inputs=inputs, hysteresis="exact normalized update, no clipping",
                        note="Candidate integration; does not change original fit acceptance flags.")
        if args.opening_dynamics:
            manifest["valve_dynamics"] = "First-order normalized opening; current-pressure orifice; exact half-step / pure pressure RK / half-step."
            manifest["snapshot_semantics"] = "[h, normalized opening, cached target opening, previous current, direction]"
            manifest["opening_time_constants_sec"] = [
                dict(valve=item["valve"], opening=2*item["parameters"][10]/item["parameters"][9],
                     closing=2*item["parameters"][12]/item["parameters"][11])
                for item in inputs]
            manifest["parameter_limitation"] = "Mapped from old flow lag tau=2*zeta/wn; not a refit for the new dynamics."
        (output / f"build_{args.optimization}{suffix}.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(f"{kind}: {library}", flush=True)


if __name__ == "__main__":
    main()
