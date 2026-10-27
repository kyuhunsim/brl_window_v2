"""Build an isolated, runtime-randomizable copy of the recalibrated Type1 plant.

The three controls affect every instance of this shared object in a process.
Set them before reset/preconditioning and keep them fixed during a rollout.
Unit controls reproduce the saved plant; production sources are never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "pneu_rl/exp/lib3_v34_pressure_calibration_20261026/candidates/type1"
DEFAULT_OUTPUT = ROOT / "pneu_rl/exp/lib3_hrr_dr_20261026/build"
MANIFEST_NAME = "build_candidate_dt0.00010000.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_hash(manifest: dict, basename: str) -> str:
    matches = [digest for name, digest in manifest["source_hashes"].items()
               if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected one recorded source hash for {basename}; found {len(matches)}")
    return matches[0]


def _verify(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Saved source/library hash mismatch: {path}")


CONTROLS = '''// Runtime plant controls; reset/time_reset leave them unchanged.
namespace {
double slew_diagnostic_tau_scale = 1.0;
double slew_diagnostic_flow3 = 1.0;
double slew_diagnostic_flow4 = 1.0;
}
extern "C" int set_slew_diagnostic(double tau_scale, double flow3, double flow4) {
    if (!(std::isfinite(tau_scale) && std::isfinite(flow3) && std::isfinite(flow4)
          && tau_scale > 0.0 && flow3 > 0.0 && flow4 > 0.0)) return -1;
    slew_diagnostic_tau_scale = tau_scale;
    slew_diagnostic_flow3 = flow3;
    slew_diagnostic_flow4 = flow4;
    return 0;
}
extern "C" void get_slew_diagnostic(double* controls) {
    controls[0] = slew_diagnostic_tau_scale;
    controls[1] = slew_diagnostic_flow3;
    controls[2] = slew_diagnostic_flow4;
}
'''


PATCHES = {
    '#include <cmath>': '#include <cmath>\n\n' + CONTROLS,
    'const double tau = (i==2?0.01:(i==3?0.10000000000000001:1.)) * 2.0 * zeta / wn;':
        'const double tau = ((i == 2 || i == 3) ? slew_diagnostic_tau_scale : 1.0)\n'
        '            * (i==2?0.01:(i==3?0.10000000000000001:1.)) * 2.0 * zeta / wn;',
    'const double area_eff = params->A_max * state->x1;':
        'const double diagnostic_gain = type == 3 ? slew_diagnostic_flow3\n'
        '                                   : (type == 4 ? slew_diagnostic_flow4 : 1.0);\n'
        '    const double area_eff = diagnostic_gain * params->A_max * state->x1;',
    'const double target_area = params->A_max * state->x2;':
        'const double target_area = diagnostic_gain * params->A_max * state->x2;',
}


def build(output_dir: Path = DEFAULT_OUTPUT, source_dir: Path = DEFAULT_SOURCE) -> Path:
    """Validate the saved snapshot, compile it with three controls, and return its .so.

    ``flow3``/``flow4`` scale effective orifice area, not a post-hoc pressure term.
    The debug area and both actual/static flow therefore remain consistent with
    the same instantaneous pressure relation. Normalized opening stays in [0, 1].
    """
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if (output_dir == source_dir or source_dir in output_dir.parents
            or output_dir in source_dir.parents):
        raise ValueError("Build output must be separate from the saved source directory")
    source_manifest = source_dir / MANIFEST_NAME
    manifest = json.loads(source_manifest.read_text())
    original_library = source_dir / Path(manifest["library"]).name
    _verify(original_library, manifest["library_sha256"])
    originals = {name: source_dir / name for name in (
        "pneumatic_simulator.cpp", "pneumatic_simulator.h", "valve_parameters.h",
        "pneumatic_CT_candidate.cpp")}
    originals["batch_replay.cpp"] = source_dir.parent / "batch_replay.cpp"
    for name, path in originals.items():
        _verify(path, _recorded_hash(manifest, name))
    command = manifest["compile_command"]
    if "-O3" not in command or "-DPNEU_OPENING_DYNAMICS" not in command:
        raise ValueError("Expected the saved O3 opening-dynamics compile configuration")

    changed = originals["pneumatic_CT_candidate.cpp"].read_text()
    for old, new in PATCHES.items():
        if changed.count(old) != 1:
            raise ValueError(f"Expected exactly one source patch target: {old}")
        changed = changed.replace(old, new)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in originals.items():
        if name != "pneumatic_CT_candidate.cpp":
            shutil.copyfile(path, output_dir / name)
    ct = output_dir / "pneumatic_CT_randomized.cpp"
    ct.write_text(changed)
    library = output_dir / "lib3_type1_randomized.so"
    mapped_sources = {
        "pneumatic_simulator.cpp": output_dir / "pneumatic_simulator.cpp",
        "pneumatic_CT_candidate.cpp": ct,
        "batch_replay.cpp": output_dir / "batch_replay.cpp",
    }
    compile_command = []
    for arg in command:
        if arg == manifest["library"]:
            arg = str(library)
        elif arg.startswith("-DPNEU_VALVE_PARAMS_HEADER="):
            arg = f'-DPNEU_VALVE_PARAMS_HEADER="{output_dir / "valve_parameters.h"}"'
        elif arg.endswith(".cpp"):
            if Path(arg).name not in mapped_sources:
                raise ValueError(f"Unexpected translation unit in saved build: {arg}")
            arg = str(mapped_sources[Path(arg).name])
        compile_command.append(arg)
    if compile_command.count(str(library)) != 1:
        raise ValueError("Saved build must name exactly one output shared object")
    # Atomic replacement also permits rebuilding while an older library is loaded.
    with tempfile.TemporaryDirectory(prefix=".compile-", dir=output_dir) as temporary:
        temporary_library = Path(temporary) / library.name
        actual_command = [str(temporary_library) if arg == str(library) else arg
                          for arg in compile_command]
        subprocess.run(actual_command, check=True)
        temporary_library.replace(library)

    copied_sources = [output_dir / name for name in originals
                      if name != "pneumatic_CT_candidate.cpp"] + [ct]
    report = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "original_library": str(original_library),
        "original_library_sha256": manifest["library_sha256"],
        "original_source_hashes": {str(path): sha256(path) for path in originals.values()},
        "command": compile_command,
        "library": str(library), "library_sha256": sha256(library),
        "source_hashes": {str(path): sha256(path) for path in copied_sources},
        "set_function": "int set_slew_diagnostic(double tau_scale, double flow3, double flow4)",
        "get_function": "void get_slew_diagnostic(double controls[3])",
        "setter_success": 0, "setter_invalid_positive_finite_argument": -1,
        "controls_survive_simulator_reset": True,
        "controls_process_global_per_loaded_shared_object": True,
        "default_controls": [1.0, 1.0, 1.0],
        "tau_semantics": "Multiply V3/V4 opening AND closing time constants relative to saved recalibrated Type1; other valves unchanged.",
        "flow_semantics": "Multiply V3/V4 effective and target orifice areas; debug area, actual/static flow, and mass flow use the same scaled area. Normalized opening remains unscaled.",
        "unchanged": ["pump coefficients", "negative valves V5/V6", "supply valves V1/V2", "hysteresis", "reference"],
        "native_step_sec": manifest["native_step_sec"],
        "pump_coeff_micro": manifest["pump_coeff_micro"],
        "baseline_opening_time_constants_sec": manifest["opening_time_constants_sec"],
        "patches": PATCHES,
        "interpretation": "Sensitivity/randomization conditions, not identified real-device parameter ranges or a closed-loop stability guarantee.",
    }
    (output_dir / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    return library


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output_dir, args.source_dir))
