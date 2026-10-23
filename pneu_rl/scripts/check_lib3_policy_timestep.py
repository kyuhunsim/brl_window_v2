"""Compare frozen bank-0 policy behavior at 0.1 ms and 0.05 ms native steps.

The final build command is reproduced with just TS and the output path changed.
The reference, control frequency, observation delay, and policy are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for package in ("pneu_env", "pneu_ref", "pneu_rl", "pneu_utils"):
    sys.path.insert(0, str(ROOT / package / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import yaml
from scipy.signal import welch
from check_lib3_closed_loop_feasibility import NativeStateSampler, analyze_arrays
from evaluate_lib3_screening import reset_to_reference_member
from train_screening import make_components
from pneu_rl.sac_with_loss import SAC

REVIEW = ROOT / "pneu_rl/exp/lib3_physical_training_review_20261023"
TUNING_RESULTS = ROOT.parent / "pneu_tune/exp"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def resolve_manifest_source(path):
    """Locate moved build inputs without rewriting their historical manifest."""
    original = Path(path)
    old_results = ROOT / "docs/img/lib3_physical_model_20261023"
    try:
        relative = original.relative_to(old_results)
    except ValueError:
        return original
    relocated = TUNING_RESULTS / "lib3_physical_model_20261023" / relative
    return relocated if relocated.is_file() else original


def fine_library(kind, directory):
    manifest_path = TUNING_RESULTS / "lib3_physical_model_20261023/final" / kind / "build_O3_opening_pumpfit.json"
    manifest = json.loads(manifest_path.read_text())
    resolved_sources = {}
    for path, expected in manifest["source_hashes"].items():
        resolved = resolve_manifest_source(path)
        if digest(resolved) != expected:
            raise ValueError(f"Current source differs from final build manifest: {path}")
        resolved_sources[path] = str(resolved)
    assert manifest["opening_dynamics"] and not manifest["fast_flow"]
    assert manifest["optimization"] == "O3" and manifest["time_step"] == .0001
    command = manifest["command"].copy()
    for index, token in enumerate(command):
        for original, resolved in resolved_sources.items():
            token = token.replace(original, resolved)
        command[index] = token
    positions = [i for i, token in enumerate(command) if token.startswith("-DTS=")]
    assert len(positions) == 1
    command[positions[0]] = "-DTS=0.00005"
    directory.mkdir(parents=True, exist_ok=True)
    library = (directory / f"libpneumatic_simulator_{kind}_O3_opening_pumpfit_dt0p00005.so").resolve()
    command[command.index("-o") + 1] = str(library)
    if not library.is_file():
        subprocess.run(command, check=True, capture_output=True, text=True)
    result = dict(type=kind, native_time_step_sec=.00005, command=command,
                  final_build_manifest=str(manifest_path), final_manifest_sha256=digest(manifest_path),
                  original_library=manifest["library"], original_library_sha256=manifest["library_sha256"],
                  library=str(library), library_sha256=digest(library),
                  unchanged_source_hashes=manifest["source_hashes"],
                  resolved_source_paths=resolved_sources,
                  pump_coeff_micro=manifest["pump_coeff_micro"])
    write_json(directory / f"build_{kind}.json", result)
    return library, result


def spectrum(values, lower, upper):
    frequency, power = welch(values, fs=50., nperseg=min(1024, len(values)), detrend="linear")
    selected = (frequency >= lower) & (frequency <= upper)
    return float(np.sqrt(np.sum(power[selected]) * (frequency[1] - frequency[0])))


def metrics(data):
    time = data["time"]
    mask = time > 5.
    result = {}
    for i, name in enumerate(("act_pos", "act_neg")):
        pressure = data["pressure_observed"][0, :, i + 2]
        error = pressure - data["reference"][0, :, i]
        result[name] = dict(rmse_whole_kpa=float(np.sqrt(np.mean(error**2))),
                            rmse_after5_kpa=float(np.sqrt(np.mean(error[mask]**2))),
                            bias_after5_kpa=float(np.mean(error[mask])),
                            error_rms_01_2_hz_kpa=spectrum(error[mask], .1, 2.),
                            error_rms_2_20_hz_kpa=spectrum(error[mask], 2., 20.),
                            error_rms_8_12_hz_kpa=spectrum(error[mask], 8., 12.),
                            pressure_peak_to_peak_after5_kpa=float(np.ptp(pressure[mask])))
    return result


def compare(model_name, checkpoint, output):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    folder = ROOT / "pneu_rl/models" / model_name
    cfg_path = folder / "cfg.yaml"
    checkpoint_path = folder / f"model_epi_{checkpoint:04d}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = yaml.full_load(cfg_path.read_text())
    kind = cfg["plant_provenance"]["calibration_type"]
    with np.load(REVIEW / model_name / "rollout.npz") as archive:
        baseline = {key: archive[key][:1] if archive[key].ndim >= 2 else archive[key]
                    for key in archive.files}
    library, build = fine_library(kind, output)
    assert digest(cfg["obs"]["library_path"]) == build["original_library_sha256"]
    cfg["obs"]["library_path"] = str(library)
    cfg["env"]["verbose"] = False
    env = make_components(cfg)
    sampler = NativeStateSampler(env.obs)
    model = SAC(env=env, **cfg["model"])
    model.load_model(str(checkpoint_path), evaluate=True)
    steps = len(baseline["time"])
    shapes = dict(pressure_observed=(4,), reference=(2,), physical_command=(6,),
                  native_state=(37,), valve_debug=(6, 9), mass_flowrate=(10,))
    data = {name: np.empty((1, steps) + shape) for name, shape in shapes.items()}
    initial = None
    try:
        state, _ = reset_to_reference_member(env, 0)
        initial = env.curr_full_press.tolist()
        for step in range(steps):
            action = model.predict(state, evaluate=True)
            state, _, _, _, info = env.step(action)
            native = sampler.sample()
            applied = np.asarray(info["ctrl_input"]).reshape(-1, 6)[0]
            data["pressure_observed"][0, step] = env.curr_full_press
            data["reference"][0, step] = [info["obs"]["ref_act_pos"], info["obs"]["ref_act_neg"]]
            data["physical_command"][0, step] = env.obs.scale_low + (env.obs.scale_high - env.obs.scale_low) * applied
            for name in ("native_state", "valve_debug", "mass_flowrate"):
                data[name][0, step] = native[name]
            if (step + 1) % cfg["model"]["horizon"] == 0 and step + 1 < steps:
                before = state.copy()
                state, _ = env.reset()
                np.testing.assert_array_equal(state, before)
    finally:
        env.close()
    data["time"] = baseline["time"]
    np.testing.assert_array_equal(data["reference"], baseline["reference"])
    destination = output / model_name
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "rollout_fine.npz", **data)
    report = dict(model=model_name, checkpoint_episode=checkpoint, calibration_type=kind,
                  bank_member=0, steps=steps, control_frequency_hz=50, observation_delay_sec=.1,
                  preconditioning_sec=30, initial_observed_pressure_fine_kpa=initial,
                  checkpoint_sha256=digest(checkpoint_path), config_sha256=digest(cfg_path),
                  fine_build=build, coarse_metrics=metrics(baseline), fine_metrics=metrics(data),
                  fine_feasibility=analyze_arrays(data),
                  pointwise_difference_warning="Nonlinear closed-loop feedback can accumulate phase differences; use aggregate error and spectral metrics, not exact path equality.",
                  interpretation="A two-step comparison on one bank member is a gross numerical sensitivity check, not a convergence proof or real stability guarantee.")
    report["pressure_pointwise_difference_from_coarse_kpa"] = {
        name: dict(rmse=float(np.sqrt(np.mean((data["pressure_observed"][0, :, i] - baseline["pressure_observed"][0, :, i])**2))),
                   max_abs=float(np.max(np.abs(data["pressure_observed"][0, :, i] - baseline["pressure_observed"][0, :, i]))))
        for i, name in enumerate(("ch_pos", "ch_neg", "act_pos", "act_neg"))}
    write_json(destination / "comparison.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    for i, name in enumerate(("ch_pos", "ch_neg", "act_pos", "act_neg")):
        axis = axes.flat[i]
        axis.plot(data["time"], baseline["pressure_observed"][0, :, i], label="Native step 0.10 ms", lw=1.)
        axis.plot(data["time"], data["pressure_observed"][0, :, i], label="Native step 0.05 ms", lw=1., alpha=.8)
        if i >= 2:
            axis.plot(data["time"], data["reference"][0, :, i-2], "k--", label="Reference", lw=.9)
        axis.set(title=name, xlabel="Policy time (s)", ylabel="Absolute pressure (kPa)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle(model_name + ": frozen policy numerical step check")
    figure.savefig(destination / "comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"model": model_name, "coarse": report["coarse_metrics"], "fine": report["fine_metrics"],
                      "fine_physical_passed": report["fine_feasibility"]["passed"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=REVIEW / "feasibility/numerical")
    args = parser.parse_args()
    compare(args.model, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
