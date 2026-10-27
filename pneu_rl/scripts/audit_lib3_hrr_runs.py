"""One-shot local audit of the matched HRR runs, without training or Real I/O.

Only completed, immutable episode-2000 checkpoints are evaluated. The standard
factory intentionally evaluates the nominal plant for both policies, including
the policy trained with dynamics randomization. All artifacts go to the ignored
experiment directory. This command is suitable for a one-shot scheduled job.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "pneu_rl/exp/lib3_hrr_dr_20261026"
DEFAULT_OUTPUT = EXPERIMENT / "scheduled_check"
for package in ("pneu_env", "pneu_ref", "pneu_rl", "pneu_utils"):
    sys.path.insert(0, str(ROOT / package / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def diagnostic_rows(path):
    """Ignore an incomplete final CSV row while the trainer is appending."""
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                converted = {key: float(value) for key, value in row.items()}
                if not all(math.isfinite(value) for value in converted.values()):
                    continue
                rows.append(converted)
            except (ValueError, TypeError):
                continue
    return rows


def run_status(manifest, now):
    requested = int(manifest.get("episodes_each", 2000))
    results = {}
    for variant, name in zip(("baseline", "dr"), manifest["models"]):
        folder = ROOT / "pneu_rl/models" / name
        log_path = Path(manifest["training_logs"][variant])
        log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        episodes = [int(x) for x in re.findall(r"\|epi\|\s*(\d+)", log)]
        wall = json.loads((folder / "wall_time.json").read_text())
        rows = diagnostic_rows(folder / "train_diagnostics.csv")
        latest_diagnostic = int(max((row["epi"] for row in rows), default=0))
        latest_episode = max(episodes + [latest_diagnostic, 0])
        checkpoint = folder / f"model_epi_{requested:04d}.pth"
        complete = (wall.get("completed") is True and checkpoint.is_file()
                    and int(wall.get("episodes_requested", -1)) == requested
                    and latest_episode >= requested)
        elapsed = float(wall.get("elapsed_seconds", now - wall["started_unix"]))
        rate = elapsed / latest_episode if latest_episode else None
        active_elapsed = float(wall.get("estimated_active_seconds", elapsed))
        if wall.get("active_recovery_started_unix") and not complete:
            segment = wall["recovery_segments"][-1]
            recovery_seconds = max(0., now - wall["active_recovery_started_unix"])
            active_elapsed = max(0., segment["replay_modified_unix"] - wall["started_unix"]) + recovery_seconds
            recovered_episodes = latest_episode - int(segment["checkpoint_episode"])
            if recovered_episodes > 0:
                rate = recovery_seconds / recovered_episodes
        failures = [line for line in log.splitlines() if re.search(
            r"Traceback \(most recent call last\)|(?:Runtime|Value|Type|Assertion|FloatingPoint|OSError|FileNotFound)Error:|CUDA out of memory", line)]
        results[variant] = {
            "name": name, "folder": str(folder), "training_log": str(log_path),
            "status": "completed" if complete else "pending",
            "latest_episode": latest_episode, "diagnostics_episodes": latest_diagnostic,
            "episodes_requested": requested, "wall_time": wall,
            "elapsed_hours": elapsed / 3600., "mean_seconds_per_episode": rate,
            "estimated_active_hours_excluding_interruption": active_elapsed / 3600.,
            "estimated_remaining_minutes": (max(0, requested - latest_episode) * rate / 60.
                                            if rate is not None and not complete else 0.),
            "checkpoint": str(checkpoint), "checkpoint_exists": checkpoint.is_file(),
            "error_lines": failures[-20:], "log_modified_unix": log_path.stat().st_mtime if log_path.exists() else None,
        }
    return results


def plot_rewards(status, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    summaries = {}
    for variant, result in status.items():
        rows = diagnostic_rows(Path(result["folder"]) / "train_diagnostics.csv")
        if not rows:
            summaries[variant] = {"available": False}
            continue
        x = np.asarray([row["epi"] for row in rows])
        reward = np.asarray([row["reward"] for row in rows])
        color = "C0" if variant == "baseline" else "C1"
        axes[0].plot(x, reward, color=color, alpha=.15, lw=.6)
        window = min(50, len(rows))
        smooth = np.convolve(reward, np.ones(window) / window, mode="valid")
        axes[0].plot(x[window-1:], smooth, color=color, label=f"{variant}, {window}-episode mean")
        for channel, style in (("pos", "-"), ("neg", ":")):
            values = np.asarray([row[f"{channel}_rmse"] for row in rows])
            axes[1].plot(x[window-1:], np.convolve(values, np.ones(window)/window, mode="valid"),
                         color=color, ls=style, label=f"{variant} {channel}")
        tail = rows[-min(100, len(rows)):]
        summaries[variant] = {"available": True, "episodes": len(rows),
            "last_100_or_fewer_episode_means": {key: float(np.mean([row[key] for row in tail]))
                                               for key in ("reward", "pos_rmse", "neg_rmse", "pos_bias", "neg_bias")}}
    axes[0].set_ylabel("Training return")
    axes[1].set_ylabel("Training RMSE (kPa)")
    axes[1].set_xlabel("Episode")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("HRR baseline vs HRR + DR: training curves (stochastic training)")
    fig.savefig(output / "reward_curves.png", dpi=135)
    plt.close(fig)
    return summaries


def band_rms(values, fs):
    import numpy as np
    from scipy.signal import welch
    f, power = welch(values, fs=fs, nperseg=min(1024, len(values)), detrend="linear")
    selected = (f >= 2.) & (f <= 20.)
    return float(np.sqrt(np.sum(power[selected]) * (f[1]-f[0])))


def trace_metrics(arrays, fs, *, last_seconds=None):
    import numpy as np
    keep = slice(-round(last_seconds * fs), None) if last_seconds else slice(None)
    pressure = arrays["observed_pressure"][keep, 2:4]
    reference = arrays["reference"][keep]
    error = pressure - reference
    command = arrays["physical_command"][keep]
    output = {"samples": len(error), "duration_seconds": len(error)/fs,
              "command_total_variation_per_second": (np.abs(np.diff(command, axis=0)).sum(axis=0)
                                                        / ((len(command)-1)/fs)).tolist()}
    for index, channel in enumerate(("positive", "negative")):
        e = error[:, index]
        output[channel] = {"rmse_kpa": float(np.sqrt(np.mean(e**2))),
            "bias_kpa": float(e.mean()), "error_std_kpa": float(e.std()),
            "pressure_band_2_20hz_rms_kpa": band_rms(pressure[:, index], fs),
            "error_band_2_20hz_rms_kpa": band_rms(e, fs),
            "pressure_peak_to_peak_kpa": float(np.ptp(pressure[:, index]))}
    return output


def rollout(env, model, steps, member, hold_target=None, p0=None):
    import numpy as np
    from evaluate_lib3_screening import reset_to_reference_member
    from check_lib3_closed_loop_feasibility import NativeStateSampler, analyze_arrays
    original_goal = env.goal.ref.get_goal
    # Carry the entire diagnostic hold; do not re-run all-open preparation at
    # the training's ordinary four-block boundary during a 120-second hold.
    env.episode_carry_blocks = max(env.episode_carry_blocks, math.ceil(steps / 512) + 1)
    state, _ = reset_to_reference_member(env, member)
    if hold_target is not None:
        def hold_goal(t):
            x = np.clip(float(t) / 3., 0., 1.)
            blend = 10*x**3 - 15*x**4 + 6*x**5
            return tuple(p0 + blend*(hold_target-p0))
        env.goal.ref.get_goal = hold_goal
        env.goal.buf = np.tile(p0, env.goal.num_ref)
        env.prev_ref_current = p0.copy()
        state, _ = env._state_from_current(env.last_policy_action)
    sampler = NativeStateSampler(env.obs)
    shapes = {"observed_pressure": (4,), "reference": (2,), "physical_command": (6,),
              "native_state": (37,), "valve_debug": (6, 9), "mass_flowrate": (10,)}
    arrays = {key: np.empty((steps,) + shape) for key, shape in shapes.items()}
    initial = {"pressure_kpa": env.curr_full_press.tolist(), "policy_time": float(env.t)}
    try:
        for step in range(steps):
            if step and step % 512 == 0:
                before = state.copy()
                state, _ = env.reset()
                np.testing.assert_array_equal(before, state)
            action = model.predict(state, evaluate=True)
            state, reward, done, truncated, info = env.step(action)
            if done or truncated or not np.isfinite(state).all() or not np.isfinite(reward):
                raise RuntimeError(f"Invalid deterministic rollout at member {member}, step {step}")
            native = sampler.sample()
            for key in ("native_state", "valve_debug", "mass_flowrate"):
                arrays[key][step] = native[key]
            arrays["observed_pressure"][step] = env.curr_full_press
            arrays["reference"][step] = [info["obs"]["ref_act_pos"], info["obs"]["ref_act_neg"]]
            raw = np.asarray(info["ctrl_input"]).reshape(-1, 6)[0]
            arrays["physical_command"][step] = env.obs.scale_low + (env.obs.scale_high-env.obs.scale_low)*raw
    finally:
        env.goal.ref.get_goal = original_goal
    feasibility = analyze_arrays({key: arrays[key][None] for key in
        ("native_state", "valve_debug", "mass_flowrate", "physical_command")})
    return arrays, {"initial": initial, "metrics_whole": trace_metrics(arrays, env.obs.freq),
        "feasibility": feasibility, "reference_sha256": hashlib.sha256(arrays["reference"].tobytes()).hexdigest()}


def evaluate_completed(result, output, representative):
    import numpy as np
    import torch
    import yaml
    from train_screening import make_components, seed_everything
    from pneu_rl.sac_with_loss import SAC
    folder = Path(result["folder"])
    cfg_path = folder / "cfg.yaml"
    cfg = yaml.full_load(cfg_path.read_text())
    cfg["env"]["verbose"] = False
    library = Path(cfg["obs"]["library_path"])
    reference = ROOT / cfg["reference"]["kwargs"]["trajectory_npz"]
    frozen_paths = (cfg_path, Path(result["checkpoint"]), library, reference)
    frozen_hashes = {str(path): digest(path) for path in frozen_paths}
    provenance = cfg["plant_provenance"]
    if frozen_hashes[str(library)] != provenance["simulator_library_sha256"]:
        raise RuntimeError("Nominal simulator differs from the saved training provenance")
    if frozen_hashes[str(reference)] != provenance["reference_bank_sha256"]:
        raise RuntimeError("Reference bank differs from saved training provenance")
    seed_everything(cfg["train_seed"])
    env = make_components(cfg)  # Deliberately ignores training-only DR metadata.
    model = SAC(env=env, **cfg["model"])
    model.load_model(result["checkpoint"], evaluate=True)
    if env.observation_space.shape != (162,) or env.obs.freq != 50:
        raise RuntimeError("Expected 162-dimensional HRR at 50 Hz")
    output_metrics = {"policy": "frozen deterministic mean; no replay or updates",
        "evaluation_plant": "saved nominal library for both policies",
        "provenance_sha256": frozen_hashes, "fixed_bank": [], "constant_holds": []}
    variant = "dr" if "_DR_" in result["name"] else "baseline"
    try:
        for member in range(16):
            arrays, row = rollout(env, model, 2048, member)
            row["member"] = member
            output_metrics["fixed_bank"].append(row)
            if member == 0:
                for key in ("observed_pressure", "reference", "physical_command"):
                    representative[f"{variant}__bank0__{key}"] = arrays[key].copy()
            print(f"{variant}: nominal bank member {member} evaluated", flush=True)
        p0 = np.asarray(cfg["reference"]["design"]["initialization"]["policy_start_observation_kpa"])[2:4]
        for member, target in enumerate(((120.,75.), (130.,70.), (140.,80.))):
            arrays, row = rollout(env, model, 6000, 0, np.asarray(target), p0)
            row.update(target_kpa=list(target), duration_seconds=120., smooth_start_seconds=3.,
                       metrics_last30_seconds=trace_metrics(arrays, 50., last_seconds=30.))
            output_metrics["constant_holds"].append(row)
            for key in ("observed_pressure", "reference", "physical_command"):
                representative[f"{variant}__hold{member}__{key}"] = arrays[key].copy()
            print(f"{variant}: hold {target} evaluated", flush=True)
    finally:
        env.close()
    if frozen_hashes != {str(path): digest(path) for path in frozen_paths}:
        raise RuntimeError("A frozen model/library/reference changed during evaluation")
    def aggregate(rows, key):
        reports = [row[key] for row in rows]
        return {channel: {"rmse_kpa": float(np.sqrt(np.mean([r[channel]["rmse_kpa"]**2 for r in reports]))),
                          "mean_bias_kpa": float(np.mean([r[channel]["bias_kpa"] for r in reports])),
                          "mean_pressure_band_2_20hz_rms_kpa": float(np.mean([r[channel]["pressure_band_2_20hz_rms_kpa"] for r in reports]))}
                for channel in ("positive", "negative")}
    output_metrics["fixed_bank_overall"] = aggregate(output_metrics["fixed_bank"], "metrics_whole")
    output_metrics["holds_last30_overall"] = aggregate(output_metrics["constant_holds"], "metrics_last30_seconds")
    output_metrics["all_sampled_physical_invariants_passed"] = all(row["feasibility"]["passed"]
        for row in output_metrics["fixed_bank"] + output_metrics["constant_holds"])
    all_rows = output_metrics["fixed_bank"] + output_metrics["constant_holds"]
    first_envelope = all_rows[0]["feasibility"]["operating_envelope_separate_from_physical_invariants"]
    output_metrics["operating_envelope_violation_sample_counts"] = {
        key: sum(row["feasibility"]["operating_envelope_separate_from_physical_invariants"]
                 ["out_of_bounds_sample_counts"][key] for row in all_rows)
        for key in first_envelope["out_of_bounds_sample_counts"]}
    return output_metrics


def plot_evaluations(data, output):
    import numpy as np
    import matplotlib.pyplot as plt
    variants = [variant for variant in ("baseline", "dr") if f"{variant}__bank0__reference" in data]
    for kind, members, filename in (("bank", [0], "nominal_bank_member0.png"),
                                     ("hold", [0,1,2], "constant_holds.png")):
        fig, axes = plt.subplots(4, len(members), figsize=(7*len(members), 9), squeeze=False,
                                 constrained_layout=True)
        for column, member in enumerate(members):
            for variant in variants:
                prefix = f"{variant}__{kind}{member}__"
                if prefix+"reference" not in data:
                    continue
                pressure, reference, commands = (data[prefix+key] for key in
                    ("observed_pressure", "reference", "physical_command"))
                t = (np.arange(len(pressure))+1)/50.
                for channel in range(2):
                    axes[channel,column].plot(t, pressure[:,channel+2], lw=.75, label=variant)
                axes[2,column].plot(t, commands[:,2], lw=.65, label=variant+" V3")
                axes[3,column].plot(t, commands[:,3], lw=.65, label=variant+" V4")
            for channel in range(2):
                axes[channel,column].plot(t, reference[:,channel], "k--", lw=.8, label="reference")
            axes[0,column].set_title(f"Nominal bank member {member}" if kind == "bank" else f"Hold {(120,130,140)[member]}/{(75,70,80)[member]} kPa")
            for row, label in enumerate(("Pact+ (kPa)", "Pact− (kPa)", "Physical V3", "Physical V4")):
                axes[row,column].set_ylabel(label)
                axes[row,column].grid(alpha=.2)
                axes[row,column].legend(fontsize=7)
            axes[3,column].set_xlabel("Policy time (s)")
        fig.savefig(output/filename, dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-only", action="store_true", help="Read logs and plot training curves; never load policy checkpoints")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(EXPERIMENT.resolve()):
        parser.error("output directory must stay within the ignored HRR experiment")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    os.chdir(ROOT)
    now = time.time()
    report = {"checked_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "status_only": args.status_only, "evaluation_errors": {}, "scope": [
            "No training, online adaptation, production settings changes, or Real I/O.",
            "Physics checks use sampled native 50 Hz states, not every integrator substep.",
            "Nominal-bank/hold performance does not establish Real transfer or unseen dynamics robustness.",
            "Training returns are stochastic and are distinct from frozen deterministic evaluation."]}
    manifest = json.loads((EXPERIMENT/"runs.json").read_text())
    report["runs"] = run_status(manifest, now)
    write_json(output/"status.json", report)
    report["training_curves"] = plot_rewards(report["runs"], output)
    data = {}
    report["evaluations"] = {}
    if not args.status_only:
        import torch
        torch.set_num_threads(1)
        for variant, result in report["runs"].items():
            if result["status"] != "completed":
                continue
            try:
                report["evaluations"][variant] = evaluate_completed(result, output, data)
                write_json(output/f"{variant}_metrics.json", report["evaluations"][variant])
            except Exception:
                report["evaluation_errors"][variant] = traceback.format_exc()
            write_json(output/"status.json", report)
        if data:
            import numpy as np
            np.savez_compressed(output/"representative_traces.npz", **data)
            plot_evaluations(data, output)
    report["elapsed_check_seconds"] = time.time()-now
    write_json(output/"status.json", report)
    write_json(output/"metrics.json", report["evaluations"])
    lines = ["HRR baseline / HRR+DR 예약 검사", report["checked_at_utc"], ""]
    for variant, result in report["runs"].items():
        lines.append(f"{variant}: {result['status']}; episode {result['latest_episode']}/{result['episodes_requested']}; elapsed {result['elapsed_hours']:.2f} h")
        if result["status"] != "completed":
            lines.append("  학습 미완료: checkpoint를 읽지 않았으며 자동 재학습하지 않음.")
        if result["error_lines"]:
            lines.extend("  LOG ERROR: "+line for line in result["error_lines"])
        if variant in report["evaluations"]:
            evaluation = report["evaluations"][variant]
            lines.append("  nominal bank 전체: "+json.dumps(evaluation["fixed_bank_overall"], ensure_ascii=False))
            lines.append("  hold 마지막 30초: "+json.dumps(evaluation["holds_last30_overall"], ensure_ascii=False))
            lines.append(f"  sampled physical invariants passed: {evaluation['all_sampled_physical_invariants_passed']}")
            lines.append("  operating envelope violation samples: "+json.dumps(
                evaluation["operating_envelope_violation_sample_counts"], ensure_ascii=False))
        if variant in report["evaluation_errors"]:
            lines.append("  EVALUATION ERROR: "+report["evaluation_errors"][variant])
    lines.extend(["", "학습 return은 stochastic이며 frozen deterministic 성능과 구분해야 함.",
        "물리 검사와 공급압 운전 범위 위반은 별도로 기록함. Real zero-shot 개선 보장은 아님."])
    (output/"analysis_summary.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
    return 1 if report["evaluation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
