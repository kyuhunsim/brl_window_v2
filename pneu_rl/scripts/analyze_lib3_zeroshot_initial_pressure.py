"""Analyze downloaded lib3 zero-shot logs without connecting to hardware.

Writes reproducible CSV/JSON summaries and figures. Logged timestamps and command
echoes are not treated as verified actuator application times or valve positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/brl_lib3_analysis_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "pneu_rl/exp/drive_zeroshot_20261015"
DEFAULT_OUTPUT = ROOT / "docs/img/lib3_zeroshot_20261015"
BANK = ROOT / "pneu_rl/config/reference/lib3_capacity_random_medium_16x4x512_seed61098.npz"
COUNTERFACTUAL = ROOT / "pneu_rl/exp/lib3_initial_pressure_counterfactual_261015"
PRESSURES = ["press_pos", "press_neg", "act_pos_press", "act_neg_press"]
REFERENCES = ["ref_act_pos", "ref_act_neg"]
COMMANDS = [
    "ctrl_ch_pos", "ctrl_ch_neg", "ctrl_act_pos_in", "ctrl_act_pos_out",
    "ctrl_act_neg_in", "ctrl_act_neg_out",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return os.path.relpath(path, ROOT)


def tracking_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"samples": 0}
    # Match saved quick-viz bias convention, reference minus measured pressure.
    error = frame[REFERENCES].to_numpy(float) - frame[PRESSURES[2:]].to_numpy(float)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return {
        "samples": len(frame),
        "rmse_pos_kpa": float(rmse[0]),
        "rmse_neg_kpa": float(rmse[1]),
        "combined_rmse_kpa": float(np.sqrt(np.mean(error**2))),
        "mae_pos_kpa": float(np.abs(error[:, 0]).mean()),
        "mae_neg_kpa": float(np.abs(error[:, 1]).mean()),
        "bias_ref_minus_obs_pos_kpa": float(error[:, 0].mean()),
        "bias_ref_minus_obs_neg_kpa": float(error[:, 1].mean()),
        "act_pos_mean_kpa": float(frame.act_pos_press.mean()),
        "act_neg_mean_kpa": float(frame.act_neg_press.mean()),
        "act_pos_std_kpa": float(frame.act_pos_press.std(ddof=0)),
        "act_neg_std_kpa": float(frame.act_neg_press.std(ddof=0)),
        "pos_reserve_mean_kpa": float((frame.press_pos - frame.act_pos_press).mean()),
        "neg_reserve_mean_kpa": float((frame.act_neg_press - frame.press_neg).mean()),
    }


def interpolate_bank(time: np.ndarray, pressure: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, time, pressure[:, i]) for i in range(2)])


def analyze(csv: Path, bank_time: np.ndarray, bank_pressure: np.ndarray) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(csv)
    cfg = yaml.safe_load((csv.parent / "cfg.yaml").read_text(encoding="utf-8"))
    required = ["curr_time"] + PRESSURES + REFERENCES + COMMANDS
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError(f"nonfinite input in {csv}")
    if frame.curr_time.diff().dropna().le(0).any():
        raise ValueError(f"non-increasing timestamps in {csv}")
    # For this bank, zero absolute-pressure references are logger placeholders.
    # This is a dataset-specific exclusion, not a general target-validity rule.
    valid = frame.loc[(frame.ref_act_pos > 1) & (frame.ref_act_neg > 1)].copy()
    if valid.empty:
        raise ValueError(f"no non-placeholder reference rows in {csv}")
    start = float(valid.curr_time.iloc[0])
    valid["active_elapsed_sec"] = valid.curr_time - start
    dt = frame.curr_time.diff().dropna().to_numpy(float)
    regular_dt = dt[(dt > 0) & (dt < 0.25)]
    if regular_dt.size == 0:
        raise ValueError(f"no regular timestamp intervals in {csv}")
    mode = "HR" if "_HR_smooth_" in csv.parent.name else "H"
    stamp = csv.parent.name.split("_260831_", 1)[0]
    windows = {
        "all_logged_including_placeholders": frame,
        "non_placeholder_reference": valid,
        "first_5_active_seconds": valid[valid.active_elapsed_sec <= 5],
        "after_5_active_seconds": valid[valid.active_elapsed_sec > 5],
        "absolute_30_to_40_seconds": valid[valid.curr_time.between(30, 40)],
        "absolute_first_40_96_seconds": valid[valid.curr_time.between(0, 40.96)],
    }
    summary = {
        "run": stamp,
        "policy": mode,
        "csv": relative(csv),
        "csv_sha256": sha256(csv),
        "cfg_sha256": sha256(csv.parent / "cfg.yaml"),
        "model_name": cfg["model_name"],
        "reference_mode": cfg.get("ref_mode"),
        "runtime": cfg.get("runtime", {}),
        "configured_frequency_hz": cfg["env"]["real"]["freq"],
        "configured_init_pressure": cfg["env"]["init_press"],
        "samples": len(frame),
        "placeholder_rows": len(frame) - len(valid),
        "first_logged_pressure_kpa": frame[PRESSURES].iloc[0].to_numpy(float).tolist(),
        "first_non_placeholder_row": int(valid.index[0]),
        "first_non_placeholder_time_sec": start,
        "first_non_placeholder_pressure_kpa": valid[PRESSURES].iloc[0].to_numpy(float).tolist(),
        "logged_end_sec": float(frame.curr_time.iloc[-1]),
        "non_placeholder_logged_span_sec": float(valid.curr_time.iloc[-1] - start),
        "first_timestamp_gap_sec": float(dt[0]),
        "large_timestamp_gaps_sec": dt[dt >= 0.25].tolist(),
        "regular_timestamp_dt_median_sec": float(np.median(regular_dt)),
        "regular_timestamp_dt_p95_sec": float(np.percentile(regular_dt, 95)),
        "inverse_regular_timestamp_dt_hz": float(1 / np.median(regular_dt)),
        "logged_command_min": frame[COMMANDS].min().to_numpy(float).tolist(),
        "logged_command_max": frame[COMMANDS].max().to_numpy(float).tolist(),
        "metrics": {name: tracking_metrics(rows) for name, rows in windows.items()},
    }
    saved = cfg.get("metrics", {}).get("full", {})
    for channel in ("pos", "neg"):
        actual = summary["metrics"]["all_logged_including_placeholders"][f"rmse_{channel}_kpa"]
        expected = saved.get(f"rmse_{channel}")
        if expected is not None and not np.isclose(actual, expected, rtol=1e-9, atol=1e-8):
            raise ValueError(f"RMSE does not reproduce saved cfg: {csv}, {channel}")

    if start > 1:
        # Describe which clock the stored reference follows; this fit is not a
        # physical latency estimate. Exclude transient synthetic-history rows.
        fit_rows = valid[(valid.active_elapsed_sec > 0.5) & (valid.curr_time <= 40)].copy()
        observed = fit_rows[REFERENCES].to_numpy(float)
        query = fit_rows.curr_time.to_numpy(float)
        offsets = np.linspace(0, 0.12, 121)
        scores = [float(np.sqrt(np.mean((interpolate_bank(bank_time, bank_pressure, query + shift) - observed)**2))) for shift in offsets]
        best = int(np.argmin(scores))
        active_prediction = interpolate_bank(bank_time, bank_pressure, query - start)
        summary["reference_clock_fit"] = {
            "samples": len(fit_rows),
            "absolute_logged_clock_best_offset_sec": float(offsets[best]),
            "absolute_logged_clock_best_rmse_kpa": scores[best],
            "active_elapsed_clock_rmse_kpa": float(np.sqrt(np.mean((active_prediction - observed)**2))),
            "scope": "Descriptive interpolation fit, not an estimate of physical actuator latency.",
        }
    return summary, frame


def plot_initialization(runs: list[dict], frames: list[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    locations = np.arange(4)
    for i, run in enumerate(runs):
        label = run["run"][7:].replace("_", ":") + " " + run["policy"]
        axes[0].bar(locations + (i - 1.5) * 0.19, run["first_non_placeholder_pressure_kpa"], 0.19, label=label)
    axes[0].axhline(101.325, color="black", linestyle="--", linewidth=1, label="training reset: 101.325")
    axes[0].set_xticks(locations, ["source +", "source -", "load +", "load -"])
    axes[0].set_ylabel("Absolute pressure (kPa)")
    axes[0].set_title("First row with non-placeholder reference")
    axes[0].legend(fontsize=8)
    for x, (run, frame) in enumerate(zip(runs, frames)):
        dt = frame.curr_time.diff().dropna().to_numpy(float)
        regular = dt[(dt > 0) & (dt < 0.25)] * 1000
        # Error bars show 5th--95th percentile; they do not measure RT timing.
        median = float(np.median(regular))
        lo, hi = np.percentile(regular, [5, 95])
        axes[1].errorbar(x, median, yerr=[[median - lo], [hi - median]], fmt="o", capsize=4)
    axes[1].axhline(20, color="black", linestyle="--", label="configured: 20 ms")
    axes[1].set_xticks(np.arange(len(runs)), [r["run"][7:].replace("_", ":") + "\n" + r["policy"] for r in runs])
    axes[1].set_ylabel("Regular logged interval (ms)")
    axes[1].set_title("CSV timing (large startup gaps excluded)")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "initial_pressure_and_timing.png", dpi=160)
    plt.close(fig)


def plot_late_tracking(runs: list[dict], frames: list[pd.DataFrame], bank_time: np.ndarray, bank_pressure: np.ndarray, output: Path) -> None:
    late = [(r, f) for r, f in zip(runs, frames) if r["placeholder_rows"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2), sharex=True, constrained_layout=True)
    for column, (run, frame) in enumerate(late):
        valid = frame[(frame.ref_act_pos > 1) & (frame.ref_act_neg > 1)]
        start = float(valid.curr_time.iloc[0])
        for channel, pressure, ref in ((0, "act_pos_press", "ref_act_pos"), (1, "act_neg_press", "ref_act_neg")):
            ax = axes[channel, column]
            ax.axvspan(0, start, color="0.9", label="unlogged startup gap")
            ax.plot(valid.curr_time, valid[ref], color="black", linestyle="--", label="logged reference")
            ax.plot(valid.curr_time, valid[pressure], color="#2374ab", label="measured pressure")
            elapsed_ref = interpolate_bank(bank_time, bank_pressure, valid.curr_time.to_numpy(float) - start)
            ax.plot(valid.curr_time, elapsed_ref[:, channel], color="#c16e13", alpha=0.8, label="bank from active start")
            ax.set_xlim(0, 40)
            ax.set_ylabel(("Positive" if channel == 0 else "Negative") + " (kPa abs)")
            ax.grid(alpha=0.2)
        axes[0, column].set_title(run["policy"] + " " + run["run"][7:].replace("_", ":"))
        axes[1, column].set_xlabel("Logged time (s); not rebased")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.savefig(output / "late_tracking_and_reference_clock.png", dpi=160)
    plt.close(fig)


def counterfactual_evidence(runs: list[dict], frames: list[pd.DataFrame], output: Path) -> dict | None:
    source = COUNTERFACTUAL / "summary.json"
    if not source.exists():
        return None
    data = json.loads(source.read_text(encoding="utf-8"))
    # Compact trackable numerical evidence; detailed states/rollouts stay local.
    compact = []
    for run in data["results"]:
        compact.append({key: run[key] for key in (
            "model", "case", "history_mode", "checkpoint", "input_sha256",
            "original_model_files_unchanged", "initialization", "hidden_reset_assumption",
            "metrics", "scope",
        )})
    result = {"source": relative(source), "source_sha256": sha256(source), "simulation_only": True, "results": compact}
    own = "261015_18_59_30"
    real = next(frame for run, frame in zip(runs, frames) if run["run"] == own)
    # Display the first 40 s, before the old real-run reference repeats at 40.96.
    real = real[real.curr_time.between(0, 40)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    for index, channel in enumerate(("pos", "neg")):
        ax = axes[index]
        ax.plot(real.curr_time, real[f"ref_act_{channel}"], "k--", linewidth=1.2, label="real logged reference")
        ax.plot(real.curr_time, real[f"act_{channel}_press"], color="#b33030", label="real HR 18:59")
        for case, label, color in (("atmospheric", "sim HR: atmospheric", "#2374ab"), (own, "sim HR: own real initial pressures", "#29934c")):
            frame = pd.read_csv(COUNTERFACTUAL / "HR_smooth" / f"{case}_consistent_history" / "rollout.csv")
            frame = frame[frame.control_elapsed_sec <= 40]
            ax.plot(frame.control_elapsed_sec, frame[f"act_{channel}"], color=color, linewidth=1, label=label)
        ax.set_ylabel(("Positive" if channel == "pos" else "Negative") + " (kPa abs)")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 40)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Seconds (sim: applied steps / 50; real: logged timestamp)")
    fig.savefig(output / "hr_initial_pressure_counterfactual.png", dpi=160)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    bank = np.load(BANK)
    time, pressure = bank["time"][0], bank["pressure"][0]
    paths = sorted(args.input_root.glob("*/*.csv"))
    if len(paths) != 4:
        raise ValueError(f"expected four experiment CSVs, found {len(paths)}")
    pairs = [analyze(path, time, pressure) for path in paths]
    runs, frames = list(map(list, zip(*pairs)))
    summary = {
        "analysis_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_manifest": json.loads((args.input_root / "drive_download_manifest.json").read_text(encoding="utf-8")),
        "reference_bank": relative(BANK), "reference_bank_sha256": sha256(BANK),
        "pressure_order": PRESSURES, "runs": runs,
        "metric_convention": "Sample-weighted metrics; bias = reference minus observation. Placeholder exclusion is dataset-specific.",
        "limits": "No verified runtime commit/checkpoint hash, RT application times, physical valve opening, or actual initial policy state in source logs.",
    }
    plot_initialization(runs, frames, args.output_root)
    plot_late_tracking(runs, frames, time, pressure, args.output_root)
    summary["counterfactual"] = counterfactual_evidence(runs, frames, args.output_root)
    (args.output_root / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for run in runs:
        for name, values in run["metrics"].items():
            rows.append({"run": run["run"], "policy": run["policy"], "window": name, **values})
    pd.DataFrame(rows).to_csv(args.output_root / "tracking_metrics.csv", index=False)
    print(json.dumps({"output": relative(args.output_root), "runs": len(runs), "counterfactual_runs": len((summary["counterfactual"] or {}).get("results", []))}, ensure_ascii=False))
    for run in runs:
        print(run["run"], run["policy"], "start", run["first_non_placeholder_time_sec"], "dt_ms", run["regular_timestamp_dt_median_sec"] * 1000, "RMSE", run["metrics"]["non_placeholder_reference"]["rmse_pos_kpa"], run["metrics"]["non_placeholder_reference"]["rmse_neg_kpa"])


if __name__ == "__main__":
    main()
