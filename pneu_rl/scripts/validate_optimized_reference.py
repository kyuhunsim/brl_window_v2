#!/usr/bin/env python3
"""Validate an optimized trajectory bank by simulator replay.

This is an oracle/open-loop feasibility check: the command waveform stored
with each trajectory is replayed from the same simulator initial condition.
It does not claim that a fixed PID can track every trajectory; that is a
separate feedback-difficulty experiment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_env.sim2 import PneuSim


ROOT = Path(__file__).resolve().parents[2]


def run_one(profile: dict, trajectory: dict, args: argparse.Namespace) -> dict:
    time = np.asarray(trajectory["time"], dtype=np.float64)
    reference = np.asarray(trajectory["pressure"], dtype=np.float64)
    action = np.asarray(trajectory["raw_action"], dtype=np.float64)
    if len(time) != len(reference) or len(time) != len(action):
        raise ValueError("time, pressure and raw_action lengths do not match")
    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    sim.set_init_press(args.init_pos, args.init_neg)
    observed = np.empty_like(reference)
    observed[0] = [args.init_pos, args.init_neg]
    for index in range(1, len(time)):
        obs, _ = sim.observe(action[index], goal=reference[index - 1])
        observed[index] = obs[1:3]
    error = observed - reference
    return {
        "trajectory_id": int(trajectory.get("id", -1)),
        "pos_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "neg_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "pos_max_abs_error": float(np.max(np.abs(error[:, 0]))),
        "neg_max_abs_error": float(np.max(np.abs(error[:, 1]))),
        "stored_pos_min": float(np.min(reference[:, 0])),
        "stored_pos_max": float(np.max(reference[:, 0])),
        "stored_neg_min": float(np.min(reference[:, 1])),
        "stored_neg_max": float(np.max(reference[:, 1])),
        "raw_pos_min": float(np.min(action[:, 0])),
        "raw_pos_max": float(np.max(action[:, 0])),
        "raw_neg_min": float(np.min(action[:, 1])),
        "raw_neg_max": float(np.max(action[:, 1])),
        "observed": observed,
        "reference": reference,
        "time": time,
        "action": action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--init-pos", type=float, default=101.325)
    parser.add_argument("--init-neg", type=float, default=101.325)
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--max-trajectories", type=int, default=0)
    args = parser.parse_args()
    with args.reference_profile.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    kwargs = profile["kwargs"]
    trajectories = kwargs.get("trajectories")
    if trajectories is None and kwargs.get("trajectory_npz") is not None:
        with np.load(kwargs["trajectory_npz"]) as data:
            time = np.asarray(data["time"], dtype=np.float64)
            pressure = np.asarray(data["pressure"], dtype=np.float64)
            action = np.asarray(data["raw_action"], dtype=np.float64)
        if pressure.ndim == 2:
            pressure = pressure[None, ...]
            action = action[None, ...]
        if time.ndim == 1:
            time = np.repeat(time[None, :], pressure.shape[0], axis=0)
        trajectories = [
            {"id": index, "time": time[index], "pressure": pressure[index], "raw_action": action[index]}
            for index in range(pressure.shape[0])
        ]
    if args.max_trajectories > 0:
        trajectories = trajectories[: args.max_trajectories]
    results = [run_one(profile, trajectory, args) for trajectory in trajectories]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for result in results:
        csv_rows.append({key: value for key, value in result.items() if not isinstance(value, np.ndarray)})
    with (args.output_dir / "oracle_replay_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "reference_profile": str(args.reference_profile),
        "trajectories_checked": len(results),
        "max_pos_rmse": max(item["pos_rmse"] for item in results),
        "max_neg_rmse": max(item["neg_rmse"] for item in results),
        "max_pos_abs_error": max(item["pos_max_abs_error"] for item in results),
        "max_neg_abs_error": max(item["neg_max_abs_error"] for item in results),
        "interpretation": "stored reference is simulator-reachable when oracle commands reproduce with near-zero error",
    }
    with (args.output_dir / "oracle_replay_summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.turbo(np.linspace(0.03, 0.97, len(results)))
    for color, result in zip(colors, results):
        time = result["time"]
        reference = result["reference"]
        observed = result["observed"]
        action = result["action"]
        axes[0].plot(time, reference[:, 0], color=color, alpha=0.55)
        axes[0].plot(time, observed[:, 0], color=color, linestyle="--", alpha=0.8)
        axes[1].plot(time, reference[:, 1], color=color, alpha=0.55)
        axes[1].plot(time, observed[:, 1], color=color, linestyle="--", alpha=0.8)
        axes[2].plot(time, action[:, 0], color=color, alpha=0.7)
        axes[2].plot(time, action[:, 1], color=color, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("POS [kPa]\nsolid ref / dashed replay")
    axes[1].set_ylabel("NEG [kPa]\nsolid ref / dashed replay")
    axes[2].set_ylabel("raw action\nsolid POS / dashed NEG")
    axes[2].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle("Optimized reference oracle replay validation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "oracle_replay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] {summary}")


if __name__ == "__main__":
    main()
