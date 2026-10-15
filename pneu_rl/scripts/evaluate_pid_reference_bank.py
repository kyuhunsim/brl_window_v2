"""Evaluate the fixed feed-forward + I-term controller on every bank member.

This is deliberately separate from ``evaluate_pid_reference.py``, which is a
single-trajectory diagnostic.  Running the same controller on the complete
reset bank prevents one randomly selected trajectory from being mistaken for
the behavior of the whole reference distribution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_env.env import PneuEnv
from pneu_env.sim2 import PneuSim

from pneu_ref.factory import make_pressure_reference


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "pneu_rl" / "exp" / "_analysis" / "pid_reference_bank"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_trajectories(profile: Path) -> list[dict[str, Any]]:
    with profile.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    kwargs = dict(spec.get("kwargs", {}))
    if kwargs.get("trajectories"):
        return list(kwargs["trajectories"])

    npz_value = kwargs.get("trajectory_npz")
    if npz_value is None:
        raise ValueError("profile must contain kwargs.trajectories or kwargs.trajectory_npz")
    with np.load(_resolve_path(str(npz_value))) as data:
        times = np.asarray(data["time"], dtype=np.float64)
        pressure = np.asarray(data["pressure"], dtype=np.float64)
    if pressure.ndim == 2:
        pressure = pressure[None, ...]
    if times.ndim == 1:
        times = np.repeat(times[None, :], pressure.shape[0], axis=0)
    if times.shape[:2] != pressure.shape[:2] or pressure.shape[2] != 2:
        raise ValueError("trajectory_npz time/pressure shapes are incompatible")
    return [
        {"id": index, "time": times[index], "pressure": pressure[index]}
        for index in range(pressure.shape[0])
    ]


def _run_one(args: argparse.Namespace, item: dict[str, Any], index: int) -> tuple[dict[str, float], list[dict[str, float]]]:
    trajectory = {
        "id": int(item.get("id", index)),
        "time": np.asarray(item["time"], dtype=np.float64).tolist(),
        "pressure": np.asarray(item["pressure"], dtype=np.float64).tolist(),
    }
    ref_spec = {
        "kind": "optimized_trajectory_bank",
        "kwargs": {
            "trajectories": [trajectory],
            "seed": 61098,
            "sample_mode": "random",
            "repeat_after_end": False,
        },
    }
    sim = PneuSim(
        freq=args.freq,
        delay=args.delay,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    env = PneuEnv(
        obs=sim,
        ref=make_pressure_reference(ref_spec),
        pred=None,
        num_prev=0,
        num_pred=1,
        num_act=1,
        rwd_kwargs=dict(
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_curr_rwd_coeff=0.3,
            neg_curr_rwd_coeff=0.3,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
        ),
        action_low=0.0,
        action_high=1.0,
        verbose=False,
        episode_carry_state=False,
        reference_reset_to_observation=False,
    )
    env.set_pid(
        Kp_pos=0.0,
        Ki_pos=args.ki_pos,
        Kd_pos=0.0,
        Kp_neg=0.0,
        Ki_neg=args.ki_neg,
        Kd_neg=0.0,
        Ka=args.ka,
    )

    env.reset()
    rows: list[dict[str, float]] = []
    base_action = np.array([args.base_pos, args.base_neg], dtype=np.float64)
    for step in range(args.steps):
        _, reward, _, _, info = env.step(base_action.copy())
        obs_info = info["obs"]
        rows.append(
            {
                "step": float(step),
                "time": float(obs_info["curr_time"]),
                "ref_pos": float(obs_info["ref_pos"]),
                "ref_neg": float(obs_info["ref_neg"]),
                "sen_pos": float(obs_info["sen_pos"]),
                "sen_neg": float(obs_info["sen_neg"]),
                "ctrl_pos": float(obs_info["ctrl_pos"]),
                "ctrl_neg": float(obs_info["ctrl_neg"]),
                "reward": float(reward),
            }
        )
    env.close()

    data = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
    pos_error = data["ref_pos"] - data["sen_pos"]
    neg_error = data["ref_neg"] - data["sen_neg"]
    start_index = min(len(rows), max(0, int(round(args.ignore_initial_sec * args.freq))))
    post_pos = pos_error[start_index:]
    post_neg = neg_error[start_index:]
    metrics = {
        "trajectory_index": float(index),
        "trajectory_id": float(trajectory["id"]),
        "steps": float(len(rows)),
        "pos_rmse": float(np.sqrt(np.mean(pos_error**2))),
        "neg_rmse": float(np.sqrt(np.mean(neg_error**2))),
        "pos_rmse_after_initial": float(np.sqrt(np.mean(post_pos**2))),
        "neg_rmse_after_initial": float(np.sqrt(np.mean(post_neg**2))),
        "pos_mae": float(np.mean(np.abs(pos_error))),
        "neg_mae": float(np.mean(np.abs(neg_error))),
        "pos_bias": float(np.mean(pos_error)),
        "neg_bias": float(np.mean(neg_error)),
        "ctrl_pos_min": float(np.min(data["ctrl_pos"])),
        "ctrl_pos_max": float(np.max(data["ctrl_pos"])),
        "ctrl_neg_min": float(np.min(data["ctrl_neg"])),
        "ctrl_neg_max": float(np.max(data["ctrl_neg"])),
        "ctrl_pos_saturation_fraction": float(np.mean(data["ctrl_pos"] >= 0.999)),
        "ctrl_neg_saturation_fraction": float(np.mean(data["ctrl_neg"] >= 0.999)),
    }
    return metrics, rows


def run(args: argparse.Namespace) -> None:
    trajectories = _load_trajectories(args.reference_profile)
    output_dir = OUT_ROOT / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, float]] = []
    for index, item in enumerate(trajectories):
        metrics, rows = _run_one(args, item, index)
        all_metrics.append(metrics)
        trajectory_dir = output_dir / f"trajectory_{index:02d}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        with (trajectory_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (trajectory_dir / "metrics.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metrics, handle, sort_keys=False)

    metric_names = [
        "pos_rmse", "neg_rmse", "pos_rmse_after_initial", "neg_rmse_after_initial",
        "pos_mae", "neg_mae", "pos_bias", "neg_bias",
        "ctrl_pos_saturation_fraction", "ctrl_neg_saturation_fraction",
    ]
    summary: dict[str, Any] = {
        "reference_profile": str(args.reference_profile),
        "trajectory_count": len(all_metrics),
        "init_pos": args.init_pos,
        "init_neg": args.init_neg,
        "base_pos_raw": args.base_pos,
        "base_neg_raw": args.base_neg,
        "scale_low": args.scale_low,
        "ki_pos": args.ki_pos,
        "ki_neg": args.ki_neg,
        "ka": args.ka,
        "ignore_initial_sec": args.ignore_initial_sec,
        "metrics": {
            name: {
                "mean": float(np.mean([row[name] for row in all_metrics])),
                "median": float(np.median([row[name] for row in all_metrics])),
                "min": float(np.min([row[name] for row in all_metrics])),
                "max": float(np.max([row[name] for row in all_metrics])),
            }
            for name in metric_names
        },
        "per_trajectory": all_metrics,
    }
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    positions = np.arange(len(all_metrics))
    axes[0, 0].bar(positions - 0.18, [row["pos_rmse"] for row in all_metrics], 0.36, label="POS")
    axes[0, 0].bar(positions + 0.18, [row["neg_rmse"] for row in all_metrics], 0.36, label="NEG")
    axes[0, 0].set_title("PID RMSE by trajectory")
    axes[0, 0].set_ylabel("RMSE [kPa]")
    axes[0, 0].legend()
    axes[0, 1].bar(positions - 0.18, [row["pos_bias"] for row in all_metrics], 0.36, label="POS")
    axes[0, 1].bar(positions + 0.18, [row["neg_bias"] for row in all_metrics], 0.36, label="NEG")
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_title("PID bias by trajectory")
    axes[0, 1].set_ylabel("REF - measured [kPa]")
    axes[0, 1].legend()
    axes[1, 0].plot(positions, [row["ctrl_pos_saturation_fraction"] for row in all_metrics], "o-", label="POS")
    axes[1, 0].plot(positions, [row["ctrl_neg_saturation_fraction"] for row in all_metrics], "o-", label="NEG")
    axes[1, 0].set_title("Control saturation fraction")
    axes[1, 0].set_xlabel("Trajectory index")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].legend()
    axes[1, 1].bar(positions - 0.18, [row["pos_rmse_after_initial"] for row in all_metrics], 0.36, label="POS")
    axes[1, 1].bar(positions + 0.18, [row["neg_rmse_after_initial"] for row in all_metrics], 0.36, label="NEG")
    axes[1, 1].set_title(f"RMSE after first {args.ignore_initial_sec:g} s")
    axes[1, 1].set_xlabel("Trajectory index")
    axes[1, 1].set_ylabel("RMSE [kPa]")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, color="lightgray", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=180)
    plt.close(fig)
    print(f"[ INFO] PID bank summary saved to {output_dir / 'summary.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--init-pos", type=float, default=101.325)
    parser.add_argument("--init-neg", type=float, default=101.325)
    parser.add_argument("--base-pos", type=float, default=0.5)
    parser.add_argument("--base-neg", type=float, default=0.5)
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--ki-pos", type=float, default=0.01)
    parser.add_argument("--ki-neg", type=float, default=0.01)
    parser.add_argument("--ka", type=float, default=10.0)
    parser.add_argument("--ignore-initial-sec", type=float, default=5.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
