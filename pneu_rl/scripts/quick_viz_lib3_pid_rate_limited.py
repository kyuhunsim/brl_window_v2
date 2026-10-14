"""Quick-viz lib3 PID with command-line per-channel gain overrides."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_lib3_matched_pid_and_valve_sweep import add_rates, plot_pid
from tune_lib3_pid_rate_limited import run_candidate


DEFAULT_PID = {
    "act_pos_in": {"kp": 0.030, "ki": 0.00100, "kd": 0.001},
    "act_pos_out": {"kp": 0.050, "ki": 0.00025, "kd": 0.001},
    "act_neg_in": {"kp": 0.038, "ki": 0.00135, "kd": 0.001},
    "act_neg_out": {"kp": 0.075, "ki": 0.00060, "kd": 0.001},
    "deadband": 0.5,
    "integral_limit": 100.0,
    "max_open": 1.0,
}


def add_gain_arguments(parser: argparse.ArgumentParser) -> None:
    for direction in ("pos_in", "pos_out", "neg_in", "neg_out"):
        cli_direction = direction.replace("_", "-")
        parser.add_argument(f"--kp-{cli_direction}", type=float, default=None)
        parser.add_argument(f"--ki-{cli_direction}", type=float, default=None)
        parser.add_argument(f"--kd-{cli_direction}", type=float, default=None)


def make_pid(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(DEFAULT_PID)
    for direction in ("pos_in", "pos_out", "neg_in", "neg_out"):
        channel = f"act_{direction}"
        for gain in ("kp", "ki", "kd"):
            value = getattr(args, f"{gain}_{direction}")
            if value is not None:
                config[channel][gain] = float(value)
    return config


def plot_valve_diagnostics(data, path: Path) -> None:
    """Visualize hidden valve quantities for diagnosis, not policy input."""
    channels = ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for channel in channels:
        axes[0].plot(data.time, data[f"{channel}_z"], label=channel)
        axes[1].plot(data.time, data[f"{channel}_area_eff"], label=channel)
        axes[2].plot(data.time, data[f"{channel}_q_pred_lpm"], label=channel)
    axes[0].set_ylabel("z")
    axes[1].set_ylabel("area_eff")
    axes[2].set_ylabel("q_pred [L/min]")
    axes[2].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=4)
    fig.suptitle("lib3 hidden valve-state diagnostics (not policy observation)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="output folder name")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=512)
    parser.add_argument("--no-carry", action="store_true")
    parser.add_argument("--deadband", type=float, default=0.5)
    parser.add_argument("--integral-limit", type=float, default=100.0)
    parser.add_argument("--max-open", type=float, default=1.0)
    add_gain_arguments(parser)
    args = parser.parse_args()

    pid_config = make_pid(args)
    pid_config["deadband"] = args.deadband
    pid_config["integral_limit"] = args.integral_limit
    pid_config["max_open"] = args.max_open

    output_dir = Path("pneu_rl/exp/lib3_pid_quick_viz") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    data = run_candidate(
        1.0,
        1.0,
        blocks=args.blocks,
        block_steps=args.block_steps,
        carry=not args.no_carry,
        pid_config=pid_config,
    )
    data.to_csv(output_dir / "trajectory.csv", index=False)
    plot_pid(data, output_dir / "trajectory.png", episode=0)
    plot_valve_diagnostics(data, output_dir / "valve_diagnostics.png")

    d = add_rates(data)
    rise = d["rate_ref_pos"] > 1.0
    pos_err = d["act_pos"] - d["ref_pos"]
    neg_err = d["act_neg"] - d["ref_neg"]
    summary = {
        "name": args.name,
        "pid": pid_config,
        "protocol": {
            "blocks": args.blocks,
            "block_steps": args.block_steps,
            "carry": not args.no_carry,
            "total_steps": args.blocks * args.block_steps,
            "freq_hz": 50.0,
            "delay_sec": 0.1,
            "scale_low": 0.8,
            "reference": "rate_limited_random seed=61098",
        },
        "metrics": {
            "pos_rmse_kpa": float(np.sqrt(np.mean(pos_err**2))),
            "neg_rmse_kpa": float(np.sqrt(np.mean(neg_err**2))),
            "rise_pos_mae_kpa": float(np.mean(np.abs(pos_err[rise]))),
            "rise_ref_rate_mean_kpa_s": float(d.loc[rise, "rate_ref_pos"].mean()),
            "rise_act_rate_mean_kpa_s": float(d.loc[rise, "rate_act_pos"].mean()),
            "act_pos_rate_abs_max_kpa_s": float(np.max(np.abs(d["rate_act_pos"]))),
            "act_neg_rate_abs_max_kpa_s": float(np.max(np.abs(d["rate_act_neg"]))),
            "rise_u3_physical_mean": float(d.loc[rise, "ctrl3_physical"].mean()),
            "rise_u3_physical_max": float(d.loc[rise, "ctrl3_physical"].max()),
        },
    }
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(yaml.safe_dump(summary, sort_keys=False))
    print(f"saved: {output_dir / 'trajectory.png'}")
    print(f"saved: {output_dir / 'valve_diagnostics.png'}")


if __name__ == "__main__":
    main()
