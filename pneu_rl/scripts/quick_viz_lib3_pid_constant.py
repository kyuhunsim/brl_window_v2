"""PID feasibility check under the same constant-target lib3 tasks as RL.

This is deliberately not the RL allocator.  It uses the existing directional
PID gains, keeps inactive branch valves at physical 0.8, and reports all four
pressures so a fixed branch drift cannot be mistaken for a successful target.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim3 import PneuSim

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from baseline_lib3 import BASELINE_CONFIG, chamber_allocator, pid_allocator


ATM = 101.325
DT = 1.0 / 50.0
FIXED_PHYSICAL = 0.8
DEFAULT_PID = {
    "act_pos_in": {"kp": 0.09, "ki": 0.005, "kd": 0.001},
    "act_pos_out": {"kp": 0.052, "ki": 0.00045, "kd": 0.001},
    "act_neg_in": {"kp": 0.058, "ki": 0.00135, "kd": 0.001},
    "act_neg_out": {"kp": 0.075, "ki": 0.0006, "kd": 0.001},
    "deadband": 0.5,
    "integral_limit": 100.0,
    "max_open": 1.0,
}


def make_sim() -> PneuSim:
    return PneuSim(
        freq=50.0,
        delay=0.1,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=0.8,
        scale_high=1.0,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )


def run(args: argparse.Namespace) -> pd.DataFrame:
    sim = make_sim()
    pid_cfg = copy.deepcopy(DEFAULT_PID)
    if args.use_baseline_gains:
        pid_cfg = copy.deepcopy(BASELINE_CONFIG["pid"])
    for channel in ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
        for term in ("kp", "ki", "kd"):
            value = getattr(args, f"{channel}_{term}", None)
            if value is not None:
                pid_cfg[channel][term] = value
    for key in ("deadband", "integral_limit"):
        value = getattr(args, key, None)
        if value is not None:
            pid_cfg[key] = value

    fixed_indices = {
        "pos": (1, 4, 5),
        "neg": (0, 2, 3),
        "both": (),
    }[args.task]
    prev_err = np.zeros(4, dtype=np.float64)
    integral_err = np.zeros(4, dtype=np.float64)
    rows: list[dict[str, float]] = []

    sim.set_init_press(ATM, ATM, ATM, ATM)
    for step in range(args.steps):
        current_act = sim.obs[2:4].copy()
        target = np.array([args.target_pos, args.target_neg], dtype=np.float64)
        controller_ref = target.copy()
        if args.task == "pos":
            controller_ref[1] = current_act[1]
        elif args.task == "neg":
            controller_ref[0] = current_act[0]

        chamber_ctrl, chamber_target = chamber_allocator(
            sim.obs[0:2],
            controller_ref,
            mode="reserve",
            chamber_raw=0.0,
            margin=35.0,
            kp=0.04,
            deadband=1.0,
            min_open=0.15,
            max_open=0.85,
            ch_pos_min=ATM,
            ch_pos_max=200.0,
            ch_neg_min=15.0,
            ch_neg_max=ATM,
        )
        action, prev_err, integral_err = pid_allocator(
            current_act,
            controller_ref,
            prev_err,
            integral_err,
            dt=DT,
            pid_cfg=pid_cfg,
            chamber_ctrl=chamber_ctrl,
        )
        for index in fixed_indices:
            action[index] = 0.0  # normalized 0 -> physical 0.8

        obs, _ = sim.observe(action, goal=target)
        debug = sim.get_valve_debug()
        row: dict[str, float] = {
            "step": float(step),
            "time": float((step + 1) * DT),
            "ref_pos": float(target[0]),
            "ref_neg": float(target[1]),
            "chamber_pos": float(obs[1]),
            "chamber_neg": float(obs[2]),
            "act_pos": float(obs[3]),
            "act_neg": float(obs[4]),
            "chamber_target_pos": float(chamber_target[0]),
            "chamber_target_neg": float(chamber_target[1]),
        }
        for index, value in enumerate(action):
            row[f"ctrl{index + 1}_raw"] = float(value)
            row[f"ctrl{index + 1}_physical"] = float(0.8 + 0.2 * value)
        for key, value in debug.items():
            row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def add_rates(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for name in ("ref_pos", "ref_neg", "chamber_pos", "chamber_neg", "act_pos", "act_neg"):
        data[f"rate_{name}"] = np.gradient(data[name].to_numpy(), DT)
    return data


def plot(data: pd.DataFrame, path: Path, task: str) -> None:
    data = add_rates(data)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(data.time, data.ref_pos, "k--", label="REF POS")
    axes[0].plot(data.time, data.act_pos, "r", label="ACT POS")
    axes[0].plot(data.time, data.ref_neg, "k:", label="REF NEG")
    axes[0].plot(data.time, data.act_neg, "b", label="ACT NEG")
    axes[0].set_ylabel("Actuator [kPa]")
    axes[0].legend(ncol=4)
    axes[1].plot(data.time, data.chamber_pos, label="CH POS")
    axes[1].plot(data.time, data.chamber_neg, label="CH NEG")
    axes[1].plot(data.time, data.chamber_target_pos, "--", label="CH POS target")
    axes[1].plot(data.time, data.chamber_target_neg, "--", label="CH NEG target")
    axes[1].set_ylabel("Chamber [kPa]")
    axes[1].legend(ncol=4)
    axes[2].plot(data.time, data.rate_act_pos, "r", label="dACT POS")
    axes[2].plot(data.time, data.rate_act_neg, "b", label="dACT NEG")
    axes[2].axhline(0.0, color="k", linewidth=0.8)
    axes[2].set_ylabel("Rate [kPa/s]")
    axes[2].legend(ncol=2)
    for index in range(6):
        axes[3].plot(data.time, data[f"ctrl{index + 1}_physical"], label=f"u{index + 1}")
    axes[3].set_ylabel("Physical command")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylim(0.78, 1.02)
    axes[3].legend(ncol=6)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"lib3 constant-target PID feasibility | task={task}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("pos", "neg", "both"), required=True)
    parser.add_argument("--target-pos", type=float, default=120.0)
    parser.add_argument("--target-neg", type=float, default=70.0)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--name", required=True)
    parser.add_argument("--use-baseline-gains", action="store_true")
    for channel in ("act_pos_in", "act_pos_out", "act_neg_in", "act_neg_out"):
        for term in ("kp", "ki", "kd"):
            parser.add_argument(f"--{channel.replace('_', '-')}-{term}", dest=f"{channel}_{term}", type=float)
    parser.add_argument("--deadband", type=float)
    parser.add_argument("--integral-limit", dest="integral_limit", type=float)
    args = parser.parse_args()

    output_dir = Path("pneu_rl/exp/lib3_pid_constant") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    data = run(args)
    data.to_csv(output_dir / "trajectory.csv", index=False)
    plot(data, output_dir / "trajectory.png", args.task)

    d = add_rates(data)
    metrics = {
        "task": args.task,
        "target_pos_kpa": args.target_pos,
        "target_neg_kpa": args.target_neg,
        "pos_rmse_kpa": float(np.sqrt(np.mean((d.act_pos - d.ref_pos) ** 2))),
        "neg_rmse_kpa": float(np.sqrt(np.mean((d.act_neg - d.ref_neg) ** 2))),
        "pos_tolerance_fraction": float(np.mean(np.abs(d.act_pos - d.ref_pos) <= 5.0)),
        "neg_tolerance_fraction": float(np.mean(np.abs(d.act_neg - d.ref_neg) <= 5.0)),
        "act_pos_rate_abs_max_kpa_s": float(np.max(np.abs(d.rate_act_pos))),
        "act_neg_rate_abs_max_kpa_s": float(np.max(np.abs(d.rate_act_neg))),
        "chamber_pos_min_kpa": float(d.chamber_pos.min()),
        "chamber_pos_max_kpa": float(d.chamber_pos.max()),
        "chamber_neg_min_kpa": float(d.chamber_neg.min()),
        "chamber_neg_max_kpa": float(d.chamber_neg.max()),
        "fixed_physical_command": FIXED_PHYSICAL,
        "steps": args.steps,
        "freq_hz": 50.0,
        "delay_sec": 0.1,
        "scale_low": 0.8,
    }
    with (output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics, handle, sort_keys=False)
    print(yaml.safe_dump(metrics, sort_keys=False))
    print(f"saved: {output_dir / 'trajectory.png'}")


if __name__ == "__main__":
    main()
