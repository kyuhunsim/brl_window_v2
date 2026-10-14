"""Characterize lib3 pressure-rate capacity under a fixed action mask.

The sweep is an open-loop plant diagnostic, not a PID or RL controller.  Each
candidate action is evaluated from the same initialized pressure state, so the
result gives a local reachable pressure-rate envelope for the configured mask.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.sim3 import PneuSim


ATM = 101.325
DEFAULT_STATES = (
    (ATM, ATM, ATM, ATM),
    (140.0, 40.0, 130.0, 70.0),
    (120.0, 20.0, 115.0, 60.0),
    (180.0, 80.0, 145.0, 80.0),
)


def full_action(
    policy_action: tuple[float, ...], fixed_indices: tuple[int, int]
) -> np.ndarray:
    free_indices = [index for index in range(6) if index not in fixed_indices]
    full = np.empty(6, dtype=np.float64)
    full[list(fixed_indices)] = 1.0
    full[free_indices] = np.asarray(policy_action, dtype=np.float64)
    return full


def sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    grid = np.linspace(0.0, 1.0, args.grid_points)
    policy_actions = list(itertools.product(grid, repeat=4))
    fixed_indices = tuple(sorted(args.fixed_control_indices))
    free_indices = [index for index in range(6) if index not in fixed_indices]
    warmup_steps = max(int(round(args.warmup_sec * args.freq)), 1)
    measure_steps = max(int(round(args.measure_sec * args.freq)), 2)
    rows: list[dict[str, float]] = []

    for state_id, initial_state in enumerate(DEFAULT_STATES):
        for policy_action in policy_actions:
            action = full_action(policy_action, tuple(args.fixed_control_indices))
            sim.set_init_press(*initial_state)
            for _ in range(warmup_steps):
                sim.observe(action)
            start, _ = sim.observe(action)
            for _ in range(measure_steps - 1):
                end, _ = sim.observe(action)
            elapsed = measure_steps / args.freq
            rate = (np.asarray(end[3:5]) - np.asarray(start[3:5])) / elapsed
            rows.append(
                {
                    "state_id": float(state_id),
                    "init_ch_pos": float(initial_state[0]),
                    "init_ch_neg": float(initial_state[1]),
                    "init_act_pos": float(initial_state[2]),
                    "init_act_neg": float(initial_state[3]),
                    "pos_rate_kpa_s": float(rate[0]),
                    "neg_rate_kpa_s": float(rate[1]),
                }
            )
            row = rows[-1]
            for index in range(6):
                row[f"normalized_u{index}"] = float(action[index])
                row[f"physical_u{index}"] = float(
                    args.scale_low + (1.0 - args.scale_low) * action[index]
                )
                if index in fixed_indices:
                    row[f"fixed_u{index}"] = float(action[index])
                else:
                    row[f"policy_u{index}"] = float(action[index])
    return rows


def summarize(df: pd.DataFrame, fixed_control_indices: tuple[int, int]) -> dict:
    fixed_control_indices = tuple(sorted(fixed_control_indices))
    summary = {
        "grid_points_per_policy_control": int(df["policy_u0"].nunique()),
        "candidate_count": int(len(df)),
        "fixed_controls": {
            f"u{index}": 1.0 for index in fixed_control_indices
        },
        "policy_controls": [
            f"u{index}" for index in range(6) if index not in fixed_control_indices
        ],
        "global": {
            "pos_rate_min_kpa_s": float(df["pos_rate_kpa_s"].min()),
            "pos_rate_max_kpa_s": float(df["pos_rate_kpa_s"].max()),
            "neg_rate_min_kpa_s": float(df["neg_rate_kpa_s"].min()),
            "neg_rate_max_kpa_s": float(df["neg_rate_kpa_s"].max()),
        },
        "per_state": {},
    }
    for state_id, group in df.groupby("state_id"):
        summary["per_state"][str(int(state_id))] = {
            "pos_rate_min_kpa_s": float(group["pos_rate_kpa_s"].min()),
            "pos_rate_max_kpa_s": float(group["pos_rate_kpa_s"].max()),
            "neg_rate_min_kpa_s": float(group["neg_rate_kpa_s"].min()),
            "neg_rate_max_kpa_s": float(group["neg_rate_kpa_s"].max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--warmup-sec", type=float, default=0.2)
    parser.add_argument("--measure-sec", type=float, default=0.5)
    parser.add_argument("--scale-low", type=float, default=0.8)
    parser.add_argument(
        "--fixed-control-indices",
        type=int,
        nargs=2,
        default=(2, 4),
        metavar=("INDEX_A", "INDEX_B"),
        help="zero-based controls fixed at normalized 1.0 (default: 2 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pneu_rl/exp/lib3_masked_rate_capacity"),
    )
    args = parser.parse_args()
    if args.grid_points < 2:
        parser.error("--grid-points must be at least 2")
    if len(set(args.fixed_control_indices)) != 2 or any(
        index < 0 or index >= 6 for index in args.fixed_control_indices
    ):
        parser.error("fixed control indices must be two distinct values in [0, 5]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = sweep(args)
    df = pd.DataFrame(rows)
    summary = summarize(df, tuple(args.fixed_control_indices))
    summary.update(
        {
            "simulator": {
                "library": "lib3",
                "scale_low": args.scale_low,
                "scale_high": 1.0,
                "freq_hz": args.freq,
                "warmup_sec": args.warmup_sec,
                "measure_sec": args.measure_sec,
            },
        "reference_candidate_rate_limits_kpa_s": {
                "pos_abs": 12.0,
                "neg_abs": 6.0,
            },
            "note": (
                "This is an open-loop local rate envelope. It does not prove "
                "closed-loop tracking or target-pressure reachability."
            ),
        }
    )
    df.to_csv(args.output_dir / "rate_sweep.csv", index=False)
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["pos_rate_kpa_s"],
        df["neg_rate_kpa_s"],
        c=df["state_id"],
        cmap="viridis",
        s=18,
        alpha=0.7,
    )
    ax.axvline(12.0, color="red", linestyle="--", label="candidate POS rate")
    ax.axvline(-12.0, color="red", linestyle="--")
    ax.axhline(6.0, color="blue", linestyle="--", label="candidate NEG rate")
    ax.axhline(-6.0, color="blue", linestyle="--")
    ax.set_xlabel("dP POS/dt [kPa/s]")
    ax.set_ylabel("dP NEG/dt [kPa/s]")
    ax.set_title("lib3 fixed-mask local pressure-rate sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="initial state id")
    fig.tight_layout()
    fig.savefig(args.output_dir / "rate_reachability.png", dpi=160)
    plt.close(fig)

    print(yaml.safe_dump(summary, sort_keys=False))
    print(f"saved rate sweep: {args.output_dir}")


if __name__ == "__main__":
    main()
