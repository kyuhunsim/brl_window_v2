#!/usr/bin/env python3
"""Measure whether a screening trajectory had locally useful action room.

For selected time points the recorded plant state is reconstructed by replaying
the recorded actions.  Around that state, the original command sequence is
perturbed for a short horizon in POS, NEG, and joint directions.  A lower
physical POS command is the expected direction for increasing POS pressure;
for NEG the useful direction depends on the sign of the tracking error, so
both directions are tested.  This is a local dynamic test, not merely a plot
of distance to an action bound.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pneu_env.sim2 import PneuSim


def array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def raw_from_physical(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0)


def rollout_cost(
    actions: np.ndarray,
    refs: np.ndarray,
    index: int,
    delta: np.ndarray,
    horizon: int,
    args: argparse.Namespace,
) -> tuple[float, np.ndarray]:
    # Reconstruct a fresh simulator for every tested direction.  This is more
    # expensive than copying pressure alone, but it also restores the hidden
    # pump/valve state produced by the recorded action prefix.
    local = PneuSim(
        freq=args.freq,
        delay=0.0,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    local.set_init_press(args.init_pos, args.init_neg)
    for step in range(index + 1):
        local.observe(raw_from_physical(actions[step], args.scale_low, args.scale_high))
    observed = []
    stop = min(len(actions), index + 1 + horizon)
    for step in range(index + 1, stop):
        command = np.clip(actions[step] + delta, args.scale_low, args.scale_high)
        obs, _ = local.observe(raw_from_physical(command, args.scale_low, args.scale_high))
        observed.append(obs[1:3].copy())
    observed_array = np.asarray(observed, dtype=np.float64)
    target = refs[index + 1 : stop]
    if len(observed_array) == 0:
        return float("nan"), observed_array
    error = observed_array - target
    return float(np.mean(np.sum(error**2, axis=1))), observed_array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--max-points", type=int, default=30)
    parser.add_argument("--transient-sec", type=float, default=2.0)
    parser.add_argument("--delta-physical", type=float, default=0.005)
    args = parser.parse_args()
    trajectory_path = args.screen_dir / "trajectory.csv"
    metrics_path = args.screen_dir / "metrics.yaml"
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = yaml.safe_load(handle) or {}
    args.init_pos = float(metrics.get("evaluation_init_pos", metrics.get("init_pos", 101.325)))
    args.init_neg = float(metrics.get("evaluation_init_neg", metrics.get("init_neg", 101.325)))
    args.scale_low = float(metrics.get("action_scale_low", 0.85))
    args.scale_high = float(metrics.get("action_scale_high", 1.0))
    refs = np.c_[array(rows, "ref_pos"), array(rows, "ref_neg")]
    actions = np.c_[array(rows, "ctrl_pos"), array(rows, "ctrl_neg")]
    times = array(rows, "time")
    first = int(np.searchsorted(times, args.transient_sec))
    last = len(rows) - args.horizon - 2
    if first >= last:
        raise ValueError("not enough trajectory samples after transient for perturbation")
    indices = list(range(first, last, max(args.stride, 1)))[: args.max_points]
    results = []
    directions = {
        "pos_lower": np.array([-args.delta_physical, 0.0]),
        "pos_higher": np.array([args.delta_physical, 0.0]),
        "neg_lower": np.array([0.0, -args.delta_physical]),
        "neg_higher": np.array([0.0, args.delta_physical]),
        "both_lower": np.array([-args.delta_physical, -args.delta_physical]),
        "both_higher": np.array([args.delta_physical, args.delta_physical]),
        # Cross-sign changes are essential in a coupled two-pressure plant:
        # e.g. raise POS by closing its valve while simultaneously raising NEG
        # by opening its valve.  Same-sign joint perturbations cannot test it.
        "pos_lower_neg_higher": np.array([-args.delta_physical, args.delta_physical]),
        "pos_higher_neg_lower": np.array([args.delta_physical, -args.delta_physical]),
    }
    # Remove an accidental duplicate spelling while keeping the output stable.
    for index in indices:
        base_cost, _ = rollout_cost(
            actions, refs, index, np.zeros(2), args.horizon, args
        )
        row = {
            "step": int(index),
            "time": float(times[index]),
            "ref_pos": float(refs[index, 0]),
            "ref_neg": float(refs[index, 1]),
            "sen_pos": float(array(rows, "sen_pos")[index]),
            "sen_neg": float(array(rows, "sen_neg")[index]),
            "ctrl_pos": float(actions[index, 0]),
            "ctrl_neg": float(actions[index, 1]),
            "base_cost": float(base_cost),
        }
        costs = {}
        for name, delta in directions.items():
            cost, _ = rollout_cost(
                actions, refs, index, delta, args.horizon, args
            )
            costs[name] = float(cost)
            row[f"cost_{name}"] = float(cost)
            row[f"improvement_{name}"] = float(base_cost - cost)
        best_name = min(costs, key=costs.get)
        row["best_direction"] = best_name
        row["best_improvement"] = float(base_cost - costs[best_name])
        row["pos_lower_headroom"] = float((actions[index, 0] - args.scale_low) / (args.scale_high - args.scale_low))
        row["pos_higher_headroom"] = float((args.scale_high - actions[index, 0]) / (args.scale_high - args.scale_low))
        row["neg_lower_headroom"] = float((actions[index, 1] - args.scale_low) / (args.scale_high - args.scale_low))
        row["neg_higher_headroom"] = float((args.scale_high - actions[index, 1]) / (args.scale_high - args.scale_low))
        results.append(row)

    output = args.output or (args.screen_dir / "local_action_perturbation.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "screen_dir": str(args.screen_dir),
                "horizon_steps": int(args.horizon),
                "delta_physical": float(args.delta_physical),
                "points": results,
                "summary": {
                    "points": len(results),
                    "improved_point_ratio": float(np.mean([row["best_improvement"] > 0.0 for row in results])),
                    "strongly_improved_ratio": float(np.mean([row["best_improvement"] > 1.0 for row in results])),
                    "best_improvement_median": float(np.median([row["best_improvement"] for row in results])),
                    "best_improvement_max": float(np.max([row["best_improvement"] for row in results])),
                    "best_direction_counts": {
                        name: int(sum(row["best_direction"] == name for row in results))
                        for name in directions
                    },
                    "interpretation": "positive improvement means a local command perturbation can reduce short-horizon tracking cost; zero/negative means no tested local direction helped.",
                },
            },
            handle,
            sort_keys=False,
        )
    figure_path = output.with_suffix(".png")
    point_time = np.asarray([row["time"] for row in results], dtype=np.float64)
    improvement = np.asarray([row["best_improvement"] for row in results], dtype=np.float64)
    controls = np.asarray([[row["ctrl_pos"], row["ctrl_neg"]] for row in results], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].plot(point_time, improvement, "o-", color="darkgreen")
    axes[0].set_ylabel("Best cost improvement")
    axes[0].set_title("Local action perturbation opportunity")
    axes[1].plot(point_time, controls[:, 0], "o-", color="red", label="POS control")
    axes[1].plot(point_time, controls[:, 1], "o-", color="blue", label="NEG control")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Physical control")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] wrote {output}")
    print(f"[ INFO] wrote {figure_path}")
    print(yaml.safe_dump({"points": len(results), "summary": {
        "improved_point_ratio": float(np.mean([row["best_improvement"] > 0.0 for row in results])),
        "strongly_improved_ratio": float(np.mean([row["best_improvement"] > 1.0 for row in results])),
        "best_improvement_median": float(np.median([row["best_improvement"] for row in results])),
    }}, sort_keys=False))


if __name__ == "__main__":
    main()
