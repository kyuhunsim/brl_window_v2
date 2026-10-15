"""Create a joint-reachable reference profile from command-map CSV data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pos", type=float, default=120.0)
    parser.add_argument("--max-pos", type=float, default=188.0)
    parser.add_argument("--min-neg", type=float, default=10.0)
    parser.add_argument("--max-neg", type=float, default=27.0)
    parser.add_argument("--initial-pos", type=float, default=150.0)
    parser.add_argument("--initial-neg", type=float, default=22.0)
    parser.add_argument("--pos-rate-up", type=float, default=7.863142133619062)
    parser.add_argument("--pos-rate-down", type=float, default=16.002594937364414)
    parser.add_argument("--neg-rate-up", type=float, default=2.1533693320255414)
    parser.add_argument("--neg-rate-down", type=float, default=4.470560960537573)
    parser.add_argument("--hold-min", type=float, default=1.5)
    parser.add_argument("--hold-max", type=float, default=4.0)
    parser.add_argument("--min-transition", type=float, default=2.0)
    parser.add_argument("--max-frequency-hz", type=float, default=0.17507043740108488)
    parser.add_argument("--initial-hold", type=float, default=1.0)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=20.0,
        help="maximum pressure-space jump between measured joint waypoints",
    )
    parser.add_argument("--seed", type=int, default=61098)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "coverage", "coverage_sweep", "coverage_sweep_neg"),
        default="random",
        help=(
            "waypoint selection: random, least-visited coverage, or least-visited "
            "coverage with farthest-point sweep tie breaking (joint or NEG-axis)"
        ),
    )
    parser.add_argument(
        "--coverage-tie-margin",
        type=int,
        default=0,
        help="allow points within this visit-count margin in coverage modes",
    )
    parser.add_argument(
        "--include-valid",
        action="store_true",
        help="include finite valid command-map points, even if they are not classified as bounded",
    )
    args = parser.parse_args()

    rows = []
    with args.command_map.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            validity_key = "valid" if args.include_valid else "bounded"
            if str(row.get(validity_key, "")).lower() not in {"true", "1", "1.0"}:
                continue
            pos = float(row["tail_pos_kpa"])
            neg = float(row["tail_neg_kpa"])
            if args.min_pos <= pos <= args.max_pos and args.min_neg <= neg <= args.max_neg:
                rows.append(
                    {
                        "point": (pos, neg),
                        "command": (float(row["physical_pos"]), float(row["physical_neg"])),
                    }
                )
    if len(rows) < 2:
        raise RuntimeError("filter left fewer than two jointly reachable points")

    points = np.asarray([row["point"] for row in rows], dtype=np.float64)
    points = np.unique(np.round(points, decimals=3), axis=0)
    initial = np.array([args.initial_pos, args.initial_neg], dtype=np.float64)
    initial = points[np.argmin(np.linalg.norm(points - initial, axis=1))]

    profile = {
        "kind": "joint_reachable_random",
        "kwargs": {
            "points": points.tolist(),
            "initial_pos": float(initial[0]),
            "initial_neg": float(initial[1]),
            "pos_rate_up": args.pos_rate_up,
            "pos_rate_down": args.pos_rate_down,
            "neg_rate_up": args.neg_rate_up,
            "neg_rate_down": args.neg_rate_down,
            "hold_min": args.hold_min,
            "hold_max": args.hold_max,
            "min_transition": args.min_transition,
            "max_frequency_hz": args.max_frequency_hz,
            "initial_hold": args.initial_hold,
            "min_distance": args.min_distance,
            "max_distance": args.max_distance,
            "seed": args.seed,
            "selection_mode": args.selection_mode,
            "coverage_tie_margin": args.coverage_tie_margin,
        },
        "design": {
            "source": str(args.command_map),
            "selection": (
                "valid command-map points, jointly filtered"
                if args.include_valid
                else "bounded command-map points, jointly filtered"
            ),
            "point_count": int(len(points)),
            "requested_initial": [args.initial_pos, args.initial_neg],
            "selected_initial": [float(initial[0]), float(initial[1])],
            "filter": {
                "pos": [args.min_pos, args.max_pos],
                "neg": [args.min_neg, args.max_neg],
            },
            "action_scale": [0.85, 1.0],
            "coverage": {
                "selection_mode": args.selection_mode,
                "coverage_tie_margin": args.coverage_tie_margin,
                "guarantee": (
                    "least-visited jointly reachable point within the local transition "
                    "neighborhood; coverage is balanced across episodes, not every point "
                    "in every single episode"
                ),
            },
            "command_points": [
                {
                    "pressure": [float(row["point"][0]), float(row["point"][1])],
                    "physical_command": [
                        float(row["command"][0]),
                        float(row["command"][1]),
                    ],
                }
                for row in rows
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)
    print(f"[INFO] points: {len(points)}")
    print(f"[INFO] selected initial target: {initial.tolist()}")
    print(f"[INFO] profile: {args.output}")


if __name__ == "__main__":
    main()
