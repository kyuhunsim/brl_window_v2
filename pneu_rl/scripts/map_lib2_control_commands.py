"""Map constant lib2 valve commands to jointly reachable pressures.

This is intentionally different from the directional-rate sweep.  The rate
sweep answers how fast pressure can move locally; this script answers which
positive/negative pressure pairs are obtained simultaneously under a fixed
pair of physical valve commands.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_utils" / "src",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.sim2 import PneuSim


def command_grid(low: float, high: float, points: int) -> np.ndarray:
    if points < 2:
        raise ValueError("--points must be at least 2")
    return np.linspace(float(low), float(high), int(points))


def linear_slope(time: np.ndarray, values: np.ndarray) -> float:
    if len(time) < 2:
        return float("nan")
    centered = time - time.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(centered, values - values.mean()) / denom)


def run_command(
    sim: PneuSim,
    physical_pos: float,
    physical_neg: float,
    *,
    init_pos: float,
    init_neg: float,
    freq: float,
    duration: float,
    window: float,
    pressure_floor: float,
    pressure_ceiling: float,
) -> dict[str, float | int | bool]:
    sim.set_init_press(float(init_pos), float(init_neg))
    raw = np.array(
        [
            (physical_pos - sim.scale_low) / (sim.scale_high - sim.scale_low),
            (physical_neg - sim.scale_low) / (sim.scale_high - sim.scale_low),
        ],
        dtype=np.float64,
    )
    goal = np.array([init_pos, init_neg], dtype=np.float64)
    time_values: list[float] = []
    pos_values: list[float] = []
    neg_values: list[float] = []
    finite = True
    steps = int(math.ceil(float(duration) * float(freq)))
    for _ in range(steps):
        obs, _ = sim.observe(raw, goal)
        if not np.all(np.isfinite(obs)):
            finite = False
            break
        time_values.append(float(obs[0]))
        pos_values.append(float(obs[1]))
        neg_values.append(float(obs[2]))
        if (
            obs[1] < pressure_floor
            or obs[1] > pressure_ceiling
            or obs[2] < pressure_floor
            or obs[2] > pressure_ceiling
        ):
            finite = False
            break

    time_arr = np.asarray(time_values, dtype=np.float64)
    pos_arr = np.asarray(pos_values, dtype=np.float64)
    neg_arr = np.asarray(neg_values, dtype=np.float64)
    if len(time_arr) == 0:
        return dict(
            physical_pos=physical_pos,
            physical_neg=physical_neg,
            raw_pos=float(raw[0]),
            raw_neg=float(raw[1]),
            valid=False,
            bounded=False,
            samples=0,
        )

    start_time = max(float(time_arr[-1]) - float(window), float(time_arr[0]))
    mask = time_arr >= start_time
    if int(mask.sum()) < 2:
        mask = np.ones(len(time_arr), dtype=bool)
    tail_t = time_arr[mask]
    tail_pos = pos_arr[mask]
    tail_neg = neg_arr[mask]
    pos_drift = linear_slope(tail_t, tail_pos)
    neg_drift = linear_slope(tail_t, tail_neg)
    pos_std = float(np.std(tail_pos))
    neg_std = float(np.std(tail_neg))
    bounded = bool(
        finite
        and abs(pos_drift) <= 1.0
        and abs(neg_drift) <= 1.0
        and pos_std <= 8.0
        and neg_std <= 8.0
    )
    return dict(
        physical_pos=float(physical_pos),
        physical_neg=float(physical_neg),
        raw_pos=float(raw[0]),
        raw_neg=float(raw[1]),
        valid=bool(finite),
        bounded=bounded,
        samples=int(len(time_arr)),
        tail_pos_kpa=float(np.mean(tail_pos)),
        tail_neg_kpa=float(np.mean(tail_neg)),
        tail_pos_std_kpa=float(pos_std),
        tail_neg_std_kpa=float(neg_std),
        tail_pos_drift_kpa_s=float(pos_drift),
        tail_neg_drift_kpa_s=float(neg_drift),
        final_pos_kpa=float(pos_arr[-1]),
        final_neg_kpa=float(neg_arr[-1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=11)
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--scale-high", type=float, default=1.0)
    parser.add_argument("--init-pos", type=float, default=101.325)
    parser.add_argument("--init-neg", type=float, default=101.325)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=40.96)
    parser.add_argument("--tail-window", type=float, default=8.0)
    parser.add_argument("--pressure-floor", type=float, default=1.0)
    parser.add_argument("--pressure-ceiling", type=float, default=450.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "pneu_rl" / "exp" / "_analysis" / "lib2_command_map",
    )
    args = parser.parse_args()
    if args.scale_high <= args.scale_low:
        parser.error("--scale-high must be greater than --scale-low")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
    )
    rows = []
    grid = command_grid(args.scale_low, args.scale_high, args.points)
    for physical_pos in grid:
        for physical_neg in grid:
            rows.append(
                run_command(
                    sim,
                    float(physical_pos),
                    float(physical_neg),
                    init_pos=args.init_pos,
                    init_neg=args.init_neg,
                    freq=args.freq,
                    duration=args.duration,
                    window=args.tail_window,
                    pressure_floor=args.pressure_floor,
                    pressure_ceiling=args.pressure_ceiling,
                )
            )

    csv_path = args.output_dir / "command_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row.get("valid")]
    bounded = [row for row in rows if row.get("bounded")]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    if valid:
        x = [row["tail_pos_kpa"] for row in valid]
        y = [row["tail_neg_kpa"] for row in valid]
        color = [row["physical_pos"] for row in valid]
        axes[0].scatter(x, y, c=color, cmap="Reds", s=45, label="valid command")
    if bounded:
        x = [row["tail_pos_kpa"] for row in bounded]
        y = [row["tail_neg_kpa"] for row in bounded]
        axes[1].scatter(x, y, c=[row["physical_neg"] for row in bounded], cmap="Blues", s=55)
    for axis, title in zip(
        axes,
        ("All bounded simulator responses", "Jointly bounded responses"),
    ):
        axis.set_xlabel("Tail mean POS pressure [kPa]")
        axis.set_ylabel("Tail mean NEG pressure [kPa]")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    figure.savefig(args.output_dir / "command_map.png", dpi=180)
    plt.close(figure)

    summary = dict(
        scale_low=float(args.scale_low),
        scale_high=float(args.scale_high),
        grid_points=int(args.points),
        total_commands=len(rows),
        valid_commands=len(valid),
        bounded_commands=len(bounded),
        physical_init=[float(args.init_pos), float(args.init_neg)],
        duration_sec=float(args.duration),
        tail_window_sec=float(args.tail_window),
        bounded_pos_range=[
            float(min(row["tail_pos_kpa"] for row in bounded)) if bounded else None,
            float(max(row["tail_pos_kpa"] for row in bounded)) if bounded else None,
        ],
        bounded_neg_range=[
            float(min(row["tail_neg_kpa"] for row in bounded)) if bounded else None,
            float(max(row["tail_neg_kpa"] for row in bounded)) if bounded else None,
        ],
    )
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print(f"[INFO] command map: {csv_path}")
    print(f"[INFO] bounded commands: {len(bounded)}/{len(rows)}")
    print(f"[INFO] bounded POS range: {summary['bounded_pos_range']}")
    print(f"[INFO] bounded NEG range: {summary['bounded_neg_range']}")


if __name__ == "__main__":
    main()
