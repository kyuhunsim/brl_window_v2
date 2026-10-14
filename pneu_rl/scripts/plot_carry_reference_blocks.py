#!/usr/bin/env python3
"""Plot every 512-step segment of a continuous carry reference separately.

The figure is a task-distribution preview, not a control result: each column
is one consecutive RL horizon and POS/NEG use separate y-axes so the small
negative-pressure motion remains visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    with args.reference_profile.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    design = dict(profile.get("design", {}))
    npz_path = resolve_path(str(profile["kwargs"]["trajectory_npz"]))
    with np.load(npz_path) as data:
        time = np.asarray(data["time"], dtype=np.float64)
        pressure = np.asarray(data["pressure"], dtype=np.float64)
    freq = 1.0 / float(np.median(np.diff(time)))
    horizon = int(round(float(design.get("horizon_512_sec", 10.24)) * freq))
    block_count = (len(pressure) - 1) // horizon
    columns = max(1, min(args.columns, block_count))
    rows = int(np.ceil(block_count / columns))
    figure, axes = plt.subplots(2 * rows, columns, figsize=(3.8 * columns, 3.0 * rows * 2), sharex=False)
    axes = np.asarray(axes).reshape(2 * rows, columns)
    for block in range(block_count):
        row, col = divmod(block, columns)
        start = block * horizon
        stop = start + horizon + 1
        local_time = time[start:stop] - time[start]
        pos = pressure[start:stop, 0]
        neg = pressure[start:stop, 1]
        pos_axis = axes[2 * row, col]
        neg_axis = axes[2 * row + 1, col]
        pos_axis.plot(local_time, pos, color="red", linewidth=1.8)
        neg_axis.plot(local_time, neg, color="blue", linewidth=1.8)
        pos_axis.set_title(f"carry block {block + 1} | POS span {np.ptp(pos):.1f} kPa", fontsize=10)
        neg_axis.set_title(f"NEG span {np.ptp(neg):.1f} kPa", fontsize=10)
        pos_axis.set_ylabel("POS [kPa]")
        neg_axis.set_ylabel("NEG [kPa]")
        neg_axis.set_xlabel("local time [s]")
        for axis in (pos_axis, neg_axis):
            axis.grid(True, which="major", alpha=0.32)
            axis.minorticks_on()
            axis.grid(True, which="minor", alpha=0.12)
    for block in range(block_count, rows * columns):
        row, col = divmod(block, columns)
        axes[2 * row, col].axis("off")
        axes[2 * row + 1, col].axis("off")
    figure.suptitle(
        f"{args.reference_profile.name}: 10.24-s carry tasks (separate POS / NEG scales)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"[ INFO] saved: {args.output}")


if __name__ == "__main__":
    main()
