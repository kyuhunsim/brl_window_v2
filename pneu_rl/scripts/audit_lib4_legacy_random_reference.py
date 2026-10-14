#!/usr/bin/env python3
"""Characterize and visualize the legacy lib4 random pressure/displacement ref.

This is intentionally an *audit*, not a feasibility certificate.  The legacy
reference independently samples pressure sinusoids and derives displacement
from a static pressure-difference fit plus clipping; it has no associated
six-valve trajectory.  Use the resulting plot to retain it as a regression or
OOD stress task, while keeping oracle-feasible banks as the causal ablation
training source.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "pneu_ref" / "src", ROOT / "pneu_utils" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from pneu_ref.disp_ref import PressureDiffDisplacementRef
from pneu_ref.random_ref import RandomRef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cfg", default="0625_lib4_Ours")
    parser.add_argument("--duration-sec", type=float, default=40.96)
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    if args.duration_sec <= 0.0 or args.frequency_hz <= 0.0:
        parser.error("duration and frequency must be positive")

    cfg_path = ROOT / "pneu_rl" / "models" / args.model_cfg / "cfg.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pressure = RandomRef(**cfg["rnd_ref"])
    mapping = cfg["pressure_diff_disp_ref"]
    reference = PressureDiffDisplacementRef(pressure, **mapping)
    dt = 1.0 / args.frequency_hz
    time = np.arange(int(round(args.duration_sec / dt)) + 1, dtype=np.float64) * dt
    values = np.asarray([reference.get_goal(float(t)) for t in time], dtype=np.float64)
    raw_disp = mapping["dp_to_disp_slope"] * (values[:, 0] - values[:, 1]) + mapping["dp_to_disp_intercept"]
    clipped_low = raw_disp < mapping["min_disp"]
    clipped_high = raw_disp > mapping["max_disp"]
    rate = np.gradient(values, dt, axis=0)

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    axes[0].plot(time, values[:, 0], color="red", label="legacy ACT POS ref")
    axes[0].plot(time, values[:, 1], color="blue", label="legacy ACT NEG ref")
    axes[0].set_ylabel("Pressure [kPa]")
    axes[1].plot(time, values[:, 2], color="tab:green", label="legacy displacement ref")
    axes[1].plot(time, raw_disp, color="black", linestyle="--", alpha=.7, label="unclipped static ΔP fit")
    axes[1].axhline(mapping["min_disp"], color="gray", linestyle=":", label="min/max clamp")
    axes[1].axhline(mapping["max_disp"], color="gray", linestyle=":")
    axes[1].set_ylabel("Contraction [mm]")
    axes[2].plot(time, rate[:, 0], color="darkred", label="dPOS ref/dt")
    axes[2].plot(time, rate[:, 1], color="navy", label="dNEG ref/dt")
    axes[2].plot(time, rate[:, 2], color="darkgreen", label="dDisp ref/dt")
    axes[2].set_ylabel("Reference rate")
    axes[3].plot(values[:, 0] - values[:, 1], values[:, 2], color="tab:purple", label="legacy pressure-diff mapping")
    axes[3].set_xlabel("ACT POS ref − ACT NEG ref [kPa]")
    axes[3].set_ylabel("Contraction ref [mm]")
    for step in range(512, len(time), 512):
        for axis in axes[:3]:
            axis.axvline(time[step], color="black", linestyle=":", linewidth=.8)
    for axis in axes:
        axis.grid(alpha=.28)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle(
        "lib4 legacy random reference audit — no oracle action/replay proof\n"
        "dotted lines: 512-step boundaries; use as stress/regression, not A0–A5 training source"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "kind": "legacy_random_pressure_diff_displacement_audit",
        "model_cfg": str(cfg_path),
        "duration_sec": float(time[-1]),
        "frequency_hz": args.frequency_hz,
        "reference_min": values.min(axis=0).tolist(),
        "reference_max": values.max(axis=0).tolist(),
        "reference_span": np.ptp(values, axis=0).tolist(),
        "max_abs_reference_rate": np.max(np.abs(rate), axis=0).tolist(),
        "displacement_clamp_low_fraction": float(np.mean(clipped_low)),
        "displacement_clamp_high_fraction": float(np.mean(clipped_high)),
        "oracle_feasibility_proven": False,
        "approved_role": "legacy regression / OOD stress only until separately oracle- or PID-certified",
        "plot": str(args.output),
    }
    summary_path = args.summary_output or args.output.with_suffix(".yaml")
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"plot": str(args.output), "summary": str(summary_path), **summary}, sort_keys=False))


if __name__ == "__main__":
    main()
