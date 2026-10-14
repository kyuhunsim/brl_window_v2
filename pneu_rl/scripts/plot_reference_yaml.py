#!/usr/bin/env python3
"""Plot the exact pressure reference generated from a reference-profile YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_ref.factory import make_pressure_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--duration", type=float, default=40.96)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=40.96,
        help="episode boundary spacing to draw for carry-mode previews",
    )
    parser.add_argument("--initial-pos", type=float, default=None)
    parser.add_argument("--initial-neg", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.profile.open() as f:
        profile = yaml.safe_load(f)

    t = np.arange(0.0, args.duration + 0.5 * args.dt, args.dt)
    reference = make_pressure_reference(profile)
    if (args.initial_pos is None) != (args.initial_neg is None):
        parser.error("--initial-pos and --initial-neg must be supplied together")
    if args.initial_pos is not None:
        if hasattr(reference, "set_initial_state"):
            reference.set_initial_state(args.initial_pos, args.initial_neg)
        else:
            raise ValueError("this reference does not support an overridden initial state")
    # PneuEnv currently initializes PneuRef once in its constructor and then
    # resets it again in env.reset().  JointReachableRandomRef intentionally
    # keeps its RNG stream across resets, so those two resets consume two
    # different random waypoint sequences.  Reproduce that lifecycle here so
    # the preview PNG is the same sequence seen by screen/quick-viz.
    reference.time_reset()
    reference.get_goal(args.dt)
    reference.time_reset()
    reference.get_goal(args.dt)
    ref = np.asarray([reference.get_goal(float(time_value)) for time_value in t])
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"expected reference shape (N, 2), got {ref.shape}")

    rate = np.gradient(ref, args.dt, axis=0)
    sample_times = [0.0, 10.24]
    if args.episode_duration > 0.0:
        sample_times.extend(
            np.arange(
                args.episode_duration,
                args.duration + 0.5 * args.episode_duration,
                args.episode_duration,
            ).tolist()
        )
    sample_times = sorted({round(float(value), 6) for value in sample_times if value <= args.duration})
    sample_idx = [int(np.argmin(np.abs(t - value))) for value in sample_times]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    ax_ref, ax_rate = axes

    ax_ref.plot(t, ref[:, 0], color="red", label="POS ref")
    ax_ref.plot(t, ref[:, 1], color="blue", label="NEG ref")
    ax_ref.axvline(10.24, color="black", linestyle="--", linewidth=1.3,
                   label="512 steps = 10.24 s")
    if args.episode_duration > 0.0:
        for boundary in np.arange(
            args.episode_duration,
            args.duration + 0.5 * args.episode_duration,
            args.episode_duration,
        ):
            ax_ref.axvline(float(boundary), color="gray", linestyle=":", linewidth=1.3)
    ax_ref.set_ylabel("Reference pressure [kPa]")
    ax_ref.set_title(f"Reference generated from {args.profile.name}")
    ax_ref.grid(True, alpha=0.3)
    ax_ref.legend(loc="best", ncol=3)

    ax_rate.plot(t, rate[:, 0], color="darkred", label="dPOS ref/dt")
    ax_rate.plot(t, rate[:, 1], color="darkblue", label="dNEG ref/dt")
    ax_rate.axhline(0.0, color="black", linewidth=0.9)
    ax_rate.axvline(10.24, color="black", linestyle="--", linewidth=1.3)
    if args.episode_duration > 0.0:
        for boundary in np.arange(
            args.episode_duration,
            args.duration + 0.5 * args.episode_duration,
            args.episode_duration,
        ):
            ax_rate.axvline(float(boundary), color="gray", linestyle=":", linewidth=1.3)
    ax_rate.set_xlabel("Time [s]")
    ax_rate.set_ylabel("Reference rate [kPa/s]")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend(loc="best")

    rows = []
    for idx, sample_time in zip(sample_idx, sample_times):
        rows.append(
            f"{sample_time:g}s: POS={ref[idx, 0]:.1f}, NEG={ref[idx, 1]:.1f} kPa"
        )
    ax_ref.text(
        0.995,
        0.03,
        "\n".join(rows),
        transform=ax_ref.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved: {args.output}")
    print("samples:")
    for idx, sample_time in zip(sample_idx, sample_times):
        print(
            f"  t={sample_time:5.2f}s  pos={ref[idx, 0]:8.3f}  "
            f"neg={ref[idx, 1]:8.3f}  "
            f"dpos={rate[idx, 0]:8.3f}  dneg={rate[idx, 1]:8.3f}"
        )


if __name__ == "__main__":
    main()
