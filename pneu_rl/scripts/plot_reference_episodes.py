#!/usr/bin/env python3
"""Plot several consecutive reference episodes from a reference-profile YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_ref.factory import make_pressure_reference


def generate_episodes(profile: dict, episodes: int, duration: float, dt: float):
    reference = make_pressure_reference(profile)
    times = np.arange(0.0, duration + 0.5 * dt, dt)

    # Match PneuEnv/PneuRef construction: one reset during wrapper
    # construction, then one reset at the beginning of every episode.
    reference.time_reset()
    reference.get_goal(dt)

    traces = []
    for _ in range(episodes):
        reference.time_reset()
        reference.get_goal(dt)
        trace = np.asarray(
            [reference.get_goal(float(time_value)) for time_value in times],
            dtype=np.float64,
        )
        traces.append(trace)
    return times, np.asarray(traces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--duration", type=float, default=40.96)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    with args.profile.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    times, traces = generate_episodes(profile, args.episodes, args.duration, args.dt)

    pos = traces[:, :, 0]
    neg = traces[:, :, 1]
    neg_rate = np.gradient(neg, args.dt, axis=1)
    cmap = plt.get_cmap("viridis", args.episodes)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "lines.linewidth": 1.6,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax_pos, ax_neg, ax_neg_rate = axes

    for episode in range(args.episodes):
        color = cmap(episode)
        label = f"ep {episode + 1}" if episode in {0, args.episodes - 1} else None
        ax_pos.plot(times, pos[episode], color=color, alpha=0.78, label=label)
        ax_neg.plot(times, neg[episode], color=color, alpha=0.78, label=label)
        ax_neg_rate.plot(times, neg_rate[episode], color=color, alpha=0.68, label=label)

    ax_pos.plot(times, np.min(pos, axis=0), "k--", linewidth=1.0, alpha=0.65, label="10-episode envelope")
    ax_pos.plot(times, np.max(pos, axis=0), "k--", linewidth=1.0, alpha=0.65)
    ax_neg.plot(times, np.min(neg, axis=0), "k--", linewidth=1.0, alpha=0.65, label="10-episode envelope")
    ax_neg.plot(times, np.max(neg, axis=0), "k--", linewidth=1.0, alpha=0.65)

    pos_range = (float(pos.min()), float(pos.max()))
    neg_range = (float(neg.min()), float(neg.max()))
    rate_range = (float(neg_rate.min()), float(neg_rate.max()))
    neg_pad = max(0.5, 0.08 * (neg_range[1] - neg_range[0]))

    ax_pos.set_ylabel("POS ref [kPa]")
    ax_pos.set_title(
        f"{args.profile.name} | {args.episodes} consecutive episodes | "
        f"POS range {pos_range[0]:.1f}–{pos_range[1]:.1f} kPa"
    )
    ax_neg.set_ylabel("NEG ref [kPa]")
    ax_neg.set_title(
        f"NEG detail | range {neg_range[0]:.1f}–{neg_range[1]:.1f} kPa"
    )
    ax_neg.set_ylim(neg_range[0] - neg_pad, neg_range[1] + neg_pad)
    ax_neg_rate.set_ylabel("dNEG/dt [kPa/s]")
    ax_neg_rate.set_xlabel("Time [s]")
    ax_neg_rate.set_title(
        f"NEG local variation | rate range {rate_range[0]:.1f}–{rate_range[1]:.1f} kPa/s"
    )

    for axis in axes:
        axis.grid(True, which="major", alpha=0.32)
        axis.minorticks_on()
        axis.grid(True, which="minor", alpha=0.12)
        axis.axvline(10.24, color="black", linestyle=":", linewidth=1.0, alpha=0.65)
    ax_pos.legend(loc="upper right", ncol=3)
    ax_neg.legend(loc="upper right", ncol=3)
    fig.suptitle("Reference coverage preview", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved: {args.output}")
    print(f"POS range: {pos_range[0]:.3f} .. {pos_range[1]:.3f} kPa")
    print(f"NEG range: {neg_range[0]:.3f} .. {neg_range[1]:.3f} kPa")
    print(f"NEG rate:  {rate_range[0]:.3f} .. {rate_range[1]:.3f} kPa/s")


if __name__ == "__main__":
    main()
