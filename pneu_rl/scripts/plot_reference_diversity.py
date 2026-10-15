#!/usr/bin/env python3
"""Visualize reference diversity for reset and carried episode semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_ref.factory import make_pressure_reference


def anchor_reference(reference, pos: float, neg: float) -> None:
    if not hasattr(reference, "set_initial_state"):
        raise ValueError("reference does not support an initial pressure override")
    reference.set_initial_state(pos, neg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--mode", choices=("reset", "carry"), required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--episode-duration", type=float, default=40.96)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--initial-pos", type=float, default=140.0)
    parser.add_argument("--initial-neg", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.profile.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    reference = make_pressure_reference(profile)
    anchor_reference(reference, args.initial_pos, args.initial_neg)
    reference.time_reset()

    steps = int(round(args.episode_duration / args.dt))
    local_time = np.arange(steps, dtype=float) * args.dt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, args.episodes))

    if args.mode == "reset":
        for episode in range(args.episodes):
            anchor_reference(reference, args.initial_pos, args.initial_neg)
            reference.time_reset()
            values = np.asarray(
                [reference.get_goal(float(time_value)) for time_value in local_time]
            )
            label = f"episode {episode + 1}" if episode in (0, args.episodes - 1) else None
            axes[0].plot(local_time, values[:, 0], color=colors[episode], alpha=0.75, label=label)
            axes[0].plot(local_time, values[:, 1], color=colors[episode], alpha=0.75, linestyle="--")
        title = f"Reset reference diversity ({args.episodes} independent episodes)"
        xlabel = "Episode-local time [s]"
    else:
        global_time = np.arange(args.episodes * steps, dtype=float) * args.dt
        values = np.asarray(
            [reference.get_goal(float(time_value)) for time_value in global_time]
        )
        for episode in range(args.episodes):
            start = episode * steps
            stop = (episode + 1) * steps
            label = f"episode {episode + 1}" if episode in (0, args.episodes - 1) else None
            axes[0].plot(global_time[start:stop], values[start:stop, 0], color=colors[episode], alpha=0.75, label=label)
            axes[0].plot(global_time[start:stop], values[start:stop, 1], color=colors[episode], alpha=0.75, linestyle="--")
            if episode < args.episodes:
                boundary = episode * args.episode_duration
                if boundary > 0.0:
                    axes[0].axvline(boundary, color="gray", linestyle=":", linewidth=1.0)
                    axes[1].axvline(boundary, color="gray", linestyle=":", linewidth=1.0)
        local_time = global_time
        title = f"Carry reference diversity ({args.episodes} continuous episodes)"
        xlabel = "Continuous time [s]"

    rate = np.gradient(values, args.dt, axis=0)
    axes[1].plot(local_time, rate[:, 0], color="darkred", label="dPOS ref/dt")
    axes[1].plot(local_time, rate[:, 1], color="darkblue", label="dNEG ref/dt")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(title)
    axes[0].set_ylabel("Reference pressure [kPa]")
    axes[1].set_ylabel("Reference rate [kPa/s]")
    axes[1].set_xlabel(xlabel)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
