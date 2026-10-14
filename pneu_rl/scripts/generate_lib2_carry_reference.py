#!/usr/bin/env python3
"""Generate a continuous simulator-backed reference for carried horizons.

Unlike a reset episode bank, this tool keeps one PneuSim instance alive for
the complete requested duration.  The stored reference therefore contains
the same continuous physical trajectory across 512-step episode boundaries.
Use a duration covering the intended carried training window; for example,
40.96 seconds covers four 512-step episodes at 50 Hz.
"""

from __future__ import annotations

import argparse
import math
import random
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
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_utils" / "src",
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.sim2 import PneuSim


def simulate(args: argparse.Namespace, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / args.freq
    time = np.arange(0.0, args.duration + 0.5 * dt, dt, dtype=np.float64)
    phase_pos = params["phase_pos"]
    phase_neg = params["phase_neg"]
    pos_signal = np.sin(2.0 * math.pi * params["freq_pos"] * time + phase_pos)
    neg_signal = np.sin(2.0 * math.pi * params["freq_neg"] * time + phase_neg)
    # A small incommensurate second component gives local variation without
    # introducing command discontinuities at carried episode boundaries.
    pos_signal = 0.78 * pos_signal + 0.22 * np.sin(
        2.0 * math.pi * params["freq_pos2"] * time + 1.3 * phase_pos
    )
    neg_signal = 0.78 * neg_signal + 0.22 * np.sin(
        2.0 * math.pi * params["freq_neg2"] * time + 1.3 * phase_neg
    )
    action = np.c_[
        np.clip(params["base_pos"] + params["amp_pos"] * pos_signal, 0.0, 1.0),
        np.clip(params["base_neg"] + params["amp_neg"] * neg_signal, 0.0, 1.0),
    ]
    sim = PneuSim(
        freq=args.freq,
        delay=0.0,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    sim.set_init_press(args.init_pos, args.init_neg)
    pressure = np.empty((len(time), 2), dtype=np.float64)
    pressure[0] = [args.init_pos, args.init_neg]
    for index in range(1, len(time)):
        obs, _ = sim.observe(action[index])
        pressure[index] = obs[1:3]
    return time, pressure, action


def score_candidate(args: argparse.Namespace, time: np.ndarray, pressure: np.ndarray) -> tuple[bool, float, dict]:
    start = min(len(time) - 1, int(round(args.ignore_initial_sec * args.freq)))
    checked = pressure[start:]
    rate = np.gradient(pressure, 1.0 / args.freq, axis=0)[start:]
    feasible = bool(
        np.all(np.isfinite(pressure))
        and np.all(checked[:, 0] >= args.pos_min)
        and np.all(checked[:, 0] <= args.pos_max)
        and np.all(checked[:, 1] >= args.neg_min)
        and np.all(checked[:, 1] <= args.neg_max)
        and np.max(np.abs(rate[:, 0])) <= args.pos_rate_max
        and np.max(np.abs(rate[:, 1])) <= args.neg_rate_max
    )
    span = np.ptp(checked, axis=0)
    score = float(
        span[0] / max(args.pos_max - args.pos_min, 1e-9)
        + span[1] / max(args.neg_max - args.neg_min, 1e-9)
        + 0.1 * np.mean(np.abs(rate[:, 0])) / max(args.pos_rate_max, 1e-9)
        + 0.1 * np.mean(np.abs(rate[:, 1])) / max(args.neg_rate_max, 1e-9)
    )
    metadata = {
        "pressure_min_after_startup": checked.min(axis=0).tolist(),
        "pressure_max_after_startup": checked.max(axis=0).tolist(),
        "pressure_span_after_startup": span.tolist(),
        "reference_rate_abs_max_after_startup": np.max(np.abs(rate), axis=0).tolist(),
        "full_pressure_min_including_startup": pressure.min(axis=0).tolist(),
        "full_pressure_max_including_startup": pressure.max(axis=0).tolist(),
        "score": score,
        "feasible": feasible,
    }
    return feasible, score, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--duration", type=float, default=40.96)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--init-pos", type=float, default=101.325)
    parser.add_argument("--init-neg", type=float, default=101.325)
    parser.add_argument("--scale-low", type=float, default=0.85)
    parser.add_argument("--pos-min", type=float, default=115.0)
    parser.add_argument("--pos-max", type=float, default=220.0)
    parser.add_argument("--neg-min", type=float, default=10.0)
    parser.add_argument("--neg-max", type=float, default=30.0)
    parser.add_argument("--pos-rate-max", type=float, default=32.0)
    parser.add_argument("--neg-rate-max", type=float, default=9.5)
    parser.add_argument("--ignore-initial-sec", type=float, default=5.0)
    parser.add_argument(
        "--fixed-params-profile",
        type=Path,
        default=None,
        help="reuse design.parameters from an existing carry profile and skip candidate search",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    best = None
    if args.fixed_params_profile is not None:
        with args.fixed_params_profile.open(encoding="utf-8") as handle:
            fixed_spec = yaml.safe_load(handle)
        params = dict((fixed_spec.get("design") or {}).get("parameters") or {})
        required = {
            "base_pos", "base_neg", "amp_pos", "amp_neg", "freq_pos", "freq_neg",
            "freq_pos2", "freq_neg2", "phase_pos", "phase_neg",
        }
        missing = sorted(required.difference(params))
        if missing:
            raise ValueError(f"fixed profile is missing carry parameters: {missing}")
        candidate_params = [params]
    else:
        candidate_params = []
        for _ in range(args.candidates):
            candidate_params.append(
                {
                    "base_pos": rng.uniform(0.38, 0.58),
                    "base_neg": rng.uniform(0.72, 0.86),
                    "amp_pos": rng.uniform(0.20, 0.38),
                    "amp_neg": rng.uniform(0.08, 0.18),
                    "freq_pos": rng.uniform(0.015, 0.09),
                    "freq_neg": rng.uniform(0.025, 0.16),
                    "freq_pos2": rng.uniform(0.02, 0.11),
                    "freq_neg2": rng.uniform(0.03, 0.19),
                    "phase_pos": rng.uniform(-math.pi, math.pi),
                    "phase_neg": rng.uniform(-math.pi, math.pi),
                }
            )
    for params in candidate_params:
        time, pressure, action = simulate(args, params)
        feasible, score, metadata = score_candidate(args, time, pressure)
        if feasible and (best is None or score > best["score"]):
            best = {"time": time, "pressure": pressure, "action": action, "params": params, "score": score, "metadata": metadata}
    if best is None:
        raise RuntimeError("no feasible continuous carry candidate; reduce command amplitude or rate limits")
    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        time=best["time"].astype(np.float32),
        pressure=best["pressure"].astype(np.float32),
        raw_action=best["action"].astype(np.float32),
    )
    profile = {
        "kind": "optimized_trajectory_bank",
        "kwargs": {
            "trajectory_npz": str(args.npz_output),
            "seed": args.seed,
            "sample_mode": "coverage",
            "repeat_after_end": False,
        },
        "design": {
            "library": 2,
            "continuous_carry": True,
            "duration_sec": args.duration,
            "horizon_512_sec": 512.0 / args.freq,
            "episode_boundaries_inside_rollout": int(args.duration / (512.0 / args.freq)),
            "generator": "one continuous lib2 simulator rollout",
            "rate_filter_ignore_initial_sec": args.ignore_initial_sec,
            "parameters": best["params"],
            "validation": best["metadata"],
            "note": "Do not use beyond duration when repeat_after_end is false; generate a longer rollout for longer training.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(profile, handle, sort_keys=False)
    rate = np.gradient(best["pressure"], 1.0 / args.freq, axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(best["time"], best["pressure"][:, 0], "r", label="POS")
    axes[0].plot(best["time"], best["pressure"][:, 1], "b", label="NEG")
    axes[1].plot(best["time"], rate[:, 0], "r", label="dPOS/dt")
    axes[1].plot(best["time"], rate[:, 1], "b", label="dNEG/dt")
    axes[2].plot(best["time"], best["action"][:, 0], "r", label="u POS")
    axes[2].plot(best["time"], best["action"][:, 1], "b", label="u NEG")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Pressure [kPa]")
    axes[1].set_ylabel("dP/dt [kPa/s]")
    axes[2].set_ylabel("raw action")
    axes[2].set_xlabel("Continuous time [s]")
    fig.suptitle("Continuous lib2 carry reference")
    fig.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[ INFO] profile: {args.output}")
    print(f"[ INFO] npz: {args.npz_output}")
    print(f"[ INFO] metadata: {best['metadata']}")


if __name__ == "__main__":
    main()
