#!/usr/bin/env python3
"""Preflight the exact lib3 reference used by screening experiments.

This is a reference/controller plumbing check, not an RL evaluation.  It
reproduces the PneuRef lifecycle used by ``train_screening.py``, reports the
reference envelope and rate limits, and saves a plot with the 512-step carry
boundaries.  With ``--chamber-controller dynamic_reserve`` it also verifies
that the policy receives four actions and the simulator receives six actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_ref.pneu_ref import PneuRef
from pneu_ref.factory import make_pressure_reference

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from train_screening import build_config, make_components


ATM = 101.325


def collect_reference(
    *,
    ref_seed: int,
    blocks: int,
    block_steps: int,
    freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = build_config(
        lib=3,
        variant="sac",
        seed=0,
        ref_seed=ref_seed,
        reference_kind="rate_limited_random",
        carry=True,
        scale_low=0.8,
    )
    reference = make_pressure_reference(cfg["reference"])
    goal = PneuRef(reference, num_prev=0, num_pred=1, ctrl_freq=freq)
    goal.reset(initial_state=np.array([ATM, ATM], dtype=np.float64))

    total_steps = int(blocks * block_steps)
    dt = 1.0 / float(freq)
    times = np.arange(total_steps, dtype=np.float64) * dt
    values = []
    for time_value in times:
        buffer = goal.get_ref(float(time_value)).reshape(-1, 2)
        values.append(buffer[-1].copy())
    return times, np.asarray(values, dtype=np.float64)


def make_plot(
    times: np.ndarray,
    values: np.ndarray,
    *,
    block_steps: int,
    freq: float,
    output: Path,
) -> dict[str, float]:
    dt = 1.0 / float(freq)
    rates = np.gradient(values, dt, axis=0)
    chamber_pos = np.clip(values[:, 0] + 35.0, 101.325, 200.0)
    chamber_neg = np.clip(values[:, 1] - 35.0, 15.0, 101.325)
    cutoff = min(1.0, float(times[-1]))
    evaluated = times >= cutoff
    startup = ~evaluated
    if not np.any(startup):
        startup = np.zeros_like(evaluated, dtype=bool)
        startup[0] = True
    metrics = {
        "pos_min_kpa": float(values[evaluated, 0].min()),
        "pos_max_kpa": float(values[evaluated, 0].max()),
        "neg_min_kpa": float(values[evaluated, 1].min()),
        "neg_max_kpa": float(values[evaluated, 1].max()),
        "pos_rate_abs_max_after_1s_kpa_s": float(
            np.max(np.abs(rates[evaluated, 0]))
        ),
        "neg_rate_abs_max_after_1s_kpa_s": float(
            np.max(np.abs(rates[evaluated, 1]))
        ),
        "pos_rate_abs_max_first_1s_kpa_s": float(
            np.max(np.abs(rates[startup, 0]))
        ),
        "neg_rate_abs_max_first_1s_kpa_s": float(
            np.max(np.abs(rates[startup, 1]))
        ),
        "chamber_pos_ref_min_kpa": float(chamber_pos.min()),
        "chamber_pos_ref_max_kpa": float(chamber_pos.max()),
        "chamber_neg_ref_min_kpa": float(chamber_neg.min()),
        "chamber_neg_ref_max_kpa": float(chamber_neg.max()),
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(times, values[:, 0], "r", label="POS actuator ref")
    axes[0].plot(times, values[:, 1], "b", label="NEG actuator ref")
    axes[0].plot(times, chamber_pos, "r--", linewidth=1.4, label="POS chamber ref (+35)")
    axes[0].plot(times, chamber_neg, "b--", linewidth=1.4, label="NEG chamber ref (-35)")
    axes[0].set_ylabel("Pressure [kPa]")
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(times, chamber_pos, "darkred", label="POS chamber ref")
    axes[1].plot(times, chamber_neg, "darkblue", label="NEG chamber ref")
    axes[1].plot(times, values[:, 0], "r:", label="POS actuator ref")
    axes[1].plot(times, values[:, 1], "b:", label="NEG actuator ref")
    axes[1].set_ylabel("Chamber target [kPa]")
    axes[1].legend(ncol=4, fontsize=8)

    axes[2].plot(times, rates[:, 0], "darkred", label="dPOS/dt")
    axes[2].plot(times, rates[:, 1], "darkblue", label="dNEG/dt")
    axes[2].axhline(12.0, color="red", linestyle=":", alpha=0.7, label="POS limit")
    axes[2].axhline(-12.0, color="red", linestyle=":", alpha=0.7)
    axes[2].axhline(6.0, color="blue", linestyle=":", alpha=0.7, label="NEG limit")
    axes[2].axhline(-6.0, color="blue", linestyle=":", alpha=0.7)
    axes[2].axvspan(0.0, 1.0, color="gray", alpha=0.12, label="first 1 s excluded")
    axes[2].set_ylabel("Rate [kPa/s]")
    axes[2].legend(ncol=3, fontsize=8)

    axes[3].plot(times, values[:, 0] - values[:, 1], color="purple", label="POS - NEG")
    axes[3].set_ylabel("Pressure separation [kPa]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()

    boundary = float(block_steps) / float(freq)
    for axis in axes:
        for value in np.arange(boundary, times[-1] + 0.5 * boundary, boundary):
            axis.axvline(value, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
        axis.grid(True, alpha=0.3)
    fig.suptitle(
        "lib3 rate-limited random reference preflight | "
        f"seed=61098 | {len(values)} steps | carry boundaries"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return metrics


def controller_smoke(
    *,
    ref_seed: int,
    chamber_controller: str,
    steps: int,
) -> dict[str, object]:
    cfg = build_config(
        lib=3,
        variant="sac",
        seed=0,
        ref_seed=ref_seed,
        reference_kind="rate_limited_random",
        carry=True,
        scale_low=0.8,
        chamber_controller=chamber_controller,
    )
    env = make_components(cfg)
    rows = []
    try:
        state, _ = env.reset()
        del state
        action = np.full(env.action_space.shape, 0.5, dtype=np.float64)
        for _ in range(int(steps)):
            _, _, _, _, info = env.step(action)
            applied = np.asarray(info["ctrl_input"], dtype=np.float64).reshape(-1, 6)[0]
            rows.append(
                {
                    "policy_action_dim": int(env.action_space.shape[0]),
                    "applied_action_dim": int(applied.size),
                    "chamber_ctrl": np.asarray(info.get("chamber_ctrl", [np.nan, np.nan])).tolist(),
                    "chamber_target": np.asarray(info.get("chamber_target", [np.nan, np.nan])).tolist(),
                    "applied_ctrl": applied.tolist(),
                }
            )
    finally:
        env.close()
    if not rows:
        raise RuntimeError("controller smoke produced no steps")
    return {
        "policy_action_dim": rows[-1]["policy_action_dim"],
        "applied_action_dim": rows[-1]["applied_action_dim"],
        "first_step": rows[0],
        "last_step": rows[-1],
        "dynamic_reserve_mapping_ok": (
            chamber_controller != "dynamic_reserve"
            or (
                rows[-1]["policy_action_dim"] == 4
                and rows[-1]["applied_action_dim"] == 6
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-seed", type=int, default=61098)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--block-steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument(
        "--chamber-controller",
        choices=("none", "dynamic_reserve"),
        default="none",
    )
    parser.add_argument("--smoke-steps", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pneu_rl/exp/lib3_reference_preflight/rate_limited_seed61098"),
    )
    args = parser.parse_args()

    times, values = collect_reference(
        ref_seed=args.ref_seed,
        blocks=args.blocks,
        block_steps=args.block_steps,
        freq=args.freq,
    )
    metrics = make_plot(
        times,
        values,
        block_steps=args.block_steps,
        freq=args.freq,
        output=args.output_dir / "reference.png",
    )
    smoke = controller_smoke(
        ref_seed=args.ref_seed,
        chamber_controller=args.chamber_controller,
        steps=args.smoke_steps,
    )
    report = {
        "reference": {
            "kind": "rate_limited_random",
            "seed": args.ref_seed,
            "blocks": args.blocks,
            "block_steps": args.block_steps,
            "carry_preview": True,
            "metrics_after_first_1s": metrics,
        },
        "controller_smoke": smoke,
        "pass": bool(smoke["dynamic_reserve_mapping_ok"]),
        "note": (
            "This validates reference envelope/rate and action plumbing. "
            "It is not proof that a learned policy can track every waypoint."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(yaml.safe_dump(report, sort_keys=False))
    print(f"saved: {args.output_dir / 'reference.png'}")


if __name__ == "__main__":
    main()
