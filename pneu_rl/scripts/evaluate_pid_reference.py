"""Evaluate a fixed feed-forward command plus the lib2 I-term/PID.

This is a no-training diagnostic for separating reference/initial-state
feasibility from an RL policy failure.  The fixed action is normalized to
[0, 1] and the simulator maps it to the configured physical valve range.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pneu_env.env import PneuEnv
from pneu_env.sim2 import PneuSim
from pneu_ref.factory import make_pressure_reference


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "pneu_rl" / "exp" / "_analysis" / "pid_reference"


def run(args: argparse.Namespace) -> None:
    with args.reference_profile.open(encoding="utf-8") as handle:
        reference_spec = yaml.safe_load(handle)

    obs = PneuSim(
        freq=50,
        delay=0.1,
        init_pos_press=args.init_pos,
        init_neg_press=args.init_neg,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    env = PneuEnv(
        obs=obs,
        ref=make_pressure_reference(reference_spec),
        pred=None,
        num_prev=0,
        num_pred=1,
        num_act=1,
        rwd_kwargs=dict(
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_curr_rwd_coeff=0.3,
            neg_curr_rwd_coeff=0.3,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
        ),
        action_low=0.0,
        action_high=1.0,
        verbose=False,
        episode_carry_state=False,
    )
    env.set_pid(
        Kp_pos=0.0,
        Ki_pos=args.ki_pos,
        Kd_pos=0.0,
        Kp_neg=0.0,
        Ki_neg=args.ki_neg,
        Kd_neg=0.0,
        Ka=args.ka,
    )

    state = env.reset()[0]
    rows: list[dict[str, float]] = []
    base_action = np.array([args.base_pos, args.base_neg], dtype=np.float64)
    for step in range(args.steps):
        # sim2.observe adds the PID correction in-place; keep the fixed
        # feed-forward command immutable across steps.
        state, reward, _, _, info = env.step(base_action.copy())
        obs_info = info["obs"]
        rows.append(
            {
                "step": float(step),
                "time": float(obs_info["curr_time"]),
                "ref_pos": float(obs_info["ref_pos"]),
                "ref_neg": float(obs_info["ref_neg"]),
                "sen_pos": float(obs_info["sen_pos"]),
                "sen_neg": float(obs_info["sen_neg"]),
                "ctrl_pos": float(obs_info["ctrl_pos"]),
                "ctrl_neg": float(obs_info["ctrl_neg"]),
                "reward": float(reward),
            }
        )
    env.close()

    data = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
    output_dir = OUT_ROOT / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pos_error = data["ref_pos"] - data["sen_pos"]
    neg_error = data["ref_neg"] - data["sen_neg"]
    metrics = {
        "reference_profile": str(args.reference_profile),
        "init_pos": args.init_pos,
        "init_neg": args.init_neg,
        "base_pos_raw": args.base_pos,
        "base_neg_raw": args.base_neg,
        "scale_low": args.scale_low,
        "ki_pos": args.ki_pos,
        "ki_neg": args.ki_neg,
        "ka": args.ka,
        "steps": args.steps,
        "pos_rmse": float(np.sqrt(np.mean(pos_error**2))),
        "neg_rmse": float(np.sqrt(np.mean(neg_error**2))),
        "pos_mae": float(np.mean(np.abs(pos_error))),
        "neg_mae": float(np.mean(np.abs(neg_error))),
        "pos_bias": float(np.mean(pos_error)),
        "neg_bias": float(np.mean(neg_error)),
        "ctrl_pos_min": float(np.min(data["ctrl_pos"])),
        "ctrl_pos_max": float(np.max(data["ctrl_pos"])),
        "ctrl_neg_min": float(np.min(data["ctrl_neg"])),
        "ctrl_neg_max": float(np.max(data["ctrl_neg"])),
    }
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics, handle, sort_keys=False)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(data["time"], data["ref_pos"], "k", linewidth=2, label="REF POS")
    axes[0].plot(data["time"], data["sen_pos"], "r", linewidth=2, label="POS")
    axes[0].plot(data["time"], data["ref_neg"], "k:", linewidth=2, label="REF NEG")
    axes[0].plot(data["time"], data["sen_neg"], "b", linewidth=2, label="NEG")
    axes[0].set_ylabel("Pressure [kPa]")
    axes[0].legend(ncol=4)
    axes[1].plot(data["time"], pos_error, "r", linewidth=2, label="POS error")
    axes[1].plot(data["time"], neg_error, "b", linewidth=2, label="NEG error")
    axes[1].set_ylabel("Error [kPa]")
    axes[1].legend()
    axes[2].plot(data["time"], data["ctrl_pos"], "r", linewidth=2, label="POS")
    axes[2].plot(data["time"], data["ctrl_neg"], "b", linewidth=2, label="NEG")
    axes[2].set_ylabel("Control")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylim(0.8, 1.0)
    axes[2].legend()
    for axis in axes:
        axis.grid(True, color="lightgray")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(fig)
    print(f"[ INFO] {args.tag}: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--init-pos", type=float, default=101.325)
    parser.add_argument("--init-neg", type=float, default=101.325)
    parser.add_argument("--base-pos", type=float, default=0.5)
    parser.add_argument("--base-neg", type=float, default=0.5)
    parser.add_argument("--scale-low", type=float, default=0.8)
    parser.add_argument("--ki-pos", type=float, default=0.01)
    parser.add_argument("--ki-neg", type=float, default=0.01)
    parser.add_argument("--ka", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
