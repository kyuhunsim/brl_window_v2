"""Smoke-test lib3 with an arbitrary fixed-action mask and legacy RandomRef.

This is deliberately not an RL evaluation.  It verifies that the policy sees
four controls, the simulator receives the intended six-control vector, PID is
not enabled, and the legacy lib3 reference has reasonable target/rate values
before a full training run is started.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pneu_env.env3 import PneuEnv3
from pneu_env.sim3 import PneuSim
from pneu_ref.feasible_random_ref import FeasibleRandomRef
from pneu_ref.random_ref import RandomRef


ATM = 101.325
REF_KWARGS = dict(
    pos_min_off=115,
    pos_max_off=140,
    neg_min_off=60,
    neg_max_off=80,
    pos_max_ts=10,
    neg_max_ts=10,
    pos_max_amp=10,
    neg_max_amp=5,
    seed=61098,
)
SMOOTH_REF_KWARGS = dict(
    # Preserve the approximate target envelope of the legacy lib3 RandomRef:
    # POS offset +/- amplitude ~= 105--150, NEG ~= 55--85 kPa.
    pos_min=105.0,
    pos_max=150.0,
    neg_min=55.0,
    neg_max=85.0,
    pos_rate_up=12.0,
    pos_rate_down=12.0,
    neg_rate_up=6.0,
    neg_rate_down=6.0,
    hold_min=1.5,
    hold_max=4.0,
    min_transition=2.5,
    max_frequency_hz=0.2,
    initial_pos=ATM,
    initial_neg=ATM,
    initial_hold=0.0,
    min_pos_delta=5.0,
    min_neg_delta=2.0,
    seed=61098,
)


def build_env(reference_kind: str) -> PneuEnv3:
    sim = PneuSim(
        freq=50,
        delay=0.1,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=0.8,
        scale_high=1.0,
        init_pos_press=ATM,
        init_neg_press=ATM,
        init_act_pos_press=ATM,
        init_act_neg_press=ATM,
    )
    if reference_kind == "legacy":
        ref = RandomRef(**REF_KWARGS)
    elif reference_kind == "smoothed":
        ref = FeasibleRandomRef(**SMOOTH_REF_KWARGS)
    else:
        raise ValueError(f"unknown reference kind: {reference_kind}")
    # 1-based ctrl 3 and 5 are u[2] and u[4] in the simulator.  They are
    # fixed at normalized 1.0; the policy receives u[0], u[1], u[3], u[5].
    return PneuEnv3(
        obs=sim,
        ref=ref,
        pred=None,
        num_prev=0,
        num_pred=1,
        num_act=1,
        verbose=False,
        action_low=0.0,
        action_high=1.0,
        episode_carry_state=False,
        rwd_kwargs=dict(
            pos_curr_rwd_coeff=0.0,
            neg_curr_rwd_coeff=0.0,
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
            action_delta_rwd_coeff=0.0,
            conflict_rwd_coeff=0.0,
            chamber_reserve_rwd_coeff=0.0,
            chamber_margin_kpa=15.0,
            chamber_deadband_kpa=5.0,
        ),
        fixed_action_mask=(False, False, True, False, True, False),
        fixed_action_values=(0.0, 0.0, 1.0, 0.0, 1.0, 0.0),
    )


def run_episode(env: PneuEnv3, steps: int) -> pd.DataFrame:
    state, _ = env.reset()
    del state
    rows = []
    for step in range(steps):
        # Deterministic neutral four-policy action.  This smoke test checks
        # plumbing and reference aggressiveness, not policy performance.
        action = np.full(env.action_space.shape, 0.5, dtype=np.float64)
        _, _, _, _, info = env.step(action)
        obs = info["obs"]
        applied = np.asarray(info["ctrl_input"], dtype=np.float64).reshape(-1, 6)[0]
        rows.append(
            dict(
                step=step,
                time=float(obs["curr_time"]),
                ref_pos=float(obs["ref_act_pos"]),
                ref_neg=float(obs["ref_act_neg"]),
                act_pos=float(obs["sen_act_pos"]),
                act_neg=float(obs["sen_act_neg"]),
                chamber_pos=float(obs["sen_pos"]),
                chamber_neg=float(obs["sen_neg"]),
                ctrl1=float(applied[0]),
                ctrl2=float(applied[1]),
                ctrl3=float(applied[2]),
                ctrl4=float(applied[3]),
                ctrl5=float(applied[4]),
                ctrl6=float(applied[5]),
            )
        )
    return pd.DataFrame(rows)


def rate(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values, dt)


def summarize(df: pd.DataFrame) -> dict[str, float | bool]:
    dt = float(np.median(np.diff(df["time"])))
    ref_pos_rate = rate(df["ref_pos"].to_numpy(), dt)
    ref_neg_rate = rate(df["ref_neg"].to_numpy(), dt)
    act_pos_rate = rate(df["act_pos"].to_numpy(), dt)
    act_neg_rate = rate(df["act_neg"].to_numpy(), dt)
    return {
        "steps": int(len(df)),
        "duration_sec": float(df["time"].iloc[-1]),
        "ref_pos_min": float(df["ref_pos"].min()),
        "ref_pos_max": float(df["ref_pos"].max()),
        "ref_neg_min": float(df["ref_neg"].min()),
        "ref_neg_max": float(df["ref_neg"].max()),
        "ref_pos_abs_rate_max_kpa_s": float(np.max(np.abs(ref_pos_rate))),
        "ref_neg_abs_rate_max_kpa_s": float(np.max(np.abs(ref_neg_rate))),
        "ref_pos_abs_rate_p95_kpa_s": float(np.percentile(np.abs(ref_pos_rate), 95)),
        "ref_neg_abs_rate_p95_kpa_s": float(np.percentile(np.abs(ref_neg_rate), 95)),
        "act_pos_abs_rate_max_kpa_s": float(np.max(np.abs(act_pos_rate))),
        "act_neg_abs_rate_max_kpa_s": float(np.max(np.abs(act_neg_rate))),
        "chamber_pos_min": float(df["chamber_pos"].min()),
        "chamber_pos_max": float(df["chamber_pos"].max()),
        "chamber_neg_min": float(df["chamber_neg"].min()),
        "chamber_neg_max": float(df["chamber_neg"].max()),
        "fixed_ctrl3_max_abs_error": float(np.max(np.abs(df["ctrl3"] - 1.0))),
        "fixed_ctrl5_max_abs_error": float(np.max(np.abs(df["ctrl5"] - 1.0))),
        "pid_enabled": False,
        "policy_action_dim": 4,
        "applied_action_dim": 6,
    }


def plot_episode(df: pd.DataFrame, summary: dict, path: Path, episode: int) -> None:
    time = df["time"].to_numpy()
    dt = float(np.median(np.diff(time)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, df["ref_pos"], "k", label="REF POS")
    axes[0].plot(time, df["act_pos"], "r", label="ACT POS")
    axes[0].plot(time, df["ref_neg"], "gray", label="REF NEG")
    axes[0].plot(time, df["act_neg"], "b", label="ACT NEG")
    axes[0].set_ylabel("Actuator pressure [kPa]")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, rate(df["ref_pos"].to_numpy(), dt), "r", label="dREF POS/dt")
    axes[1].plot(time, rate(df["act_pos"].to_numpy(), dt), "--", color="darkred", label="dACT POS/dt")
    axes[1].plot(time, rate(df["ref_neg"].to_numpy(), dt), "b", label="dREF NEG/dt")
    axes[1].plot(time, rate(df["act_neg"].to_numpy(), dt), "--", color="navy", label="dACT NEG/dt")
    axes[1].set_ylabel("Rate [kPa/s]")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for column, color, label in (
        ("ctrl1", "tab:purple", "u1 policy"),
        ("ctrl2", "tab:orange", "u2 policy"),
        ("ctrl3", "black", "u3 fixed"),
        ("ctrl4", "tab:green", "u4 policy"),
        ("ctrl5", "gray", "u5 fixed"),
        ("ctrl6", "tab:brown", "u6 policy"),
    ):
        axes[2].plot(time, df[column], color=color, label=label)
    axes[2].set_ylabel("Normalized control")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(
        f"lib3 masked RandomRef smoke episode {episode} | "
        f"ref rate max POS/NEG={summary['ref_pos_abs_rate_max_kpa_s']:.2f}/"
        f"{summary['ref_neg_abs_rate_max_kpa_s']:.2f} kPa/s | PID off"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--reference", choices=("legacy", "smoothed"), default="legacy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pneu_rl/exp/lib3_masked_randomref_smoke"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.reference)
    summaries = []
    try:
        for episode in range(1, args.episodes + 1):
            df = run_episode(env, args.steps)
            summary = summarize(df)
            summaries.append({"episode": episode, **summary})
            df.to_csv(args.output_dir / f"episode_{episode:02d}.csv", index=False)
            plot_episode(
                df,
                summary,
                args.output_dir / f"episode_{episode:02d}.png",
                episode,
            )
            print(
                f"episode={episode} "
                f"ref_range_pos={summary['ref_pos_min']:.2f}:{summary['ref_pos_max']:.2f} "
                f"neg={summary['ref_neg_min']:.2f}:{summary['ref_neg_max']:.2f} "
                f"rate_max={summary['ref_pos_abs_rate_max_kpa_s']:.2f}/"
                f"{summary['ref_neg_abs_rate_max_kpa_s']:.2f} kPa/s "
                f"fixed_err={summary['fixed_ctrl3_max_abs_error']:.3g}/"
                f"{summary['fixed_ctrl5_max_abs_error']:.3g}"
            )
    finally:
        env.close()

    report = {
        "reference": {
            "kind": "legacy_random" if args.reference == "legacy" else "rate_limited_random",
            "kwargs": REF_KWARGS if args.reference == "legacy" else SMOOTH_REF_KWARGS,
        },
        "simulator": {
            "library": "lib3",
            "scale_low": 0.8,
            "scale_high": 1.0,
            "freq_hz": 50.0,
            "delay_sec": 0.1,
        },
        "mask": {
            "policy_controls": [0, 1, 3, 5],
            "fixed_controls": {"2": 1.0, "4": 1.0},
            "policy_action_dim": 4,
            "applied_action_dim": 6,
        },
        "pid_enabled": False,
        "note": (
            "This smoke test reports reference rate and masked-control plumbing. "
            "It does not certify dynamic feasibility of the reference."
        ),
        "episodes": summaries,
    }
    with (args.output_dir / "summary.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved smoke report: {args.output_dir / 'summary.yaml'}")


if __name__ == "__main__":
    main()
