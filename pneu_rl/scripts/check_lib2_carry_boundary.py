#!/usr/bin/env python3
"""Assert that a lib2 carry reset is a non-physical episode boundary.

This is intentionally an environment-level check rather than a learning
test.  It proves that a trainer can split a continuous physical rollout into
512-step SAC horizons without rewinding simulator time, pressure history,
reference time, or the most recent command at the boundary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for source_dir in (ROOT / "pneu_env" / "src", ROOT / "pneu_ref" / "src"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pneu_env.env import PneuEnv
from pneu_env.sim2 import PneuSim
from pneu_ref.factory import make_pressure_reference


def close(a: np.ndarray | float, b: np.ndarray | float, tolerance: float = 1e-10) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=tolerance, equal_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--scale-low", type=float, default=0.85)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    with args.reference_profile.open(encoding="utf-8") as handle:
        reference_spec = yaml.safe_load(handle)
    sim = PneuSim(
        freq=args.freq,
        delay=args.delay,
        init_pos_press=101.325,
        init_neg_press=101.325,
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=args.scale_low,
        scale_high=1.0,
    )
    env = PneuEnv(
        obs=sim,
        ref=make_pressure_reference(reference_spec),
        pred=None,
        num_prev=10,
        num_pred=10,
        num_act=1,
        rwd_kwargs={
            "pos_prev_rwd_coeff": 0.0,
            "neg_prev_rwd_coeff": 0.0,
            "pos_curr_rwd_coeff": 0.3,
            "neg_curr_rwd_coeff": 0.3,
            "pos_fut_rwd_coeff": 0.0,
            "neg_fut_rwd_coeff": 0.0,
            "pos_pred_rwd_coeff": 0.0,
            "neg_pred_rwd_coeff": 0.0,
        },
        action_low=0.0,
        action_high=1.0,
        verbose=False,
        episode_carry_state=True,
        reference_reset_to_observation=False,
    )
    env.reset()
    # This action is deliberately non-trivial; it must be preserved by the
    # carry reset, not replaced by an action-space lower bound/reset command.
    action = np.array([0.37, 0.73], dtype=np.float64)
    for _ in range(args.steps):
        env.step(action)

    before = {
        "simulator_time_sec": float(sim.lib.get_time()),
        "environment_time_sec": float(env.t),
        "pressure_kpa": np.asarray(sim.obs, dtype=np.float64).copy(),
        "observation_history": np.asarray(env.obs_traj, dtype=np.float64).copy(),
        "reference_buffer": np.asarray(env.goal.buf, dtype=np.float64).copy(),
        "last_action_normalized": np.asarray(env.last_action, dtype=np.float64).copy(),
    }
    state_after_reset, _ = env.reset()
    after = {
        "simulator_time_sec": float(sim.lib.get_time()),
        "environment_time_sec": float(env.t),
        "pressure_kpa": np.asarray(sim.obs, dtype=np.float64).copy(),
        "observation_history": np.asarray(env.obs_traj, dtype=np.float64).copy(),
        "reference_buffer": np.asarray(env.goal.buf, dtype=np.float64).copy(),
        "last_action_normalized": np.asarray(env.last_action, dtype=np.float64).copy(),
    }
    next_state, _, _, _, _ = env.step(action)
    next_simulator_time = float(sim.lib.get_time())

    checks = {
        "simulator_time_unchanged_at_reset": close(before["simulator_time_sec"], after["simulator_time_sec"]),
        "environment_time_unchanged_at_reset": close(before["environment_time_sec"], after["environment_time_sec"]),
        "pressure_unchanged_at_reset": close(before["pressure_kpa"], after["pressure_kpa"]),
        "observation_history_unchanged_at_reset": close(before["observation_history"], after["observation_history"]),
        "reference_buffer_unchanged_at_reset": close(before["reference_buffer"], after["reference_buffer"]),
        "previous_action_preserved_at_reset": close(before["last_action_normalized"], after["last_action_normalized"]),
        "state_rebuilt_without_advance": close(state_after_reset[-len(before["reference_buffer"]):], before["reference_buffer"]),
        "one_physical_step_after_boundary": close(next_simulator_time - before["simulator_time_sec"], 1.0 / args.freq, 1e-9),
        "next_state_finite": bool(np.all(np.isfinite(next_state))),
    }
    report = {
        "reference_profile": str(args.reference_profile),
        "steps_before_boundary": int(args.steps),
        "frequency_hz": float(args.freq),
        "delay_sec": float(args.delay),
        "plant_initial_pressure_kpa": [101.325, 101.325],
        "command_used_normalized": action.tolist(),
        "before_boundary": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in before.items()},
        "after_boundary_reset": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in after.items()},
        "next_simulator_time_sec": next_simulator_time,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    env.close()
    print(f"[ INFO] carry boundary report: {args.output}")
    print(f"[ INFO] all checks pass: {report['all_checks_pass']}")
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
