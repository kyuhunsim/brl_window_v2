#!/usr/bin/env python3
"""Phase-0 checks for the lib4 RL path.

This is deliberately an environment/reference plumbing test, not an RL
evaluation.  It proves that the exact lib4 simulator configuration intended
for new training can reset without a hidden command, preserve a carried plant
for a bounded number of SAC horizons, and expose a consistent 3D reference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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

from pneu_env.env4 import PneuEnv4
from pneu_env.sim4 import PneuSim
from pneu_ref.disp_ref import PressureDiffDisplacementRef
from pneu_ref.factory import make_nd_reference
from pneu_ref.random_ref import RandomRef


TUNED_DIR = (
    ROOT
    / "pneu_env"
    / "coeff_result"
    / "260624_11_21_35_tuner4_output48plus49_linit34807_leak_tuned"
)


class _TunerCfgLoader(yaml.SafeLoader):
    """Read the historical tuner cfg without executing arbitrary YAML tags."""


def _construct_python_tuple(loader: yaml.SafeLoader, node: Any) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


_TunerCfgLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)


def load_model_cfg(value: str) -> tuple[dict[str, Any], Path]:
    candidate = Path(value)
    if candidate.is_file():
        path = candidate
    else:
        path = ROOT / "pneu_rl" / "models" / value / "cfg.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def tuned_setup() -> dict[str, float | str]:
    params_path = TUNED_DIR / "params.yaml"
    cfg_path = TUNED_DIR / "cfg.yaml"
    with params_path.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    with cfg_path.open(encoding="utf-8") as handle:
        tuner_cfg = yaml.load(handle, Loader=_TunerCfgLoader)
    data_names = tuner_cfg.get("data_names") or []
    if not data_names:
        raise ValueError(f"No data_names in {cfg_path}")
    csv_path = ROOT / "pneu_env" / str(data_names[0])
    with csv_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        row = handle.readline().strip().split(",")
    first = dict(zip(header, row))
    return {
        "params_path": str(params_path),
        "state_csv": str(csv_path),
        "init_length": float(params["l_init_total"]) / 2.0,
        "init_pos_press": float(first["press_pos"]),
        "init_neg_press": float(first["press_neg"]),
        "init_act_pos_press": float(first["act_pos_press"]),
        "init_act_neg_press": float(first["act_neg_press"]),
        "pump_in_coeff": float(params["pump_in_coeff"]),
        "pump_out_coeff": float(params["pump_out_coeff"]),
        "leak_pos_atm": float(params["leak_pos_atm"]),
        "leak_neg_atm": float(params["leak_neg_atm"]),
        "leak_cross": float(params["leak_cross"]),
    }


def build_sim(cfg: dict[str, Any], source: str) -> tuple[PneuSim, dict[str, Any]]:
    sim_cfg = copy.deepcopy(cfg["obs"])
    sim_cfg.update(noise=False, noise_std=0.0)
    tuned = tuned_setup()
    if source == "current_tuned":
        for key in (
            "init_length",
            "init_pos_press",
            "init_neg_press",
            "init_act_pos_press",
            "init_act_neg_press",
        ):
            sim_cfg[key] = tuned[key]
    sim = PneuSim(**sim_cfg)
    sim.set_discharge_coeff(
        inlet_pump_coeff=float(tuned["pump_in_coeff"]),
        outlet_pump_coeff=float(tuned["pump_out_coeff"]),
    )
    sim.set_leak_coefficients(
        pos_atm=float(tuned["leak_pos_atm"]),
        neg_atm=float(tuned["leak_neg_atm"]),
        cross=float(tuned["leak_cross"]),
    )
    return sim, {"state_source": source, "sim_cfg": sim_cfg, "tuned": tuned}


def close(a: np.ndarray | float, b: np.ndarray | float, atol: float = 1e-10) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True))


def snapshot(env: PneuEnv4, sim: PneuSim) -> dict[str, Any]:
    return {
        "sim_time": float(sim.lib.get_time_c()),
        "env_time": float(env.t),
        "sim_observation": np.asarray(sim.obs, dtype=np.float64).copy(),
        "obs_history": np.asarray(env.obs_traj, dtype=np.float64).copy(),
        "reference_buffer": np.asarray(env.goal.buf, dtype=np.float64).copy(),
        "last_policy_action": np.asarray(env.last_policy_action, dtype=np.float64).copy(),
        "previous_applied_action": np.asarray(env.prev_action, dtype=np.float64).copy(),
        "carry_block_index": int(env.carry_block_index),
        "hard_reset_count": int(env.hard_reset_count),
    }


def compact(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cfg", default="0625_lib4_Ours")
    parser.add_argument(
        "--state-source",
        choices=("model", "current_tuned"),
        default="current_tuned",
        help="initial state for the upcoming lib4 path; model keeps a legacy cfg unchanged",
    )
    parser.add_argument("--carry-blocks", type=int, default=4)
    parser.add_argument("--steps-per-block", type=int, default=8)
    parser.add_argument(
        "--reference-profile",
        type=Path,
        default=None,
        help="optional trajectory_bank_nd profile; omit to inspect the legacy random reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.carry_blocks < 1 or args.steps_per_block < 1:
        parser.error("--carry-blocks and --steps-per-block must be positive")

    cfg, cfg_path = load_model_cfg(args.model_cfg)
    sim, sim_record = build_sim(cfg, args.state_source)
    if args.reference_profile is None:
        reference = PressureDiffDisplacementRef(
            RandomRef(**cfg["rnd_ref"]),
            **cfg["pressure_diff_disp_ref"],
        )
        reference_record: dict[str, Any] = {
            "kind": "legacy_random_pressure_diff_displacement",
            "random_kwargs": cfg["rnd_ref"],
            "pressure_to_displacement": cfg["pressure_diff_disp_ref"],
        }
    else:
        with args.reference_profile.open(encoding="utf-8") as handle:
            profile = yaml.safe_load(handle)
        reference = make_nd_reference(profile)
        reference_record = {
            "kind": str(profile.get("kind")),
            "profile": str(args.reference_profile),
            "design": profile.get("design", {}),
        }
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg.update(
        episode_carry_state=True,
        episode_carry_blocks=args.carry_blocks,
        verbose=False,
    )
    env = PneuEnv4(obs=sim, ref=reference, pred=None, **env_cfg)
    action = np.linspace(0.15, 0.85, env.action_space.shape[0], dtype=np.float64)
    expected_physical = sim.scale_low + (sim.scale_high - sim.scale_low) * action[:6]
    initial_time = float(sim.lib.get_time_c())
    try:
        initial_state, _ = env.reset()
        after_initial_reset_time = float(sim.lib.get_time_c())
        action_mapping_ok = False
        boundaries: list[dict[str, Any]] = []
        for block in range(1, args.carry_blocks + 1):
            for _ in range(args.steps_per_block):
                state, _, _, _, info = env.step(action)
            if block == 1:
                applied = np.asarray(
                    [
                        info["obs"]["ctrl_pos"],
                        info["obs"]["ctrl_neg"],
                        info["obs"]["ctrl_act_pos_in"],
                        info["obs"]["ctrl_act_pos_out"],
                        info["obs"]["ctrl_act_neg_in"],
                        info["obs"]["ctrl_act_neg_out"],
                    ],
                    dtype=np.float64,
                )
                action_mapping_ok = close(applied, expected_physical)
            before = snapshot(env, sim)
            reset_state, _ = env.reset()
            after = snapshot(env, sim)
            carried = block < args.carry_blocks
            checks = {"expected_carry": carried, "reset_state_finite": bool(np.all(np.isfinite(reset_state)))}
            if carried:
                checks.update(
                    sim_time_unchanged=close(before["sim_time"], after["sim_time"]),
                    env_time_unchanged=close(before["env_time"], after["env_time"]),
                    sim_observation_unchanged=close(before["sim_observation"], after["sim_observation"]),
                    obs_history_unchanged=close(before["obs_history"], after["obs_history"]),
                    reference_buffer_unchanged=close(before["reference_buffer"], after["reference_buffer"]),
                    last_policy_action_preserved=close(before["last_policy_action"], after["last_policy_action"]),
                )
            else:
                checks["hard_reset_happened"] = after["hard_reset_count"] == before["hard_reset_count"] + 1
                # The C++ getter returns the last emitted observation, which
                # is stale until the first post-reset plant step.  Verify the
                # clock through that first actual simulator output instead.
                post_state, _, _, _, post_info = env.step(action)
                checks["first_physical_step_restarts_at_dt"] = close(
                    float(post_info["obs"]["curr_time"]), 1.0 / float(sim.freq), 1e-9
                )
                checks["post_hard_reset_state_finite"] = bool(np.all(np.isfinite(post_state)))
            boundaries.append({"block": block, "before": before, "after": after, "checks": checks})

        report = {
            "phase": "phase0_lib4_environment_and_carry_preflight",
            "model_cfg": str(cfg_path),
            "model_cfg_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
            "simulator": sim_record,
            "reference": reference_record,
            "dimensions": {
                "policy_observation": int(env.observation_space.shape[0]),
                "policy_action": int(env.action_space.shape[0]),
                "full_applied_action": int(env.dim_act),
                "reference": int(env.dim_ref),
            },
            "carry": {
                "blocks_before_hard_reset": args.carry_blocks,
                "steps_per_checked_block": args.steps_per_block,
                "initial_reset_did_not_advance_simulator": close(initial_time, after_initial_reset_time),
                "physical_action_mapping_ok": action_mapping_ok,
                "boundaries": boundaries,
            },
        }
        checks = [
            report["carry"]["initial_reset_did_not_advance_simulator"],
            report["carry"]["physical_action_mapping_ok"],
            *[
                bool(
                    all(
                        value
                        for key, value in item["checks"].items()
                        if key != "expected_carry"
                    )
                )
                for item in boundaries
            ],
        ]
        report["all_checks_pass"] = bool(all(checks))
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(compact(report), handle, sort_keys=False)
    print(json.dumps({"output": str(args.output), "all_checks_pass": report["all_checks_pass"]}, indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
