#!/usr/bin/env python3
"""Fast, non-training checks for the six-cell lib3 overnight matrix."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import yaml

from pneu_rl.sac_with_loss import SAC

from train_screening import build_config, make_components, seed_everything


MATRIX = (
    (1, "history_sac", "pressure_history_sac", 96),
    (2, "history_sac+caps", "pressure_history_caps", 96),
    (3, "mpobs1_sac", "mpobs1_sac", 100),
    (4, "sac+mpobs1+caps", "mpobs1_caps", 100),
    (5, "history_action_sac", "pressure_action_history_sac", 156),
    (6, "history_action_sac+caps", "pressure_action_history_caps", 156),
)


def make_cfg(variant: str, seed: int, ref_seed: int) -> dict[str, Any]:
    cfg = build_config(
        lib=3,
        variant=variant,
        seed=seed,
        ref_seed=ref_seed,
        reference_kind="rate_limited_random",
        carry=True,
        carry_reset_blocks=4,
        scale_low=0.8,
        caps_temporal_weight=0.5,
        caps_spatial_weight=0.1,
        chamber_reserve_rwd_coeff=0.05,
        chamber_bound_rwd_coeff=0.10,
        normalize_observation=True,
        caps_mode="mean_action",
        caps_spatial_noise_normalized=0.01,
        warmup_physical_action=0.95,
        warmup_mode="pink",
        warmup_noise_std=0.20,
        warmup_noise_seed=seed + 200_003,
        update_during_warmup=False,
        reset_after_warmup=True,
        action_history_len=10,
    )
    cfg["model"].update(
        horizon=512,
        start_epi=10,
        gamma=0.99,
        caps_start_steps=5_120,
        caps_ramp_steps=51_200,
        exploration_noise="none",
        exploration_noise_std=0.0,
        exploration_noise_octaves=8,
        exploration_noise_seed=seed + 100_003,
        exploration_use_policy_mean=False,
    )
    return cfg


def check_reset_schedule(env: Any) -> dict[str, Any]:
    """Check four carried blocks followed by one hard reset."""
    state, _ = env.reset()
    del state
    action = np.full(env.action_space.shape, 0.75, dtype=np.float64)
    records = []
    for logical_block in range(1, 6):
        env.step(action)
        time_before_reset = float(env.t)
        hard_before = int(env.hard_reset_count)
        env.reset()
        records.append(
            {
                "completed_block": logical_block,
                "time_before_reset": time_before_reset,
                "time_after_reset": float(env.t),
                "hard_reset": int(env.hard_reset_count) > hard_before,
                "next_block_index": int(env.carry_block_index),
            }
        )
    expected = [False, False, False, True, False]
    observed = [row["hard_reset"] for row in records]
    if observed != expected:
        raise AssertionError(
            f"unexpected four-block carry reset schedule: {observed} != {expected}"
        )
    hard_before = int(env.hard_reset_count)
    env.request_hard_reset()
    env.reset()
    explicit_reset_ok = int(env.hard_reset_count) == hard_before + 1 and float(env.t) == 0.0
    if not explicit_reset_ok:
        raise AssertionError("explicit post-warmup hard reset request was ignored")
    return {
        "pass": True,
        "records": records,
        "explicit_post_warmup_reset": explicit_reset_ok,
    }


def check_variant(
    experiment_id: int,
    variant: str,
    label: str,
    expected_state_dim: int,
    seed: int,
    ref_seed: int,
) -> tuple[dict[str, Any], Any]:
    seed_everything(seed)
    cfg = make_cfg(variant, seed, ref_seed)
    env = make_components(cfg)
    model = SAC(env=env, **cfg["model"])
    state, _ = env.reset()

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    if state_dim != expected_state_dim:
        raise AssertionError(
            f"{variant}: state_dim={state_dim}, expected={expected_state_dim}"
        )
    if action_dim != 6 or int(env.dim_act) != 6 or int(env.num_act) != 1:
        raise AssertionError(
            f"{variant}: expected one 6-valve command, got "
            f"policy={action_dim}, full={env.dim_act}, num_act={env.num_act}"
        )
    if np.asarray(env.policy_state_scale).shape != (state_dim,):
        raise AssertionError(f"{variant}: invalid policy_state_scale")

    deterministic_initial_action = model.evaluate_action(state)
    action_history_mapping_ok = True
    if env.action_history_len > 0:
        test_action = np.linspace(0.50, 1.00, 6, dtype=np.float64)
        env.step(test_action)
        action_history_mapping_ok = bool(
            np.allclose(env.action_history[-1], test_action, atol=1e-12)
        )
        if not action_history_mapping_ok:
            raise AssertionError(f"{variant}: applied action was not appended to history")
    warmup = np.stack([model._warmup_action()[0] for _ in range(256)], axis=0)
    physical_warmup = 0.8 + 0.2 * warmup
    result = {
        "id": experiment_id,
        "label": label,
        "variant": variant,
        "state_dim": state_dim,
        "policy_action_dim": action_dim,
        "num_act": int(env.num_act),
        "num_prev": int(env.num_prev),
        "num_pred": int(env.num_pred),
        "predictor": bool(env.pred is not None),
        "action_history_len": int(env.action_history_len),
        "action_history_mapping_ok": action_history_mapping_ok,
        "initial_alpha": float(model.alpha.detach().cpu().item()),
        "initial_policy_mean_raw": deterministic_initial_action.tolist(),
        "initial_policy_mean_physical": (0.8 + 0.2 * deterministic_initial_action).tolist(),
        "warmup_raw_min": float(warmup.min()),
        "warmup_raw_max": float(warmup.max()),
        "warmup_raw_std": float(warmup.std()),
        "warmup_physical_min": float(physical_warmup.min()),
        "warmup_physical_max": float(physical_warmup.max()),
        "warmup_physical_std": float(physical_warmup.std()),
        "update_during_warmup": bool(model.update_during_warmup),
        "reset_after_warmup": bool(model.reset_after_warmup),
        "carry_reset_blocks": int(env.episode_carry_blocks),
        "caps_mode": model.caps_mode,
        "caps_weights": [float(model.temporal_weight), float(model.spatial_weight)],
    }
    return result, env


def check_warmup_rollout(seed: int, ref_seed: int) -> dict[str, Any]:
    """Replay the exact ten-episode randomized warm-up without any updates."""
    seed_everything(seed)
    cfg = make_cfg("history_sac", seed, ref_seed)
    env = make_components(cfg)
    model = SAC(env=env, **cfg["model"])
    pressures = []
    controls = []
    references = []
    try:
        for _ in range(10):
            env.reset()
            for _ in range(512):
                action, _ = model._warmup_action()
                _, _, _, _, info = env.step(action)
                obs = info["obs"]
                pressures.append(
                    [
                        obs["sen_pos"],
                        obs["sen_neg"],
                        obs["sen_act_pos"],
                        obs["sen_act_neg"],
                    ]
                )
                controls.append(
                    [
                        obs["ctrl_pos"],
                        obs["ctrl_neg"],
                        obs["ctrl_act_pos_in"],
                        obs["ctrl_act_pos_out"],
                        obs["ctrl_act_neg_in"],
                        obs["ctrl_act_neg_out"],
                    ]
                )
                references.append([obs["ref_act_pos"], obs["ref_act_neg"]])
        hard_before = int(env.hard_reset_count)
        env.request_hard_reset()
        env.reset()
        post_reset_pressure = np.asarray(env.curr_full_press, dtype=np.float64)
        post_reset_ok = (
            int(env.hard_reset_count) == hard_before + 1
            and np.allclose(post_reset_pressure, 101.325, atol=1e-9)
        )
    finally:
        env.close()

    pressure = np.asarray(pressures, dtype=np.float64)
    control = np.asarray(controls, dtype=np.float64)
    reference = np.asarray(references, dtype=np.float64)
    finite = bool(
        np.isfinite(pressure).all()
        and np.isfinite(control).all()
        and np.isfinite(reference).all()
    )
    report = {
        "steps": int(len(pressure)),
        "updates_performed": 0,
        "finite": finite,
        "chamber_pos_range_kpa": [float(pressure[:, 0].min()), float(pressure[:, 0].max())],
        "chamber_neg_range_kpa": [float(pressure[:, 1].min()), float(pressure[:, 1].max())],
        "actuator_pos_range_kpa": [float(pressure[:, 2].min()), float(pressure[:, 2].max())],
        "actuator_neg_range_kpa": [float(pressure[:, 3].min()), float(pressure[:, 3].max())],
        "physical_control_range": [float(control.min()), float(control.max())],
        "physical_control_std": float(control.std()),
        "reference_pos_range_kpa": [float(reference[:, 0].min()), float(reference[:, 0].max())],
        "reference_neg_range_kpa": [float(reference[:, 1].min()), float(reference[:, 1].max())],
        "post_warmup_atmosphere_reset": bool(post_reset_ok),
    }
    report["pass"] = bool(finite and post_reset_ok)
    if not report["pass"]:
        raise AssertionError(f"warm-up rollout preflight failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref-seed", type=int, default=61098)
    args = parser.parse_args()

    rows = []
    reset_report = None
    for experiment in MATRIX:
        row, env = check_variant(*experiment, seed=args.seed, ref_seed=args.ref_seed)
        try:
            if reset_report is None:
                reset_report = check_reset_schedule(env)
            rows.append(row)
        finally:
            env.close()

    warmup_rollout = check_warmup_rollout(args.seed, args.ref_seed)

    report = {
        "pass": True,
        "note": "No training was run; this checks configuration and simulator plumbing only.",
        "common": {
            "reference": "rate_limited_random",
            "ref_seed": args.ref_seed,
            "horizon": 512,
            "carry_reset_blocks": 4,
            "physical_rollout_steps": 2048,
            "warmup": "pink around physical 0.95 for 10 episodes",
            "post_warmup_external_noise": "none (SAC sampling remains active)",
        },
        "reset_schedule": reset_report,
        "warmup_rollout": warmup_rollout,
        "experiments": rows,
    }
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
