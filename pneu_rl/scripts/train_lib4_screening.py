#!/usr/bin/env python3
"""Run the minimal lib4 Direct→History→rate/reserve→K=2 matrix.

This runner intentionally does not reuse ``train4.py``'s interactive random
reference/PID/MPOBS/CAPS presets.  Every variant below receives the same
displacement-inclusive oracle-feasible profile, tuned simulator, reward,
physical horizon, and bounded carry schedule.  Only the named ablation
component changes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (
    ROOT / "pneu_env" / "src",
    ROOT / "pneu_ref" / "src",
    ROOT / "pneu_rl" / "src",
    ROOT / "pneu_utils" / "src",
    SCRIPT_DIR,
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from preflight_lib4 import build_sim, load_model_cfg
from pneu_env.env4 import PneuEnv4
from pneu_ref.factory import make_nd_reference
from pneu_rl.sac_with_loss import SAC


VARIANTS = (
    "sac",
    "history_sac",
    "history_rate",
    "history_reserve",
    "history_rate_reserve",
    # Short names used in the structured-history experiment manifest.
    "hrr",
    "pide",
    "gpide",
    "history_edot",
    "history_rate_reserve_edot",
    "history_rate_reserve_k2",
    "history_rate_reserve_integral",
    "history_rate_reserve_smooth",
    "history_rate_reserve_k2_hold",
    "history_rate_reserve_k2_consistency",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_profile(path: Path) -> tuple[dict[str, Any], Path]:
    with path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if profile.get("kind") != "trajectory_bank_nd":
        raise ValueError("lib4 screening requires a trajectory_bank_nd profile")
    copied = copy.deepcopy(profile)
    npz_value = Path(copied["kwargs"]["trajectory_npz"])
    if not npz_value.is_absolute():
        copied["kwargs"]["trajectory_npz"] = str(ROOT / npz_value)
    return copied, path


def variant_flags(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown lib4 variant: {variant}")
    history = variant != "sac"
    physics_mode = {
        "sac": "none",
        "history_sac": "none",
        "history_rate": "rate",
        "history_reserve": "reserve",
        "history_rate_reserve": "rate_reserve",
        "hrr": "rate_reserve",
        "pide": "pide",
        "gpide": "gpide",
        "history_edot": "edot",
        "history_rate_reserve_edot": "rate_reserve_edot",
        "history_rate_reserve_k2": "rate_reserve",
        "history_rate_reserve_integral": "rate_reserve",
        "history_rate_reserve_smooth": "rate_reserve",
        "history_rate_reserve_k2_hold": "rate_reserve",
        "history_rate_reserve_k2_consistency": "rate_reserve",
    }[variant]
    chunk = 2 if "_k2" in variant else 1
    chunk_mode = "hold" if variant.endswith("_k2_hold") else "free"
    return dict(
        history=history,
        physics=physics_mode != "none",
        physics_mode=physics_mode,
        integral=variant.endswith("_integral"),
        chunk=chunk,
        chunk_mode=chunk_mode,
        smooth=variant.endswith("_smooth"),
        chunk_consistency=variant.endswith("_k2_consistency"),
    )


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    flags = variant_flags(args.variant)
    profile, profile_path = resolve_profile(args.reference_profile)
    source_cfg, source_cfg_path = load_model_cfg(args.model_cfg)
    if args.physical_horizon % flags["chunk"] != 0:
        raise ValueError("physical horizon must be divisible by the action chunk size")
    policy_horizon = args.physical_horizon // flags["chunk"]
    env_cfg = dict(
        num_prev=(args.history_len if flags["history"] else 0),
        num_pred=(args.reference_lookahead if flags["history"] else 1),
        num_act=1,
        rwd_kwargs=dict(
            pos_prev_rwd_coeff=0.0,
            neg_prev_rwd_coeff=0.0,
            pos_curr_rwd_coeff=args.pos_reward,
            neg_curr_rwd_coeff=args.neg_reward,
            pos_fut_rwd_coeff=0.0,
            neg_fut_rwd_coeff=0.0,
            pos_pred_rwd_coeff=0.0,
            neg_pred_rwd_coeff=0.0,
            diff_curr_rwd_coeff=args.diff_reward,
            diff_fut_rwd_coeff=0.0,
            diff_pred_rwd_coeff=0.0,
            disp_prev_rwd_coeff=0.0,
            disp_curr_rwd_coeff=args.disp_reward,
            disp_fut_rwd_coeff=0.0,
            disp_pred_rwd_coeff=0.0,
            disp_vel_rwd_coeff=0.0,
            action_delta_rwd_coeff=(
                args.action_delta_reward if flags["smooth"] else 0.0
            ),
        ),
        pos_pred_rnd_offset_range=0.0,
        neg_pred_rnd_offset_range=0.0,
        verbose=False,
        action_low=0.0,
        action_high=1.0,
        episode_carry_state=True,
        episode_carry_blocks=args.carry_blocks,
        action_history_len=(args.history_len if flags["history"] else 0),
        include_physics_features=flags["physics"],
        physics_feature_mode=flags["physics_mode"],
        include_integral_error=flags["integral"],
        integral_error_decay=args.integral_decay,
        integral_error_limit=args.integral_limit,
        integral_error_scale=args.integral_scale,
        action_chunk_steps=flags["chunk"],
        action_chunk_gamma=args.gamma,
        action_chunk_mode=flags["chunk_mode"],
        fixed_chamber_ctrl=None,
        steady_chamber_ctrl=None,
        displacement_obs_key="sen_total_contraction_from_rest",
        displacement_scale=1000.0,
        velocity_scale=1000.0,
        include_displacement_velocity_obs=True,
        include_prev_action_obs=False,
        include_pressure_diff_velocity_obs=False,
    )
    model_cfg = dict(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        automatic_entropy_tunning=True,
        target_entropy=args.target_entropy,
        hidden_dim=args.hidden_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epoch=1,
        horizon=policy_horizon,
        start_epi=args.start_epi,
        max_grad_norm=0.5,
        log_std_min=-10,
        log_std_max=1,
        temporal_weight=(args.temporal_weight if flags["smooth"] else 0.0),
        spatial_weight=0.0,
        noise_std=0.0,
        caps_mode="mean_action",
        caps_start_steps=0,
        caps_ramp_steps=0,
        normalize_observation=True,
        train_diag_interval=1,
        train_diag_atm_band=8.0,
        train_verbose_interval=args.verbose_every,
        exploration_noise="none",
        exploration_noise_std=0.0,
        exploration_noise_seed=args.seed + 100_003,
        warmup_action_raw=0.75,
        warmup_mode="fixed",
        update_during_warmup=False,
        reset_after_warmup=True,
        update_every=1,
        chunk_consistency_weight=(
            args.chunk_consistency_weight if flags["chunk_consistency"] else 0.0
        ),
    )
    return dict(
        schema="lib4_screening_v2",
        lib=4,
        variant=args.variant,
        train_seed=args.seed,
        physical_horizon=args.physical_horizon,
        policy_horizon=policy_horizon,
        action_chunk_steps=flags["chunk"],
        action_chunk_mode=flags["chunk_mode"],
        episodes=args.episodes,
        model_cfg_source=str(source_cfg_path),
        model_cfg_sha256=hashlib.sha256(source_cfg_path.read_bytes()).hexdigest(),
        simulator_state_source=args.state_source,
        source_model_config=source_cfg,
        reference=profile,
        reference_profile_source=str(profile_path),
        reference_profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        reference_artifact_sha256=hashlib.sha256(
            Path(profile["kwargs"]["trajectory_npz"]).read_bytes()
        ).hexdigest(),
        structured_history=dict(
            name=(
                "HRR" if args.variant in {"hrr", "history_rate_reserve"}
                else args.variant.upper()
            ),
            physics_feature_mode=flags["physics_mode"],
            source="deterministic feature map from observation/action history",
            reward_excluded=True,
        ),
        env=env_cfg,
        model=model_cfg,
    )


def make_components(cfg: dict[str, Any]) -> PneuEnv4:
    sim, _ = build_sim(cfg["source_model_config"], cfg["simulator_state_source"])
    reference = make_nd_reference(cfg["reference"])
    env = PneuEnv4(obs=sim, ref=reference, pred=None, **cfg["env"])
    env.action_space.seed(int(cfg["train_seed"]))
    return env


def save_config(model_name: str, cfg: dict[str, Any]) -> Path:
    path = ROOT / "pneu_rl" / "models" / model_name / "cfg.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


def dry_run(env: PneuEnv4, cfg: dict[str, Any]) -> dict[str, Any]:
    state, _ = env.reset()
    if state.shape != env.observation_space.shape:
        raise AssertionError(f"reset state {state.shape} != {env.observation_space.shape}")
    action = np.full(env.action_space.shape, 0.5, dtype=np.float64)
    next_state, reward, _, _, info = env.step(action)
    if not np.all(np.isfinite(next_state)) or not np.isfinite(reward):
        raise AssertionError("non-finite state or reward in screening dry run")
    obs_traj, _, ref_traj = env._split_state(next_state)
    expected_disp_reward = -float(cfg["env"]["rwd_kwargs"]["disp_curr_rwd_coeff"]) * abs(
        float(ref_traj[env.num_obs - 1, 2] - obs_traj[-1, env.disp_idx])
    )
    actual_disp_reward = float(info["reward"]["disp_curr_reward"])
    if not np.isclose(actual_disp_reward, expected_disp_reward, atol=1e-12):
        raise AssertionError(
            "displacement reward is not aligned with the current 3D reference: "
            f"{actual_disp_reward} != {expected_disp_reward}"
        )
    state_scale = np.asarray(env.policy_state_scale, dtype=np.float64)
    if state_scale.shape != env.observation_space.shape:
        raise AssertionError("policy_state_scale does not match observation shape")
    if np.any(~np.isfinite(state_scale)) or np.any(state_scale <= 0.0):
        raise AssertionError("policy_state_scale must be finite and positive")
    model = SAC(env=env, **cfg["model"])
    policy_action = model.evaluate_action(next_state)
    if policy_action.shape != env.action_space.shape or not np.all(np.isfinite(policy_action)):
        raise AssertionError("normalized SAC policy/action construction failed")
    value_estimate = model.estimate_value(next_state, policy_action)
    if not np.isfinite(value_estimate):
        raise AssertionError("deterministic critic value construction failed")
    diagnostics = model._episode_diagnostics(epi=0, reward=float(reward), infos=[info])
    required_displacement_diagnostics = {
        "disp_rmse",
        "disp_bias",
        "disp_span",
        "disp_ref_span",
        "disp_direction_agreement",
        "disp_tracking_reward_sum",
    }
    if not required_displacement_diagnostics.issubset(diagnostics):
        raise AssertionError("lib4 displacement learning diagnostics are incomplete")
    warmup_action, _ = model._warmup_action()
    warmup_physical = np.stack(
        [
            env._raw_to_physical_action(row)
            for row in env.decode_action_chunk(warmup_action)
        ],
        axis=0,
    )
    if not np.allclose(warmup_action, 0.75) or not np.allclose(warmup_physical, 0.95):
        raise AssertionError("lib3-matched raw 0.75 / physical 0.95 warm-up mapping failed")
    physical_steps = int(info.get("chunk_steps", 1))
    before_time = float(env.t)
    hard_before = int(env.hard_reset_count)
    state_after_carry, _ = env.reset()
    if not np.all(np.isfinite(state_after_carry)) or not np.isclose(env.t, before_time):
        raise AssertionError("carry boundary advanced or corrupted the state")
    hard_reset_pattern = [int(env.hard_reset_count) > hard_before]
    for _ in range(3):
        env.step(action)
        hard_before = int(env.hard_reset_count)
        env.reset()
        hard_reset_pattern.append(int(env.hard_reset_count) > hard_before)
    if hard_reset_pattern != [False, False, False, True]:
        raise AssertionError(
            "four-block carry schedule mismatch: "
            f"{hard_reset_pattern} != {[False, False, False, True]}"
        )
    return dict(
        dry_run=True,
        variant=cfg["variant"],
        observation_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        full_valve_action_dim=int(env.dim_act),
        physical_steps_per_policy_step=physical_steps,
        action_chunk_mode=env.action_chunk_mode,
        policy_horizon=cfg["policy_horizon"],
        physical_horizon=cfg["physical_horizon"],
        carry_blocks=cfg["env"]["episode_carry_blocks"],
        normalize_observation=cfg["model"]["normalize_observation"],
        warmup_raw_to_physical="0.75 -> 0.95",
        hard_reset_after_blocks=4,
        displacement_reward_wiring="pass",
        displacement_diagnostics="pass",
        value_trace="pass",
        reward=float(reward),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--model-cfg", default="0625_lib4_Ours")
    parser.add_argument("--state-source", choices=("model", "current_tuned"), default="current_tuned")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--physical-horizon", type=int, default=512)
    parser.add_argument("--carry-blocks", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--reference-lookahead", type=int, default=15)
    parser.add_argument("--pos-reward", type=float, default=0.3)
    parser.add_argument("--neg-reward", type=float, default=0.3)
    parser.add_argument("--disp-reward", type=float, default=0.2)
    parser.add_argument("--diff-reward", type=float, default=0.0)
    parser.add_argument("--action-delta-reward", type=float, default=0.05)
    parser.add_argument("--temporal-weight", type=float, default=0.0)
    parser.add_argument("--chunk-consistency-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="override SAC target entropy; leave unset for -action_dim",
    )
    parser.add_argument("--integral-decay", type=float, default=1.0)
    parser.add_argument("--integral-limit", type=float, default=100.0)
    parser.add_argument("--integral-scale", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-epi", type=int, default=10)
    parser.add_argument("--verbose-every", type=int, default=25)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.physical_horizon < 2 or args.carry_blocks < 1:
        parser.error("physical horizon must be >= 2 and carry blocks must be positive")

    seed_everything(args.seed)
    cfg = build_config(args)
    env = make_components(cfg)
    try:
        if args.dry_run:
            print(yaml.safe_dump(dry_run(env, cfg), sort_keys=False))
            return
        name = args.name or f"lib4_{args.variant}_seed{args.seed}"
        model = SAC(env=env, **cfg["model"])
        model.set_logger(name)
        config_path = save_config(name, cfg)
        print(json.dumps({"model": name, "config": str(config_path), "variant": args.variant, "physical_horizon": cfg["physical_horizon"], "policy_horizon": cfg["policy_horizon"]}, indent=2))
        model.train(episode=args.episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()
