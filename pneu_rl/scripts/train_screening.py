"""Run the lib2/lib3 SAC screening matrix.

The runner intentionally keeps the first comparison small and explicit:

    sac
    sac+caps
    sac+mpobs+caps
    sac+mpobs+caps+iterm
    sac+mpobs1+caps
    sac+mpobs1+caps+iterm
    history_sac
    mpobs1_sac

All runs use the same reference seed, initial state, reward, action scaling,
episode reset policy, and SAC hyperparameters within a library.  Only the
requested architecture components and training seed are changed.

Example:
    python pneu_rl/scripts/train_screening.py \
        --lib 3 --variant sac --seed 0 \
        --name 260722_screen_lib3_sac_seed0
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from pneu_rl.sac_with_loss import SAC
from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_utils.utils import save_yaml


VARIANTS = (
    "sac",
    "sac+iterm",
    "sac+caps",
    "sac+mpobs+caps",
    "sac+mpobs+caps+iterm",
    "sac+mpobs1+caps",
    "sac+mpobs1+caps+iterm",
    "history_sac",
    "history_sac+caps",
    "history_action_sac",
    "history_action_sac+caps",
    "history_action_sac+mpobs1",
    "history_action_sac+mpobs1+caps",
    "history_action_rate",
    "history_action_rate_k2",
    "history_action_rate_k2_caps",
    "history_action_rate_k2_mpobs1",
    "history_action_rate_k2_mpobs1_caps",
    "mpobs1_sac",
)


def seed_everything(seed: int) -> None:
    """Seed training randomness without touching the reference RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reward_kwargs(
    lib: int,
    tracking_loss_mode: str = "abs",
    rate_rwd_coeff: float = 0.0,
    rate_rwd_scale_pos: float = 12.0,
    rate_rwd_scale_neg: float = 6.0,
    action_delta_rwd_coeff: float = 0.0,
    second_action_delta_rwd_coeff: float = 0.0,
    conflict_rwd_coeff: float = 0.0,
    chamber_reserve_rwd_coeff: float = 0.0,
    chamber_bound_rwd_coeff: float = 0.0,
    future_rwd_coeff: float = 0.0,
    tracking_channels: str = "both",
) -> dict[str, Any]:
    # The first screening uses only current actuator-pressure tracking.
    # Future/prediction, action-delta, conflict, and chamber-reserve terms are
    # all disabled so that CAPS is not accidentally mixed into the reward.
    if tracking_channels not in ("both", "pos", "neg"):
        raise ValueError(f"unknown tracking_channels: {tracking_channels}")
    track_pos = tracking_channels in ("both", "pos")
    track_neg = tracking_channels in ("both", "neg")
    values = dict(
        pos_prev_rwd_coeff=0.0,
        neg_prev_rwd_coeff=0.0,
        pos_curr_rwd_coeff=0.3 if track_pos else 0.0,
        neg_curr_rwd_coeff=0.3 if track_neg else 0.0,
        pos_fut_rwd_coeff=float(future_rwd_coeff) if track_pos else 0.0,
        neg_fut_rwd_coeff=float(future_rwd_coeff) if track_neg else 0.0,
        pos_pred_rwd_coeff=float(future_rwd_coeff) if track_pos else 0.0,
        neg_pred_rwd_coeff=float(future_rwd_coeff) if track_neg else 0.0,
        tracking_loss_mode=str(tracking_loss_mode),
        tracking_huber_delta=1.0,
        pos_rate_rwd_coeff=float(rate_rwd_coeff) if track_pos else 0.0,
        neg_rate_rwd_coeff=float(rate_rwd_coeff) if track_neg else 0.0,
        pos_rate_scale_kpa_s=float(rate_rwd_scale_pos),
        neg_rate_scale_kpa_s=float(rate_rwd_scale_neg),
        tracking_channels=tracking_channels,
    )
    if lib == 3:
        values.update(
            action_delta_rwd_coeff=float(action_delta_rwd_coeff),
            second_action_delta_rwd_coeff=float(second_action_delta_rwd_coeff),
            conflict_rwd_coeff=float(conflict_rwd_coeff),
            chamber_reserve_rwd_coeff=float(chamber_reserve_rwd_coeff),
            chamber_margin_kpa=15.0,
            chamber_deadband_kpa=5.0,
            chamber_bound_rwd_coeff=float(chamber_bound_rwd_coeff),
            chamber_pos_min_kpa=101.325,
            chamber_pos_max_kpa=220.0,
            chamber_neg_min_kpa=10.0,
            chamber_neg_max_kpa=101.325,
        )
    return values


def _pid_kwargs(lib: int) -> dict[str, float]:
    if lib == 2:
        return dict(
            Kp_pos=0.0,
            # The previous 0.01 I-term was large enough to dominate the RL
            # action during long tracking errors.  Keep the I-term variant as
            # a small-bias comparison rather than a second controller.
            Ki_pos=0.002,
            Kd_pos=0.0,
            Kp_neg=0.0,
            Ki_neg=0.002,
            Kd_neg=0.0,
            Ka=1.0,
        )
    return dict(
        Kp_act_pos_in=0.0,
        Ki_act_pos_in=0.002,
        Kd_act_pos_in=0.0,
        Kp_act_pos_out=0.0,
        Ki_act_pos_out=0.002,
        Kd_act_pos_out=0.0,
        Kp_act_neg_in=0.0,
        Ki_act_neg_in=0.002,
        Kd_act_neg_in=0.0,
        Kp_act_neg_out=0.0,
        Ki_act_neg_out=0.002,
        Kd_act_neg_out=0.0,
        Ka=1.0,
    )


def build_config(
    lib: int,
    variant: str,
    seed: int,
    ref_seed: int,
    reference_kind: str = "legacy_random",
    fixed_action_indices: Optional[tuple[int, ...]] = None,
    carry: bool = False,
    scale_low: Optional[float] = None,
    init_pos: float = 101.325,
    init_neg: float = 101.325,
    caps_temporal_weight: Optional[float] = None,
    caps_spatial_weight: Optional[float] = None,
    caps_noise_std: Optional[float] = None,
    physical_action_low: float = 0.90,
    warmup_physical_action: float = 0.95,
    tracking_loss_mode: str = "abs",
    rate_rwd_coeff: float = 0.0,
    rate_rwd_scale_pos: float = 12.0,
    rate_rwd_scale_neg: float = 6.0,
    action_delta_rwd_coeff: float = 0.0,
    second_action_delta_rwd_coeff: float = 0.0,
    conflict_rwd_coeff: float = 0.0,
    chamber_reserve_rwd_coeff: float = 0.0,
    future_rwd_coeff: float = 0.0,
    include_valve_z: bool = False,
    include_physics_features: bool = False,
    physics_feature_mode: str = "all",
    physics_rate_scale_kpa_s: float = 50.0,
    include_valve_diagnostics: bool = False,
    constant_pos: float = 140.0,
    constant_neg: float = 70.0,
    tracking_channels: str = "both",
    fixed_control_physical: Optional[float] = None,
    chamber_controller: str = "none",
    carry_reset_blocks: int = 0,
    action_history_len: int = 10,
    chamber_bound_rwd_coeff: float = 0.0,
    normalize_observation: bool = False,
    caps_mode: str = "legacy",
    caps_spatial_noise_normalized: float = 0.01,
    warmup_mode: str = "fixed",
    warmup_noise_std: float = 0.0,
    warmup_noise_seed: Optional[int] = None,
    update_during_warmup: bool = True,
    reset_after_warmup: bool = False,
    include_integral_error: bool = False,
    integral_error_decay: float = 1.0,
    integral_error_limit: float = 100.0,
    integral_error_scale: float = 20.0,
    action_chunk_steps: int = 1,
    gamma: Optional[float] = None,
    checkpoint_interval: int = 0,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if chamber_controller not in ("none", "dynamic_reserve"):
        raise ValueError(f"Unknown chamber_controller: {chamber_controller}")

    use_action_history = variant.startswith("history_action")
    use_history = (
        variant.startswith("history_")
        or variant == "mpobs1_sac"
        or "mpobs" in variant
    )
    use_mpobs = variant == "mpobs1_sac" or "mpobs" in variant
    use_caps = "caps" in variant
    use_iterm = "iterm" in variant

    # Keep the reference task fixed across all variants for one library.
    legacy_ref_common = dict(
        seed=ref_seed,
        pos_max_ts=5,
        neg_max_ts=5,
        pos_max_amp=20 if lib == 2 else 10,
        neg_max_amp=5,
        pos_max_per=10,
        neg_max_per=10,
    )
    if lib == 2:
        legacy_ref_common.update(
            pos_min_off=140,
            pos_max_off=240,
            neg_min_off=10,
            neg_max_off=32,
        )
    else:
        legacy_ref_common.update(
            pos_min_off=115,
            pos_max_off=140,
            neg_min_off=60,
            neg_max_off=80,
        )

    if reference_kind == "legacy_random":
        reference_cfg = dict(kind="legacy_random", kwargs=copy.deepcopy(legacy_ref_common))
        legacy_rnd_ref = copy.deepcopy(legacy_ref_common)
    elif reference_kind == "rate_limited_random":
        if lib != 3:
            raise ValueError("rate_limited_random is currently intended for lib3")
        # Preserve the historical lib3 target envelope while replacing the
        # discontinuous RandomRef parameter jumps with minimum-jerk segments.
        smooth_ref = dict(
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
            initial_pos=101.325,
            initial_neg=101.325,
            initial_hold=0.0,
            min_pos_delta=5.0,
            min_neg_delta=2.0,
            seed=ref_seed,
        )
        reference_cfg = dict(kind="rate_limited_random", kwargs=smooth_ref)
        legacy_rnd_ref = None
    elif reference_kind == "constant":
        reference_cfg = dict(
            kind="constant",
            kwargs={"pos": float(constant_pos), "neg": float(constant_neg)},
        )
        legacy_rnd_ref = None
    else:
        raise ValueError(f"Unknown reference_kind: {reference_kind}")

    # lib2's compiled positive-valve model has a cracking region around
    # physical command 0.84--0.85.  Its historical action mapping was
    # 0.85--1.0; using 0.8 as the lower bound can effectively close the
    # positive exhaust valve and make pressure run away.  Keep lib3's
    # screening range unchanged until its own valve sweep is characterized.
    if scale_low is None:
        scale_low = 0.85 if lib == 2 else 0.8

    # ``scale_low`` is the physical command produced when normalized action 0
    # is supplied.  The learned SAC policy always keeps continuous [0, 1]
    # (therefore physical [scale_low, 1.0]).  ``physical_action_low`` only
    # bounds a *uniform* warm-up fallback; the default warm-up is instead a
    # fixed safe action.
    scale_high = 1.0
    if not scale_low <= physical_action_low <= scale_high:
        raise ValueError(
            "physical_action_low must lie inside the simulator command scale: "
            f"got {physical_action_low} for [{scale_low}, {scale_high}]"
        )
    warmup_uniform_low = (float(physical_action_low) - float(scale_low)) / (
        float(scale_high) - float(scale_low)
    )
    warmup_action_raw = (float(warmup_physical_action) - float(scale_low)) / (
        float(scale_high) - float(scale_low)
    )
    if not scale_low <= warmup_physical_action <= scale_high:
        raise ValueError(
            "warmup_physical_action must lie inside the simulator command scale: "
            f"got {warmup_physical_action} for [{scale_low}, {scale_high}]"
        )

    obs_cfg: dict[str, Any] = dict(
        freq=50,
        delay=0.1,
        init_pos_press=float(init_pos),
        init_neg_press=float(init_neg),
        noise=False,
        noise_std=0.0,
        scale=True,
        scale_low=float(scale_low),
        scale_high=scale_high,
    )

    # MPObs in the current implementation means observation history plus a
    # multi-step model prediction.  This intentionally preserves its current
    # interface; the resulting action dimension is recorded in cfg.yaml.
    pred_cfg: dict[str, Any] | None = None
    if use_mpobs:
        pred_cfg = copy.deepcopy(obs_cfg)
        pred_cfg.update(delay=0.0)

    if use_history:
        env_cfg: dict[str, Any] = dict(
            num_prev=10,
            num_pred=15,
            # Only the explicitly named legacy multi-action MPObs variant
            # outputs a five-command trajectory.  History-only policies and
            # MPObs1 all issue one six-valve command per 20 ms step.
            num_act=(
                5
                if variant in (
                    "sac+mpobs+caps",
                    "sac+mpobs+caps+iterm",
                )
                else 1
            ),
        )
    else:
        env_cfg = dict(
            num_prev=0,
            num_pred=1,
            num_act=1,
        )
    env_cfg.update(
        rwd_kwargs=_reward_kwargs(
            lib,
            tracking_loss_mode=tracking_loss_mode,
            rate_rwd_coeff=rate_rwd_coeff,
            rate_rwd_scale_pos=rate_rwd_scale_pos,
            rate_rwd_scale_neg=rate_rwd_scale_neg,
            action_delta_rwd_coeff=action_delta_rwd_coeff,
            second_action_delta_rwd_coeff=second_action_delta_rwd_coeff,
            conflict_rwd_coeff=conflict_rwd_coeff,
            chamber_reserve_rwd_coeff=chamber_reserve_rwd_coeff,
            chamber_bound_rwd_coeff=chamber_bound_rwd_coeff,
            future_rwd_coeff=future_rwd_coeff,
            tracking_channels=tracking_channels,
        ),
        pos_pred_rnd_offset_range=0.0,
        neg_pred_rnd_offset_range=0.0,
        action_low=0.0,
        action_high=1.0,
        verbose=False,
        episode_carry_state=bool(carry),
    )
    if lib == 3:
        env_cfg.update(
            fixed_chamber_ctrl=None,
            steady_chamber_ctrl=None,
            dynamic_chamber_ctrl=None,
            include_valve_z=bool(include_valve_z),
            include_physics_features=bool(include_physics_features),
            physics_feature_mode=str(physics_feature_mode),
            physics_rate_scale_kpa_s=float(physics_rate_scale_kpa_s),
            include_valve_diagnostics=bool(include_valve_diagnostics),
            include_integral_error=bool(include_integral_error),
            integral_error_decay=float(integral_error_decay),
            integral_error_limit=float(integral_error_limit),
            integral_error_scale=float(integral_error_scale),
            action_chunk_steps=max(int(action_chunk_steps), 1),
            action_chunk_gamma=float(
                gamma if gamma is not None else (0.99 if lib == 3 else 0.9)
            ),
            action_history_len=(int(action_history_len) if use_action_history else 0),
            episode_carry_blocks=(int(carry_reset_blocks) if carry else 0),
        )
        if fixed_action_indices is not None:
            mask = np.zeros(6, dtype=bool)
            values = np.zeros(6, dtype=np.float64)
            fixed_physical = 1.0 if fixed_control_physical is None else float(fixed_control_physical)
            if not scale_low <= fixed_physical <= scale_high:
                raise ValueError(
                    "fixed_control_physical must lie inside the simulator command scale: "
                    f"got {fixed_physical} for [{scale_low}, {scale_high}]"
                )
            fixed_raw = (fixed_physical - float(scale_low)) / (
                float(scale_high) - float(scale_low)
            )
            for index in fixed_action_indices:
                if index < 0 or index >= 6:
                    raise ValueError(f"fixed action index must be in [0, 5], got {index}")
                mask[index] = True
                values[index] = fixed_raw
            env_cfg.update(
                fixed_action_mask=tuple(bool(value) for value in mask),
                fixed_action_values=tuple(float(value) for value in values),
            )
        if chamber_controller == "dynamic_reserve":
            if fixed_action_indices is not None:
                raise ValueError(
                    "dynamic_reserve cannot be combined with fixed_action_indices"
                )
            env_cfg["dynamic_chamber_ctrl"] = dict(
                # Match the existing lib3 reserve PID convention.  These are
                # chamber pressure targets derived from the actuator reference.
                pos_margin=35.0,
                neg_margin=35.0,
                pos_min=101.325,
                pos_max=200.0,
                neg_min=15.0,
                neg_max=101.325,
                kp=0.04,
                ki=0.001,
                deadband=1.0,
                integral_limit=100.0,
                min_open=0.15,
                max_open=0.85,
            )
    elif chamber_controller != "none":
        raise ValueError("chamber_controller is only supported for lib3")
    elif fixed_action_indices is not None:
        raise ValueError("fixed_action_indices is only supported for lib3")
    else:
        env_cfg.update(
            action_history_len=(int(action_history_len) if use_action_history else 0),
            include_physics_features=bool(include_physics_features),
            physics_feature_mode=str(physics_feature_mode),
            action_chunk_steps=max(int(action_chunk_steps), 1),
            action_chunk_gamma=float(
                gamma if gamma is not None else 0.9
            ),
        )

    model_cfg = dict(
        learning_rate=3e-4,
        gamma=(0.99 if lib == 3 else 0.9),
        tau=0.005,
        alpha=0.5,
        automatic_entropy_tunning=True,
        hidden_dim=256,
        buffer_size=500_000,
        batch_size=128,
        epoch=1,
        # carry=False screening follows the historical 2048-step episode
        # setup.  Use --horizon explicitly when running a shorter pilot.
        horizon=2048,
        start_epi=10,
        max_grad_norm=0.5,
        log_std_min=-10,
        log_std_max=1,
        # CAPS is a policy loss, not an environment reward term.
        temporal_weight=(
            (0.5 if caps_temporal_weight is None else float(caps_temporal_weight))
            if use_caps
            else 0.0
        ),
        spatial_weight=(
            (0.4 if caps_spatial_weight is None else float(caps_spatial_weight))
            if use_caps
            else 0.0
        ),
        # Spatial CAPS needs perturbed observations in ReplayBuffer.
        noise_std=(
            (
                (1.0 if str(caps_mode).lower() == "legacy" else 0.0)
                if caps_noise_std is None
                else float(caps_noise_std)
            )
            if use_caps
            else 0.0
        ),
        caps_mode=str(caps_mode),
        caps_spatial_noise_normalized=float(caps_spatial_noise_normalized),
        caps_start_steps=0,
        caps_ramp_steps=0,
        normalize_observation=bool(normalize_observation),
        train_diag_interval=1,
        train_diag_atm_band=8.0,
        train_verbose_interval=0,
        # Warm-up is a fixed, measured-safe joint operating point.  SAC uses
        # the full continuous [0, 1] policy range after start_epi.
        warmup_action_raw=float(warmup_action_raw),
        warmup_uniform_low=float(warmup_uniform_low),
        warmup_mode=str(warmup_mode),
        warmup_noise_std=float(warmup_noise_std),
        warmup_noise_octaves=8,
        warmup_noise_seed=(
            int(seed + 200_003)
            if warmup_noise_seed is None
            else int(warmup_noise_seed)
        ),
        update_during_warmup=bool(update_during_warmup),
        reset_after_warmup=bool(reset_after_warmup),
        policy_init_action=float(warmup_action_raw),
        checkpoint_interval=max(int(checkpoint_interval), 0),
    )

    return dict(
        lib=lib,
        variant=variant,
        control_mode="rl_plus_iterm" if use_iterm else "pure_rl",
        pid_enabled=use_iterm,
        iterm_enabled=use_iterm,
        train_seed=seed,
        ref_seed=ref_seed,
        reference_kind=reference_kind,
        fixed_action_indices=(
            None if fixed_action_indices is None else list(fixed_action_indices)
        ),
        fixed_control_physical=(
            None if fixed_control_physical is None else float(fixed_control_physical)
        ),
        chamber_controller=chamber_controller,
        tracking_channels=tracking_channels,
        obs=obs_cfg,
        pred=pred_cfg,
        reference=reference_cfg,
        rnd_ref=legacy_rnd_ref,
        env=env_cfg,
        model=model_cfg,
        pid=_pid_kwargs(lib) if use_iterm else None,
        action_range_physical=[float(scale_low), float(scale_high)],
        warmup_action_range_physical=[float(physical_action_low), float(scale_high)],
        warmup_physical_action=[
            float(warmup_physical_action),
            float(warmup_physical_action),
        ],
        structured_history=(
            dict(
                name={
                    "hrr": "HRR",
                    "pide": "PIDE",
                    "gpide": "GPIDE",
                    "edot": "E_DOT",
                    "rate_reserve_edot": "HRR_E_DOT",
                }.get(str(physics_feature_mode).lower(), str(physics_feature_mode).upper()),
                physics_feature_mode=str(physics_feature_mode).lower(),
                source="deterministic feature map from lib3 observation/action history",
                reward_excluded=True,
            )
            if include_physics_features
            else None
        ),
    )


def make_components(cfg: dict[str, Any]):
    reference = make_pressure_reference(reference_spec_from_model_config(cfg))
    if cfg["lib"] == 2:
        from pneu_env.env import PneuEnv
        from pneu_env.pred2 import PneuPred
        from pneu_env.sim2 import PneuSim

        obs = PneuSim(**cfg["obs"])
        pred = PneuPred(**cfg["pred"]) if cfg["pred"] is not None else None
        env = PneuEnv(obs=obs, ref=reference, pred=pred, **cfg["env"])
    else:
        from pneu_env.env3 import PneuEnv3
        from pneu_env.pred3 import PneuPred
        from pneu_env.sim3 import PneuSim

        obs = PneuSim(**cfg["obs"])
        pred = PneuPred(**cfg["pred"]) if cfg["pred"] is not None else None
        env = PneuEnv3(obs=obs, ref=reference, pred=pred, **cfg["env"])

    env.action_space.seed(cfg["train_seed"])
    if cfg["pid"] is not None:
        env.set_pid(**cfg["pid"])
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=int, choices=(2, 3), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0, help="training seed")
    parser.add_argument("--ref-seed", type=int, default=61098)
    parser.add_argument(
        "--reference-kind",
        choices=("legacy_random", "rate_limited_random", "constant"),
        default=None,
        help="lib3 reference generator; default is rate_limited_random for lib3",
    )
    parser.add_argument(
        "--fixed-control-indices",
        type=int,
        nargs="+",
        default=None,
        help="zero-based lib3 controls fixed at --fixed-control-physical, e.g. 1 4 5",
    )
    parser.add_argument(
        "--fixed-control-physical",
        type=float,
        default=None,
        help=(
            "physical value for controls selected by --fixed-control-indices; "
            "default remains 1.0 for backward compatibility, use 0.8 for the neutral fixed branch"
        ),
    )
    parser.add_argument(
        "--chamber-controller",
        choices=("none", "dynamic_reserve"),
        default="none",
        help=(
            "lib3 chamber-side controller; dynamic_reserve derives chamber "
            "targets from the current actuator reference and leaves u3-u6 to RL"
        ),
    )
    parser.add_argument("--constant-pos", type=float, default=140.0)
    parser.add_argument("--constant-neg", type=float, default=70.0)
    parser.add_argument(
        "--tracking-channels",
        choices=("both", "pos", "neg"),
        default="both",
        help="pressure channels included in the tracking reward",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--start-epi", type=int, default=10)
    parser.add_argument(
        "--carry-reset-blocks",
        type=int,
        default=0,
        help=(
            "hard-reset lib3 after this many carried episode blocks; 0 keeps "
            "the historical unlimited carry"
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="SAC discount factor; use this for the gamma-only diagnostic",
    )
    parser.add_argument(
        "--num-prev",
        type=int,
        default=None,
        help="override observation history length without changing the variant",
    )
    parser.add_argument(
        "--scale-low",
        type=float,
        default=None,
        help=(
            "physical command produced by normalized action 0; defaults to "
            "0.85 for lib2 and 0.8 for lib3"
        ),
    )
    parser.add_argument(
        "--action-low-physical",
        type=float,
        default=0.90,
        help=(
            "minimum physical valve command for the uniform warm-up fallback; "
            "it does not restrict the learned continuous policy (default: 0.90)"
        ),
    )
    parser.add_argument(
        "--warmup-physical-action",
        type=float,
        default=0.95,
        help="fixed physical command used during the initial warm-up (default: 0.95)",
    )
    parser.add_argument(
        "--warmup-mode",
        choices=("fixed", "uniform", "pink"),
        default="fixed",
        help="initial replay collection mode for the first --start-epi episodes",
    )
    parser.add_argument(
        "--warmup-pink-sigma",
        type=float,
        default=0.0,
        help="pink-noise std in normalized action units during warm-up only",
    )
    parser.add_argument(
        "--defer-updates-during-warmup",
        action="store_true",
        help="collect initial replay first and begin actor/critic updates afterward",
    )
    parser.add_argument(
        "--reset-after-warmup",
        action="store_true",
        help="hard-reset plant/reference once before the first policy episode",
    )
    parser.add_argument(
        "--caps-temporal-weight",
        type=float,
        default=None,
        help="override CAPS temporal-loss weight (default 0.5 when CAPS is enabled)",
    )
    parser.add_argument(
        "--caps-spatial-weight",
        type=float,
        default=None,
        help="override CAPS spatial-loss weight (default 0.4 when CAPS is enabled)",
    )
    parser.add_argument(
        "--caps-noise-std",
        type=float,
        default=None,
        help="override CAPS replay perturbation std (default 1.0 when CAPS is enabled)",
    )
    parser.add_argument(
        "--caps-mode",
        choices=("legacy", "mean_action"),
        default="legacy",
        help="legacy stochastic-sample CAPS or entropy-compatible mean-action CAPS",
    )
    parser.add_argument(
        "--caps-spatial-noise-normalized",
        type=float,
        default=0.01,
        help="state perturbation std after fixed observation normalization",
    )
    parser.add_argument(
        "--caps-start-after-warmup",
        action="store_true",
        help="activate CAPS only after the initial replay-collection episodes",
    )
    parser.add_argument(
        "--caps-ramp-episodes",
        type=int,
        default=0,
        help="linearly ramp CAPS weights over this many episodes",
    )
    parser.add_argument(
        "--tracking-loss",
        choices=("abs", "huber", "square"),
        default="abs",
        help="pressure/rate tracking loss; abs preserves the historical reward",
    )
    parser.add_argument(
        "--rate-rwd-coeff",
        type=float,
        default=0.0,
        help="coefficient for POS/NEG reference-rate tracking (default: disabled)",
    )
    parser.add_argument("--rate-rwd-scale-pos", type=float, default=12.0)
    parser.add_argument("--rate-rwd-scale-neg", type=float, default=6.0)
    parser.add_argument(
        "--action-delta-rwd-coeff",
        type=float,
        default=0.0,
        help="penalty on physical command changes (default: disabled)",
    )
    parser.add_argument(
        "--second-action-delta-rwd-coeff",
        "--second-difference-rwd-coeff",
        dest="second_action_delta_rwd_coeff",
        type=float,
        default=0.0,
        help="penalty on physical command second differences (default: disabled)",
    )
    parser.add_argument("--conflict-rwd-coeff", type=float, default=0.0)
    parser.add_argument("--chamber-reserve-rwd-coeff", type=float, default=0.0)
    parser.add_argument(
        "--chamber-bound-rwd-coeff",
        type=float,
        default=0.0,
        help="penalty per kPa outside the configured lib3 chamber envelope",
    )
    parser.add_argument(
        "--include-valve-z",
        action="store_true",
        help=(
            "append signed-log-compressed simulator valve hysteresis z states "
            "to the lib3 policy observation (privileged ablation)"
        ),
    )
    parser.add_argument(
        "--include-physics-features",
        action="store_true",
        help=(
            "append the lib3 derived-coordinate group selected by "
            "--physics-feature-mode"
        ),
    )
    parser.add_argument(
        "--physics-feature-mode",
        choices=(
            "none",
            "rate",
            "zero4",
            "reserve",
            "rate_reserve",
            "hrr",
            "edot",
            "rate_reserve_edot",
            "pide",
            "gpide",
            "all",
        ),
        default="all",
        help=(
            "select lib3 derived coordinates: pressure rates, a four-zero "
            "dimension placebo, chamber reserves, HRR, error/PID features, "
            "or the historical all=18 feature set"
        ),
    )
    parser.add_argument(
        "--physics-rate-scale-kpa-s",
        type=float,
        default=50.0,
        help=(
            "fixed normalization denominator for each pressure-rate feature; "
            "50 preserves the 260810 setup and 5000 matches the amplitude of "
            "a finite difference formed from pressure/100 history"
        ),
    )
    parser.add_argument(
        "--include-valve-diagnostics",
        action="store_true",
        help="append privileged signed-log z/q_pred/mdot features to the lib3 observation",
    )
    parser.add_argument(
        "--include-integral-error",
        action="store_true",
        help="append the bounded actuator pressure-error integral to the lib3 observation",
    )
    parser.add_argument(
        "--integral-error-decay",
        type=float,
        default=1.0,
        help="forgetting factor for actuator integral error (default: 1.0)",
    )
    parser.add_argument(
        "--integral-error-limit",
        type=float,
        default=100.0,
        help="absolute kPa*s limit for actuator integral error",
    )
    parser.add_argument(
        "--integral-error-scale",
        type=float,
        default=20.0,
        help="kPa*s scale used before exposing integral error to the policy",
    )
    parser.add_argument(
        "--action-chunk-steps",
        type=int,
        default=1,
        help="number of sequential physical commands emitted per policy step",
    )
    parser.add_argument(
        "--action-history-len",
        type=int,
        default=10,
        help="number of prior full six-valve actions for history_action variants",
    )
    parser.add_argument(
        "--normalize-observation",
        action="store_true",
        help="apply fixed physical-unit scales before policy and critic networks",
    )
    parser.add_argument(
        "--future-rwd-coeff",
        type=float,
        default=0.0,
        help="coefficient for MPObs future/predicted tracking (default: disabled)",
    )
    parser.add_argument(
        "--exploration",
        choices=("none", "pink"),
        default="none",
        help="external exploration process; 'pink' adds temporally correlated action noise",
    )
    parser.add_argument("--pink-sigma", type=float, default=0.05)
    parser.add_argument("--pink-octaves", type=int, default=8)
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=0,
        metavar="N",
        help=(
            "print compact RMSE/oscillation/exploration diagnostics every N episodes; "
            "0 disables the extra summary"
        ),
    )
    parser.add_argument(
        "--env-verbose",
        action="store_true",
        help="enable the environment's detailed step-level logging (very noisy)",
    )
    parser.add_argument(
        "--verbose-history",
        action="store_true",
        help="keep every verbose step on screen instead of refreshing one display block",
    )
    parser.add_argument("--carry", action="store_true", help="carry final physical pressure into the next episode")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        metavar="N",
        help=(
            "also save model_epi_NNNN.pth every N completed episodes; "
            "0 keeps only the rolling model.pth"
        ),
    )
    parser.add_argument("--log-std-min", type=float, default=-10.0)
    parser.add_argument("--log-std-max", type=float, default=1.0)
    parser.add_argument(
        "--init-pos",
        type=float,
        default=101.325,
        help="initial positive pressure for each non-carried rollout",
    )
    parser.add_argument(
        "--init-neg",
        type=float,
        default=101.325,
        help="initial negative pressure for each non-carried rollout",
    )
    parser.add_argument(
        "--reference-profile",
        type=Path,
        default=None,
        help=(
            "YAML pressure-reference profile.  When design.library is present "
            "it must match --lib."
        ),
    )
    reference_reset_group = parser.add_mutually_exclusive_group()
    reference_reset_group.add_argument(
        "--reference-independent-reset",
        dest="reference_independent_reset",
        action="store_true",
        help=(
            "keep the reference profile's initial target when the physical "
            "simulator resets (default)"
        ),
    )
    reference_reset_group.add_argument(
        "--reference-reset-to-observation",
        dest="reference_independent_reset",
        action="store_false",
        help=(
            "explicitly replace the reference profile's initial target with "
            "the physical simulator initial pressure"
        ),
    )
    parser.set_defaults(reference_independent_reset=True)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="construct and print dimensions/config without starting training",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    reference_kind = args.reference_kind or (
        "rate_limited_random" if args.lib == 3 else "legacy_random"
    )
    gamma = args.gamma if args.gamma is not None else (0.99 if args.lib == 3 else 0.9)
    if args.carry_reset_blocks < 0:
        parser.error("--carry-reset-blocks must be non-negative")
    if args.carry_reset_blocks > 0 and not args.carry:
        parser.error("--carry-reset-blocks requires --carry")
    if args.action_history_len < 0:
        parser.error("--action-history-len must be non-negative")
    if args.action_chunk_steps < 1:
        parser.error("--action-chunk-steps must be positive")
    if args.action_chunk_steps > 1 and not args.variant.startswith("history_action"):
        parser.error("--action-chunk-steps>1 currently requires a history_action variant")
    if not 0.0 < args.integral_error_decay <= 1.0:
        parser.error("--integral-error-decay must lie in (0, 1]")
    if args.integral_error_limit <= 0.0 or args.integral_error_scale <= 0.0:
        parser.error("integral-error-limit and integral-error-scale must be positive")
    if not np.isfinite(args.physics_rate_scale_kpa_s) or args.physics_rate_scale_kpa_s <= 0.0:
        parser.error("--physics-rate-scale-kpa-s must be finite and positive")
    if args.physics_feature_mode == "zero4" and args.lib != 3:
        parser.error("--physics-feature-mode zero4 is only supported for lib3")
    if args.lib != 3 and args.physics_feature_mode in {
        "hrr",
        "edot",
        "rate_reserve_edot",
        "pide",
        "gpide",
    }:
        parser.error(
            "structured lib3 physics features (hrr/edot/pide/gpide) "
            "are only supported with --lib 3"
        )
    if args.caps_ramp_episodes < 0:
        parser.error("--caps-ramp-episodes must be non-negative")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative")
    if args.caps_mode == "mean_action" and "caps" in args.variant and not args.normalize_observation:
        parser.error("mean_action CAPS requires --normalize-observation")
    cfg = build_config(
        lib=args.lib,
        variant=args.variant,
        seed=args.seed,
        ref_seed=args.ref_seed,
        reference_kind=reference_kind,
        fixed_action_indices=(
            None
            if args.fixed_control_indices is None
            else tuple(args.fixed_control_indices)
        ),
        carry=args.carry,
        scale_low=args.scale_low,
        init_pos=args.init_pos,
        init_neg=args.init_neg,
        caps_temporal_weight=args.caps_temporal_weight,
        caps_spatial_weight=args.caps_spatial_weight,
        caps_noise_std=args.caps_noise_std,
        physical_action_low=args.action_low_physical,
        warmup_physical_action=args.warmup_physical_action,
        tracking_loss_mode=args.tracking_loss,
        rate_rwd_coeff=args.rate_rwd_coeff,
        rate_rwd_scale_pos=args.rate_rwd_scale_pos,
        rate_rwd_scale_neg=args.rate_rwd_scale_neg,
        action_delta_rwd_coeff=args.action_delta_rwd_coeff,
        second_action_delta_rwd_coeff=args.second_action_delta_rwd_coeff,
        conflict_rwd_coeff=args.conflict_rwd_coeff,
        chamber_reserve_rwd_coeff=args.chamber_reserve_rwd_coeff,
        chamber_bound_rwd_coeff=args.chamber_bound_rwd_coeff,
        future_rwd_coeff=args.future_rwd_coeff,
        include_valve_z=args.include_valve_z,
        include_physics_features=args.include_physics_features,
        physics_feature_mode=args.physics_feature_mode,
        physics_rate_scale_kpa_s=args.physics_rate_scale_kpa_s,
        include_valve_diagnostics=args.include_valve_diagnostics,
        include_integral_error=args.include_integral_error,
        integral_error_decay=args.integral_error_decay,
        integral_error_limit=args.integral_error_limit,
        integral_error_scale=args.integral_error_scale,
        action_chunk_steps=args.action_chunk_steps,
        gamma=gamma,
        constant_pos=args.constant_pos,
        constant_neg=args.constant_neg,
        tracking_channels=args.tracking_channels,
        fixed_control_physical=args.fixed_control_physical,
        chamber_controller=args.chamber_controller,
        carry_reset_blocks=args.carry_reset_blocks,
        action_history_len=args.action_history_len,
        normalize_observation=args.normalize_observation,
        caps_mode=args.caps_mode,
        caps_spatial_noise_normalized=args.caps_spatial_noise_normalized,
        warmup_mode=args.warmup_mode,
        warmup_noise_std=args.warmup_pink_sigma,
        warmup_noise_seed=args.seed + 200_003,
        update_during_warmup=not args.defer_updates_during_warmup,
        reset_after_warmup=args.reset_after_warmup,
        checkpoint_interval=args.checkpoint_interval,
    )
    if args.include_valve_z and args.lib != 3:
        parser.error("--include-valve-z is only supported for lib3")
    if args.reference_profile is not None:
        with args.reference_profile.open(encoding="utf-8") as handle:
            profile = yaml.safe_load(handle)
        if not isinstance(profile, dict):
            parser.error("--reference-profile must contain a YAML mapping")
        profile_library = (profile.get("design") or {}).get("library")
        if profile_library is not None and int(profile_library) != args.lib:
            parser.error(
                "reference profile library does not match --lib: "
                f"profile=lib{profile_library}, requested=lib{args.lib}"
            )
        cfg["reference"] = profile
        profile_scale = (cfg["reference"].get("design") or {}).get("action_scale")
        if (
            profile_scale is not None
            and len(profile_scale) >= 2
            and not np.allclose(
                np.asarray(profile_scale[:2], dtype=np.float64),
                np.asarray(
                    [cfg["obs"]["scale_low"], cfg["obs"]["scale_high"]],
                    dtype=np.float64,
                ),
            )
        ):
            parser.error(
                "reference profile action scale does not match the training scale: "
                f"profile={profile_scale}, training="
                f"[{cfg['obs']['scale_low']}, {cfg['obs']['scale_high']}]. "
                "Use the matching profile or pass --scale-low explicitly."
            )
        # Avoid two competing reference definitions in a newly saved cfg.
        cfg.pop("rnd_ref", None)
        cfg["reference_kind"] = "profile"
        cfg["reference_profile_source"] = str(args.reference_profile)
    cfg["model"]["horizon"] = args.horizon
    cfg["model"]["start_epi"] = args.start_epi
    cfg["model"]["gamma"] = gamma
    cfg["model"].update(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        buffer_size=args.buffer_size,
        tau=args.tau,
        alpha=args.alpha,
        epoch=args.epoch,
        update_every=args.update_every,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        caps_start_steps=(
            args.start_epi * args.horizon
            if args.caps_start_after_warmup
            else 0
        ),
        caps_ramp_steps=args.caps_ramp_episodes * args.horizon,
    )
    if args.num_prev is not None:
        if args.num_prev < 0:
            parser.error("--num-prev must be non-negative")
        cfg["env"]["num_prev"] = args.num_prev
    cfg["model"]["train_verbose_interval"] = args.verbose_every
    cfg["model"]["exploration_noise"] = args.exploration
    cfg["model"]["exploration_noise_std"] = args.pink_sigma if args.exploration == "pink" else 0.0
    cfg["model"]["exploration_noise_octaves"] = args.pink_octaves
    cfg["model"]["exploration_noise_seed"] = args.seed + 100_003
    cfg["model"]["exploration_use_policy_mean"] = args.exploration == "pink"
    # PneuEnv (lib2) supports overriding the reference initial target on reset.
    # PneuEnv3 currently resets its PneuRef directly and does not accept this
    # keyword; passing it through makes every lib3 screening run fail before
    # training starts.
    if args.lib == 2:
        cfg["env"]["reference_reset_to_observation"] = not args.reference_independent_reset
    cfg["env"]["verbose"] = args.env_verbose
    if args.lib == 2:
        cfg["env"]["verbose_clear"] = not args.verbose_history
    cfg["episodes"] = args.episodes

    model_name = args.name or (
        f"screen_260722_lib{args.lib}_{args.variant.replace('+', '_')}_seed{args.seed}"
    )
    env = make_components(cfg)
    model = SAC(env=env, **cfg["model"])
    if args.dry_run:
        alpha_value = (
            float(model.alpha.detach().cpu().item())
            if torch.is_tensor(model.alpha)
            else float(model.alpha)
        )
        print(
            yaml.safe_dump(
                {
                    "dry_run": True,
                    "model_name": model_name,
                    "variant": args.variant,
                    "state_dim": int(env.observation_space.shape[0]),
                    "policy_action_dim": int(env.action_space.shape[0]),
                    "full_valve_action_dim": int(getattr(env, "dim_act", env.action_space.shape[0])),
                    "action_history_len": int(getattr(env, "action_history_len", 0)),
                    "carry_reset_blocks": int(getattr(env, "episode_carry_blocks", 0)),
                    "initial_alpha": alpha_value,
                    "config": cfg,
                },
                sort_keys=False,
            )
        )
        env.close()
        return
    model.set_logger(model_name)
    save_yaml(model_name, cfg)

    print(
        f"[ INFO] screening={model_name} lib={args.lib} variant={args.variant} "
        f"train_seed={args.seed} ref_seed={args.ref_seed} "
        f"action_dim={env.action_space.shape[0]} carry={args.carry}"
    )
    model.train(episode=args.episodes)


if __name__ == "__main__":
    main()
