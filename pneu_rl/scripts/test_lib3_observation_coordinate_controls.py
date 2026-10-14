#!/usr/bin/env python3
"""Regression checks for the lib3 rate-scale and zero-4 controls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for source_dir in (
    ROOT / "pneu_env/src",
    ROOT / "pneu_ref/src",
    ROOT / "pneu_rl/src",
    ROOT / "pneu_utils/src",
    SCRIPT_DIR,
):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from train_screening import build_config, make_components


def config_for(
    mode: str,
    rate_scale: float = 50.0,
    include_features: bool = True,
) -> dict:
    return build_config(
        lib=3,
        variant="history_action_sac",
        seed=0,
        ref_seed=61098,
        reference_kind="constant",
        include_physics_features=include_features,
        physics_feature_mode=mode,
        physics_rate_scale_kpa_s=rate_scale,
        normalize_observation=True,
        checkpoint_interval=100,
    )


def short_config_for(mode: str, include_features: bool) -> dict:
    cfg = config_for(
        mode,
        rate_scale=5000.0,
        include_features=include_features,
    )
    # Five previous samples plus the current pressure span 0.10 s at 50 Hz.
    # Match the delayed-command memory with five prior six-valve actions.
    cfg["env"]["num_prev"] = 5
    cfg["env"]["action_history_len"] = 5
    return cfg


class ObservationCoordinateControlTest(unittest.TestCase):
    def test_structured_history_modes_have_expected_dimensions(self) -> None:
        expected = {"hrr": 6, "pide": 6, "gpide": 16, "edot": 2, "rate_reserve_edot": 8}
        envs = {
            mode: make_components(config_for(mode))
            for mode in expected
        }
        try:
            for mode, feature_dim in expected.items():
                with self.subTest(mode=mode):
                    env = envs[mode]
                    self.assertEqual(env.dim_physics_features, feature_dim)
                    self.assertEqual(
                        env.observation_space.shape,
                        (156 + feature_dim,),
                    )
                    state, _ = env.reset()
                    next_state, reward, _, _, _ = env.step(np.full(6, 0.5))
                    self.assertEqual(state.shape, next_state.shape)
                    self.assertTrue(np.all(np.isfinite(next_state)))
                    self.assertTrue(np.isfinite(reward))
        finally:
            for env in envs.values():
                env.close()

    def test_four_way_dimensions_and_rate_scales(self) -> None:
        base = make_components(config_for("all", include_features=False))
        rate = make_components(config_for("rate", rate_scale=50.0))
        matched = make_components(config_for("rate", rate_scale=5000.0))
        zero4 = make_components(config_for("zero4", rate_scale=50.0))
        try:
            self.assertEqual(base.observation_space.shape, (156,))
            self.assertEqual(rate.observation_space.shape, (160,))
            self.assertEqual(matched.observation_space.shape, (160,))
            self.assertEqual(zero4.observation_space.shape, (160,))

            physics_start = rate.dim_obs_traj + rate.dim_action_history
            np.testing.assert_allclose(
                rate.policy_state_scale[physics_start : physics_start + 4],
                50.0,
            )
            np.testing.assert_allclose(
                matched.policy_state_scale[physics_start : physics_start + 4],
                5000.0,
            )
            np.testing.assert_allclose(
                zero4.policy_state_scale[physics_start : physics_start + 4],
                50.0,
            )
        finally:
            base.close()
            rate.close()
            matched.close()
            zero4.close()

    def test_scale_changes_normalization_not_raw_rate(self) -> None:
        rate = make_components(config_for("rate", rate_scale=50.0))
        matched = make_components(config_for("rate", rate_scale=5000.0))
        zero4 = make_components(config_for("zero4", rate_scale=50.0))
        try:
            pressure_history = np.asarray(
                [
                    [150.0, 30.0, 110.0, 80.0],
                    [150.2, 29.9, 110.1, 79.8],
                ],
                dtype=np.float64,
            )
            action_history = (np.zeros(6), np.zeros(6))
            ref = np.zeros((1, 2), dtype=np.float64)
            raw_rate = rate._get_physics_features(
                pressure_history,
                ref,
                action_history,
            )
            raw_matched = matched._get_physics_features(
                pressure_history,
                ref,
                action_history,
            )
            raw_zero4 = zero4._get_physics_features(
                pressure_history,
                ref,
                action_history,
            )
            np.testing.assert_allclose(raw_rate, raw_matched)
            np.testing.assert_allclose(
                raw_rate,
                (pressure_history[-1] - pressure_history[-2]) * 50.0,
            )
            np.testing.assert_allclose(raw_zero4, 0.0)
            np.testing.assert_allclose(raw_rate / 5000.0, np.asarray([0.002, -0.001, 0.001, -0.002]))
        finally:
            rate.close()
            matched.close()
            zero4.close()

    def test_checkpoint_interval_reaches_model_config(self) -> None:
        cfg = config_for("zero4")
        self.assertEqual(cfg["model"]["checkpoint_interval"], 100)

    def test_short_history_pair_changes_only_rate_group(self) -> None:
        short = make_components(short_config_for("all", include_features=False))
        short_rate = make_components(short_config_for("rate", include_features=True))
        try:
            self.assertEqual(short.num_prev, 5)
            self.assertEqual(short.action_history_len, 5)
            self.assertEqual(short.num_ref, 21)
            self.assertEqual(short.observation_space.shape, (96,))
            self.assertEqual(short_rate.observation_space.shape, (100,))
            self.assertEqual(short.dim_physics_features, 0)
            self.assertEqual(short_rate.dim_physics_features, 4)
            physics_start = short_rate.dim_obs_traj + short_rate.dim_action_history
            np.testing.assert_allclose(
                short_rate.policy_state_scale[physics_start : physics_start + 4],
                5000.0,
            )
        finally:
            short.close()
            short_rate.close()


if __name__ == "__main__":
    unittest.main()
