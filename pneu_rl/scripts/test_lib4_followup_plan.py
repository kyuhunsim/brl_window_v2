#!/usr/bin/env python3
"""Regression checks for the gated lib4 follow-up infrastructure."""

from __future__ import annotations

import sys
import unittest
from argparse import Namespace
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

from lib4_evaluation_protocol import load_protocol, summarize_validation
from train_lib4_screening import build_config, make_components
from pneu_rl.sac_with_loss import SAC
from pneu_rl.lambda_discrepancy import discrepancy, lambda_returns


def args_for(variant: str) -> Namespace:
    return Namespace(
        variant=variant,
        reference_profile=ROOT / "pneu_rl/config/reference/lib4_capacity_core_16x4x512_seed64001.yaml",
        model_cfg="0625_lib4_Ours",
        state_source="current_tuned",
        seed=0,
        episodes=1,
        physical_horizon=8,
        carry_blocks=4,
        history_len=10,
        reference_lookahead=15,
        pos_reward=0.3,
        neg_reward=0.3,
        disp_reward=0.2,
        diff_reward=0.0,
        action_delta_reward=0.05,
        temporal_weight=0.0,
        chunk_consistency_weight=1.0,
        target_entropy=None,
        integral_decay=1.0,
        integral_limit=100.0,
        integral_scale=20.0,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.5,
        hidden_dim=32,
        buffer_size=128,
        batch_size=4,
        start_epi=10,
        verbose_every=0,
    )


class ChunkModeTest(unittest.TestCase):
    def test_hold_repeats_one_six_dimensional_action(self) -> None:
        cfg = build_config(args_for("history_rate_reserve_k2_hold"))
        env = make_components(cfg)
        try:
            self.assertEqual(env.action_space.shape, (6,))
            action = np.linspace(0.1, 0.6, 6)
            decoded = env.decode_action_chunk(action)
            self.assertEqual(decoded.shape, (2, 6))
            np.testing.assert_allclose(decoded[0], decoded[1])
        finally:
            env.close()


class StructuredHistoryTest(unittest.TestCase):
    def test_hrr_pide_gpide_have_finite_matched_states(self) -> None:
        expected_dims = {"hrr": 210, "pide": 210, "gpide": 222}
        for variant, expected_dim in expected_dims.items():
            with self.subTest(variant=variant):
                cfg = build_config(args_for(variant))
                env = make_components(cfg)
                try:
                    state, _ = env.reset()
                    next_state, reward, _, _, _ = env.step(np.full(6, 0.5))
                    self.assertEqual(state.shape, (expected_dim,))
                    self.assertEqual(next_state.shape, (expected_dim,))
                    self.assertTrue(np.all(np.isfinite(next_state)))
                    self.assertTrue(np.isfinite(reward))
                    self.assertEqual(
                        cfg["structured_history"]["physics_feature_mode"],
                        {"hrr": "rate_reserve", "pide": "pide", "gpide": "gpide"}[variant],
                    )
                finally:
                    env.close()


class LambdaDiscrepancyTest(unittest.TestCase):
    def test_td_zero_matches_one_step_bootstrap(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        expected = rewards + 0.9 * values[1:]
        np.testing.assert_allclose(lambda_returns(rewards, values, 0.9, 0.0), expected)

    def test_discrepancy_is_zero_for_equal_lambda(self) -> None:
        report = discrepancy(
            np.array([1.0, 2.0]),
            np.array([0.2, 0.3, 0.4]),
            gamma=0.99,
            lambda_low=0.5,
            lambda_high=0.5,
        )
        self.assertAlmostEqual(report["mean_abs"], 0.0, places=12)

    def test_free_k2_preserves_both_subactions_and_logs_them(self) -> None:
        cfg = build_config(args_for("history_rate_reserve_k2_consistency"))
        env = make_components(cfg)
        try:
            self.assertEqual(env.action_space.shape, (12,))
            action = np.r_[np.full(6, 0.25), np.full(6, 0.75)]
            decoded = env.decode_action_chunk(action)
            np.testing.assert_allclose(decoded[0], 0.25)
            np.testing.assert_allclose(decoded[1], 0.75)
            env.reset()
            _, reward, _, _, info = env.step(action)
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(len(info["chunk_substep_infos"]), 2)
            model = SAC(env=env, **cfg["model"])
            diagnostics = model._episode_diagnostics(0, reward, [info])
            self.assertEqual(diagnostics["n"], 2)
        finally:
            env.close()


class EvaluationProtocolTest(unittest.TestCase):
    def test_protocol_separates_chunk_deltas_and_emits_gates(self) -> None:
        protocol, _ = load_protocol(
            ROOT / "pneu_rl/config/evaluation/lib4_p0_protocol.yaml"
        )
        rows = []
        for index in range(100):
            reference = 110.0 + min(index, 49) * 0.02
            command = 0.85 if index % 2 == 0 else 0.95
            rows.append(
                {
                    "policy_step": index // 2,
                    "sub_action_index": index % 2,
                    "time": index * 0.02,
                    "ch_pos": 140.0,
                    "ch_neg": 30.0,
                    "act_pos": reference - 0.2,
                    "act_neg": 80.0,
                    "disp": 5.0,
                    "ref_pos": reference,
                    "ref_neg": 80.0,
                    "ref_disp": 5.0,
                    "ctrl_ch_pos": command,
                    "ctrl_ch_neg": command,
                    "ctrl_act_pos_in": command,
                    "ctrl_act_pos_out": command,
                    "ctrl_act_neg_in": command,
                    "ctrl_act_neg_out": command,
                }
            )
        summary = summarize_validation(
            rows,
            member=0,
            action_chunk_steps=2,
            protocol=protocol,
        )
        self.assertIn("pos_transition_rmse", summary)
        self.assertIn("disp_plateau_oscillation_std", summary)
        self.assertIn("within_chunk_mean_abs_delta_u", summary)
        self.assertIn("boundary_mean_abs_delta_u", summary)
        self.assertIn("gate_all", summary)
        self.assertGreater(summary["within_chunk_mean_abs_delta_u"], 0.09)


if __name__ == "__main__":
    unittest.main()
