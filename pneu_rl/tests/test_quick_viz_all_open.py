"""Prepared simulation evaluation must not consume rollout time during preparation."""
from unittest.mock import Mock, patch
import unittest
import numpy as np
from test_lib3_all_open_reset import config
from quick_viz3 import build_viz_env, model_bank_reference, collect_rollout, infos_to_datas
from pneu_env.sim3 import PneuSim
from pneu_env.env3 import PneuEnv3
from pneu_env.env3_all_open import PneuEnv3AllOpen


class QuickVizPreparedTests(unittest.TestCase):
    def test_full_40_seconds_and_reference_zero_for_both_policies(self):
        for policy, dimension in [('H', 156), ('HR', 160)]:
            cfg = config(policy)
            cfg['initialization'] = dict(mode='pump_on_all_open', precondition_seconds=30)
            ref, _ = model_bank_reference(cfg)
            env = build_viz_env(cfg, PneuSim(**cfg['obs']), ref)
            self.assertIsInstance(env, PneuEnv3AllOpen)
            model = Mock()
            def action(state):
                self.assertEqual(len(state), dimension)
                return np.ones(6)
            model.predict.side_effect = action
            with patch.object(env, 'close', wraps=env.close) as close:
                infos = collect_rollout(env, model, ref)
            close.assert_called_once()
            self.assertEqual(len(infos), 2000)
            self.assertEqual(infos[0]['reference_time'], 0)
            self.assertEqual(infos[-1]['policy_time'], 40)
            self.assertGreater(infos[0]['obs']['curr_time'], 29)
            self.assertGreater(infos[-1]['obs']['curr_time'], 69)
            data = infos_to_datas(infos)
            np.testing.assert_allclose(np.diff(data['curr_time']), .02)
            self.assertAlmostEqual(data['reference_time'][-1], 39.98)
            np.testing.assert_allclose(
                [data['ref_act_pos'][0], data['ref_act_neg'][0]],
                ref.get_goal(0), atol=2e-4)

    def test_legacy_sim_keeps_original_env(self):
        cfg = config()
        ref, _ = model_bank_reference(cfg)
        env = build_viz_env(cfg, PneuSim(**cfg['obs']), ref)
        self.assertIs(type(env), PneuEnv3)
        env.close()


if __name__ == '__main__':
    unittest.main()
