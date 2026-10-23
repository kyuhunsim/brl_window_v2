"""No hardware I/O: portable bundles, measured startup and continuous fine-tuning."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from pneu_rl.deployment import load_deployment_config, build_deployment_env, continuous_finetune_config
from pneu_rl.sac_with_loss import SAC
from pneu_env.real.real9 import PneuReal
from pneu_env.real.real_act import PneuRealAct
from pneu_ref.factory import make_pressure_reference
from quick_viz3 import model_bank_reference

ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ROOT / 'pneu_rl/exp/lib3_real_ready_20261023/models'


class PreparedRealDeploymentTests(unittest.TestCase):
    def config(self, kind):
        name = f'261023_lib3_opening_type{kind}_HR_smooth_seed0'
        return name, load_deployment_config(name, BUNDLES)

    def test_portable_bundles_and_both_actor_weights(self):
        for kind in (1, 2):
            name, cfg = self.config(kind)
            original = Path.cwd()
            with tempfile.TemporaryDirectory() as folder:
                try:
                    os.chdir(folder)
                    loaded = load_deployment_config(name, BUNDLES)
                    ref, _ = model_bank_reference(loaded)
                finally:
                    os.chdir(original)
            self.assertEqual(ref._active_index, 0)
            self.assertEqual(ref.max_time, 40)
            real = PneuReal(**cfg['obs'])
            env = build_deployment_env(cfg, real, ref)
            calls = []
            def read(backend):
                calls.append(1)
                backend.time = 100 + len(calls) / 50
                backend.pos_press, backend.neg_press = 170. + .001 * (len(calls) % 2), 45.
                backend.act_pos_press, backend.act_neg_press = 123., 75.
            with patch.object(PneuRealAct, 'write_ctrl_file') as write, \
                 patch.object(PneuRealAct, 'read_obs_file', read), \
                 patch.object(PneuRealAct, 'wait'):
                state, _ = env.reset()
                self.assertEqual(len(calls), 1500)
                np.testing.assert_allclose(env.obs_traj, np.tile([170,45,123,75], (env.num_obs,1)), atol=.002)
                np.testing.assert_array_equal(env.action_history, np.ones((10,6)))
                self.assertEqual(env.t, 0)
                self.assertEqual(env.policy_step, 0)
                model = SAC(env, **cfg['model'])
                model.load_model(str(BUNDLES / name / 'model.pth'), evaluate=True)
                action = model.predict(state)
                self.assertEqual(action.shape, (6,))
                self.assertTrue(np.isfinite(action).all())
                _, reward, _, _, info = env.step(action)
                self.assertAlmostEqual(info['obs']['curr_time'], .02)
                self.assertAlmostEqual(env.t, .02)
                self.assertTrue(np.isfinite(reward))
                env.close()
                self.assertEqual(write.call_count, 1502)

    def test_finetuning_prepares_once_across_200_boundaries(self):
        _, cfg = self.config(1)
        cfg = continuous_finetune_config(cfg)
        ref = make_pressure_reference(cfg['reference'])
        env = build_deployment_env(cfg, PneuReal(**cfg['obs']), ref)
        calls = []
        def read(backend):
            calls.append(1)
            backend.time = len(calls) * .02
            backend.pos_press, backend.neg_press = 170. + .001 * (len(calls) % 2),45.
            backend.act_pos_press, backend.act_neg_press = 123.,75.
        with patch.object(PneuRealAct, 'write_ctrl_file'), \
             patch.object(PneuRealAct, 'read_obs_file', read), \
             patch.object(PneuRealAct, 'wait'):
            for _ in range(200):
                env.reset()
                env.step(np.full(6,.5))
            self.assertEqual(env.hard_reset_count, 1)
            self.assertEqual(len(calls), 1500 + 200)
            self.assertEqual(env.policy_step, 200)
            self.assertAlmostEqual(env.t, 4)
            env.close()

    def test_continuous_reference_joins_and_lookahead_for_both_types(self):
        for kind in (1,2):
            _, cfg = self.config(kind)
            ref = make_pressure_reference(continuous_finetune_config(cfg)['reference'])
            initial = ref.get_goal(1)
            ref.get_goal(2050)  # Full 200-episode horizon plus preview.
            np.testing.assert_array_equal(ref.get_goal(1), initial)
            self.assertGreater(len(ref.segments), 40)
            for start in ref.starts[1:]:
                np.testing.assert_allclose(ref.get_goal(start-1e-7), ref.get_goal(start+1e-7), atol=1e-4)
            for segment in ref.segments:
                if segment[0] != 'bridge':
                    continue
                _, start, duration, a, b = segment
                delta = b-a
                limits = np.where(delta>=0, ref.rate_limits[:,0], ref.rate_limits[:,1])
                self.assertTrue(np.all(1.875*np.abs(delta)/duration <= limits+1e-9))
                for t in np.linspace(start,start+duration,11):
                    value = np.array(ref.get_goal(t))
                    self.assertTrue(np.all(value >= np.minimum(a,b)-1e-6))
                    self.assertTrue(np.all(value <= np.maximum(a,b)+1e-6))
            adapted = continuous_finetune_config(cfg)
            evaluation_ref, _ = model_bank_reference(adapted)
            np.testing.assert_array_equal(evaluation_ref.trajectories[0]['pressure'],
                                          ref.trajectories[0]['pressure'])


if __name__ == '__main__':
    unittest.main()
