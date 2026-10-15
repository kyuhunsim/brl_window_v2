"""Lib3 deployment checks without file I/O to the bridge or hardware access."""
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import yaml

from pneu_env.real.real9 import PneuReal
from pneu_env.real.real_act import PneuRealAct
from pneu_env.env3 import PneuEnv3
from pneu_ref.factory import make_pressure_reference, reference_spec_from_model_config
from pneu_rl.sac_with_loss import SAC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pneu_rl/scripts'))
from quick_viz3 import model_bank_reference, collect_rollout


class Lib3RealPathsTest(unittest.TestCase):
    def test_six_channel_physical_commands(self):
        with patch.object(PneuRealAct, 'write_ctrl_file'), \
             patch.object(PneuRealAct, 'read_obs_file'), \
             patch.object(PneuRealAct, 'wait'):
            for scale, low in [(True, 0.8), (True, 0.85), (False, 0.8)]:
                real = PneuReal(scale=scale, scale_low=low)
                for action in [np.zeros(6), np.ones(6), np.linspace(0, 1, 6)]:
                    _, info = real.observe(action)
                    values = info['Observation']
                    actual = [values[k] for k in (
                        'ctrl_pos', 'ctrl_neg', 'ctrl_act_pos_in',
                        'ctrl_act_pos_out', 'ctrl_act_neg_in', 'ctrl_act_neg_out')]
                    expected = low + (1 - low) * action if scale else action
                    np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_shutdown_writes_all_open_once_without_pid_or_sensor_wait(self):
        real = PneuReal(scale_low=0.8, scale_high=0.95)
        real.is_pid = True
        real.pid = Mock()
        with tempfile.TemporaryDirectory() as folder:
            real.backend.labview_path = folder
            with patch.object(real.backend, 'write_ctrl_file', wraps=real.backend.write_ctrl_file) as write, \
                 patch.object(real.backend, 'read_obs_file') as read, \
                 patch.object(real.backend, 'wait') as wait:
                PneuEnv3.close(SimpleNamespace(obs=real))
            write.assert_called_once()
            read.assert_not_called()
            wait.assert_not_called()
            real.pid.get_action.assert_not_called()
            payload = json.loads(Path(folder, 'ctrl_act.json').read_text())
            for field in ('pos_ctrl', 'neg_ctrl', 'act_pos_ctrl1', 'act_pos_ctrl2',
                          'act_neg_ctrl1', 'act_neg_ctrl2'):
                self.assertEqual(payload[field], 1.0)

    def test_rollout_closes_once_on_completion_interrupt_and_error(self):
        for error in (None, KeyboardInterrupt(), RuntimeError('sensor failure')):
            env, model = Mock(), Mock()
            env.reset.return_value = (np.zeros(4), {})
            env.step.return_value = (np.zeros(4), 0, False, False, {'obs': {'curr_time': 1}})
            if error is not None:
                env.step.side_effect = error
            if isinstance(error, RuntimeError):
                with self.assertRaisesRegex(RuntimeError, 'sensor failure'):
                    collect_rollout(env, model, SimpleNamespace(max_time=1))
            else:
                result = collect_rollout(env, model, SimpleNamespace(max_time=1))
                self.assertEqual(len(result), 1 if error is None else 0)
            env.close.assert_called_once()

    def test_august_models_reference_and_policy(self):
        for folder in sorted((ROOT / 'pneu_rl/models').glob('260831_*')):
            with self.subTest(model=folder.name):
                config = yaml.safe_load((folder / 'cfg.yaml').read_text())
                self.assertNotIn('rnd_ref', config)
                training_ref = make_pressure_reference(reference_spec_from_model_config(config))
                self.assertFalse(training_ref.repeat_after_end)
                self.assertEqual(len(training_ref.trajectories), 16)
                ref, spec = model_bank_reference(config)
                self.assertEqual(ref.max_time, 40)
                duration = float(ref.trajectories[0]['duration'])
                self.assertAlmostEqual(duration, 40.96, places=5)
                self.assertFalse(ref.repeat_after_end)
                self.assertFalse(ref.repeat_same)
                self.assertLess(ref.max_time + (config['env']['num_pred'] + 1) / config['obs']['freq'], duration)
                np.testing.assert_allclose(ref.get_goal(40),
                    [np.interp(40, ref.trajectories[0]['time'], ref.trajectories[0]['pressure'][:, i]) for i in range(2)])
                np.testing.assert_allclose(ref.get_goal(100), ref.get_goal(duration))
                ref.time_reset()
                self.assertEqual(ref._active_index, 0)
                self.assertFalse(config['reference']['kwargs']['repeat_after_end'])
                env = PneuEnv3(obs=PneuReal(**config['obs']), ref=ref,
                               pred=None, **config['env'])
                state, _ = env.reset()
                self.assertEqual(env.dim_policy_act, 6)
                model = SAC(env, **config['model'])
                model.load_model(str(folder / 'model.pth'), evaluate=True)
                action = model.predict(state)
                self.assertEqual(action.shape, (6,))
                self.assertTrue(np.all(np.isfinite(action)))
                # Exercise real wrapper/env state and reward construction while
                # mocking only the bridge-facing methods.
                with patch.object(PneuRealAct, 'write_ctrl_file'), \
                     patch.object(PneuRealAct, 'read_obs_file'), \
                     patch.object(PneuRealAct, 'wait'):
                    next_state, reward, _, _, _ = env.step(action)
                self.assertEqual(next_state.shape, state.shape)
                self.assertTrue(np.isfinite(reward))

    def test_retrain_import_does_not_prompt_or_construct_hardware(self):
        spec = importlib.util.spec_from_file_location('retrain3_import', ROOT / 'pneu_rl/scripts/retrain3.py')
        module = importlib.util.module_from_spec(spec)
        with patch('builtins.input', side_effect=AssertionError('unexpected prompt')), \
             patch.object(PneuRealAct, '__init__', side_effect=AssertionError('hardware init')):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))


if __name__ == '__main__':
    unittest.main()
