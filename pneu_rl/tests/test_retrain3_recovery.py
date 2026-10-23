"""Recovery CLI keeps the saved real checkpoint and replay in a new folder."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/retrain3.py'
spec = importlib.util.spec_from_file_location('retrain3_recovery', SCRIPT)
retrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain)


class RetrainRecoveryTests(unittest.TestCase):
    def run_entrypoint(self, extra=()):
        args = retrain.parse_args(['--model', 'saved-run', '--name', 'recovered-run',
                                  '--obs', 'real', '--pid', 'off', '--episodes', '93', *extra])
        cfg = dict(model=dict(update_every=1, batch_size=128), obs={}, pred=None,
                   env={}, pid=None, reference={})
        model = Mock()
        model.last_epi, model.total_steps = 2107, 1078784
        model.update_every, model.batch_size = 5, 128
        model.buffer = []
        model.set_retrain.return_value = 'recovered-run'
        with tempfile.TemporaryDirectory() as folder:
            model.logger.folder_path = folder
            Path(folder, 'async_state.json').write_text(json.dumps(dict(steps=1078800)))
            with patch.object(retrain, 'parse_args', return_value=args), \
                    patch.object(retrain, 'load_deployment_config', return_value=cfg), \
                    patch.object(retrain, 'continuous_finetune_config', side_effect=copy.deepcopy), \
                    patch.object(retrain, 'make_pressure_reference'), \
                    patch.object(retrain, 'reference_spec_from_model_config'), \
                    patch.object(retrain, 'PneuReal'), \
                    patch.object(retrain, 'build_deployment_env'), \
                    patch.object(retrain, 'SAC', return_value=model) as constructor, \
                    patch.object(retrain, 'save_yaml') as save, \
                    patch('pneu_rl.async_training.train_async') as train, \
                    patch('builtins.input', side_effect=AssertionError('unexpected prompt')):
                retrain.main()
            return model, constructor, save, train

    def test_fresh_real_run_defaults_to_one_update_per_five_steps(self):
        model, constructor, save, train = self.run_entrypoint()
        self.assertEqual(constructor.call_args.kwargs['update_every'], 5)
        model.load.assert_called_once_with(name='saved-run', train=True, load_buffer=False)
        model.set_retrain.assert_called_once_with('recovered-run', load_buffer=False, copy_buffer=False)
        model.clear_buffer.assert_called_once()
        self.assertTrue(save.call_args.args[1]['finetuning_protocol']['fresh_replay'])
        train.assert_called_once_with(model, episode=93, queue_capacity=4096,
                                      sync_every=5, shutdown_timeout=180)

    def test_recovery_loads_replay_without_clearing_it_and_restores_partial_step_count(self):
        model, constructor, save, train = self.run_entrypoint(['--load-buffer', '--update-every', '7'])
        self.assertEqual(constructor.call_args.kwargs['update_every'], 7)
        model.load.assert_called_once_with(name='saved-run', train=True, load_buffer=True)
        model.set_retrain.assert_called_once_with('recovered-run', load_buffer=True, copy_buffer=True)
        model.clear_buffer.assert_not_called()
        self.assertEqual(model.total_steps, 1078800)
        self.assertFalse(save.call_args.args[1]['finetuning_protocol']['fresh_replay'])


if __name__ == '__main__':
    unittest.main()
