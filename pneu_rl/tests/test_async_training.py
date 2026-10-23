"""Hardware-free integration tests, including real SAC updates in spawn workers."""
import os
import io
import json
import pickle
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium.spaces import Box

from pneu_rl.async_training import AsyncLearner, train_async, _save_model
from pneu_rl.buffer import ReplayBuffer
from pneu_rl.logger import Logger
from pneu_rl.sac_with_loss import SAC


class FakeEnv:
    observation_space = Box(-1, 1, shape=(4,), dtype=np.float32)
    action_space = Box(0, 1, shape=(2,), dtype=np.float32)
    dim_ref_traj = 1
    dim_policy_act = 2

    def __init__(self):
        self.closed = False
        self.steps = 0
        self.resets = 0

    def __getstate__(self):
        raise AssertionError('Hardware environment must never be pickled')

    def reset(self):
        self.resets += 1
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.full(4, 0.1, dtype=np.float32), 1.0, False, False, {}

    def close(self):
        self.closed = True


class SlowSAC(SAC):
    def update_parameters(self):
        time.sleep(0.5)
        super().update_parameters()


class BrokenSAC(SAC):
    def update_parameters(self):
        raise RuntimeError('injected learner failure')


def make_model(cls=SAC):
    return cls(FakeEnv(), hidden_dim=8, batch_size=2, buffer_size=100,
               horizon=6, start_epi=0, update_every=2, train_diag_interval=0)


def transition():
    return (np.zeros(4, dtype=np.float32), np.zeros(2, dtype=np.float32),
            1.0, np.ones(4, dtype=np.float32), False)


class AsyncTrainingTest(unittest.TestCase):
    def test_no_extra_reset_without_warmup_and_final_checkpoint(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as folder:
            logger = Logger.__new__(Logger)
            logger.folder_path, logger.infos = folder, {}
            model.logger, model.log = logger, True
            train_async(model, episode=2, sync_every=1)
            self.assertEqual(model.env.resets, 2)
            self.assertEqual(model.env.steps, 12)
            self.assertTrue(os.path.isfile(f'{folder}/model_epi_0002.pth'))

    def test_training_warmup_checkpoint_and_actor_isolation(self):
        model = make_model()
        original = {k: v.clone() for k, v in model.policy.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            # Avoid Logger's repository-directory creation.
            logger = Logger.__new__(Logger)
            logger.folder_path, logger.infos = folder, {}
            model.logger, model.log = logger, True
            result = train_async(model, episode=1, warmup_steps=4, sync_every=1)
            self.assertEqual(result['buffer_size'], 10)
            self.assertEqual(result['updates'], 3)
            self.assertEqual(result['steps'], 10)
            self.assertTrue(os.path.exists(logger.model_path))
            self.assertTrue(os.path.exists(logger.buffer_path))
            checkpoint = torch.load(logger.model_path, weights_only=True)
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, checkpoint['policy_state_dict'][key])
            self.assertTrue(any(not torch.equal(original[k], v)
                                for k, v in model.policy.state_dict().items()))
        self.assertEqual(len(model.buffer), 0)
        self.assertTrue(model.env.closed)

    def test_slow_update_does_not_block_actor_or_share_parameters(self):
        model = make_model(SlowSAC)
        original = {k: v.clone() for k, v in model.policy.state_dict().items()}
        worker = AsyncLearner(model, sync_every=1)
        try:
            start = time.monotonic()
            for step in range(5):
                worker.submit('transition', (transition(), step, True, None, None))
            self.assertLess(time.monotonic() - start, 0.4)
            result = worker.finish()
            self.assertEqual(result['updates'], 2)
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(worker.sync(model.policy))
        finally:
            worker.abort()

    def test_queue_overflow_is_explicit_and_nonblocking(self):
        worker = AsyncLearner.__new__(AsyncLearner)
        worker.check = lambda: None
        worker.experience = queue.Queue(maxsize=1)
        worker.submit('transition', None)
        with self.assertRaisesRegex(RuntimeError, 'queue full'):
            worker.submit('transition', None)
        self.assertEqual(worker.pending_message, ('transition', None))

    def test_error_drain_saves_overflow_transition_without_training_backlog(self):
        model = make_model(SlowSAC)
        model.update_every = 1
        with tempfile.TemporaryDirectory() as folder:
            logger = Logger.__new__(Logger)
            logger.folder_path, logger.infos = folder, {}
            model.logger, model.log = logger, True
            worker = AsyncLearner(model, queue_capacity=1)
            try:
                failed_step = None
                for step in range(50):
                    try:
                        worker.submit('transition', (transition(), step, True, None, None))
                    except RuntimeError as exc:
                        self.assertIn('queue full', str(exc))
                        failed_step = step
                        break
                self.assertIsNotNone(failed_step)
                result = worker.finish(timeout=10, train_pending=False)
                self.assertEqual(result['buffer_size'], failed_step + 1)
                self.assertEqual(result['steps'], failed_step + 1)
                with open(logger.buffer_path, 'rb') as stream:
                    self.assertEqual(len(pickle.load(stream)), failed_step + 1)
                state = json.loads(Path(folder, 'async_state.json').read_text())
                self.assertEqual(state['steps'], failed_step + 1)
                self.assertTrue(Path(folder, f'model_step_{failed_step + 1}.pth').is_file())
            finally:
                worker.abort()

    def test_error_drain_skips_queued_gradient_updates(self):
        model = make_model(SlowSAC)
        model.update_every = 1
        with tempfile.TemporaryDirectory() as folder:
            logger = Logger.__new__(Logger)
            logger.folder_path, logger.infos = folder, {}
            model.logger, model.log = logger, True
            worker = AsyncLearner(model, queue_capacity=100)
            try:
                for step in range(30):
                    worker.submit('transition', (transition(), step, True, None, None))
                start = time.monotonic()
                result = worker.finish(timeout=10, train_pending=False)
                self.assertLess(time.monotonic() - start, 5)
                self.assertEqual(result['buffer_size'], 30)
                self.assertLess(result['updates'], 3)
                state = json.loads(Path(folder, 'async_state.json').read_text())
                self.assertGreater(state['skipped_pending_updates'], 20)
            finally:
                worker.abort()

    def test_cleanup_failure_preserves_original_error_after_closing_control(self):
        model = make_model()
        worker = Mock()
        worker.submit.side_effect = RuntimeError('original queue full')
        def fail_finish(**kwargs):
            self.assertTrue(model.env.closed)
            raise TimeoutError('secondary shutdown timeout')
        worker.finish.side_effect = fail_finish
        stderr = io.StringIO()
        with patch('pneu_rl.async_training.AsyncLearner', return_value=worker), \
                patch('sys.stderr', stderr), \
                self.assertRaisesRegex(RuntimeError, 'original queue full'):
            train_async(model, episode=1)
        worker.finish.assert_called_once_with(timeout=180, train_pending=False)
        worker.abort.assert_called_once()
        self.assertIn('secondary shutdown timeout', stderr.getvalue())

    def test_failed_checkpoint_and_replay_save_preserve_previous_files(self):
        with tempfile.TemporaryDirectory() as folder:
            model = make_model()
            path = Path(folder, 'model.pth')
            _save_model(model, str(path))
            original = path.read_bytes()
            def fail_checkpoint(temporary):
                Path(temporary).write_bytes(b'incomplete')
                raise OSError('disk error')
            with patch.object(model, 'save_model', side_effect=fail_checkpoint), \
                    self.assertRaisesRegex(OSError, 'disk error'):
                _save_model(model, str(path))
            self.assertEqual(path.read_bytes(), original)
            replay = ReplayBuffer(capacity=100, dim_goal=1)
            replay.add(*transition())
            buffer_path = Path(folder, 'buffer.pkl')
            replay.save_buffer(buffer_path)
            original = buffer_path.read_bytes()
            def fail_replay(value, stream):
                stream.write(b'incomplete')
                raise OSError('disk error')
            with patch('pneu_rl.buffer.pickle.dump', side_effect=fail_replay), \
                    self.assertRaisesRegex(OSError, 'disk error'):
                replay.save_buffer(buffer_path)
            self.assertEqual(buffer_path.read_bytes(), original)
            self.assertEqual(Path(folder, 'buffer_backup.pkl').read_bytes(), original)

    def test_worker_error_closes_environment(self):
        model = make_model(BrokenSAC)
        with self.assertRaisesRegex(RuntimeError, 'injected learner failure'):
            train_async(model, episode=1)
        self.assertTrue(model.env.closed)


if __name__ == '__main__':
    unittest.main()
