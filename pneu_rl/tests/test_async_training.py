"""Hardware-free integration tests, including real SAC updates in spawn workers."""
import os
import queue
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from pneu_rl.async_training import AsyncLearner, train_async
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

    def __getstate__(self):
        raise AssertionError('Hardware environment must never be pickled')

    def reset(self):
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

    def test_worker_error_closes_environment(self):
        model = make_model(BrokenSAC)
        with self.assertRaisesRegex(RuntimeError, 'injected learner failure'):
            train_async(model, episode=1)
        self.assertTrue(model.env.closed)


if __name__ == '__main__':
    unittest.main()
