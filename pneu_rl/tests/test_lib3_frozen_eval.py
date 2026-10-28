"""Frozen evaluation must preserve sampling semantics and partial evidence."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from gymnasium.spaces import Box

from evaluate_lib3_frozen import FrozenActor, actor_hash, collect, main, sha256
from pneu_rl.networks import SACPolicy
from pneu_rl.sac_with_loss import SAC


class FrozenActorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(74)
        self.space = Box(low=0., high=1., shape=(6,), dtype=np.float32)
        self.config = dict(hidden_dim=8, log_std_min=-3., log_std_max=.5)
        self.policy = SACPolicy(4, 6, action_space=self.space, **self.config)
        self.weights = self.policy.state_dict()
        self.scale = np.array([100., 50., 1., 2.], dtype=np.float32)
        self.state = np.array([130., 70., .3, -.2], dtype=np.float32)

    def actor(self, seed=123):
        return FrozenActor(self.weights, self.config, action_space=self.space,
                           state_scale=self.scale, seed=seed)

    def sac_predictor(self):
        # Exercise the production inference method without creating training state.
        sac = object.__new__(SAC)
        sac.policy = self.policy
        sac.normalize_observation = True
        sac.state_scale = torch.from_numpy(self.scale)
        return sac

    def test_deterministic_matches_production_predict_and_does_not_consume_sampling_rng(self):
        actor = self.actor()
        before = actor.generator.get_state().clone()
        action, trace = actor.action(self.state, 'deterministic')
        expected = self.sac_predictor().predict(self.state, evaluate=True)
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(trace['policy_input'], self.state/self.scale)
        np.testing.assert_array_equal(trace['epsilon'], np.zeros(6))
        self.assertTrue(torch.equal(before, actor.generator.get_state()))

    def test_stochastic_matches_production_predict_for_identical_rng_start(self):
        actor = self.actor(seed=19)
        action, trace = actor.action(self.state, 'stochastic')
        torch.manual_seed(19)
        expected = self.sac_predictor().predict(self.state, evaluate=False)
        np.testing.assert_array_equal(action, expected)
        latent = trace['pre_tanh_mu'] + np.exp(trace['pre_tanh_log_std'])*trace['epsilon']
        np.testing.assert_allclose(action, .5*np.tanh(latent)+.5, atol=6e-8)
        self.assertFalse(any(p.requires_grad for p in actor.policy.parameters()))

    def test_seed_reproducible_without_global_rng_or_parameter_changes(self):
        a, b = self.actor(36), self.actor(36)
        before = actor_hash(a.policy)
        for _ in range(5):
            first, _ = a.action(self.state, 'stochastic')
            torch.randn(100)
            second, _ = b.action(self.state, 'stochastic')
            np.testing.assert_array_equal(first, second)
        self.assertEqual(before, actor_hash(a.policy))
        other, _ = self.actor(37).action(self.state, 'stochastic')
        self.assertFalse(np.array_equal(other, first))

    def test_invalid_state_and_mode_rejected(self):
        for state, mode in [(np.zeros(3), 'deterministic'),
                            (np.full(4, np.nan), 'stochastic'),
                            (self.state, 'unknown')]:
            with self.assertRaises(ValueError):
                self.actor().action(state, mode)


class FakeEnv:
    def __init__(self, fail_at=2, close_failure=False):
        self.fail_at = fail_at
        self.close_failure = close_failure
        self.closed = 0
        self.calls = 0
        self.precondition_steps = 1
        self.curr_full_press = np.array([150., 30., 120., 75.])
        self.obs = SimpleNamespace(freq=50., scale_low=.8, scale_high=1., observe=self.observe)
        self.t = 0.

    def observe(self, action, goal):
        return np.r_[0., self.curr_full_press], {'Observation': {'flowrate1': 2.5}}

    def reset(self):
        self.obs.observe(np.ones(6), [120., 75.])
        return np.array([130., 70., .3, -.2], dtype=np.float32), {}

    def step(self, action):
        self.calls += 1
        if self.calls == self.fail_at:
            raise TimeoutError('synthetic sensor timeout')
        self.t = self.calls/50
        row = {'curr_time': self.t, 'ref_act_pos': 120., 'ref_act_neg': 75.}
        return np.ones(4, dtype=np.float32), -.5, False, False, {'obs': row}

    def close(self):
        self.closed += 1
        if self.close_failure:
            raise RuntimeError('synthetic close failure')


class FrozenRolloutTests(unittest.TestCase):
    setUp = FrozenActorTests.setUp
    actor = FrozenActorTests.actor

    def test_timeout_preserves_completed_rows_attempted_action_and_shutdown_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            actor = self.actor()
            env = FakeEnv(close_failure=True)
            ref = SimpleNamespace(max_time=.08, bank_seconds=.04, hold_seconds=.04,
                                  get_goal=lambda t: (120., 75.),
                                  get_phase=lambda t: 'hold' if t<=.04 else 'bank')
            manifest = {'actor_sha256_before': actor_hash(actor.policy)}
            rows = collect(env, actor, ref, 'stochastic', output, manifest)
            self.assertEqual(env.closed, 1)
            self.assertEqual(len(rows), 1)
            saved = json.loads((output/'manifest.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('synthetic sensor timeout', saved['error'])
            self.assertIn('synthetic close failure', saved['close_error'])
            self.assertEqual(saved['attempted_transitions'], 2)
            self.assertEqual(saved['completed_transitions'], 1)
            self.assertTrue(saved['actor_unchanged'])
            trace = np.load(output/'policy_trace.npz')
            np.testing.assert_array_equal(trace['completed'], [True, False])
            self.assertTrue(np.isnan(trace['next_state'][1]).all())
            self.assertEqual(len((output/'rollout.csv').read_text().splitlines()), 2)
            self.assertTrue((output/'preparation.csv').is_file())

    def test_real_dry_run_does_not_open_backend_or_change_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/'model'; model.mkdir()
            time = np.arange(101)/50.
            np.savez(model/'reference.npz', time=time,
                     pressure=np.array([np.c_[120+.1*time,75-.1*time]]))
            cfg = {'obs': {'freq':50., 'scale':True, 'delay':0.},
                   'env': {'action_chunk_steps':1, 'action_low':0., 'action_high':1.,
                           'num_prev':1, 'num_pred':1},
                   'model':self.config, 'initialization': {'mode':'pump_on_all_open'},
                   'reference': {'kwargs':{'trajectory_npz':'/missing/oldpc/ref.npz'}}}
            (model/'cfg.yaml').write_text(yaml.safe_dump(cfg))
            torch.save({'policy_state_dict':self.weights},model/'model.pth')
            digest = sha256(model/'model.pth')
            with patch('pneu_env.sim3.PneuSim') as sim, patch('pneu_env.real.real9.PneuReal') as real:
                rc = main(['--model',str(model),'--obs','real','--dry-run',
                           '--hold-seconds','.1','--bank-seconds','.2',
                           '--preparation-seconds','.1','--output',str(root/'output')])
                sim.assert_not_called(); real.assert_not_called()
            self.assertEqual(rc,0)
            saved = json.loads((root/'output/manifest.json').read_text())
            self.assertEqual(saved['status'],'validated')
            self.assertEqual(saved['checkpoint_sha256'],digest)
            self.assertEqual(saved['policy_weight_updates'],0)
            self.assertEqual(sha256(model/'model.pth'),digest)
            self.assertFalse((root/'output/rollout.csv').exists())


if __name__ == '__main__':
    unittest.main()
