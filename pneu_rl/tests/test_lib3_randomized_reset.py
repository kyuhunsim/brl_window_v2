"""Native integration checks for training-only dynamics randomization.

Set LIB3_DR_TEST_LIBRARY to an isolated lib3 build exposing the diagnostic API.
The local diagnostic artifact is a fallback; tests skip if neither is available.
Every test environment uses a unique copy, because native state is global.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for package in ("pneu_env", "pneu_ref", "pneu_utils"):
    sys.path.insert(0, str(ROOT / package / "src"))

from pneu_env.env3_all_open import PneuEnv3AllOpen
from pneu_env.env3_randomized import PneuEnv3Randomized
from pneu_env.sim3 import PneuSim

LIBRARY = Path(os.environ.get(
    "LIB3_DR_TEST_LIBRARY",
    ROOT / "pneu_rl/exp/lib3_hr_v34_slew_check_20261026/build/lib3_type1_slew_diagnostic.so",
))


class ConstantReference:
    def get_goal(self, _time):
        return 125.0, 70.0


def scenarios():
    return [
        dict(name="nominal", delay_seconds=.1, tau_scale=1.,
             flow3_scale=1., flow4_scale=1., weight=3.),
        dict(name="fast_delay", delay_seconds=.02, tau_scale=1.,
             flow3_scale=1., flow4_scale=1., weight=1.),
        dict(name="slow_valve", delay_seconds=.1, tau_scale=1.5,
             flow3_scale=1., flow4_scale=1., weight=1.),
        dict(name="flow_balance", delay_seconds=.1, tau_scale=1.,
             flow3_scale=1.1, flow4_scale=.9, weight=1.),
    ]


@unittest.skipUnless(LIBRARY.is_file(), "isolated diagnostic lib3 library not available")
class RandomizedResetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="lib3_dr_test_")
        self.root = Path(self.directory.name)
        self.env_count = 0

    def tearDown(self):
        self.directory.cleanup()

    def make_env(self, config=None, parent=False):
        self.env_count += 1
        library = self.root / f"lib3_{self.env_count}.so"
        shutil.copyfile(LIBRARY, library)
        obs = PneuSim(library_path=library, delay=.1)
        kwargs = dict(obs=obs, ref=ConstantReference(), num_prev=10,
                      num_pred=15, action_history_len=10,
                      include_physics_features=True, physics_feature_mode="rate_reserve",
                      episode_carry_state=True, episode_carry_blocks=4,
                      precondition_seconds=.5, verbose=False)
        if parent:
            return PneuEnv3AllOpen(**kwargs)
        if config is None:
            config = dict(enabled=True, seed=610260, first_nominal=True,
                          scenarios=scenarios())
        return PneuEnv3Randomized(**kwargs, dynamics_randomization=config)

    def test_disabled_and_nominal_match_parent_pressure_and_action_history(self):
        parent = self.make_env(parent=True)
        disabled = self.make_env(dict(enabled=False))
        nominal = self.make_env(dict(enabled=True, scenarios=[scenarios()[0]]))
        expected, _ = parent.reset()
        for env in (disabled, nominal):
            state, _ = env.reset()
            np.testing.assert_array_equal(state, expected)
            np.testing.assert_array_equal(env.obs.obs_buf, parent.obs.obs_buf)
            self.assertEqual(state.size, 162)
        for action in (np.full(6, .7), np.array([.8, .7, .6, .9, .7, .5])):
            expected = parent.step(action)
            for env in (disabled, nominal):
                actual = env.step(action)
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1:4], expected[1:4])
                np.testing.assert_array_equal(env.action_history, parent.action_history)

    def test_carry_preserves_condition_fifo_history_and_native_time(self):
        log_path = self.root / "schedule.jsonl"
        env = self.make_env(dict(enabled=True, first_nominal=False,
                                scenarios=[scenarios()[1]], log_path=str(log_path)))
        _, info = env.reset()
        self.assertEqual(info["dynamics_randomization"]["scenario"]["name"], "fast_delay")
        self.assertEqual(env.obs.obs_buf.maxlen, 2)
        self.assertAlmostEqual(env.obs.lib.get_time_c(), .5)
        self.assertAlmostEqual(env.obs.obs_buf[-1][0] - env.obs.obs_buf[0][0], .02)
        for _ in range(3):
            env.step(np.full(6, .75))
            before = (np.asarray(env.obs.obs_buf).copy(), env.obs_traj.copy(),
                      env.action_history.copy(), env.goal.buf.copy(),
                      env.obs.lib.get_time_c(), env.t)
            env.reset()
            after = (np.asarray(env.obs.obs_buf), env.obs_traj, env.action_history,
                     env.goal.buf, env.obs.lib.get_time_c(), env.t)
            for actual, expected in zip(after, before):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(env._dr_draw_count, 1)
        env.reset()
        self.assertEqual(env.hard_reset_count, 2)
        self.assertEqual(env._dr_draw_count, 2)
        self.assertEqual(env.t, 0.)
        self.assertEqual(len(log_path.read_text().splitlines()), 2)
        self.assertEqual(json.loads(log_path.read_text().splitlines()[1])["scenario"]["name"],
                         "fast_delay")

    def test_forced_reset_resamples_reproducibly_without_global_rng(self):
        envs = [self.make_env(), self.make_env()]
        sequence = []
        for env in envs:
            observed = []
            for _ in range(12):
                env.request_hard_reset()
                env.reset()
                observed.append(env.current_dynamics_scenario["name"])
            sequence.append(observed)
        self.assertEqual(sequence[0], sequence[1])
        self.assertEqual(sequence[0][0], "nominal")
        self.assertGreater(len(set(sequence[0])), 1)
        state_before = np.random.get_state()
        envs[0]._select_dynamics()
        state_after = np.random.get_state()
        for before, after in zip(state_before, state_after):
            np.testing.assert_array_equal(before, after)

    def test_invalid_physical_or_delay_configuration_rejected(self):
        for key, value in (("tau_scale", 0), ("flow3_scale", -1),
                           ("flow4_scale", float("nan")),
                           ("delay_seconds", .015), ("delay_seconds", .5)):
            bad = scenarios()[0]
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.make_env(dict(enabled=True, first_nominal=False, scenarios=[bad]))


if __name__ == "__main__":
    unittest.main()
