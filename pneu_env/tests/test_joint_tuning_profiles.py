"""Offline checks for the collection protocol; never open a hardware bridge."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for package in ("pneu_env", "pneu_utils"):
    sys.path.insert(0, str(ROOT / package / "src"))

from pneu_env.real.joint_tuning_profiles import (
    JointTuningConfig, JointTuningSuite, _limit_supply_runs,
)


class JointTuningProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = JointTuningSuite(JointTuningConfig())

    def test_complete_six_channel_protocol_and_duration(self):
        suite = self.suite
        self.assertEqual(suite.commands.shape, (32850, 6))
        self.assertEqual(suite.duration, 657)
        self.assertEqual(len(suite.profiles), 30)
        np.testing.assert_allclose(np.diff(suite.times), .02, atol=1e-12)
        self.assertTrue(np.isfinite(suite.commands).all())
        self.assertGreaterEqual(suite.commands.min(), .8)
        self.assertLessEqual(suite.commands.max(), 1)
        self.assertTrue((np.ptp(suite.commands, axis=0) > .19).all())
        self.assertEqual([p["start_sec"] for p in suite.profiles], [
            0, 5, 35, 47, 52, 82, 94, 99, 129, 141, 186, 216,
            228, 273, 303, 315, 360, 390, 402, 447, 492, 497,
            542, 547, 552, 577, 582, 607, 612, 652])
        for t in (0, 4.98, 47, 94, 547, 652, 657, 700):
            command, _ = suite.sample(t)
            np.testing.assert_array_equal(command, np.ones(6))

    def test_supply_budget_is_continuous_across_all_profiles(self):
        maxima = self.suite.max_low_seconds()
        self.assertLessEqual(maxima[0], .2)
        self.assertLessEqual(maxima[1], .2)
        # Actuator low holds are deliberately longer than the supply budget.
        self.assertTrue(all(t >= 5 for t in maxima[2:]))
        for valve in (1, 2):
            for mode in ("triangle_wave", "random_staircase", "ramp"):
                profile = next(p for p in self.suite.profiles
                               if p["channel"] == valve and p["mode"] == mode)
                begin = round(profile["start_sec"] * 50)
                end = round(profile["end_sec"] * 50)
                values = self.suite.commands[begin:end, valve - 1]
                self.assertAlmostEqual(values.min(), .8)

    def test_triangle_period_and_actuator_waveform(self):
        for start, period in ((5, 10), (35, 2), (52, 10), (82, 2)):
            valve = 0 if start < 47 else 1
            self.assertAlmostEqual(self.suite.sample(start)[0][valve], .8)
            self.assertAlmostEqual(self.suite.sample(start + .1)[0][valve], .85)
            self.assertAlmostEqual(self.suite.sample(start + period / 2)[0][valve], 1)
            self.assertAlmostEqual(self.suite.sample(start + period)[0][valve], .8)
        # Valve 3 retains the ordinary 0.1 Hz linear triangle.
        self.assertAlmostEqual(self.suite.sample(100)[0][2], .84)
        self.assertAlmostEqual(self.suite.sample(104)[0][2], 1)

    def test_supply_stairs_are_compressed_actuator_stairs_keep_two_second_holds(self):
        for valve in (1, 2):
            p = next(p for p in self.suite.profiles
                     if p["mode"] == "random_staircase" and p["channel"] == valve)
            self.assertLess(p["signal_duration_sec"], p["duration"])
            self.assertEqual(self.suite.sample(p["start_sec"] + .2)[0][valve - 1] >= .85, True)
            np.testing.assert_equal(self.suite.sample(p["end_sec"] - .02)[0][valve - 1], 1)
        for valve in (3, 4, 5, 6):
            p = next(p for p in self.suite.profiles
                     if p["mode"] == "random_staircase" and p["channel"] == valve)
            for t in (0, .2, 1, 1.98):
                self.assertAlmostEqual(self.suite.sample(p["start_sec"] + t)[0][valve - 1], .8)

    def test_low_pulse_retains_actuator_hold_and_removes_supply_tail(self):
        np.testing.assert_allclose(self.suite.sample(542)[0], .8)
        command, metadata = self.suite.sample(542.2)
        np.testing.assert_allclose(command, [1, 1, .8, .8, .8, .8])
        np.testing.assert_array_equal(metadata["low_limited"], [True, True])
        np.testing.assert_allclose(self.suite.sample(546.98)[0], [1, 1, .8, .8, .8, .8])
        for t, target in ((576.98, 0), (606.98, 1)):
            self.assertEqual(self.suite.sample(t)[0][target], 1)
        for t in (577, 582 - .02, 607, 612 - .02):
            np.testing.assert_allclose(self.suite.sample(t)[0], np.ones(6))

    def test_limiter_does_not_restart_on_another_low_value(self):
        requested = np.ones((50, 6))
        requested[:35, 0] = .8
        requested[5:35, 0] = .82
        requested[20:35, 0] = .84
        requested[36:, 0] = .81
        requested[:, 2] = .8
        actual, limited = _limit_supply_runs(requested, .85, 10, 1)
        np.testing.assert_allclose(actual[:10, 0], requested[:10, 0])
        np.testing.assert_allclose(actual[10:35, 0], 1)
        np.testing.assert_allclose(actual[36:46, 0], .81)
        np.testing.assert_allclose(actual[46:, 0], 1)
        np.testing.assert_allclose(actual[:, 2], .8)
        self.assertTrue(limited[10:35, 0].all())

    def test_repeatable_seed_and_configurable_budget(self):
        again = JointTuningSuite(JointTuningConfig())
        np.testing.assert_array_equal(again.commands, self.suite.commands)
        shorter = JointTuningSuite(JointTuningConfig(supply_low_max_sec=.1))
        self.assertTrue(all(t <= .1 for t in shorter.max_low_seconds()[:2]))
        np.testing.assert_array_equal(shorter.commands[:, 2:], self.suite.commands[:, 2:])
        for kwargs in (dict(freq=0), dict(command_min=.9), dict(supply_low_max_sec=.001),
                       dict(idle_ramp_period_sec=.1), dict(idle_ramp_fade_sec=0),
                       dict(solo_prefix_sec=-1)):
            with self.assertRaises(ValueError):
                JointTuningConfig(**kwargs)

    def test_idle_ramps_only_replace_non_target_open_channels_after_solo_prefix(self):
        suite = self.suite
        baseline = JointTuningSuite(JointTuningConfig(idle_ramps=False))
        self.assertTrue(suite.idle_ramp_mask.any(axis=0).all())
        np.testing.assert_allclose(suite.primary_requested[suite.idle_ramp_mask], 1)
        np.testing.assert_array_equal(suite.requested[~suite.idle_ramp_mask],
                                      baseline.requested[~suite.idle_ramp_mask])
        for p in suite.profiles:
            start, end = round(p["start_sec"] * 50), round(p["end_sec"] * 50)
            if p["mode"] in ("hold", "random"):
                self.assertFalse(suite.idle_ramp_mask[start:end].any())
                np.testing.assert_array_equal(suite.commands[start:end], baseline.commands[start:end])
            else:
                solo_end = start + round(p["solo_duration_sec"] * 50)
                self.assertFalse(suite.idle_ramp_mask[start:solo_end].any())
                np.testing.assert_array_equal(suite.commands[start:solo_end], baseline.commands[start:solo_end])
                target = p["channel"] - 1
                self.assertFalse(suite.idle_ramp_mask[start:end, target].any())
                np.testing.assert_array_equal(suite.commands[start:end, target], baseline.commands[start:end, target])

    def test_idle_ramps_reduce_open_fraction_without_extending_collection(self):
        baseline = JointTuningSuite(JointTuningConfig(idle_ramps=False))
        self.assertEqual(self.suite.duration, baseline.duration)
        for previous, current in zip(baseline.coverage(), self.suite.coverage()):
            self.assertGreater(current["idle_ramp_sec"], 300)
            self.assertLess(current["full_open_percent"], previous["full_open_percent"] - 30)
        for i in range(2):
            self.assertLessEqual(self.suite.max_low_seconds()[i], .2)

    def test_saved_types_preserve_both_command_sequences(self):
        suites = [JointTuningSuite.from_preset(name) for name in ("type1", "type2")]
        first, second = suites
        np.testing.assert_array_equal(first.commands,
            JointTuningSuite(JointTuningConfig(idle_ramps=False)).commands)
        np.testing.assert_array_equal(second.commands, self.suite.commands)
        self.assertEqual(first.config.freq, second.config.freq)
        self.assertEqual(first.config.seed, second.config.seed)
        self.assertFalse(first.config.idle_ramps)
        self.assertTrue(second.config.idle_ramps)
        for name, suite in zip(("type1", "type2"), suites):
            manifest = suite.manifest()
            self.assertEqual(manifest["collection_type"], name)
            self.assertEqual(manifest["saved_preset"]["contents"]["collection_type"], name)
            self.assertEqual(len(manifest["saved_preset"]["sha256"]), 64)
            self.assertEqual(suite.duration, 657)
            self.assertTrue(all(t <= .2 for t in suite.max_low_seconds()[:2]))

    def test_preview_exits_before_bridge_io(self):
        path = ROOT / "pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py"
        spec = importlib.util.spec_from_file_location("collection_runner_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        original = JointTuningSuite.write_preview
        def without_plot(suite, folder):
            return original(suite, folder, plot=False)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runner, "_resolve_tcpip_dir", side_effect=AssertionError("bridge access")), \
             patch.object(runner, "_write_json_atomic", side_effect=AssertionError("control write")), \
             patch.object(JointTuningSuite, "write_preview", without_plot):
            for name in ("type1", "type2"):
                destination = Path(folder) / name
                runner.main(["--type", name, "--preview", str(destination)])
                config = json.loads((destination / "config.json").read_text())
                self.assertEqual(config["duration_sec"], 657)
                self.assertEqual(config["sample_count"], 32850)
                self.assertEqual(config["collection_type"], name)
                self.assertEqual(config["commands_sha256"], JointTuningSuite.from_preset(name).manifest()["commands_sha256"])
                self.assertTrue((destination / "commands.csv").is_file())
                self.assertTrue((destination / "coverage.csv").is_file())

    def test_runner_publishes_full_suite_using_monotonic_time_and_saves(self):
        path = ROOT / "pneu_env/src/pneu_env/real/real_act_flowrate_6ctrl.py"
        spec = importlib.util.spec_from_file_location("collection_runner_loop_test", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        now = [0.0]
        latest = {}
        sent = []
        def sleep(seconds):
            now[0] += max(0, seconds)
        def write(path, payload):
            latest.clear()
            latest.update(payload)
            sent.append((now[0], [payload[k] for k in ("pos_ctrl", "neg_ctrl",
                "act_pos_ctrl1", "act_pos_ctrl2", "act_neg_ctrl1", "act_neg_ctrl2")]))
        def read(**kwargs):
            result = runner._initial_obs_state()
            result.update(latest)
            return result
        for name in ("type1", "type2"):
            now[0] = 0
            sent.clear()
            with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
                patches.enter_context(patch.object(runner, "_resolve_tcpip_dir", return_value=folder))
                patches.enter_context(patch.object(runner, "_write_json_atomic", side_effect=write))
                patches.enter_context(patch.object(runner, "_read_obs_state", side_effect=read))
                patches.enter_context(patch.object(runner, "get_pkg_path", return_value=folder))
                patches.enter_context(patch.object(runner.time, "perf_counter", side_effect=lambda: now[0]))
                patches.enter_context(patch.object(runner.time, "time", side_effect=lambda: 1000 + now[0]))
                patches.enter_context(patch.object(runner.time, "sleep", side_effect=sleep))
                patches.enter_context(patch("builtins.print"))
                runner.main(["--type", name])
                output = list((Path(folder) / "exp").glob("*.csv"))
                self.assertEqual(len(output), 1)
                self.assertIn(f"_{name}_", output[0].name)
                manifest = json.loads(output[0].with_suffix(".json").read_text())
                self.assertEqual(manifest["collection_type"], name)
                self.assertGreaterEqual(manifest["logged_rows"], 32850)
                self.assertEqual(manifest["config"]["supply_low_max_sec"], .2)
                np.testing.assert_allclose(sent[0][1], np.ones(6))
                np.testing.assert_allclose(sent[-1][1], np.ones(6))
                for channel in (0, 1):
                    started = None
                    for t, command in sent:
                        if command[channel] < .85 - 1e-12:
                            if started is None:
                                started = t
                        elif started is not None:
                            self.assertLessEqual(t - started, .2 + 1e-6)
                            started = None


if __name__ == "__main__":
    unittest.main()
