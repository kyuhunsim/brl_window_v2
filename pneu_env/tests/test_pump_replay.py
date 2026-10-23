"""Calibration-clock and optimizer-selection regression checks."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneu_env.pump_replay import (CHANNEL_NAMES, PumpReplayTuner, Trajectory,
                                  load_trajectory, replay)


class FakeSim:
    def reset(self, pressure, coeff):
        self.time = 0.
        self.controls = []

    def step(self, control, dt):
        self.controls.append(control.copy())
        self.time += dt
        return self.time, np.arange(10.) + self.time * 10.


class PumpReplayTests(unittest.TestCase):
    def test_physical_clock_command_changes_and_common_channel_alignment(self):
        times = np.array([0., .02, .04, .06])
        controls = np.repeat(np.arange(4.)[:, None] / 4., 6, axis=1)
        truth = np.arange(10.)[None] + times[:, None] * 10.
        data = Trajectory(Path("example.csv"), times, times + 4., controls, truth)
        sim = FakeSim()
        output_time, prediction, aligned = replay(sim, data, np.ones(2))
        np.testing.assert_allclose(output_time, times[1:])
        np.testing.assert_allclose(np.array(sim.controls), controls[:-1])
        np.testing.assert_allclose(prediction, aligned)

    def test_optimizer_coefficients_come_from_result_x_not_last_evaluation(self):
        tuner = PumpReplayTuner(None, [])
        expected = np.array([2., 3.])
        result = OptimizeResult(x=expected, fun=0., success=True)
        def minimize(objective, *args, **kwargs):
            objective(np.array([8., 9.]))
            return result
        with patch("pneu_env.pump_replay.optimize.minimize", minimize):
            tuner.tune()
        np.testing.assert_equal(tuner.params, expected)

    def test_replay_rejects_wrong_native_clock(self):
        class DelayedSim(FakeSim):
            def step(self, control, dt):
                time, values = super().step(control, dt)
                return time - .1, values
        time = np.array([0., .02])
        data = Trajectory(Path("example.csv"), time, time, np.zeros((2, 6)), np.zeros((2, 10)))
        with self.assertRaisesRegex(RuntimeError, "native time"):
            replay(DelayedSim(), data, np.ones(2))

    def test_loading_uses_last_duplicate_and_preserves_raw_signed_flow(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            columns = {"time": [4., 4.02, 4.04]}
            columns.update({name: [0., -.02, .03] for name in CHANNEL_NAMES})
            columns.update({f"ctrl{i}": [.8, .9, 1.] for i in range(1, 7)})
            frame = pd.DataFrame(columns)
            frame.to_csv(path, index=False)
            trajectory = load_trajectory(path, start_sec=4.02)
            np.testing.assert_allclose(trajectory.time, [0., .02])
            self.assertEqual(trajectory.values[0, 4], -.02)
            frame.loc[1, "time"] = 4.
            frame.to_csv(path, index=False)
            duplicate = load_trajectory(path)
            self.assertEqual(duplicate.duplicate_time_rows_dropped, 1)
            self.assertEqual(duplicate.values[0, 4], -.02)
            frame.loc[1, "time"] = 3.99
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "nondecreasing"):
                load_trajectory(path)


if __name__ == "__main__":
    unittest.main()
