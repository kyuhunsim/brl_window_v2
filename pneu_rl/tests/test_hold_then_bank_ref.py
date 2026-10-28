"""Protocol checks for frozen hold+bank evaluation, without hardware I/O."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pneu_ref.hold_then_bank_ref import HoldThenBankRef
from pneu_ref.pneu_ref import PneuRef


class HoldThenBankReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "bank.npz"
        t = np.arange(2049, dtype=float) / 50.0
        pressure = np.stack(
            [np.c_[120 + .1*t, 75 - .2*t], np.c_[130 - .3*t, 65 + .1*t]]
        )
        np.savez(self.path, time=t, pressure=pressure)

    def make_ref(self, **kwargs):
        return HoldThenBankRef(trajectory_npz=self.path, **kwargs)

    def test_hold_bank_boundary_and_no_rebase(self):
        ref = self.make_ref(trajectory_index=1)
        self.assertEqual(ref.max_time, 60)
        for t in (-1, 0, 5, 19.98, 20):
            np.testing.assert_allclose(ref.get_goal(t), [130, 65], atol=1e-12)
        np.testing.assert_allclose(ref.get_goal(21), [129.7, 65.1], atol=1e-12)
        self.assertEqual(ref.get_phase(19.98), "hold")
        self.assertEqual(ref.get_phase(20), "bank")
        self.assertEqual(ref.get_phase(60), "end")
        # A preview reads future bank targets without changing current targets.
        ref.get_goal(30)
        np.testing.assert_allclose(ref.get_goal(19), [130, 65], atol=1e-12)

    def test_preview_clamps_at_requested_endpoint_not_source_tail(self):
        ref = self.make_ref(bank_seconds=40)
        expected = [124, 67]
        for t in (60, 60.32, 60.96, 61, 100):
            np.testing.assert_allclose(ref.get_goal(t), expected, atol=1e-12)
        self.assertGreater(ref.source_duration_seconds, ref.bank_seconds)

    def test_pinned_reset_and_hash_metadata(self):
        ref = self.make_ref(trajectory_index=1)
        for _ in range(3):
            ref.get_goal(59.5)
            ref.time_reset()
            np.testing.assert_allclose(ref.get_goal(0), [130, 65])
        metadata = ref.metadata()
        self.assertEqual(metadata["trajectory_index"], 1)
        self.assertEqual(metadata["hold_target_kpa"], [130, 65])
        self.assertFalse(metadata["pressure_rebasing"])
        self.assertEqual(metadata["trajectory_npz_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_existing_history_buffer_fills_during_hold_without_startup_jump(self):
        ref = self.make_ref()
        buf = PneuRef(ref, num_prev=10, num_pred=15, ctrl_freq=50)
        for i in range(1050):
            state_time = (i+1)/50.0
            window = buf.get_ref(i/50.0).reshape(26, 2)
            # Current slot becomes exact after warmup; during warmup the
            # reference is constant, so it is exact from the first step too.
            np.testing.assert_allclose(window[10], ref.get_goal(state_time), atol=1e-12)
            if i >= 25:
                times = state_time + np.arange(-10, 16)/50.0
                expected = np.asarray([ref.get_goal(t) for t in times])
                np.testing.assert_allclose(window, expected, atol=1e-12)
        buf.reset()
        np.testing.assert_allclose(buf.buf.reshape(26, 2), np.tile([120, 75], (26, 1)))

    def test_invalid_protocol_rejected(self):
        for kwargs in ({"hold_seconds": -1}, {"hold_seconds": float("nan")},
                       {"bank_seconds": 0}, {"bank_seconds": float("inf")},
                       {"bank_seconds": 41}, {"trajectory_index": 2},
                       {"trajectory_index": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make_ref(**kwargs)
        ref = self.make_ref()
        for t in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ref.get_goal(t)


if __name__ == "__main__":
    unittest.main()
