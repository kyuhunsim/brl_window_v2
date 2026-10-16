import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from pneu_env.real.real_act import PneuRealAct


class CommandTimingTests(unittest.TestCase):
    def test_publications_are_spaced_and_late_steps_do_not_catch_up(self):
        backend = PneuRealAct(freq=50)
        clock = [10.0]
        def sleep(duration):
            clock[0] += duration
        with patch('pneu_env.real.real_act.time.monotonic', side_effect=lambda: clock[0]), patch('pneu_env.real.real_act.time.sleep', side_effect=sleep):
            backend.wait()
            backend.wait()
            self.assertAlmostEqual(clock[0], 10.02)
            clock[0] = 10.1
            backend.wait()
            backend.wait()
            self.assertAlmostEqual(clock[0], 10.12)

    def test_old_session_timestamp_is_rejected(self):
        backend = PneuRealAct(freq=50)
        backend.obs_wait_timeout = .005
        backend._sent_time = .1
        with tempfile.TemporaryDirectory() as directory:
            backend.labview_path = directory
            Path(directory, 'obs_act.json').write_text(json.dumps({'time': 40, 'time_ctrl_sent': 40}))
            with self.assertRaises(TimeoutError):
                backend.read_obs_file()

    def test_current_command_accepted_after_clock_reset(self):
        backend = PneuRealAct(freq=50)
        backend.time = 40
        backend._sent_time = .1
        with tempfile.TemporaryDirectory() as directory:
            backend.labview_path = directory
            Path(directory, 'obs_act.json').write_text(json.dumps({'time': .1, 'time_ctrl_sent': .1, 'pos_press': 150}))
            backend.read_obs_file()
            self.assertEqual(backend.time, .1)
            self.assertEqual(backend.pos_press, 150)

    def test_recorder_saves_partial_data_and_opens_on_error(self):
        import importlib.util
        script = Path(__file__).resolve().parents[1] / 'scripts/record_all_open.py'
        spec = importlib.util.spec_from_file_location('record_all_open_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class FakeBackend:
            _sent_time = .02
            calls = 0
            opened = False
            def observe(self, ctrl, goal):
                self.calls += 1
                if self.calls == 2:
                    raise TimeoutError('test bridge stopped')
                return None, {'Observation': {'time': .02, 'pos_press': 150}}
            def write_ctrl_file(self):
                self.opened = all(getattr(self, k) == 1 for k in module.CONTROL_KEYS)
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'record.csv'
            with patch.object(module, 'PneuRealAct', return_value=backend):
                with self.assertRaises(TimeoutError):
                    module.record(60, output)
            self.assertTrue(backend.opened)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                module.record(60, output)
