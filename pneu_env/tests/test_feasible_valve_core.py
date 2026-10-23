"""Numerical checks for the isolated calibration core, without hardware I/O."""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import unittest
from scipy.integrate import solve_ivp


class FeasibleValveCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/"scripts/retry_lib3_valve_feasible.py"
        spec = importlib.util.spec_from_file_location("feasible_pilot", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        library, _, _ = module.build_core()
        library.advance_h_feasible.argtypes = [ctypes.c_double]*4
        library.advance_h_feasible.restype = ctypes.c_double
        cls.core = library

    def test_exact_hysteresis_matches_independent_ode(self):
        cases = [(-.9, .12, 390., .999), (.9, -.12, 912., .9),
                 (-.5, .12, 30., .5), (.5, -.12, 10., .75)]
        for initial, delta, rate, fraction in cases:
            with self.subTest(initial=initial, rate=rate):
                sign = np.sign(delta)
                def rhs(_, h):
                    return rate*(sign-fraction*h-(1-fraction)*sign*abs(h))
                reference = solve_ivp(rhs, (0, abs(delta)), [initial],
                                      rtol=1e-11, atol=1e-12).y[0, -1]
                result = self.core.advance_h_feasible(initial, delta, rate, fraction)
                self.assertLess(abs(result-reference), 2e-9)
                self.assertLessEqual(abs(result), 1+1e-12)

    def test_hysteresis_is_invariant_to_monotone_command_subdivision(self):
        coarse = self.core.advance_h_feasible(-.9, .12, 912., .75)
        fine = -.9
        for _ in range(100):
            fine = self.core.advance_h_feasible(fine, .0012, 912., .75)
        self.assertLess(abs(coarse-fine), 1e-12)

    def test_hysteresis_stays_bounded_for_repeated_large_reversals(self):
        h = 0.
        for index in range(1000):
            h = self.core.advance_h_feasible(h, .12 if index % 2 else -.12, 2000., .5)
            self.assertLessEqual(abs(h), 1+1e-12)

    def test_startup_does_not_depend_on_later_measured_flow_mean(self):
        ptr = ctypes.POINTER(ctypes.c_double)
        n = 100
        current = np.full(n, .3)
        direction = np.ones(n)
        pin = np.full(n, 101.325)
        phi = np.full(n, .68)
        parameters = np.array([.32, 50., .2, 0., .01, 100., 75., 25., 1.,
                               15., .6, 15., .6])
        outputs = []
        for tail in (1., 20.):
            measured = np.r_[np.full(10, 20.), np.full(n-10, tail)]
            buffers = [np.empty(n) for _ in range(4)]
            status = self.core.simulate_feasible(n, *[a.ctypes.data_as(ptr) for a in
                (current, direction, pin, phi, measured)], .02,
                parameters.ctypes.data_as(ptr),
                *[a.ctypes.data_as(ptr) for a in buffers])
            self.assertEqual(status, 0)
            outputs.append(buffers[0])
        self.assertTrue(np.array_equal(*outputs))


if __name__ == "__main__":
    unittest.main()
