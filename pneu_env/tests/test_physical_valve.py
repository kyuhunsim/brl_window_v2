"""Physical invariants tested against the actual lib3 translation unit.

No recorded trajectory is needed.  These checks target properties that an
in-sample flow RMSE or a numerical parity test cannot establish.
"""
from pathlib import Path
import ctypes
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PTR = ctypes.POINTER(ctypes.c_double)
ATM = 101.325
WRAPPER = r'''
#include "pneumatic_CT.cpp"
extern "C" void inspect_derivative(const double* x, const double* u,
    const double* initial_runtime, double* dx, double* flow,
    double* after_runtime, double* repeated_dx) {
    const auto old = std::cout.rdstate();
    std::cout.setstate(std::ios_base::failbit);
    PneumaticCT model;
    std::cout.clear(old);
    model.set_valve_runtime_states(initial_runtime);
    model.model(x, u, dx, 1);
    const double* actual = model.get_mass_flowrate();
    for (int i=0;i<10;++i) flow[i]=actual[i];
    model.get_valve_runtime_states(after_runtime);
    model.model(x, u, repeated_dx, 1);
}
extern "C" double inspect_valve(int valve, double pin, double pout,
    const double* initial_runtime, double* debug) {
    const auto old = std::cout.rdstate();
    std::cout.setstate(std::ios_base::failbit);
    PneumaticCT model;
    std::cout.clear(old);
    model.set_valve_runtime_states(initial_runtime);
    const double result=model.solenoid_valve(pin,pout,1.,valve,1.);
    const double* actual=model.get_valve_debug()+(valve-1)*9;
    for(int i=0;i<9;++i) debug[i]=actual[i];
    return result;
}
extern "C" void inspect_opening(int n, const double* x, const double* u,
    const double* initial_runtime, double dt, double* final_runtime) {
    const auto old = std::cout.rdstate();
    std::cout.setstate(std::ios_base::failbit);
    PneumaticCT model;
    std::cout.clear(old);
    model.set_valve_runtime_states(initial_runtime);
    for(int i=0;i<n;++i) model.prepare_valve_openings(x,u,dt);
    model.get_valve_runtime_states(final_runtime);
}
'''


class PhysicalValveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="test_physical_valve_")
        directory = Path(cls.temporary.name)
        source = ROOT / "pneu_env/src/pneu_env/lib3"
        header = ROOT.parent / "pneu_tune/exp/lib3_feasible_integration_20261022/type1/valve_parameters.h"
        wrapper = directory / "wrapper.cpp"
        wrapper.write_text(WRAPPER)
        library = directory / "test.so"
        command = ["g++", "-std=c++17", "-O3", "-shared", "-fPIC",
                   "-DPNEU_EXACT_VALVE_HYSTERESIS", "-DPNEU_OPENING_DYNAMICS",
                   f'-DPNEU_VALVE_PARAMS_HEADER="{header}"',
                   "-I", str(source), str(wrapper),
                   str(source / "pneumatic_simulator.cpp"), "-o", str(library)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.inspect_derivative.argtypes = [PTR]*7
        cls.lib.inspect_derivative.restype = None
        cls.lib.inspect_valve.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, PTR, PTR]
        cls.lib.inspect_valve.restype = ctypes.c_double
        cls.lib.inspect_opening.argtypes = [ctypes.c_int, PTR, PTR, PTR, ctypes.c_double, PTR]
        cls.lib.inspect_opening.restype = None
        cls.lib.get_augmented_state_schema_c.restype = ctypes.c_int
        cls.lib.get_augmented_state_c.argtypes = [PTR]
        cls.lib.set_augmented_state_checked_c.argtypes = [PTR, ctypes.c_int]
        cls.lib.set_augmented_state_checked_c.restype = ctypes.c_int
        cls.lib.set_init_env_act_c.argtypes = [ctypes.c_double] * 4

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def runtime():
        # Every valve is already open, rather than resetting a flow lag to zero
        # and accidentally passing the zero-head test through initialization.
        value = np.zeros((6, 5))
        value[:, 1] = .5
        value[:, 3] = .3
        value[:, 4] = 1.
        return value.ravel()

    def derivative(self, x, control=None):
        x = np.ascontiguousarray(x, dtype=float)
        control = np.ones(6) if control is None else np.asarray(control, dtype=float)
        runtime = self.runtime()
        outputs = [np.full(n, np.nan) for n in (8, 10, 30, 8)]
        self.lib.inspect_derivative(*[a.ctypes.data_as(PTR) for a in
                                    (x, control, runtime, *outputs)])
        return runtime, outputs

    def test_runge_kutta_derivative_does_not_advance_persistent_valve_state(self):
        x = [0., 200., 40., .8, 250., 30., 150., 70.]
        runtime, (dx, flows, after, again) = self.derivative(x)
        np.testing.assert_array_equal(after, runtime)
        np.testing.assert_array_equal(again, dx)
        self.assertTrue(np.isfinite(flows).all())

    def test_all_four_fixed_volume_nodes_obey_mass_balance(self):
        rng = np.random.default_rng(409)
        factor = 1000*.287
        volumes = [.75, .4, .100, .156]
        temperatures = [293.15, 293.15, 323.15, 323.15]
        for _ in range(20):
            x = np.r_[0., rng.uniform(120, 350), rng.uniform(20, 80),
                      rng.uniform(0, 2*np.pi), rng.uniform(20, 350, 2),
                      rng.uniform(85, 290), rng.uniform(25, 110)]
            _, (dx, flow, _, _) = self.derivative(x)
            observed = dx[[1, 2, 6, 7]] * volumes/(factor*np.array(temperatures))
            expected = np.array([flow[0]+flow[2]-flow[4]-flow[6],
                                 -flow[1]-flow[3]+flow[5]+flow[9],
                                 flow[6]-flow[7], flow[8]-flow[9]])
            np.testing.assert_allclose(observed, expected, rtol=5e-13, atol=1e-18)

    def test_previously_open_valves_have_no_flow_at_equal_or_reverse_head(self):
        runtime = self.runtime()
        debug = np.empty(9)
        for valve in range(1, 7):
            for pin, pout in ((101.325, 101.325), (60., 90.), (1e-6, 100.)):
                with self.subTest(valve=valve, pin=pin, pout=pout):
                    flow = self.lib.inspect_valve(valve, pin, pout,
                             runtime.ctypes.data_as(PTR), debug.ctypes.data_as(PTR))
                    self.assertEqual(flow, 0.)
                    self.assertEqual(debug[7], 0.)

    def test_atmospheric_actuator_cannot_be_exhausted_below_atmosphere(self):
        x = [0., 200., 40., .8, 250., 30., ATM, 70.]
        _, (dx, flow, _, _) = self.derivative(x)
        self.assertEqual(flow[7], 0.)
        self.assertGreaterEqual(dx[6], 0.)

    def test_atmospheric_negative_actuator_cannot_be_filled_from_atmosphere(self):
        x = [0., 200., 40., .8, 250., 30., 150., ATM]
        _, (dx, flow, _, _) = self.derivative(x)
        self.assertEqual(flow[8], 0.)
        self.assertLessEqual(dx[7], 0.)

    def opening(self, runtime, command, dt, repeats=1):
        x = np.array([0., 200., 40., .8, 250., 30., 150., 70.])
        command = np.ascontiguousarray(command, dtype=float)
        runtime = np.ascontiguousarray(runtime, dtype=float)
        final = np.empty(30)
        self.lib.inspect_opening(repeats, x.ctypes.data_as(PTR), command.ctypes.data_as(PTR),
             runtime.ctypes.data_as(PTR), dt, final.ctypes.data_as(PTR))
        return final

    def test_area_response_has_no_overshoot_and_is_consistent_under_subdivision(self):
        initial = self.runtime()
        command = np.ones(6)
        for dt in (1e-6, .001, .02, 1., 100.):
            final = self.opening(initial, command, dt).reshape(6, 5)
            target = final[:, 2]
            self.assertTrue(np.all(final[:, 1] >= np.minimum(.5, target)-1e-14))
            self.assertTrue(np.all(final[:, 1] <= np.maximum(.5, target)+1e-14))
        coarse = self.opening(initial, command, .2)
        fine = self.opening(initial, command, .02, repeats=10)
        np.testing.assert_allclose(coarse, fine, rtol=1e-13, atol=1e-14)

    def test_repeated_command_reversals_preserve_area_and_hysteresis_bounds(self):
        state = self.runtime()
        rng = np.random.default_rng(46)
        for _ in range(1000):
            state = self.opening(state, rng.uniform(.5, 1., 6), rng.uniform(.00001, .2))
            valve = state.reshape(6, 5)
            self.assertTrue(np.isfinite(valve).all())
            self.assertTrue(np.all(np.abs(valve[:, 0]) <= 1.+1e-12))
            self.assertTrue(np.all((valve[:, 1] >= 0.) & (valve[:, 1] <= 1.)))

    def test_snapshot_requires_opening_schema_and_rejects_invalid_state_without_mutation(self):
        self.lib.set_init_env_act_c(200., 40., 150., 70.)
        original = np.empty(37)
        self.lib.get_augmented_state_c(original.ctypes.data_as(PTR))
        self.assertEqual(self.lib.get_augmented_state_schema_c(), 2)
        self.assertFalse(hasattr(self.lib, "set_augmented_state_c"))
        self.assertEqual(self.lib.set_augmented_state_checked_c(original.ctypes.data_as(PTR), 1), 1)
        for index, value, status in ((8, 1.1, 4), (7, 1.1, 4), (0, -1., 3), (0, np.nan, 2)):
            bad = original.copy()
            bad[index] = value
            self.assertEqual(self.lib.set_augmented_state_checked_c(bad.ctypes.data_as(PTR), 2), status)
            after = np.empty(37)
            self.lib.get_augmented_state_c(after.ctypes.data_as(PTR))
            np.testing.assert_array_equal(after, original)
        self.assertEqual(self.lib.set_augmented_state_checked_c(original.ctypes.data_as(PTR), 2), 0)

    def test_opening_cannot_generate_flow_against_the_pressure_head(self):
        rng = np.random.default_rng(204)
        runtime, debug = self.runtime(), np.empty(9)
        for _ in range(100):
            pin, pout = rng.uniform(10., 400., 2)
            for valve in range(1, 7):
                flow = self.lib.inspect_valve(valve, pin, pout,
                         runtime.ctypes.data_as(PTR), debug.ctypes.data_as(PTR))
                self.assertGreaterEqual(flow * (pin-pout), 0.)


if __name__ == "__main__":
    unittest.main()
