"""Pressure coverage and parameterization checks for batch calibration."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from retry_lib3_all_valves_feasible import Fit, anchors_for, area_ratio


def example_data(variable=True):
    n = 1000
    pin = np.repeat([150., 200., 250.], [300, 300, 400]) if variable else np.full(n, 101.325)
    return dict(Time=np.arange(n)*.02, dt=.02, I=np.full(n, .3), u=np.ones(n),
                State=np.zeros(n), P_in_abs=pin, P_out_abs=np.full(n, 101.),
                Phi=np.full(n, .6), Q=pin*.06)


class AllValvesFeasibleTests(unittest.TestCase):
    def test_full_open_bins_cover_variable_upstream_and_atmospheric_downstream(self):
        data = example_data()
        anchors = anchors_for(data, 15.)
        bins = [a for a in anchors if a["kind"] == "full_open_pressure_bin"]
        self.assertEqual({a["pin"] for a in bins}, {150., 200., 250.})
        self.assertTrue(all(a["pout"] == 101 for a in bins))

    def test_sparse_anchors_and_parameter_roundtrip_for_both_identifiability_cases(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                data = example_data(variable)
                fit = Fit(None, data, {}, dict(wn_bounds=[2., 25.], zeta_bounds=[.4, 1.2]))
                self.assertEqual(fit.dimension, 12 if variable else 11)
                means = np.array([data["Q"][a["ids"]].mean() for a in fit.anchors])
                np.testing.assert_allclose(fit.matrix@data["Q"], means)
                x = np.full(fit.dimension, .5)
                if variable:
                    x[11] = .6
                p = fit.parameters(x)
                np.testing.assert_allclose(fit.encode(p), x, atol=1e-12)
                mid = .7+.35*x[3]
                self.assertAlmostEqual(float(area_ratio(p, mid, fit.pin_ref, 0)), .5)

    def test_reverse_pressure_cannot_define_full_open_capacity(self):
        data = example_data()
        data["Phi"][:] = 0.
        with self.assertRaisesRegex(ValueError, "Insufficient full-open"):
            Fit(None, data, {}, dict(wn_bounds=[2., 25.], zeta_bounds=[.4, 1.2]))

    def test_pressure_dependent_opening_is_checked_at_both_ends(self):
        p = np.array([.3, 50., .45, .001, 0., 100., 75., 25., 1., 10., .6, 10., .6])
        result = area_ratio(p, 1., np.array([150., 200., 250.])[:, None], np.array([-1., 0., 1.]))
        self.assertEqual(result.shape, (3, 3))
        self.assertAlmostEqual(result[0, 1], .5)
        self.assertGreater(result[-1, 1], .99)


if __name__ == "__main__":
    unittest.main()
