"""Unit tests for the N3 hard-sphere EOS benchmark."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cs_benchmark as benchmark  # noqa: E402


class EOSBenchmarkTests(unittest.TestCase):
    def test_symbolic_thermodynamic_integrals(self) -> None:
        for model in benchmark.MODELS:
            residual = sp.cancel(
                sp.diff(model.a_ex_expr, benchmark.ETA)
                - (model.z_expr - 1) / benchmark.ETA
            )
            self.assertEqual(residual, 0, model.key)

    def test_analytic_monotonicity_certificates(self) -> None:
        checks = benchmark.symbolic_consistency_checks()
        self.assertEqual(checks["mKLM_sturm_roots_on_x_0_to_13_over_12"], 0)

    def test_random_symbolic_derivatives_against_finite_differences(self) -> None:
        errors = benchmark.random_finite_difference_checks(samples=32, seed=7319)
        self.assertTrue(errors)
        for name, error in errors.items():
            self.assertLess(error, 1.0e-8, name)

    def test_dense_grid_monotonicity_and_bounds(self) -> None:
        eta = np.linspace(0.0, benchmark.ETA_MAX, 2001)
        checks = benchmark.numerical_checks(eta)
        for name, values in checks.items():
            self.assertGreater(values["min_total_beta_mu_slope_eta_gt_0"], 0, name)
            self.assertGreater(values["min_beta_mu_ex_slope_eta_gt_0"], 0, name)
            self.assertGreater(values["min_S0"], 0, name)
            self.assertLess(values["max_dW_dmu"], 1, name)

    def test_cs_matches_manuscript_closed_forms(self) -> None:
        eta = benchmark.ETA
        cs = benchmark.MODELS[0]
        expected_mu = (8 * eta - 9 * eta**2 + 3 * eta**3) / (1 - eta) ** 3
        expected_s0 = (1 - eta) ** 4 / (
            1 + 4 * eta + 4 * eta**2 - 4 * eta**3 + eta**4
        )
        self.assertEqual(sp.cancel(cs.mu_ex_expr - expected_mu), 0)
        self.assertEqual(sp.cancel(cs.s0_expr - expected_s0), 0)

    def test_coexistence_marker_propagation(self) -> None:
        expected = np.pi / (6 * benchmark.V_FLUID)
        uncertainty = expected * benchmark.V_FLUID_STD / benchmark.V_FLUID
        self.assertAlmostEqual(benchmark.ETA_FLUID, expected, places=15)
        self.assertAlmostEqual(benchmark.ETA_FLUID_STD, uncertainty, places=15)


if __name__ == "__main__":
    unittest.main()
