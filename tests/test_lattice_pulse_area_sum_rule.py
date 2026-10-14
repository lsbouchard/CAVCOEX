from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lattice_gas_phase_switch import (  # noqa: E402
    K_ISING,
    exact_pair_cavity_parameters,
    exact_spontaneous_magnetization,
    phase_theory,
)
from lattice_pulse_area_sum_rule import (  # noqa: E402
    H_MAX,
    H_MIN,
    exact_area_targets,
    weighted_polynomial_fit,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_exact_sum_rule_and_finite_window_antiderivative() -> None:
    delta_rho = exact_spontaneous_magnetization(K_ISING)
    delta_beta_w = exact_pair_cavity_parameters(K_ISING)[2]
    targets = exact_area_targets(delta_rho, delta_beta_w)

    h_dense = np.linspace(H_MIN, H_MAX, 60_001)
    numerical_truncated = simpson(
        phase_theory(h_dense, delta_rho, delta_beta_w), x=h_dense
    )
    assert np.isclose(
        numerical_truncated, targets["truncated_area"], rtol=0.0, atol=2.0e-13
    )
    assert np.isclose(
        targets["full_area"], -delta_beta_w, rtol=0.0, atol=1.0e-15
    )
    assert np.isclose(
        targets["truncated_area"] + targets["omitted_tail_area"],
        targets["full_area"],
        rtol=0.0,
        atol=1.0e-14,
    )
    assert 0.0 < targets["omitted_tail_fraction_of_full_magnitude"] < 0.01


def test_weighted_exact_and_free_fits_recover_synthetic_sequence() -> None:
    inverse_volume = 1.0 / np.array([256.0, 576.0, 1024.0, 2304.0, 4096.0])
    exact_limit = -4.3
    slope = -31.5
    curvature = 920.0
    values = exact_limit + slope * inverse_volume + curvature * inverse_volume**2
    errors = np.linspace(0.001, 0.002, len(values))

    fixed = weighted_polynomial_fit(
        inverse_volume, values, errors, degree=2, exact_limit=exact_limit
    )
    assert np.isclose(fixed["B"], slope, rtol=0.0, atol=1.0e-10)
    assert np.isclose(fixed["C"], curvature, rtol=0.0, atol=1.0e-7)
    assert fixed["chi2"] < 1.0e-20

    free = weighted_polynomial_fit(
        inverse_volume, values, errors, degree=2, exact_limit=None
    )
    assert np.isclose(
        free["thermodynamic_limit"], exact_limit, rtol=0.0, atol=1.0e-12
    )
    assert np.isclose(free["B"], slope, rtol=0.0, atol=1.0e-9)
    assert np.isclose(free["C"], curvature, rtol=0.0, atol=1.0e-6)


def test_area_artifacts_are_complete_and_support_the_exact_limit() -> None:
    area_path = ROOT / "data_processed" / "lattice_pulse_area_sum_rule.csv"
    fit_path = ROOT / "data_processed" / "lattice_pulse_area_sum_rule_fits.csv"
    figure_path = ROOT / "figures" / "lattice_pulse_area_sum_rule.pdf"
    metadata_path = ROOT / "generated" / "lattice_pulse_area_sum_rule_metadata.json"
    areas = pd.read_csv(area_path)
    fits = pd.read_csv(fit_path)
    metadata = json.loads(metadata_path.read_text())

    assert list(areas["L"]) == [16, 24, 32, 48, 64]
    assert np.all(areas["replicas"] == 12)
    assert np.all(areas["blocks_per_replica"] == 40)
    assert np.all(areas["bootstrap_draws"] == metadata["bootstrap_draws"])
    assert np.all(areas["bootstrap_draws"] >= 2_000)
    assert np.all(areas["h_min"] == H_MIN)
    assert np.all(areas["h_max"] == H_MAX)
    assert np.all(areas["grid_points"] == 601)
    assert np.all(np.isfinite(areas["signed_area"]))
    assert np.all(areas["signed_area"] < 0.0)
    assert np.all(areas["se_signed_area"] > 0.0)
    assert np.all(np.diff(np.abs(areas["residual_from_exact_truncated_area"])) < 0.0)

    target = metadata["exact_area_targets"]["truncated_area"]
    full = metadata["exact_area_targets"]["full_area"]
    tail = metadata["exact_area_targets"]["omitted_tail_area"]
    assert np.allclose(areas["exact_truncated_area"], target, rtol=0.0, atol=1e-12)
    assert np.allclose(areas["exact_full_area"], full, rtol=0.0, atol=1e-12)
    assert np.isclose(target + tail, full, rtol=0.0, atol=1.0e-12)

    expected_models = {
        "fixed_linear_all",
        "fixed_quadratic_all",
        "fixed_linear_drop_L16",
        "free_linear_all",
        "free_quadratic_all",
    }
    assert set(fits["model"]) == expected_models
    assert np.all(np.isfinite(fits[["chi2", "chi2_p_value"]]))
    assert np.all(
        (fits["centered_hierarchical_bootstrap_p_value"] >= 0.0)
        & (fits["centered_hierarchical_bootstrap_p_value"] <= 1.0)
    )
    fixed_quadratic = fits.set_index("model").loc["fixed_quadratic_all"]
    free_quadratic = fits.set_index("model").loc["free_quadratic_all"]
    assert fixed_quadratic["dof"] > 0
    assert np.isfinite(free_quadratic["difference_from_exact_in_se"])
    assert figure_path.stat().st_size > 10_000

    # Keep the statistical execution hashes immutable; a subsequent
    # typography-only figure render has its own provenance.
    output_hashes = dict(metadata["output_sha256"])
    output_hashes.update(metadata.get("figure_render", {}).get("output_sha256", {}))
    for relative_path, expected_hash in output_hashes.items():
        assert _sha256(ROOT / relative_path) == expected_hash
