import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cavity_mc import box_length, fcc_positions  # noqa: E402
from solid_cavity_mc import (  # noqa: E402
    FCC_BRAGG_CRYSTAL_MINIMUM,
    LINDEMANN_MAXIMUM,
    Q6_CRYSTAL_MINIMUM,
    SOLID_RADII,
    lattice_displacements,
    lindemann_ratio,
    load_solid_coexistence_density,
    structure_gate,
    structure_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def test_solid_radius_scan_is_sorted_and_resolves_the_rare_event_onset() -> None:
    assert np.all(np.diff(SOLID_RADII) > 0.0)
    assert np.isclose(SOLID_RADII[0], 0.50)
    assert np.isclose(SOLID_RADII[-1], 1.00)
    assert len(SOLID_RADII) == 11


def test_published_solid_density_conversion() -> None:
    source = load_solid_coexistence_density(
        ROOT / "data_raw" / "fernandez2012_coexistence.csv"
    )
    assert np.isclose(source["specific_volume"], 0.96405)
    assert np.isclose(source["eta"], np.pi / (6.0 * 0.96405))
    assert 0.5430 < source["eta"] < 0.5432
    assert source["se_eta"] > 0.0


def test_lattice_displacement_removes_periodic_global_translation() -> None:
    n_particles = 108
    length = box_length(n_particles, 0.543)
    reference = fcc_positions(n_particles, length)
    shifted = np.mod(reference + np.asarray([0.31, -0.22, 0.17]), length)
    displacement = lattice_displacements(shifted, reference, length)
    assert np.max(np.abs(displacement)) < 1.0e-12
    rms, maximum = lindemann_ratio(
        shifted, reference, length, n_particles
    )
    assert rms < 1.0e-12
    assert maximum < 1.0e-12


def test_perfect_fcc_passes_predeclared_crystal_gates() -> None:
    n_particles = 108
    length = box_length(n_particles, 0.543)
    reference = fcc_positions(n_particles, length)
    snapshot = structure_snapshot(
        reference.copy(), reference, length, n_particles
    )
    assert snapshot["global_Q6"] > 0.55
    assert snapshot["max_FCC_Bragg"] > 0.99
    assert snapshot["lindemann_ratio"] < 1.0e-12
    frame = pd.DataFrame([snapshot, snapshot])
    passed, reasons = structure_gate(frame)
    assert passed
    assert not reasons


def test_crystal_gate_rejects_each_independent_failure_mode() -> None:
    good = {
        "global_Q6": Q6_CRYSTAL_MINIMUM + 0.1,
        "max_FCC_Bragg": FCC_BRAGG_CRYSTAL_MINIMUM + 0.1,
        "lindemann_ratio": LINDEMANN_MAXIMUM - 0.1,
        "minimum_pair_distance": 1.0,
    }
    cases = [
        ({**good, "global_Q6": Q6_CRYSTAL_MINIMUM - 0.01}, "Q6"),
        (
            {
                **good,
                "max_FCC_Bragg": FCC_BRAGG_CRYSTAL_MINIMUM - 0.01,
            },
            "Bragg",
        ),
        (
            {**good, "lindemann_ratio": LINDEMANN_MAXIMUM + 0.01},
            "Lindemann",
        ),
        ({**good, "minimum_pair_distance": 0.999}, "overlap"),
    ]
    for bad, label in cases:
        passed, reasons = structure_gate(pd.DataFrame([good, bad]))
        assert not passed
        assert any(label in reason for reason in reasons)
