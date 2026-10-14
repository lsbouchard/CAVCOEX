"""Independent Legendre-addition-theorem geometry, not phase-sampling evidence."""
import itertools

import numpy as np
import pytest
from scipy.special import eval_legendre

from scripts.continuum_tempering_structure import snapshot_control


def _global_q6_from_bond_dot_products(host, length, cutoff):
    """Q6^2 = mean_{directed bonds a,b} P6(unit_a dot unit_b)."""
    differences = host[:, None] - host[None, :]
    differences -= length * np.rint(differences / length)
    distances = np.linalg.norm(differences, axis=2)
    keep = (distances > 0) & (distances < cutoff)
    if not np.any(keep):
        return 0.
    units = differences[keep] / distances[keep, None]
    return float(np.sqrt(max(0., eval_legendre(6, np.clip(units @ units.T, -1., 1.)).mean())))


def _independent_fourier(host, length):
    if not len(host):
        return 0.
    waves = 2 * np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                          [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    return float(abs(np.exp(2j * np.pi * (host @ waves.T) / length).sum(axis=0)).mean() / len(host))


def _frame(number):
    length = (32 * .96405) ** (1 / 3)
    basis = np.array([[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
    full = np.array([(np.array(cell) + point) * length / 2
                     for cell in itertools.product(range(2), repeat=3) for point in basis])
    host = full[:number]
    positions = np.full((80, 3), np.nan)
    positions[:number] = host
    center = np.array([.1, .2, .3])
    row = np.zeros(12)
    row[2], row[3] = number, _independent_fourier(host, length)
    deleted_b = [_independent_fourier(np.delete(host, tag, axis=0), length)
                 for tag in range(number)]
    row[8] = sum(value > .45 for value in deleted_b) - number * (row[3] > .45)
    difference = host - center
    difference -= length * np.rint(difference / length)
    row[7] = np.linalg.norm(difference, axis=1).min()
    return host, positions, center, row, length


@pytest.mark.parametrize("number", [19, 20, 21])
def test_q6_and_delete_one_minimum_count_rule_match_independent_addition_theorem(number):
    host, positions, center, row, length = _frame(number)
    result = snapshot_control(positions, center, number, row, length, 1., 0., 2,
                              1.4, (.35,), 20)
    independent = _global_q6_from_bond_dot_products(host, length, 1.4)
    assert independent > .35
    assert result["Q6_global"] == pytest.approx(independent, abs=2e-14)
    deleted_order = [_global_q6_from_bond_dot_products(np.delete(host, tag, axis=0), length, 1.4)
                     for tag in range(number)]
    expected_current = number >= 20 and independent > .35
    expected_shift = sum(number - 1 >= 20 and value > .35 for value in deleted_order)
    expected_shift -= number * expected_current
    label = result["Q6_labels"]["0.35"]
    assert label["ordered"] == expected_current
    assert label["signed_delete_one_label_change"] == expected_shift
    if number == 20:
        assert all(value > .35 for value in deleted_order)
        assert expected_shift == -20  # A count change, not a loss of local crystal geometry.
    else:
        assert expected_shift == 0
