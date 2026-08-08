import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hard_sphere_models import (  # noqa: E402
    tonks_grand_distribution,
    tonks_insertion_probability,
)


def test_tonks_n_zero_insertion_limit() -> None:
    np.testing.assert_allclose(
        tonks_insertion_probability(0, 10.0), 1.0, atol=3.0e-15
    )


def test_tonks_exact_grand_insertion_identity() -> None:
    for length, x in ((20.0, 0.0), (20.0, 2.0), (50.0, 1.0)):
        grand = tonks_grand_distribution(x, length)
        insertion = np.asarray(
            [
                tonks_insertion_probability(int(n), length)
                for n in grand.n
            ]
        )
        mixture = float(np.dot(grand.probability, insertion))
        exact = grand.mean / length * np.exp(-x)
        np.testing.assert_allclose(mixture, exact, rtol=5.0e-15, atol=5.0e-15)


def test_tonks_distribution_is_normalized() -> None:
    grand = tonks_grand_distribution(1.7, 31.25)
    np.testing.assert_allclose(grand.probability.sum(), 1.0, atol=2.0e-15)
