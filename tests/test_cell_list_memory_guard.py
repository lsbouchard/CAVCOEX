"""Reject pathological dense-cell allocations before requesting their memory."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cavity_mc import _build_cell_list


def test_huge_cell_count_fails_before_allocation_or_integer_overflow():
    with pytest.raises(ValueError, match="exceeds 128 MiB"):
        _build_cell_list(np.zeros((1, 3)), 10.0, 1_000_000_000)


def test_nonpositive_cell_count_is_rejected():
    with pytest.raises(ValueError, match="n_cells must be positive"):
        _build_cell_list(np.zeros((1, 3)), 10.0, 0)


def test_small_valid_cell_list_is_unchanged():
    counts, members = _build_cell_list(np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]]), 2.0, 2)
    assert counts.sum() == 2
    assert counts[0] == counts[7] == 1
    assert members[0, 0] == 0
    assert members[7, 0] == 1
