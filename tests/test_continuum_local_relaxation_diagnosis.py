import numpy as np
import pytest

from scripts.continuum_local_relaxation_diagnosis import spell_lengths, spell_summary


def test_spells_retain_singletons_and_final_partial_spell():
    assert spell_lengths([2,2,3,2,2,2,4]).tolist() == [2,1,3,1]
    summary = spell_summary([2,2,3,2,2,2,4])
    assert summary['spells'] == 4
    assert summary['one_clock_fraction'] == .5
    assert summary['maximum_clocks'] == 3


def test_empty_spells_are_unavailable_not_zero_duration():
    assert len(spell_lengths([])) == 0
    assert spell_summary([])['mean_clocks'] is None


def test_multidimensional_labels_rejected():
    with pytest.raises(ValueError, match='one dimensional'):
        spell_lengths(np.ones((2,2)))
