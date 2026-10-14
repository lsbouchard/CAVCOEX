"""Pure bookkeeping controls for the nonequilibrium path diagnostic."""
import numpy as np
import pytest

from scripts.continuum_regeneration_diagnosis import local_number_transport, radius_excursions, weight_secants


def trace(labels, numbers, refresh=()):
    rows = np.zeros((len(labels), 12))
    rows[:, 0] = np.arange(1, len(labels) + 1)
    rows[:, 1] = labels
    rows[:, 2] = numbers
    rows[list(refresh), 11] = 1
    return rows


def test_retained_deleted_particle_and_ordinary_radius_return():
    rows = trace([0, 2, 1, 0], [4, 3, 3, 3])
    result = radius_excursions(rows, np.array([0, 1, 1, 0]), 0, 0, 2, [0, 1, 2])
    excursion = result['deletion_excursions'][0]
    assert result['tagged_deletions'] == 1
    assert excursion['return_mechanism'] == 'ordinary_radius'
    assert excursion['intermediate_radius_nodes_encountered'] == [1]
    assert excursion['N_immediately_after_tag'] == 3
    assert excursion['observed_N_change_since_pre_tag'] == -1


def test_reverse_tagged_return_is_not_a_retained_deletion():
    rows = trace([0, 2, 0], [4, 3, 4])
    result = radius_excursions(rows, np.array([0, 2, 0]), 0, 0, 2, [0, 1, 2])
    assert result['tagged_insertions'] == 1
    excursion = result['deletion_excursions'][0]
    assert excursion['return_mechanism'] == 'reverse_tagged'
    assert excursion['observed_N_change_since_pre_tag'] == 0


def test_compound_gibbs_base_entrance_uses_same_stage_N():
    rows = trace([0, 2, 2], [4, 3, 2])
    result = radius_excursions(rows, np.array([0, 0, 2]), 2, 0, 2, [0, 1, 2])
    first = result['deletion_excursions'][0]
    assert first['return_mechanism'] == 'Gibbs'
    assert first['end_clock'] == 2
    assert first['observed_N_at_end'] == 3


def test_refresh_stops_an_excursion_and_is_not_a_number_change():
    rows = trace([0, 2, 3, 0], [4, 3, 8, 5], refresh=[2])
    result = radius_excursions(rows, np.array([0, 2, 3, 0]), 0, 0, 2, [0, 1, 2])
    assert result['deletion_excursions'][0]['outcome'] == 'ideal_regeneration_before_base'
    changes = local_number_transport(rows, np.array([0, 2, 3, 0]), 0, 0, 2)
    assert changes['by_starting_particle_number'] == [
        {'N_before_local_sweep': 8, 'local_clock_rows': 1, 'positive_net_changes': 0,
         'negative_net_changes': 1, 'net_change_support': [-3]}]


def test_tag_change_removed_from_subsequent_local_number_difference():
    rows = trace([3, 0, 2, 0, 0], [5, 4, 3, 4, 5], refresh=[0])
    source = np.array([3, 0, 2, 0, 0])
    result = local_number_transport(rows, source, 0, 0, 2)
    by_n = {x['N_before_local_sweep']: x for x in result['by_starting_particle_number']}
    assert by_n[3]['net_change_support'] == [0]
    assert by_n[4]['net_change_support'] == [0, 1]


def test_weight_concavity_and_cap_scope_are_descriptive_only():
    target = {'radius': np.zeros(4), 'activity': np.arange(4.),
              'pair_penalty': np.array([0., np.inf, np.inf, np.inf]),
              'umbrella_stiffness': np.zeros(4), 'number_stiffness': np.zeros(4), 'cap': 8}
    good = weight_secants(target, np.array([0., -1., -4., -8.]))
    assert good['adjacent_neutral_particle_counts_without_degree'] == [3., 4.]
    assert good['concavity_violations'] == 0
    bad = weight_secants(target, np.array([0., -1., -10., -8.]))
    assert bad['concavity_violations'] == 1
    assert bad['out_of_cap_counts'] == 2


def test_misaligned_pre_neighbor_labels_are_rejected():
    with pytest.raises(ValueError, match='Pre-neighbor'):
        radius_excursions(trace([0, 2], [4, 3]), np.array([0]), 2, 0, 2, [0, 1, 2])
