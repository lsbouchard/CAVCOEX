import numpy as np
import pytest

from scripts import continuum_single_walker_gates as gates


def target():
    return {'diameter': np.ones(4), 'radius': np.array([0., 0., 1., 0.]),
        'activity': np.array([0., 1., 1., -1.]),
        'pair_penalty': np.array([0., np.inf, np.inf, np.inf]),
        'umbrella_stiffness': np.zeros(4), 'umbrella_preferred': np.zeros(4),
        'number_stiffness': np.zeros(4), 'number_preferred': np.zeros(4),
        'base_node': 1, 'cavity_node': 2, 'ideal_node': 0,
        'volume': 4., 'cap': 5, 'log_weights': np.array([0., -1., 0., 0.]),
        'exchange_edges': np.array([[0, 3], [3, 1], [1, 2]])}


def test_activity_gate_degree_and_number_are_exact():
    data = target()
    degree = gates.neighbor_degrees(data['exchange_edges'], 4)
    n = np.array([1., 2., 3.])
    probability, support, score, tagged = gates.directional_score(
        data, 1, 3, n, np.zeros(3), np.zeros(3), np.ones(3), degree)
    np.testing.assert_allclose(score, 1. - 2. * n)
    np.testing.assert_allclose(probability, np.exp(1. - 2. * n))
    assert support.all() and not tagged


def test_tagged_edge_has_single_number_factor_and_degree():
    data = target()
    degree = gates.neighbor_degrees(data['exchange_edges'], 4)
    n = np.array([0., 2., 5.])
    probability, support, score, tagged = gates.directional_score(
        data, 1, 2, n, np.zeros(3), np.zeros(3), np.ones(3), degree)
    np.testing.assert_allclose(score[1:], np.log(n[1:] / 2.))
    np.testing.assert_allclose(probability, [0., 1., 1.])
    assert tagged and support.tolist() == [False, True, True]
    reverse, valid, reverse_score, _ = gates.directional_score(
        data, 2, 1, np.array([1., 5.]), np.zeros(2), np.zeros(2), np.ones(2), degree)
    assert valid.tolist() == [True, False]
    assert reverse[1] == 0.
    assert reverse_score[0] == pytest.approx(0.)


def test_empty_host_cavity_support_and_hard_overlap_rejection():
    data = target()
    data['radius'][3] = 10.
    degree = gates.neighbor_degrees(data['exchange_edges'], 4)
    p, support, _, _ = gates.directional_score(data, 0, 3, np.array([0., 1., 2.]),
        np.zeros(3), np.array([0., 0., 1.]), np.ones(3), degree)
    assert support.tolist() == [True, False, False]
    assert p[0] > 0 and not p[1:].any()


def test_own_support_violation_is_rejected():
    data = target()
    degree = gates.neighbor_degrees(data['exchange_edges'], 4)
    with pytest.raises(ValueError, match='current target support'):
        gates.directional_score(data, 1, 3, np.array([2.]), np.zeros(1),
                                np.ones(1), np.ones(1), degree)


@pytest.mark.parametrize('name', ['pair_penalty', 'number_stiffness', 'umbrella_stiffness'])
def test_tagged_edge_rejects_biased_physical_target(name):
    data = target()
    data[name][1] = .7
    degree = gates.neighbor_degrees(data['exchange_edges'], 4)
    with pytest.raises(ValueError, match='unbiased and hard'):
        gates.directional_score(data, 1, 2, np.array([2.]), np.zeros(1),
                                np.zeros(1), np.ones(1), degree)


def test_full_clock_observed_flows_and_last_label_are_retained():
    data = target()
    samples = np.zeros((4, 12))
    samples[:, 0] = np.arange(1, 5)
    samples[:, 1] = [3, 1, 2, 1]
    samples[:, 2] = [2, 2, 1, 2]
    samples[:, 7] = 2.
    result = gates.summarize_gate(data, samples, final_label=3)
    edges = {(r['source'], r['destination']): r for r in result['directional_edges']}
    assert edges[3, 1]['observed_accepted_hops'] == 1
    assert edges[1, 2]['observed_accepted_hops'] == 1
    assert edges[2, 1]['observed_accepted_hops'] == 1
    assert edges[1, 3]['observed_accepted_hops'] == 1
    assert edges[1, 3]['activity_edge_neutral_MH_particle_count'] == pytest.approx(.5)
    expected = np.exp(-3.) * .25 * 2
    assert edges[1, 3]['expected_accepted_hops'] == pytest.approx(expected)
    no_final = gates.summarize_gate(data, samples)
    old = next(r for r in no_final['directional_edges'] if (r['source'], r['destination']) == (1, 3))
    assert old['expected_accepted_hops'] == pytest.approx(expected / 2)
    assert old['observed_accepted_hops'] == 0


def test_non_edge_transition_and_sparse_clock_are_rejected():
    data = target()
    samples = np.zeros((2, 12))
    samples[:, 0] = [1, 2]
    samples[:, 1] = [0, 2]
    with pytest.raises(ValueError, match='not a graph edge'):
        gates.summarize_gate(data, samples)
    samples[:, 0] = [1, 3]
    with pytest.raises(ValueError, match='consecutive clock'):
        gates.summarize_gate(data, samples)


def test_dilute_high_bragg_is_not_called_dense_high_bragg():
    assert gates.bragg_strata(np.array([1, 28, 31]), np.array([1., .2, .8])).tolist() == [
        'intermediate_or_dilute', 'low_Bragg', 'high_Bragg']


def test_duplicate_and_isolated_graph_nodes_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        gates.neighbor_degrees(np.array([[0, 1], [1, 0]]), 2)
    with pytest.raises(ValueError, match='neighbor'):
        gates.neighbor_degrees(np.array([[0, 1]]), 3)


def test_exact_global_label_conditional_has_no_degree_and_respects_support():
    data = target()
    p = gates.conditional_label_probabilities(data, np.array([2.]), np.zeros(1),
                                             np.zeros(1), np.array([.5]))[0]
    logits = np.array([0., 1., -np.inf, -2.])
    expected = np.exp(logits - 1.)
    expected /= expected.sum()
    np.testing.assert_allclose(p, expected)
    assert p[2] == 0


def test_global_label_conditional_is_gauge_invariant_and_stable():
    data = target()
    values = (np.array([0., 2.]), np.zeros(2), np.zeros(2), np.ones(2))
    initial = gates.conditional_label_probabilities(data, *values)
    data['log_weights'] += 10000.
    shifted = gates.conditional_label_probabilities(data, *values)
    np.testing.assert_allclose(initial, shifted, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(initial.sum(axis=1), 1.)


def test_conditional_label_summary_chunking_is_exact():
    data = target()
    samples = np.zeros((5, 12))
    samples[:, 2] = [0, 1, 2, 2, 3]
    samples[:, 7] = 2.
    one = gates.conditional_label_diagnostics(data, samples, chunk_rows=1)
    all_rows = gates.conditional_label_diagnostics(data, samples, chunk_rows=5)
    np.testing.assert_allclose(one['mean_conditional_probability_by_node'],
                               all_rows['mean_conditional_probability_by_node'])
    assert [p['pre_hop_dwells'] for p in one['conditional_probability_by_particle_number']] == [1, 1, 2, 1]
    with pytest.raises(ValueError, match='positive integers'):
        gates.conditional_label_diagnostics(data, samples, 0)


def test_executed_Gibbs_scope_is_not_described_as_an_unexecuted_hypothesis():
    data = target()
    samples = np.zeros((2,12))
    samples[:,2] = [1.,2.]
    samples[:,7] = 2.
    legacy = gates.conditional_label_diagnostics(data,samples)
    executed = gates.conditional_label_diagnostics(data,samples,executed_gibbs=True)
    assert 'not an executed kernel' in legacy['scope']
    assert 'executed Gibbs label probabilities' in executed['scope']
    assert 'not an executed kernel' not in executed['scope']
    assert legacy['executed_gibbs_protocol'] is False
    assert executed['executed_gibbs_protocol'] is True
    np.testing.assert_array_equal(legacy['mean_conditional_probability_by_node'],
                                  executed['mean_conditional_probability_by_node'])


def test_compound_Gibbs_jump_is_not_misread_as_a_neighbor_hop():
    data = target()
    samples = np.zeros((2, 12))
    samples[:, 0] = [1, 2]
    samples[:, 1] = [0, 2]
    samples[:, 2] = [2, 1]
    samples[:, 7] = 2.
    # Global Gibbs makes 0->1, then the actual marked edge is 1->2.
    intermediate = np.array([1, 2])
    result = gates.summarize_gate(data, samples, final_label=2,neighbor_source_labels=intermediate)
    rows = {(r['source'],r['destination']):r for r in result['directional_edges']}
    assert rows[1,2]['observed_accepted_hops']==1
    assert rows[0,3]['observed_accepted_hops']==0
    assert result['gibbs_label_changes']==1
    with pytest.raises(ValueError,match='pre-neighbor'):
        gates.summarize_gate(data,samples,neighbor_source_labels=np.array([1.5,2.]))
