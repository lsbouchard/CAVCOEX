"""Exact controls for a proposed discarded physical-normalizer projection.

No sampler is imported and no Monte Carlo is performed. The hard graph below
has a certified maximum population; it is a finite probability model, not a
fluid--solid surrogate. The rational projection is an independent active-set
oracle for the constraint mathematics, not an implemented production method.
"""
import itertools
import math
from fractions import Fraction as F

import pytest


VERTICES = range(3)
EDGES = {(0, 1), (1, 2)}


def hard_hosts(cap):
    for mask in range(1 << 3):
        host = frozenset(i for i in VERTICES if mask & (1 << i))
        if len(host) <= cap and not any(set(edge) <= host for edge in EDGES):
            yield host


def excluded(center, stage):
    if stage == 0:
        return frozenset()
    if stage == 1:
        return frozenset((center,))
    return frozenset((center,)) | frozenset(
        second if first == center else first
        for first, second in EDGES if center in (first, second))


def graph_partition(activity, cap, stage):
    # Unordered subsets already have their correct sector measure. Inserting an
    # additional 1/N! here would change the target and invalidate the identity.
    return sum((activity ** len(host) * sum(
        not host.intersection(excluded(center, stage)) for center in VERTICES) / 3
        for host in hard_hosts(cap)), F(0))


def graph_mean_number(activity, cap):
    normalizer = graph_partition(activity, cap, 0)
    return sum((len(host) * activity ** len(host) for host in hard_hosts(cap)), F(0)) / normalizer


@pytest.mark.parametrize('activity', [F(1, 1000), F(1, 10), F(1), F(10), F(1000)])
def test_nontruncating_hard_normalizer_bounds_and_matched_palm_identity(activity):
    assert max(map(len, hard_hosts(3))) == 2  # independently certified support
    base = graph_partition(activity, 2, 0)
    small = graph_partition(activity, 2, 1)
    cavity = graph_partition(activity, 2, 2)
    mean = graph_mean_number(activity, 2)
    assert base >= small >= cavity > 0
    assert cavity / base == mean / (3 * activity)
    assert base >= 1 + 3 * activity
    assert mean >= 3 * activity / (1 + 3 * activity)
    assert mean <= 2
    work = math.log(float(base / cavity))
    lower = max(0., math.log(float(3 * activity / 2)))
    upper = math.log1p(float(3 * activity))
    assert lower - 1e-14 <= work <= upper + 1e-14


def test_artificial_cap_invalidates_unrestricted_endpoint_identity_and_lower_bound():
    activity = F(10)
    base = graph_partition(activity, 1, 0)
    cavity = graph_partition(activity, 1, 2)
    mean = graph_mean_number(activity, 1)
    assert (base, cavity, mean) == (F(31), F(23, 3), F(30, 31))
    assert cavity / base != mean / (3 * activity)
    assert math.log(float(base / cavity)) < math.log(float(3 * activity / 2))
    # The correct capped Campbell expression involves the next base sector:
    next_base = graph_partition(activity, 2, 0)
    next_mean = graph_mean_number(activity, 2)
    assert cavity == next_base * next_mean / (3 * activity)
    # Cavity-support nesting itself is still exact with an artificial cap.
    assert base >= graph_partition(activity, 1, 1) >= cavity


def test_equal_activity_and_hard_host_conditions_are_essential_for_endpoint_constraints():
    assert graph_partition(F(1000), 2, 2) > graph_partition(F(1), 2, 0)
    # No pair exclusions: all subsets are allowed, but the cavity still excludes
    # the tagged hard neighborhood. Soft/ideal insertion is not a void event.
    hosts = [frozenset(i for i in VERTICES if mask & (1 << i)) for mask in range(8)]
    base = F(len(hosts))
    mean = F(sum(map(len, hosts)), len(hosts))
    cavity = sum((F(sum(not host.intersection(excluded(center, 2))
                        for center in VERTICES), 3) for host in hosts), F(0))
    assert (base, mean, cavity) == (F(8), F(3, 2), F(5, 3))
    assert cavity / base != mean / 3


def dot(first, second):
    return sum((a * b for a, b in zip(first, second)), F(0))


def solve_rational(matrix, right):
    count = len(right)
    augmented = [list(row) + [rhs] for row, rhs in zip(matrix, right)]
    for column in range(count):
        pivot = next((row for row in range(column, count)
                      if augmented[row][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(count):
            if row != column:
                scale = augmented[row][column]
                augmented[row] = [a - scale * b for a, b in
                                  zip(augmented[row], augmented[column])]
    return tuple(row[-1] for row in augmented)


def constraints(nodes, lower, upper):
    rows = []
    for index in range(nodes - 1):
        row = [F(0)] * nodes
        row[index], row[index + 1] = F(-1), F(1)
        rows.append(tuple(row))
    endpoint = tuple(F(-1 if i == 0 else 1 if i == nodes - 1 else 0)
                     for i in range(nodes))
    rows.extend((endpoint, tuple(-a for a in endpoint)))
    return rows, [F(0)] * (nodes - 1) + [lower, -upper]


def rational_projection(values, lower, upper, metric=None):
    """Enumerate independent active constraints and solve weighted KKT exactly."""
    values = tuple(map(F, values))
    if not F(0) <= lower <= upper:
        raise ValueError('Nonnegative ordered span bounds required')
    metric = tuple(map(F, metric or [1] * len(values)))
    if len(metric) != len(values) or min(metric) <= 0:
        raise ValueError('Strictly positive matching metric required')
    rows, rhs = constraints(len(values), lower, upper)
    candidates = []
    # All normalizer constraints have additive-gauge lineality, hence rank <=n-1.
    for count in range(len(values)):
        for active in itertools.combinations(range(len(rows)), count):
            selected = [rows[index] for index in active]
            gram = [[sum((first[k] * second[k] / metric[k]
                          for k in range(len(values))), F(0))
                     for second in selected] for first in selected]
            multiplier = solve_rational(gram, [rhs[index] - dot(rows[index], values)
                                                for index in active])
            if multiplier is None or any(value < 0 for value in multiplier):
                continue
            candidate = tuple(values[k] + sum((multiplier[j] * selected[j][k]
                                  for j in range(count)), F(0)) / metric[k]
                              for k in range(len(values)))
            if all(dot(row, candidate) >= bound for row, bound in zip(rows, rhs)):
                objective = sum((metric[k] * (candidate[k] - values[k]) ** 2
                                 for k in range(len(values))), F(0))
                candidates.append((objective, candidate))
    assert candidates, 'The declared convex constraint set is feasible'
    return min(candidates)[1]


@pytest.mark.parametrize('values', [(10, -100, 20), (0, 20, 10), (0, 2, 4), (-3, -2, 7)])
@pytest.mark.parametrize('metric', [[1, 1, 1], [30, 1, 30]])
def test_joint_projection_is_feasible_idempotent_and_additive_gauge_equivariant(values, metric):
    lower, upper = F(3), F(6)
    projected = rational_projection(values, lower, upper, metric)
    assert projected[0] <= projected[1] <= projected[2]
    assert lower <= projected[2] - projected[0] <= upper
    assert rational_projection(projected, lower, upper, metric) == projected
    shifted = rational_projection([F(value) + 123 for value in values], lower, upper, metric)
    assert shifted == tuple(value + 123 for value in projected)


def test_true_feasible_normalizers_are_unchanged_and_one_pass_endpoint_clipping_can_fail():
    assert rational_projection([0, 2, 4], F(3), F(6)) == (F(0), F(2), F(4))
    # Isotonic projection of [0,20,10] is [0,15,15]. A subsequent one-pass upper
    # endpoint clamp to span 5 makes [0,15,5], violating the middle inequality.
    wrongly_clamped = [0, 15, 5]
    assert not wrongly_clamped[0] <= wrongly_clamped[1] <= wrongly_clamped[2]
    correct = rational_projection([0, 20, 10], F(3), F(5))
    assert correct[0] <= correct[1] <= correct[2]
    assert 3 <= correct[2] - correct[0] <= 5


def test_constraints_apply_to_quota_corrected_normalizers_not_raw_label_weights():
    activity = F(10)
    partition = [graph_partition(activity, 2, stage) for stage in range(3)]
    quota = [F(49, 100), F(2, 100), F(49, 100)]
    normalizer = [-math.log(float(value)) for value in partition]
    weights = [math.log(float(q)) + a for q, a in zip(quota, normalizer)]
    assert normalizer[0] <= normalizer[1] <= normalizer[2]
    assert weights[1] < weights[0]  # raw-b isotonicity would reject the exact root
    assert all(abs((b - math.log(float(q))) - a) < 1e-14
               for b, q, a in zip(weights, quota, normalizer))


def test_normalizer_constraints_cannot_identify_absolute_family_mass_or_phase_probabilities():
    activity = F(10)
    base_hosts = list(hard_hosts(2))
    base = graph_partition(activity, 2, 0)
    cavity = graph_partition(activity, 2, 2)
    category = sum((activity ** len(host) for host in base_hosts if len(host) == 2), F(0)) / base
    assert category == F(100, 131)
    # A common physical-family weight shift preserves every span/order constraint
    # and each within-node category law but changes total family vs auxiliary mass.
    auxiliary_mass = F(7)
    first = (base + cavity) / (base + cavity + auxiliary_mass)
    shifted = 5 * (base + cavity) / (5 * (base + cavity) + auxiliary_mass)
    assert first != shifted
    assert 5 * F(100) / (5 * base) == category
    assert (5 * base) / (5 * cavity) == base / cavity


def projection_target():
    import numpy as np
    return {'volume':27., 'activity':np.array([16.,16.,16.,-.1]),
            'diameter':np.ones(4), 'radius':np.array([0.,1.,.5,0.]),
            'pair_penalty':np.array([np.inf,np.inf,np.inf,0.]),
            'umbrella_stiffness':np.zeros(4), 'number_stiffness':np.zeros(4),
            'base_node':0, 'cavity_node':1, 'physical_nodes':np.array([0,1,2]),
            'radius_path_nodes':np.array([0,2,1])}


@pytest.mark.parametrize('values', [(10, -100, 20), (0, 20, 10), (0, 2, 4), (-3, -2, 7), (0, 8, 16)])
def test_actual_unweighted_helper_matches_independent_rational_kkt_oracle(values):
    import numpy as np
    from scripts.continuum_normalizer_projection import project_physical_normalizer_weights
    target = projection_target()
    quota, path = np.array([.30,.30,.38,.02]), target['radius_path_nodes']
    weights = np.full(4,-7.)
    weights[path] = np.array(values) + np.log(quota[path])
    before = weights.copy()
    output, record = project_physical_normalizer_weights(weights,quota,target,80)
    assert np.array_equal(weights,before)  # independent helper must not mutate input
    particle_bound = math.ceil(6 * 27 / math.pi)
    lower = max(0.,16 + math.log(27) - math.log(particle_bound))
    upper = math.log1p(math.exp(16) * 27)
    assert abs(record['lower_span'] - lower) < 1e-14
    assert abs(record['upper_span'] - upper) < 1e-14
    # Use exact rational representatives of the input/bounds supplied to SLSQP,
    # not the output or an isotonic implementation copied from the helper.
    rational_input = [F(str(value)) for value in before[path] - np.log(quota[path])]
    oracle = rational_projection(rational_input,F(str(record['lower_span'])),
                                 F(str(record['upper_span'])),[1,1,1])
    adjusted = output[path] - np.log(quota[path])
    np.testing.assert_allclose(adjusted,[float(value) for value in oracle],atol=1e-7,rtol=0)
    assert output[3] == before[3]  # unprojected auxiliary label is unchanged
    assert abs(adjusted.sum() - sum(values)) < 1e-8  # translation-invariant L2 KKT
    shifted, _ = project_physical_normalizer_weights(weights+123.,quota,target,80)
    np.testing.assert_allclose(shifted,output+123.,atol=1e-7,rtol=0)
    again, _ = project_physical_normalizer_weights(output,quota,target,80)
    np.testing.assert_allclose(again,output,atol=1e-7,rtol=0)
    if values == (0,8,16):
        assert output.tobytes() == before.tobytes()
        assert record['solver_iterations'] == 0


@pytest.mark.parametrize('mutation', ['activity','diameter','soft','order','number','cap','path'])
def test_actual_helper_rejects_unsupported_endpoint_proof_domains(mutation):
    import numpy as np
    from scripts.continuum_normalizer_projection import project_physical_normalizer_weights
    target = projection_target()
    cap = 80
    if mutation in ('activity','diameter'):
        target[mutation][2] += .01
    elif mutation == 'soft':
        target['pair_penalty'][2] = 8.
    elif mutation == 'order':
        target['umbrella_stiffness'][2] = .01
    elif mutation == 'number':
        target['number_stiffness'][2] = .01
    elif mutation == 'cap':
        cap = math.ceil(6 * 27 / math.pi) - 1
    else:
        target['radius_path_nodes'] = np.array([0,1,2])
    with pytest.raises(ValueError):
        project_physical_normalizer_weights(np.zeros(4),np.array([.30,.30,.38,.02]),target,cap)
