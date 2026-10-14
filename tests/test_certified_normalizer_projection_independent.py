"""Independent joint active-set controls for certified normalizer constraints.

Exact Fractions and a separately enumerated KKT oracle; no sampler or Monte
Carlo. Signed equality multipliers are not assumed nonnegative. The production
certificates encode each equality as two nonnegative-multiplier inequalities.
"""
import itertools
import math
from fractions import Fraction as F

import pytest

from test_physical_normalizer_constraints_independent import dot, solve_rational


def joint_rational_projection(values, rows, bounds):
    """Independent brute active-set L2 oracle, all rows expressed A*x>=b."""
    values = tuple(map(F,values))
    rows = [tuple(map(F,row)) for row in rows]
    bounds = tuple(map(F,bounds))
    candidates = []
    for count in range(len(values)):
        for active in itertools.combinations(range(len(rows)),count):
            selected = [rows[index] for index in active]
            multipliers = solve_rational([[dot(first,second) for second in selected]
                                         for first in selected],
                                        [bounds[index]-dot(rows[index],values) for index in active])
            if multipliers is None or any(value < 0 for value in multipliers):
                continue
            point = tuple(values[k]+sum((multipliers[index]*selected[index][k]
                                        for index in range(count)),F(0))
                          for k in range(len(values)))
            if all(dot(row,point)>=bound for row,bound in zip(rows,bounds)):
                squared = sum(((point[k]-values[k])**2 for k in range(len(values))),F(0))
                candidates.append((squared,point,active,multipliers))
    assert candidates
    return min(candidates)


def small_joint_constraints():
    # Coordinates: hard base, hard cavity, unbiased soft-base, ideal, duplicate.
    rows = [(-1,1,0,0,0),(1,-1,0,0,0),
            (1,0,-1,0,0),(-1,0,1,0,0),
            (0,0,0,-1,1),(0,0,0,1,-1)]
    bounds = [F(3),F(-6),F(0),F(-1,10),F(0),F(0)]
    return rows,bounds


def test_joint_optimum_matches_closed_form_and_needs_a_negative_signed_equality_multiplier():
    values = (0,-20,50,10,0)
    rows,bounds = small_joint_constraints()
    _,point,active,multipliers = joint_rational_projection(values,rows,bounds)
    # The soft datum50 pulls upward: hard-soft>=0 is active at its LOWER bound,
    # not at epsilon. With hard=soft=t and cavity=t+3, the derivative is
    # t+(t+23)+(t-50)=3t-27, hence t=9. The duplicate ideal pair averages to5.
    assert point == (F(9),F(12),F(9),F(5),F(5))
    assert active == (0,2,4) and multipliers == (F(32),F(41),F(5))
    assert point[0] - point[2] == 0  # soft-hard lower bound is active
    assert all(value>=0 for value in multipliers)  # signed-row inequality form
    equality = (0,0,0,-1,1)
    # In the ordinary equality Lagrangian .5||x-y||² + nu*(a4-a3),
    # stationarity requires nu=-5. Rejecting nu<0 would reject the true optimum.
    signed_equality_multiplier = F(-5)
    for index in (3,4):
        assert point[index]-values[index]+signed_equality_multiplier*equality[index] == 0
    assert 4 in active  # positive oriented inequality multiplier implements nu<0
    assert sum(point) == sum(values)


def test_feasible_centroid_preserving_point_can_still_be_nonoptimal_for_joint_constraints():
    values = (0,-20,50,10,0)
    rows,bounds = small_joint_constraints()
    optimum,point,_,_ = joint_rational_projection(values,rows,bounds)
    wrong = tuple(value+shift for value,shift in zip(point,(1,1,1,F(-3,2),F(-3,2))))
    assert all(dot(row,wrong)>=bound for row,bound in zip(rows,bounds))
    assert wrong[0] <= wrong[1] and 3 <= wrong[1]-wrong[0] <= 6
    assert sum(wrong) == sum(values)  # a global centroid check is insufficient
    assert sum(((wrong[k]-values[k])**2 for k in range(5)),F(0)) > optimum
    # Constraint gradients sum to zero separately on the physical and ideal
    # components, while this candidate's residual does not. No KKT multipliers
    # can represent it, despite feasibility and the full centroid check.
    assert sum((wrong[k]-values[k] for k in (0,1,2)),F(0)) == 3
    assert all(sum(row[:3]) == 0 for row in rows)


def test_equality_sign_reversal_preserves_optimum_and_common_gauge_commutes():
    values = (0,-20,50,10,0)
    rows,bounds = small_joint_constraints()
    result = joint_rational_projection(values,rows,bounds)[1]
    swapped = rows[:4]+[rows[5],rows[4]]
    assert joint_rational_projection(values,swapped,bounds)[1] == result
    shifted = joint_rational_projection([F(value)+123 for value in values],rows,bounds)[1]
    assert shifted == tuple(value+123 for value in result)


def test_projecting_blocks_separately_can_violate_previously_repaired_radius_constraints():
    # First satisfy radius span [3,6] using only the base/cavity block:
    # (0,-20,50)->(-11.5,-8.5,50). Then force soft/hard equality by averaging
    # base and soft, giving(19.25,-8.5,19.25), which breaks radius ordering.
    wrong = (F(77,4),F(-17,2),F(77,4),F(5),F(5))
    rows,bounds = small_joint_constraints()
    assert wrong[0] == wrong[2] and wrong[3] == wrong[4]
    assert dot(rows[0],wrong) < bounds[0]
    correct = joint_rational_projection((0,-20,50,10,0),rows,bounds)[1]
    assert all(dot(row,correct)>=bound for row,bound in zip(rows,bounds))


def certified_target():
    import numpy as np
    return {'volume':27., 'activity':np.array([16.,16.,16.,16.,-.1,-.1,16.]),
            'diameter':np.ones(7), 'radius':np.array([0.,1.,.5,0.,0.,0.,0.]),
            'pair_penalty':np.array([np.inf,np.inf,np.inf,25.,0.,0.,2.]),
            'umbrella_stiffness':np.array([0.,0.,0.,0.,0.,0.,.2]),
            'umbrella_preferred':np.zeros(7), 'number_stiffness':np.zeros(7),
            'number_preferred':np.zeros(7), 'base_node':0, 'cavity_node':1,
            'physical_nodes':np.array([0,1,2]),'radius_path_nodes':np.array([0,2,1])}


def independent_actual_rows(lower,upper,epsilon):
    # Fixed independently from this seven-node target's definition, not by
    # copying the helper's selected matrix or certificate metadata.
    rows = [(-1,0,1,0,0,0),(0,1,-1,0,0,0),
            (-1,1,0,0,0,0),(1,-1,0,0,0,0),
            (1,0,0,-1,0,0),(-1,0,0,1,0,0),
            (0,0,0,0,-1,1),(0,0,0,0,1,-1)]
    bounds = [F(0),F(0),lower,-upper,F(0),-epsilon,F(0),F(0)]
    return rows,bounds


@pytest.mark.parametrize('values', [(0,20,-100,50,10,0),(0,8,16,50,10,0),
                                   (-30,-10,-20,40,-5,5),(0,16,8,-.001,0,0)])
def test_actual_certified_joint_minimizer_matches_independent_fraction_oracle(values):
    import numpy as np
    from scripts.continuum_normalizer_projection import (
        project_certified_normalizer_weights,upward_soft_hard_epsilon)
    target = certified_target()
    quota = np.array([.30,.30,.095,.095,.02,.095,.095])
    weights = np.full(7,-7.)
    weights[:6] = np.array(values)+np.log(quota[:6])
    before = weights.copy()
    output,record = project_certified_normalizer_weights(weights,quota,target,80)
    assert weights.tobytes() == before.tobytes()
    assert record['selected_nodes'] == list(range(6))
    assert record['duplicate_groups'] == [{'anchor_node':4,'nodes':[4,5]}]
    certificates = record['soft_hard_certificates']
    assert len(certificates) == 1
    assert (certificates[0]['soft_node'],certificates[0]['hard_node']) == (3,0)
    epsilon = upward_soft_hard_epsilon(16.,27.,25.)['epsilon_upper']
    assert certificates[0]['epsilon_upper'] == epsilon
    lower = F(str(max(0.,16+math.log(27)-math.log(math.ceil(6*27/math.pi)))))
    upper = F(str(float(np.logaddexp(0.,16+math.log(27)))))
    rows,bounds = independent_actual_rows(lower,upper,F(str(epsilon)))
    rational_input = [F(str(value)) for value in before[:6]-np.log(quota[:6])]
    _,oracle,_,_ = joint_rational_projection(rational_input,rows,bounds)
    adjusted = output[:6]-np.log(quota[:6])
    np.testing.assert_allclose(adjusted,[float(value) for value in oracle],atol=2e-7,rtol=0)
    assert output[6] == before[6]
    assert abs(adjusted.sum()-sum(rational_input)) < 2e-7
    shifted,_ = project_certified_normalizer_weights(weights+123.,quota,target,80)
    np.testing.assert_allclose(shifted,output+123.,atol=2e-7,rtol=0)
    again,_ = project_certified_normalizer_weights(output,quota,target,80)
    np.testing.assert_allclose(again,output,atol=2e-7,rtol=0)
    if values == (0,16,8,-.001,0,0):
        assert output.tobytes() == before.tobytes()
        assert record['solver_iterations'] == 0


def independent_decimal_interval(activity,volume,penalty):
    from decimal import Decimal,ROUND_CEILING,ROUND_FLOOR,localcontext
    with localcontext() as context:
        context.prec = 256
        x,v,g = map(Decimal.from_float,(float(activity),float(volume),float(penalty)))
        nearest_log = v.ln()
        log_lower,log_upper = nearest_log.next_minus(),nearest_log.next_plus()
        context.rounding = ROUND_FLOOR
        score_lower = x+log_lower-g
        lower = score_lower.exp().next_minus()
        context.rounding = ROUND_CEILING
        score_upper = x+log_upper-g
        upper = score_upper.exp().next_plus()
        return lower,upper


@pytest.mark.parametrize('activity,volume,penalty', [(16.,30.8496,25.),(-2.,27.,4.),
                                                  (16.,30.8496,1100.),(-1000.,1.,0.),
                                                  (0.,math.nextafter(0.,1.),1.),
                                                  (-1000.,float.fromhex('0x1.fffffffffffffp+1023'),0.)])
def test_epsilon_binary_upper_exceeds_independent_high_precision_interval(activity,volume,penalty):
    from decimal import Decimal
    from scripts.continuum_normalizer_projection import upward_soft_hard_epsilon
    result = upward_soft_hard_epsilon(activity,volume,penalty)
    assert result is not None and 0 < result['epsilon_upper'] <= 1
    _,reference_upper = independent_decimal_interval(activity,volume,penalty)
    assert Decimal.from_float(result['epsilon_upper']) >= reference_upper
    assert math.isfinite(result['log_epsilon_upper'])
    if penalty >= 1000 or activity <= -1000 and volume == 1.:
        assert result['epsilon_upper'] == math.nextafter(0.,1.)


def test_extreme_gamma_stays_positive_and_non_tight_bounds_are_omitted():
    from scripts.continuum_normalizer_projection import upward_soft_hard_epsilon
    result = upward_soft_hard_epsilon(16.,30.8496,1e308)
    assert result['epsilon_upper'] == math.nextafter(0.,1.)
    assert math.isfinite(result['log_epsilon_upper']) and result['log_epsilon_upper'] < -744
    assert upward_soft_hard_epsilon(16.,30.8496,0.) is None
    # The exact epsilon=1 boundary may be conservatively omitted because the
    # certified Decimal upper is slightly above1; never include an upper>1.
    boundary = upward_soft_hard_epsilon(0.,1.,0.)
    assert boundary is None or boundary['epsilon_upper'] == 1.


@pytest.mark.parametrize('cost', ['order','number','activity','radius'])
def test_soft_hard_certificate_excludes_unequal_or_biased_targets(cost):
    import numpy as np
    from scripts.continuum_normalizer_projection import certified_normalizer_constraints
    target = certified_target()
    if cost == 'order':
        target['umbrella_stiffness'][3] = .01
    elif cost == 'number':
        target['number_stiffness'][3] = .01
    else:
        target[cost][3] += .01
    result = certified_normalizer_constraints(target,np.array([.30,.30,.095,.095,.02,.095,.095]),80)
    assert result['soft_hard_certificates'] == []


def test_unused_preferred_values_do_not_break_true_ideal_target_identity():
    import numpy as np
    from scripts.continuum_normalizer_projection import certified_normalizer_constraints
    target = certified_target()
    target['umbrella_preferred'][5],target['number_preferred'][5] = .8,17.
    result = certified_normalizer_constraints(target,np.array([.30,.30,.095,.095,.02,.095,.095]),80)
    assert result['duplicate_groups'] == [{'anchor_node':4,'nodes':[4,5]}]
