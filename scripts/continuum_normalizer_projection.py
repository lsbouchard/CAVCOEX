#!/usr/bin/env python3
"""Optional discarded-calibration projection onto exact normalizer constraints.

The projected variables are a_j=b_j-log(tau_j), not phase probabilities.
For nested hard cavities at common activity, -log Xi_R is nondecreasing in R.
The endpoint span obeys elementary insertion/packing bounds. These necessary
constraints regularize efficiency guesses; they neither supply the unknown
partition functions nor certify mixing. Frozen production never calls this.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()
METHOD = 'unweighted_euclidean_nested_normalizer_slsqp_v1'
CERTIFIED_METHOD = 'unweighted_euclidean_certified_normalizer_slsqp_v1'
EPSILON_ROUNDING = 'decimal80_ln_nextplus_ceiling_arithmetic_exp_nextplus_float_ceiling_positive_subnormal_v1'
EPSILON_DECIMAL_PRECISION = 80
SCALED_BOUND_ROUNDING = 'binary64_division_nextafter_outward_nonzero_finite_bounds_exact_zero_infinity_preserved_v1'


def nested_physical_constraints(target, quota, cap):
    """Validate the exact physical domain and return the linear constraint set."""
    if isinstance(cap,(bool,np.bool_)) or not isinstance(cap,(int,np.integer)) or cap < 1:
        raise ValueError('Nontruncating positive integer particle cap required')
    if 'radius_path_nodes' not in target:
        raise ValueError('Projection requires a declared nested physical radius path')
    weights_nodes = len(target['activity'])
    q = np.asarray(quota)
    if (q.shape != (weights_nodes,) or q.dtype.kind not in 'fiu'
            or not np.isfinite(q).all() or np.any(q <= 0)
            or abs(float(q.sum())-1.) > 1e-12):
        raise ValueError('Projection requires positive normalized real label quotas')
    path = np.asarray(target['radius_path_nodes'])
    if (path.ndim != 1 or path.dtype.kind not in 'iu' or not 2 <= len(path) <= 257
            or np.any(path < 0) or np.any(path >= weights_nodes)
            or len(np.unique(path)) != len(path)):
        raise ValueError('Nested path must contain 2..257 distinct valid integer nodes')
    for name in ('activity','diameter','radius','pair_penalty','umbrella_stiffness','number_stiffness'):
        values = np.asarray(target[name])
        if values.shape != (weights_nodes,) or values.dtype.kind not in 'fiu':
            raise ValueError('Real target arrays must match the node count')
        if name != 'pair_penalty' and not np.isfinite(values).all():
            raise ValueError('Nested physical target parameters must be finite')
    if (int(path[0]) != target['base_node'] or int(path[-1]) != target['cavity_node']
            or not np.isin(path,np.asarray(target['physical_nodes'])).all()):
        raise ValueError('Nested path must run through physical labels from base to cavity')
    radius = np.asarray(target['radius'])[path]
    diameter = np.asarray(target['diameter'])[path]
    activity = np.asarray(target['activity'])[path]
    if (not np.isposinf(np.asarray(target['pair_penalty'])[path]).all()
            or np.any(np.asarray(target['umbrella_stiffness'])[path] != 0)
            or np.any(np.asarray(target['number_stiffness'])[path] != 0)
            or not np.all(diameter == diameter[0]) or diameter[0] <= 0
            or not np.all(activity == activity[0]) or radius[0] != 0
            or radius[-1] != diameter[0] or np.any(np.diff(radius) <= 0)):
        raise ValueError('Projection requires same-activity/diameter hard unbiased cavities R0 to sigma')
    volume = target['volume']
    if isinstance(volume,(bool,np.bool_)) or not np.isscalar(volume) or not np.isfinite(volume) or volume <= 0:
        raise ValueError('Finite positive physical volume required')
    sigma = float(diameter[0])
    if float(volume)**(1/3) < sigma:
        raise ValueError('Packing proof requires box length at least sigma')
    geometric_bound = math.ceil(6*float(volume)/(math.pi*sigma**3))
    if cap < geometric_bound:
        raise ValueError('Projection endpoint bounds require a nontruncating physical cap')
    particle_bound = min(int(cap),geometric_bound)
    log_zeta_volume = float(activity[0])+math.log(float(volume))
    lower = max(0.,log_zeta_volume-math.log(particle_bound))
    upper = float(np.logaddexp(0.,log_zeta_volume))
    count = len(path)
    matrix = np.zeros((count,count))
    for index in range(count-1):
        matrix[index,index],matrix[index,index+1] = -1.,1.
    matrix[-1,0],matrix[-1,-1] = -1.,1.
    lower_bounds = np.r_[np.zeros(count-1),lower]
    upper_bounds = np.r_[np.full(count-1,np.inf),upper]
    return {'path_nodes':path.astype(np.int64).copy(),'matrix':matrix,
        'linear_lower':lower_bounds,'linear_upper':upper_bounds,
        'lower_span':lower,'upper_span':upper,
        'geometric_particle_bound':particle_bound,
        'geometric_bound_rounding':'ceiling(6V/(pi sigma^3)); conservative floating-bound guard'}


def project_physical_normalizer_weights(weights, quota, target, cap):
    """Unique unweighted Euclidean projection under ALL constraints together.

    Centering removes an arbitrary weight gauge. The feasible start and exact
    linear Jacobian are deterministic. A no-op input is returned exactly; this
    helper is never invoked at all by the default projection-none driver path.
    """
    constraint = nested_physical_constraints(target,quota,cap)
    weights = np.asarray(weights)
    q = np.asarray(quota,dtype=float)
    if (weights.shape != q.shape or weights.dtype.kind not in 'fiu'
            or not np.isfinite(weights).all()):
        raise ValueError('Projection weights must be a finite real node vector')
    path = constraint['path_nodes']
    before = np.asarray(weights,dtype=float).copy()
    a = before[path]-np.log(q[path])
    matrix,lower,upper = (constraint[k] for k in ('matrix','linear_lower','linear_upper'))
    values = matrix@a
    already_feasible = bool(np.all(values >= lower) and np.all(values <= upper))
    solver_status,solver_success,solver_message = 0,True,'Input already feasible; exact no-op'
    iterations = 0
    variable_scale = 1.
    if already_feasible:
        projected_a = a.copy()
    else:
        # SciPy is imported only by the opt-in nontrivial calibration branch.
        from scipy.optimize import LinearConstraint, minimize
        center = float(a.mean())
        y = a-center
        variable_scale = max(1.,float(np.max(np.abs(y))))
        scaled_y = y/variable_scale
        initial_span = float(np.clip(y[-1]-y[0],constraint['lower_span'],constraint['upper_span']))
        initial = np.linspace(-initial_span/2,initial_span/2,len(path))/variable_scale
        # A positive common scaling changes neither the minimizer nor the
        # Euclidean metric; it avoids line-search failure for 100-log gaps.
        result = minimize(lambda x:.5*float(np.dot(x-scaled_y,x-scaled_y)),initial,
            jac=lambda x:x-scaled_y,method='SLSQP',
            constraints=(LinearConstraint(matrix,lower/variable_scale,upper/variable_scale),),
            options={'ftol':1e-13,'maxiter':1000,'disp':False})
        solver_status,solver_success,solver_message = int(result.status),bool(result.success),str(result.message)
        iterations = int(result.nit)
        if not solver_success or not np.isfinite(result.x).all():
            raise ValueError(f'Normalizer projection failed: {solver_message}')
        projected_a = result.x*variable_scale+center
    values = matrix@projected_a
    violation = float(max(0.,np.max(lower-values),np.max(values-upper)))
    tolerance = 1e-7
    if violation > tolerance:
        raise ValueError('Normalizer projection violates its simultaneous constraints')
    after = before.copy()
    if not already_feasible:
        after[path] = projected_a+np.log(q[path])
    squared = float(np.dot(projected_a-a,projected_a-a))
    record = {'method':METHOD,'path_nodes':path.tolist(),
        'pre_projection_log_weights':before.tolist(),
        'quota_adjusted_pre':a.tolist(),'quota_adjusted_post':projected_a.tolist(),
        'lower_span':constraint['lower_span'],'upper_span':constraint['upper_span'],
        'geometric_particle_bound':constraint['geometric_particle_bound'],
        'geometric_bound_rounding':constraint['geometric_bound_rounding'],
        'solver_success':solver_success,'solver_status':solver_status,'solver_message':solver_message,
        'solver_iterations':iterations,'objective_squared_distance':squared,
        'solver_variable_scale':variable_scale,
        'solver_ftol':1e-13,'solver_maxiter':1000,
        'maximum_constraint_violation':violation,'constraint_tolerance':tolerance,
        'scope':'Discarded calibration label-normalizer efficiency regularization only; not assumed work or phase probabilities'}
    return after,record


def canonical_target_signature(target, node):
    """Exact target identity, not proposal identity or phase classification."""
    penalty = float(target['pair_penalty'][node])
    order = float(target['umbrella_stiffness'][node])
    number = float(target['number_stiffness'][node])
    return (float(target['activity'][node]),
            float(target['diameter'][node]) if penalty else 0.,
            float(target['radius'][node]), penalty, order,
            float(target['umbrella_preferred'][node]) if order else 0., number,
            float(target['number_preferred'][node]) if number else 0.)


def upward_soft_hard_epsilon(activity, volume, penalty):
    """Conservative binary64 bound on exp(x)*V*exp(-gamma), or no tight bound.

    Decimal ln/exp are correctly rounded HALF_EVEN regardless of the context
    rounding. Advancing those results one Decimal ulp, plus ceiling arithmetic
    and final float conversion, produces an upper bound. A positive subnormal
    replaces a truly smaller positive bound; underflow never means equality.
    """
    from decimal import Decimal, ROUND_CEILING, localcontext
    import sys
    if (any(isinstance(v,(bool,np.bool_)) or not np.isscalar(v) or not np.isfinite(v)
            for v in (activity,volume,penalty)) or volume <= 0 or penalty < 0):
        raise ValueError('Finite activity/penalty and positive volume required for a soft-hard certificate')
    with localcontext() as context:
        context.prec = EPSILON_DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        log_volume_upper = Decimal.from_float(float(volume)).ln().next_plus()
        log_upper = Decimal.from_float(float(activity))+log_volume_upper-Decimal.from_float(float(penalty))
        if log_upper > 0:
            return None
        smallest = math.nextafter(0.,math.inf)
        if log_upper == 0:
            upper = 1.
        elif log_upper <= Decimal.from_float(smallest).ln().next_minus():
            upper = smallest
        else:
            decimal_upper = log_upper.exp().next_plus()
            upper = float(decimal_upper)
            if Decimal.from_float(upper) < decimal_upper:
                upper = math.nextafter(upper,math.inf)
            if upper == 0:
                upper = smallest
        if upper > 1:
            return None
        log_float = float(log_upper)
        if log_float == -math.inf:
            log_float = -sys.float_info.max
        elif Decimal.from_float(log_float) < log_upper:
            log_float = math.nextafter(log_float,math.inf)
        return {'epsilon_upper':upper,'log_epsilon_upper':log_float,
                'log_epsilon_upper_decimal':str(log_upper)}


def certified_normalizer_constraints(target, quota, cap):
    """Joint nested, exact-duplicate, and tight unbiased soft-hard constraints.

    Only independent equality rows are added. Certificates are selected using
    canonical-group representatives, avoiding redundant duplicate bounds.
    All constants concern the label-normalization root, not physical odds.
    """
    nested = nested_physical_constraints(target,quota,cap)
    nodes = len(target['activity'])
    if nodes > 4096:
        raise ValueError('Certified normalizer target scan is limited to 4096 nodes')
    for name in ('activity','diameter','radius','pair_penalty','umbrella_stiffness',
                 'umbrella_preferred','number_stiffness','number_preferred'):
        values = np.asarray(target[name])
        if (values.shape != (nodes,) or values.dtype.kind not in 'fiu'
                or (name != 'pair_penalty' and not np.isfinite(values).all())):
            raise ValueError('Finite real canonical-target arrays must match the node count')
    penalty = np.asarray(target['pair_penalty'])
    if (np.any(np.isnan(penalty)) or np.any(np.isneginf(penalty)) or np.any(penalty < 0)
            or np.any(np.asarray(target['diameter']) <= 0) or np.any(np.asarray(target['radius']) < 0)
            or np.any(np.asarray(target['umbrella_stiffness']) < 0)
            or np.any(np.asarray(target['number_stiffness']) < 0)):
        raise ValueError('Canonical interactions, radii and stiffnesses must be nonnegative; diameters positive')
    groups = {}
    for node in range(nodes):
        groups.setdefault(canonical_target_signature(target,node),[]).append(node)
    duplicate_groups = [{'anchor_node':members[0],'nodes':members}
                        for members in groups.values() if len(members)>1]
    duplicate_groups.sort(key=lambda group:group['anchor_node'])
    hard_groups,soft_groups = {},[]
    for members in groups.values():
        representative = members[0]
        if (target['umbrella_stiffness'][representative] != 0
                or target['number_stiffness'][representative] != 0):
            continue
        if np.isposinf(penalty[representative]):
            key = tuple(float(target[k][representative]) for k in ('activity','diameter','radius'))
            hard_groups[key] = members
        else:
            soft_groups.append(members)
    selected = set(map(int,nested['path_nodes']))
    for group in duplicate_groups:
        selected.update(group['nodes'])
    equality_rows = sum(len(group['nodes'])-1 for group in duplicate_groups)
    if len(selected)>257 or equality_rows+len(nested['path_nodes'])>1024:
        raise ValueError('Certified joint projection requires at most 257 selected nodes and 1024 linear rows')
    certificates = []
    for soft_members in soft_groups:
        matching_groups = {}
        for member in soft_members:
            key = tuple(float(target[k][member]) for k in ('activity','diameter','radius'))
            if key in hard_groups and key not in matching_groups:
                matching_groups[key] = member
        for key,soft in matching_groups.items():
            hard_members = hard_groups[key]
            hard = hard_members[0]
            bound = upward_soft_hard_epsilon(float(target['activity'][soft]),
                                            float(target['volume']),float(penalty[soft]))
            if bound is not None:
                certificates.append({'soft_node':soft,'hard_node':hard,
                    'pair_penalty':float(penalty[soft]),**bound})
                selected.update((soft,hard))
                if len(selected)>257 or len(nested['path_nodes'])+len(certificates)+equality_rows>1024:
                    raise ValueError('Certified joint projection requires at most 257 selected nodes and 1024 linear rows')
    certificates.sort(key=lambda bound:(bound['soft_node'],bound['hard_node']))
    selected = np.array(sorted(selected),dtype=np.int64)
    inequality_rows = len(nested['path_nodes'])+len(certificates)
    if len(selected)>257 or equality_rows+inequality_rows>1024:
        raise ValueError('Certified joint projection requires at most 257 selected nodes and 1024 linear rows')
    index = {int(node):local for local,node in enumerate(selected)}
    inequality = np.zeros((inequality_rows,len(selected)))
    lower = np.zeros(inequality_rows)
    upper = np.full(inequality_rows,np.inf)
    path = nested['path_nodes']
    for row in range(len(path)-1):
        inequality[row,index[int(path[row])]] = -1.
        inequality[row,index[int(path[row+1])]] = 1.
    row = len(path)-1
    inequality[row,index[int(path[0])]],inequality[row,index[int(path[-1])]] = -1.,1.
    lower[row],upper[row] = nested['lower_span'],nested['upper_span']
    for row,bound in enumerate(certificates,start=len(path)):
        inequality[row,index[bound['soft_node']]],inequality[row,index[bound['hard_node']]] = -1.,1.
        upper[row] = bound['epsilon_upper']
    equality = np.zeros((equality_rows,len(selected)))
    row = 0
    for group in duplicate_groups:
        for member in group['nodes'][1:]:
            equality[row,index[group['anchor_node']]],equality[row,index[member]] = -1.,1.
            row += 1
    return {**nested,'selected_nodes':selected,'inequality_matrix':inequality,
        'inequality_lower':lower,'inequality_upper':upper,'equality_matrix':equality,
        'duplicate_groups':duplicate_groups,'soft_hard_certificates':certificates,
        'epsilon_rounding':EPSILON_ROUNDING,'epsilon_decimal_precision':EPSILON_DECIMAL_PRECISION}


def outward_scaled_inequality_bounds(lower, upper, scale):
    """Conservatively divide solver bounds without false subnormal equality."""
    scaled_lower,scaled_upper = np.asarray(lower)/scale,np.asarray(upper)/scale
    lower_nonzero = np.isfinite(lower)&(np.asarray(lower)!=0)
    upper_nonzero = np.isfinite(upper)&(np.asarray(upper)!=0)
    scaled_lower[lower_nonzero] = np.nextafter(scaled_lower[lower_nonzero],-np.inf)
    scaled_upper[upper_nonzero] = np.nextafter(scaled_upper[upper_nonzero],np.inf)
    return scaled_lower,scaled_upper


def project_certified_normalizer_weights(weights, quota, target, cap):
    """Unweighted L2 projection on the complete certified union simultaneously."""
    constraint = certified_normalizer_constraints(target,quota,cap)
    weights,q = np.asarray(weights),np.asarray(quota,dtype=float)
    if (weights.shape != q.shape or weights.dtype.kind not in 'fiu'
            or not np.isfinite(weights).all()):
        raise ValueError('Projection weights must be a finite real node vector')
    selected,path = constraint['selected_nodes'],constraint['path_nodes']
    before = np.asarray(weights,dtype=float).copy()
    a = before[selected]-np.log(q[selected])
    matrix,lower,upper = (constraint[k] for k in ('inequality_matrix','inequality_lower','inequality_upper'))
    equality = constraint['equality_matrix']
    values = matrix@a
    already_feasible = bool(np.all(values >= lower) and np.all(values <= upper)
                            and np.all(equality@a == 0))
    solver_status,solver_success,solver_message = 0,True,'Input already feasible; exact no-op'
    iterations,variable_scale = 0,1.
    if already_feasible:
        projected_a = a.copy()
    else:
        from scipy.optimize import LinearConstraint,minimize
        center = float(a.mean())
        y = a-center
        variable_scale = max(1.,float(np.max(np.abs(y))),constraint['lower_span'])
        if not np.isfinite(variable_scale):
            raise ValueError('Certified normalizer scaling must be finite')
        scaled_y = y/variable_scale
        # Equal-radius hosts receive the same start value: this satisfies all
        # duplicate rows and soft-hard bounds, independently of their x/gamma.
        sigma = float(target['diameter'][int(path[0])])
        initial = np.clip(np.asarray(target['radius'])[selected],0.,sigma)/sigma*constraint['lower_span']
        initial -= initial.mean()
        initial /= variable_scale
        scaled_lower,scaled_upper = outward_scaled_inequality_bounds(lower,upper,variable_scale)
        linear = [LinearConstraint(matrix,scaled_lower,scaled_upper)]
        if len(equality):
            linear.append(LinearConstraint(equality,np.zeros(len(equality)),np.zeros(len(equality))))
        result = minimize(lambda x:.5*float(np.dot(x-scaled_y,x-scaled_y)),initial,
            jac=lambda x:x-scaled_y,method='SLSQP',constraints=tuple(linear),
            options={'ftol':1e-13,'maxiter':1000,'disp':False})
        solver_status,solver_success,solver_message = int(result.status),bool(result.success),str(result.message)
        iterations = int(result.nit)
        if not solver_success or not np.isfinite(result.x).all():
            raise ValueError(f'Certified normalizer projection failed: {solver_message}')
        projected_a = result.x*variable_scale+center
    values = matrix@projected_a
    violation = float(max(0.,np.max(lower-values),np.max(values-upper),
                          np.max(np.abs(equality@projected_a)) if len(equality) else 0.))
    tolerance = 1e-7
    if violation>tolerance:
        raise ValueError('Certified normalizer projection violates its simultaneous constraints')
    after = before.copy()
    if not already_feasible:
        after[selected] = projected_a+np.log(q[selected])
    squared = float(np.dot(projected_a-a,projected_a-a))
    if not np.isfinite(after).all() or not np.isfinite(squared):
        raise ValueError('Certified normalizer projection output/objective must be finite')
    selected_index = {int(node):local for local,node in enumerate(selected)}
    path_indices = [selected_index[int(node)] for node in path]
    record = {'method':CERTIFIED_METHOD,'path_nodes':path.tolist(),
        'selected_nodes':selected.tolist(),'pre_projection_log_weights':before.tolist(),
        'quota_adjusted_pre':a[path_indices].tolist(),'quota_adjusted_post':projected_a[path_indices].tolist(),
        'selected_quota_adjusted_pre':a.tolist(),'selected_quota_adjusted_post':projected_a.tolist(),
        'duplicate_groups':constraint['duplicate_groups'],
        'soft_hard_certificates':constraint['soft_hard_certificates'],
        'epsilon_rounding':constraint['epsilon_rounding'],'epsilon_decimal_precision':constraint['epsilon_decimal_precision'],
        'scaled_bound_rounding':SCALED_BOUND_ROUNDING,
        'lower_span':constraint['lower_span'],'upper_span':constraint['upper_span'],
        'geometric_particle_bound':constraint['geometric_particle_bound'],
        'geometric_bound_rounding':constraint['geometric_bound_rounding'],
        'solver_success':solver_success,'solver_status':solver_status,'solver_message':solver_message,
        'solver_iterations':iterations,'objective_squared_distance':squared,
        'solver_variable_scale':variable_scale,'solver_ftol':1e-13,'solver_maxiter':1000,
        'maximum_constraint_violation':violation,'constraint_tolerance':tolerance,
        'scope':'Discarded calibration label-normalizer efficiency regularization only; not assumed work or phase probabilities'}
    return after,record
