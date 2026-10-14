#!/usr/bin/env python3
"""Native finite tempering graphs, not manufactured replica data.

Carnahan--Starling fluid pressure supplies only initial label-efficiency guesses.
No guessed thermodynamic weight enters a conditional hard-sphere phase law.
The dilute ideal reservoir shares x=-2 with its neighboring hard gas.
The optional compact graph uses an exact dense ideal reservoir and removable
soft, order and number bridges. Every physical cavity target is unbiased.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

try:
    from . import continuum_replica_exchange as kernel
    from . import continuum_analyze as validation
except ImportError:
    import continuum_replica_exchange as kernel
    import continuum_analyze as validation

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'cavcoex.native_tempering_graph.v1'
EXECUTION_SOURCE_BYTES = Path(__file__).read_bytes()
EXECUTION_SOURCE_SHA256 = hashlib.sha256(EXECUTION_SOURCE_BYTES).hexdigest()
VALIDATION_SOURCE_BYTES = Path(validation.__file__).read_bytes()
COMPACT_WEIGHT_MODEL = 'soft_cs_number_gaussian_bragg_v1'
COMPACT_WEIGHT_CONSTANTS = {'effective_packing_fraction_cutoff':.74,
                          'Bragg_mean':.2,'Bragg_variance':.01}


def fluid_cs_density(field):
    """Solve x=log(rho)+mu_ex^CS for unit spheres, used only for b guesses."""
    if not np.isfinite(field):
        raise ValueError('Activity field must be finite')
    lower,upper = 1e-12,.8
    for _ in range(100):
        eta = .5*(lower+upper)
        x = math.log(6*eta/np.pi)+kernel.excess_mu_cs(eta)
        if x<field:
            lower = eta
        else:
            upper = eta
    eta = .5*(lower+upper)
    rho = 6*eta/np.pi
    compressibility = (1+eta+eta**2-eta**3)/(1-eta)**3
    return rho,compressibility


def log_capped_ideal_partition(field,volume,cap):
    if (not np.isfinite(field) or not np.isfinite(volume) or volume<=0
            or not isinstance(cap,(int,np.integer)) or not 0<=cap<=100000):
        raise ValueError('Finite field/positive volume and integer cap at most 100000 required')
    values = np.array([n*(field+math.log(volume))-math.lgamma(n+1) for n in range(cap+1)])
    maximum = values.max()
    return float(maximum+math.log(np.exp(values-maximum).sum()))


def cavity_weight_guess(radius,field):
    """Finite fluid-model efficiency guess, never a phase-weight assignment.

    For R<=sigma/2, use the exact geometric form with the guessed CS density.
    Above it, a Hermite interpolation matches the model endpoint work and
    contact slopes; only the endpoint/target definitions, not these guesses,
    enter correctness of the subsequent calibrated or frozen sampler.
    """
    if not np.isfinite(radius) or not 0<=radius<=1:
        raise ValueError('Unit-sphere cavity radius must lie in [0,1]')
    rho,z = fluid_cs_density(field)
    eta = rho*np.pi/6
    if radius<=.5:
        return float(-math.log1p(-8*eta*radius**3))
    t = 2*radius-1
    left,right = -math.log1p(-eta),field-math.log(rho)
    # Slopes with respect to t (the physical radius interval has length .5).
    slope_left,slope_right = 3*eta/(1-eta),3*(z-1)
    return float((2*t**3-3*t**2+1)*left+(t**3-2*t**2+t)*slope_left
                 +(-2*t**3+3*t**2)*right+(t**3-t**2)*slope_right)


def build_activity_graph(volume,sites,cap,sigma=1.,base_field=14.8,displacement=.15,
                         cavity_radius_stages=1):
    """Unbiased hard/ideal activity chain with an optional physical radius loop."""
    if (not np.isfinite(volume) or volume<=0 or not isinstance(sites,(int,np.integer)) or not 4<=sites<=100000
            or not isinstance(cap,(int,np.integer)) or not 1<=cap<=100000 or sigma!=1.
            or not np.isfinite(base_field) or not -2<=base_field<=17
            or not np.isfinite(displacement) or displacement<=0
            or not isinstance(cavity_radius_stages,(int,np.integer))
            or not 1<=cavity_radius_stages<=256):
        raise ValueError('Positive finite volume/counts, sites/cap at most 100000, sigma=1 and valid field/step required')
    # Regular increments are exactly binary-representable. Preserve the user's
    # base field verbatim rather than silently rounding a physical activity.
    hard_fields = np.unique(np.r_[np.arange(-2.,8.001,1.),np.arange(8.5,12.001,.5),
                                 np.arange(12.25,17.001,.25),base_field])
    fields = np.r_[-2.,hard_fields,base_field]
    base = 1+int(np.flatnonzero(hard_fields==base_field)[0])
    cavity,nodes = len(fields)-1,len(fields)
    penalty = np.full(nodes,np.inf)
    penalty[0] = 0.
    radius = np.zeros(nodes)
    radius[cavity] = sigma
    edges = np.r_[np.column_stack((np.arange(len(hard_fields)),np.arange(1,len(hard_fields)+1))),
                  np.array([[base,cavity]])].astype(np.int64)
    weights = np.empty(nodes)
    weights[0] = -log_capped_ideal_partition(-2.,volume,cap)
    for index,field in enumerate(hard_fields,1):
        rho,z = fluid_cs_density(field)
        weights[index] = -rho*z*volume
    rho,_ = fluid_cs_density(base_field)
    weights[cavity] = weights[base]+base_field-math.log(rho)
    weights -= weights[0]
    target = {'volume':float(volume),'diameter':np.full(nodes,sigma),'radius':radius,
        'activity':fields,'pair_penalty':penalty,'umbrella_stiffness':np.zeros(nodes),
        'umbrella_preferred':np.zeros(nodes),'number_stiffness':np.zeros(nodes),
        'number_preferred':np.zeros(nodes),'displacement':np.full(nodes,displacement),
        'exchange_edges':edges,'base_node':base,'cavity_node':cavity,'ideal_node':0,
        'physical_nodes':np.arange(1,nodes,dtype=np.int64)}
    if cavity_radius_stages>1:
        # Append intermediate radii, keeping all original node identities and
        # the marked base<->sigma edge intact. Spacing refines near sigma.
        fractions = np.arange(1,cavity_radius_stages)/cavity_radius_stages
        intermediate_radii = sigma*(1-(1-fractions)**2)
        extras = len(intermediate_radii)
        for name in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                     'umbrella_preferred','number_stiffness','number_preferred','displacement'):
            values = (intermediate_radii if name=='radius' else np.full(extras,target[name][base]))
            target[name] = np.r_[target[name],values]
        path = np.r_[base,np.arange(nodes,nodes+extras),cavity].astype(np.int64)
        target['radius_path_nodes'] = path
        target['exchange_edges'] = np.r_[edges,np.column_stack((path[:-1],path[1:]))]
        target['physical_nodes'] = np.arange(1,nodes+extras,dtype=np.int64)
        weights = np.r_[weights,[weights[base]+cavity_weight_guess(r/sigma,base_field)
                                 for r in intermediate_radii]]
    return target,weights


def compact_weight_guess(field,volume,cap,penalty,stiffness,preferred,nu,nstar):
    """Finite soft-CS/number/Gaussian-Bragg efficiency model only.

    The modeled excluded-volume fraction is a_gamma=1-exp(-gamma), with
    a_infinity=1. Its finite N sum uses ideal combinatorial factors, CS
    excess free energy per particle (4 eta-3 eta^2)/(1-eta)^2, and the exact
    quadratic number cost. Terms with model eta>=.74 are omitted solely to
    regularize the surrogate; NO particle-number sector is removed from the
    sampler's actual target. An unrestricted Gaussian Bragg surrogate with
    mean .2 and variance .01 gives an analytic umbrella log factor. It is not
    an assertion about the physical Bragg distribution or crystal odds.
    """
    if (not np.isfinite([field,volume,stiffness,preferred,nu,nstar]).all()
            or volume<=0 or stiffness<0 or nu<0 or penalty<0 or np.isnan(penalty)
            or not isinstance(cap,(int,np.integer)) or not 1<=cap<=100000
            or not 0<=nstar<=cap):
        raise ValueError('Finite bounded scalar efficiency-model inputs required')
    if penalty==0 and stiffness==0 and nu==0:
        return -log_capped_ideal_partition(field,volume,cap)
    n = np.arange(cap+1,dtype=float)
    excluded = 1. if np.isposinf(penalty) else -math.expm1(-penalty)
    eta = np.pi*n*excluded/(6*volume)
    chosen = eta<COMPACT_WEIGHT_CONSTANTS['effective_packing_fraction_cutoff']
    n,eta = n[chosen],eta[chosen]
    excess = (4*eta-3*eta**2)/(1-eta)**2
    values = (n*(field+math.log(volume))
              -np.array([math.lgamma(number+1) for number in n])-n*excess-nu*(n-nstar)**2)
    maximum = float(values.max())
    log_partition = maximum+math.log(np.exp(values-maximum).sum())
    denominator = 1+2*stiffness*COMPACT_WEIGHT_CONSTANTS['Bragg_variance']
    log_order = (-.5*math.log(denominator)
                 -stiffness*(COMPACT_WEIGHT_CONSTANTS['Bragg_mean']-preferred)**2/denominator)
    return -log_partition-log_order


def build_compact_graph(volume,sites,cap,sigma=1.,base_field=14.8,displacement=.15,
                        cavity_radius_stages=1,ordered_hardening_number_strength=4.):
    """Self-contained two-order/number bridge sharing one unbiased endpoint.

    Fixed schedule v1: gamma=2 order gateway, low-order N*=V/1.06448,
    high-order N*=sites, both with removable strength-four number umbrellas.
    Optional 0<H<4 relaxes the high-route number strength after the ordered
    TARGET ramp at gamma=2, before hardening; it does not assert that any
    incoming host is ordered. H=4 retains the exact historical schedule.
    Site count is only an auxiliary target/reference diagnostic, never an
    occupancy constraint on any physical or soft configuration.
    """
    if (not np.isfinite(volume) or volume<=0 or not isinstance(sites,(int,np.integer))
            or not 4<=sites<=100000 or not isinstance(cap,(int,np.integer))
            or not sites<=cap<=100000 or sigma!=1. or not np.isfinite(base_field)
            or not -2<=base_field<=17 or not np.isfinite(displacement) or displacement<=0
            or not isinstance(cavity_radius_stages,(int,np.integer))
            or not 1<=cavity_radius_stages<=256):
        raise ValueError('Finite compact settings require FCC sites<=cap<=100000, sigma=1 and valid field/stages')
    hardening_strength = valid_hardening_number_strength(ordered_hardening_number_strength)
    cells = int(round((sites/4.)**(1./3.)))
    if sites!=4*cells**3:
        raise ValueError('Compact sites must equal 4*m^3 for an integer FCC cell count')
    if volume/1.06448>cap:
        raise ValueError('Auxiliary fluid number target must not exceed cap')
    ladder = list(kernel.make_compact_ladder(sites,volume/sites,base_field,cavity_stages=1,
                                            order_penalty=2.,low_order_return=True))
    # Preserve an explicitly requested volume even for non-power-of-two site
    # counts; all auxiliary number targets use that exact scalar as well.
    ladder[0] = float(volume)
    values = kernel.add_compact_number_bridge(tuple(ladder),strength=4.)
    _,diameter,radius,activity,base,penalty,kappa,preferred,cavity,edges,nu,nstar = values
    finish,gateway,ramp = base-1,20,21
    if (penalty[gateway]!=2 or kappa[gateway]!=120 or preferred[gateway]!=.2
            or penalty[finish]!=25 or kappa[finish]!=0 or radius[finish]!=0):
        raise ValueError('Imported compact helper schedule differs from fixed native construction v1')
    target = {'volume':float(volume),'diameter':diameter,'radius':radius,'activity':activity,
              'pair_penalty':penalty,'umbrella_stiffness':kappa,'umbrella_preferred':preferred,
              'number_stiffness':nu,'number_preferred':nstar,
              'displacement':np.full(len(diameter),displacement),'exchange_edges':edges,
              'base_node':int(base),'cavity_node':int(cavity[-1]),'ideal_node':0,
              'physical_nodes':np.asarray(cavity,dtype=np.int64)}
    # Existing high-order/hardening auxiliaries, not either physical endpoint.
    target['number_stiffness'][ramp:base] = 4.
    if hardening_strength<4.:
        target['number_stiffness'][39:base] = hardening_strength
    target['number_preferred'][ramp:base] = float(sites)
    entry = len(diameter)
    entry_strength = np.array([.25,.5,1.,2.,4.])
    removal_strength = np.array([hardening_strength/2,hardening_strength/4,
        hardening_strength/8,hardening_strength/16,hardening_strength/40,0.])
    templates = np.r_[np.full(len(entry_strength),gateway),np.full(len(removal_strength),finish)].astype(int)
    for name in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                 'umbrella_preferred','number_stiffness','number_preferred','displacement'):
        target[name] = np.r_[target[name],target[name][templates]]
    target['number_stiffness'][entry:] = np.r_[entry_strength,removal_strength]
    target['number_preferred'][entry:] = float(sites)
    keep = ~(((edges[:,0]==gateway)&(edges[:,1]==ramp))
             |((edges[:,0]==finish)&(edges[:,1]==base)))
    tail = entry+len(entry_strength)
    additions = ([(gateway,entry)]+[(j,j+1) for j in range(entry,tail-1)]
                 +[(tail-1,ramp),(finish,tail)]
                 +[(j,j+1) for j in range(tail,len(target['diameter'])-1)]
                 +[(len(target['diameter'])-1,base)])
    target['exchange_edges'] = np.r_[edges[keep],np.asarray(additions,dtype=np.int64)]
    if hardening_strength<4.:
        # Keep every original target identity. Two new fixed-gamma nodes
        # stage number fluctuations before the high-order hardening path.
        ordered_finish,hardening_begin = 38,39
        if (penalty[ordered_finish]!=2. or kappa[ordered_finish]!=120.
                or preferred[ordered_finish]!=.88):
            raise ValueError('Ordered target-ramp endpoint differs from fixed compact schedule')
        relax = len(target['diameter'])
        for name in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                     'umbrella_preferred','number_stiffness','number_preferred','displacement'):
            target[name] = np.r_[target[name],np.full(2,target[name][ordered_finish])]
        target['number_stiffness'][relax:] = [math.sqrt(4*hardening_strength),hardening_strength]
        keep = ~np.all(target['exchange_edges']==[ordered_finish,hardening_begin],axis=1)
        target['exchange_edges'] = np.r_[target['exchange_edges'][keep],
            np.array([[ordered_finish,relax],[relax,relax+1],[relax+1,hardening_begin]],dtype=np.int64)]
    nodes = len(target['diameter'])
    weights = np.array([compact_weight_guess(target['activity'][j],volume,cap,
        target['pair_penalty'][j],target['umbrella_stiffness'][j],target['umbrella_preferred'][j],
        target['number_stiffness'][j],target['number_preferred'][j]) for j in range(nodes)])
    weights[target['cavity_node']] = weights[base]+cavity_weight_guess(1.,base_field)
    weights -= weights[0]
    if cavity_radius_stages>1:
        fractions = np.arange(1,cavity_radius_stages)/cavity_radius_stages
        radii = sigma*(1-(1-fractions)**2)
        for name in ('diameter','radius','activity','pair_penalty','umbrella_stiffness',
                     'umbrella_preferred','number_stiffness','number_preferred','displacement'):
            target[name] = np.r_[target[name],radii if name=='radius' else np.full(len(radii),target[name][base])]
        path = np.r_[base,np.arange(nodes,nodes+len(radii)),target['cavity_node']].astype(np.int64)
        target['radius_path_nodes'] = path
        target['exchange_edges'] = np.r_[target['exchange_edges'],np.column_stack((path[:-1],path[1:]))]
        target['physical_nodes'] = np.r_[target['physical_nodes'],np.arange(nodes,nodes+len(radii))].astype(np.int64)
        weights = np.r_[weights,[weights[base]+cavity_weight_guess(r/sigma,base_field) for r in radii]]
    return target,weights


def build_native_graph(volume,sites,cap,sigma=1.,base_field=14.8,displacement=.15,
                       cavity_radius_stages=1,graph_kind='hard_activity',ordered_hardening_number_strength=4.):
    hardening_strength = valid_hardening_number_strength(ordered_hardening_number_strength)
    if graph_kind=='hard_activity':
        if hardening_strength!=4.:
            raise ValueError('Ordered hardening number strength applies only to compact graphs')
        return build_activity_graph(volume,sites,cap,sigma,base_field,displacement,cavity_radius_stages)
    if graph_kind=='compact':
        return build_compact_graph(volume,sites,cap,sigma,base_field,displacement,
                                   cavity_radius_stages,hardening_strength)
    raise ValueError('Native graph_kind must be hard_activity or compact')


def valid_hardening_number_strength(value):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating))
            or not np.isfinite(value) or not 0<value<=4.):
        raise ValueError('Ordered hardening number strength must be a finite real scalar in (0,4]')
    return float(value)


def load_native_graph(path):
    """Read only a provenance-complete graph, never interpret it as samples."""
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata.get('schema')!=SCHEMA or metadata.get('raw_sha256')!=kernel.file_sha256(path):
        raise ValueError('Native graph schema/raw digest is not verified')
    snapshots = metadata.get('source_snapshots',{})
    if not {'generator','kernel','graph_validation'}<=set(snapshots):
        raise ValueError('Native graph requires executed generator/kernel/validator snapshots')
    for name,(relative,digest) in snapshots.items():
        source = (ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or kernel.file_sha256(source)!=digest:
            raise ValueError('Native graph executed-source digest mismatch: '+name)
    inputs = metadata.get('input_sha256',{})
    explicit = metadata.get('construction_mode')=='explicit_box'
    kind = metadata.get('settings',{}).get('graph_kind','hard_activity')
    if kind not in ('hard_activity','compact') or (kind=='compact' and not explicit):
        raise ValueError('Native compact graphs require explicit-box construction')
    if kind=='compact' and (metadata.get('initial_weight_model')!=COMPACT_WEIGHT_MODEL
            or metadata.get('initial_weight_model_constants')!=COMPACT_WEIGHT_CONSTANTS):
        raise ValueError('Compact graph must bind its finite efficiency-model version/constants')
    if explicit and 'target_validation' not in snapshots:
        raise ValueError('Explicit-box graph requires the exact target-validation source')
    if not isinstance(inputs,dict) or (not inputs and not explicit):
        raise ValueError('Imported native graph requires parameter-origin input provenance')
    if explicit and inputs:
        raise ValueError('Explicit-box graph may not conceal imported parameter inputs')
    for name,digest in inputs.items():
        if kernel.file_sha256(ROOT/Path(name))!=digest:
            raise ValueError('Native graph parameter-origin digest mismatch')
    with np.load(path,allow_pickle=False) as archive:
        if str(archive['schema'])!=SCHEMA or 'samples' in archive:
            raise ValueError('Native graph must not contain fabricated trajectory samples')
        target = {key:archive[key] for key in ('volume','diameter','radius','activity','pair_penalty',
            'umbrella_stiffness','umbrella_preferred','number_stiffness','number_preferred','displacement',
            'exchange_edges','base_node','cavity_node','ideal_node','physical_nodes')}
        if 'radius_path_nodes' in archive:
            target['radius_path_nodes'] = archive['radius_path_nodes']
        declared_stages = scalar_integer(metadata.get('settings',{}).get('cavity_radius_stages',1),
                                         'cavity_radius_stages',1)
        if declared_stages>256:
            raise ValueError('Physical cavity loop exceeds the bounded stage count')
        if declared_stages>1 and 'radius_path_nodes' not in target:
            raise ValueError('Declared physical radius loop lacks its path nodes')
        if 'radius_path_nodes' in target and len(target['radius_path_nodes'])!=declared_stages+1:
            raise ValueError('Physical cavity path length differs from declared stages')
        sites,cap = scalar_integer(archive['sites'],'sites',4),scalar_integer(archive['cap'],'cap',1)
        weights = archive['initial_log_weights']
    target['volume'] = float(target['volume'])
    for name in ('base_node','cavity_node','ideal_node'):
        target[name] = scalar_integer(target[name],name,0)
    if (weights.shape!=target['activity'].shape or not np.isfinite(weights).all()
            or metadata['settings']['sites']!=sites or metadata['settings']['cap']!=cap):
        raise ValueError('Native graph initial weights/count metadata disagree')
    if explicit:
        settings = metadata['settings']
        required = ('volume','sites','cap','sigma','base_field','displacement','cavity_radius_stages')
        if not all(name in settings for name in required):
            raise ValueError('Explicit-box native graph must bind every scalar construction setting')
        expected,expected_weights = build_native_graph(settings['volume'],settings['sites'],settings['cap'],
            settings['sigma'],settings['base_field'],settings['displacement'],settings['cavity_radius_stages'],kind,
            settings.get('ordered_hardening_number_strength',4.))
        if target.keys()!=expected.keys() or any(not np.array_equal(target[name],expected[name]) for name in target):
            raise ValueError('Explicit-box target arrays differ from their recorded construction')
        if not np.array_equal(weights,expected_weights):
            raise ValueError('Explicit-box efficiency weights differ from their recorded construction')
    return target,{'initial_log_weights':weights},metadata


def scalar_integer(value,name,minimum):
    """Reject malformed archive scalars instead of lossy integer casting."""
    value = np.asarray(value)
    if (value.shape!=() or not np.isfinite(value) or value!=np.rint(value) or value<minimum):
        raise ValueError(name+' must be a finite integer scalar at least '+str(minimum))
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    construction = parser.add_mutually_exclusive_group(required=True)
    construction.add_argument('--parameters-from',type=Path,
                              help='Validated closed run supplies only volume/sites/cap/sigma')
    construction.add_argument('--volume',type=float,help='Explicit box volume; requires --sites and --cap')
    parser.add_argument('--sites',type=int)
    parser.add_argument('--cap',type=int)
    parser.add_argument('--sigma',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-field',type=float,default=14.8)
    parser.add_argument('--displacement',type=float,default=.15)
    parser.add_argument('--cavity-radius-stages',type=int,default=1,
                        help='Optional physical cavity loop, refining toward R=sigma (1 disables it)')
    parser.add_argument('--graph-kind',choices=('hard_activity','compact'),default='hard_activity',
                        help='Compact adds removable auxiliary soft/order/number routes, never physical bias')
    parser.add_argument('--ordered-hardening-number-strength',type=float,default=4.,
                        help='Compact-only auxiliary strength H in (0,4]; 4 preserves the original target schedule')
    args = parser.parse_args()
    paths = [args.output.with_suffix(suffix) for suffix in ('.npz','.json')]
    if any(path.exists() for path in paths):
        parser.error('Native graph output must be new')
    if args.parameters_from:
        if args.graph_kind=='compact':
            parser.error('Self-contained compact construction requires explicit --volume/--sites/--cap')
        if args.sites is not None or args.cap is not None or args.sigma!=1.:
            parser.error('Imported parameters cannot be overridden by explicit scalar options')
        parameter_path = args.parameters_from.resolve()
        metadata = json.loads(parameter_path.with_suffix('.json').read_text())
        data,metadata = validation.load_validated_run(parameter_path,metadata['settings']['sample_stride'])
        sites,cap = metadata['settings']['sites'],metadata['settings']['cap']
        volume,sigma = float(data['volume']),float(data['diameter'][int(data['endpoint'])])
        inputs = {validation.display_path(path):kernel.file_sha256(path)
                  for path in (parameter_path,parameter_path.with_suffix('.json'))}
        mode = 'validated_parameter_archive'
    else:
        if args.sites is None or args.cap is None:
            parser.error('Explicit --volume requires integer --sites and --cap')
        volume,sites,cap,sigma = args.volume,args.sites,args.cap,args.sigma
        inputs,mode = {},'explicit_box'
    try:
        target,weights = build_native_graph(volume,sites,cap,sigma,args.base_field,
            args.displacement,args.cavity_radius_stages,args.graph_kind,args.ordered_hardening_number_strength)
    except ValueError as error:
        parser.error(str(error))
    # Reuse strict target/graph validation without inventing execution records.
    try:
        from . import continuum_simulated_tempering as sampler
    except ImportError:
        import continuum_simulated_tempering as sampler
    try:
        sampler.validate_targets(target,cap)
    except ValueError as error:
        parser.error(str(error))
    sources = {name:sampler.capture_source(path,content) for name,path,content in (
        ('generator',__file__,EXECUTION_SOURCE_BYTES),
        ('kernel',kernel.__file__,kernel.EXECUTION_SOURCE_BYTES),
        ('graph_validation',validation.__file__,VALIDATION_SOURCE_BYTES),
        ('target_validation',sampler.__file__,sampler.EXECUTION_SOURCE_BYTES))}
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(paths[0],schema=SCHEMA,sites=sites,cap=cap,initial_log_weights=weights,**target)
    record = {'schema':SCHEMA,'raw_sha256':kernel.file_sha256(paths[0]),
        'construction_mode':mode,
        'settings':{'volume':volume,'sites':sites,'cap':cap,'sigma':sigma,
                    'base_field':args.base_field,'displacement':args.displacement,
                    'cavity_radius_stages':args.cavity_radius_stages,'graph_kind':args.graph_kind,
                    'ordered_hardening_number_strength':args.ordered_hardening_number_strength},
        'source_snapshots':sources,'input_sha256':inputs,
        'target_origin':('Native unbiased hard-activity chain with dilute ideal regeneration and optional physical cavity-radius loop; '
                         +('explicit scalar box, no imported trajectory' if mode=='explicit_box' else 'inherited box parameters only')),
        'initial_weight_origin':'Finite CS fluid-pressure/cavity contact-interpolation guesses and exact capped ideal partition; label efficiency only, not physical phase odds',
        'reproduce_argv':['python3','-m','scripts.continuum_tempering_graph',*sys.argv[1:]],
    }
    if args.graph_kind=='compact':
        record.update(initial_weight_model=COMPACT_WEIGHT_MODEL,
                      initial_weight_model_constants=COMPACT_WEIGHT_CONSTANTS,
                      target_origin='Native compact soft/order/number bridges with exact dense ideal regeneration and unbiased hard cavity endpoints; explicit box, no imported history',
                      initial_weight_origin='Capped soft-CS/number and Gaussian-Bragg surrogate plus fluid cavity interpolation; label-efficiency guesses only. Model eta cutoff does not truncate the actual target.')
    paths[1].write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'nodes':len(weights),'base_node':target['base_node'],
        'cavity_node':target['cavity_node'],'ideal_node':0,'graph':str(paths[0])}))


if __name__=='__main__':
    main()
