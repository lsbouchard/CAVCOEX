"""Independent finite-law and release-integrity checks for continuum diagnostics."""

import hashlib
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import continuum_analyze as analysis


def save_run(directory, name="run"):
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/f"{name}.npz"
    samples = np.zeros((40,2,9))
    samples[:,:,0] = np.tile(np.array([2.,3.,4.,5.]),10)[:,None]
    samples[:,:,1] = np.tile(np.array([.2,.2,.8,.8]),10)[:,None]
    samples[:,:,2] = np.array([0.,1.])
    samples[:,:,3] = 1.
    samples[:,0,8] = np.tile(np.array([1.,0.,-1.,-2.]),10)
    np.savez(path,samples=samples,endpoint=0,volume=20.,activity=np.array([2.,2.]),
             diameter=np.ones(2),radius=np.array([0.,1.]),swaps=np.array([[100,80]]))
    metadata = {"settings":{"sample_stride":10},"raw_sha256":analysis.sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path,samples


def save_graph_run(directory):
    path,linear = save_run(directory,"graph")
    samples = np.repeat(linear[:,:1,:],5,axis=1)
    samples[:,:,2] = np.tile(np.array([[0,1,2,3,4],[2,1,0,3,4]]),(20,1))
    samples[:,2,0] += 2
    samples[:,4,0] += 7
    samples[:,4,1] = .1
    samples[:,1,3] = np.tile([1.,1.,1.,0.],10)
    samples[:,4,3] = np.tile([1.,1.,0.,0.],10)
    edges = np.array([[0,1],[1,4],[4,2],[2,3],[3,0]])
    np.savez(path,samples=samples,endpoint=1,volume=20.,activity=np.full(5,2.),
             diameter=np.array([0.,1.,1.,1.,1.]),radius=np.array([0.,0.,1.,0.,.5]),
             cavity_nodes=np.array([1,4,2]),exchange_edges=edges,
             swaps=np.tile([100,80],(len(edges),1)))
    metadata = {"settings":{"sample_stride":10},"raw_sha256":analysis.sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path,samples


@pytest.mark.parametrize("block_sweeps",[0,5,15])
def test_block_lengths_must_respect_recorded_stride(tmp_path,block_sweeps):
    path,_ = save_run(tmp_path)
    with pytest.raises(ValueError,match="block|stride"):
        analysis.analyze(path,block_sweeps=block_sweeps,bootstrap=10)


def test_raw_metadata_digest_is_checked_before_analysis(tmp_path):
    path,_ = save_run(tmp_path)
    record = json.loads(path.with_suffix(".json").read_text())
    record["raw_sha256"] = "incorrect"
    path.with_suffix(".json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="digest"):
        analysis.analyze(path,block_sweeps=50,bootstrap=10)


def test_binary_deletion_basin_response_and_derivative_match_finite_law(tmp_path):
    _,samples = save_run(tmp_path)
    measured = analysis.estimates(samples,0,20.,2.)
    n = np.array([2.,3.,4.,5.])
    b = np.array([0.,0.,1.,1.])
    shift = np.array([1.,0.,-1.,-2.])
    np.testing.assert_allclose(measured["deletion_label_change_probability"],np.mean(abs(shift))/n.mean())
    for label,indicator,change in (("solid",b,shift),("fluid",1-b,-shift)):
        deletion_mass = n*indicator+change
        mean_b = np.dot(n,indicator)/indicator.sum()
        direct = np.dot(n-1,deletion_mass)/deletion_mass.sum()-mean_b
        np.testing.assert_allclose(measured[f"intraphase_response_Palm_{label}"],direct)
        def epsilon(dx):
            probability = np.exp(dx*n)
            probability /= probability.sum()
            q = np.dot(probability,indicator)
            return np.dot(probability,change)/(20*np.exp(2+dx)*q)
        step = 1e-5
        derivative = (epsilon(step)-epsilon(-step))/(2*step)
        np.testing.assert_allclose(measured[f"epsilon_derivative_{label}"],derivative,rtol=2e-9)


def test_unvisited_phase_is_explicitly_undefined(tmp_path):
    _,samples = save_run(tmp_path)
    samples[:,:,1] = .1
    samples[:,:,8] = 0
    result = analysis.estimates(samples,0,20.,2.)
    assert np.isnan(result["epsilon_solid"])
    assert np.isnan(result["epsilon_derivative_solid"])
    assert np.isnan(result["intraphase_response_Palm_solid"])
    assert result["deletion_label_change_probability"] == 0


@pytest.mark.parametrize("defect",["shape","negative","fractional","nonfinite","too_many","hard"])
def test_pair_overlap_observations_are_validated_before_analysis(tmp_path,defect):
    path,samples = save_run(tmp_path)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    overlaps = np.zeros(samples.shape[:2])
    if defect == 'shape':overlaps = overlaps[:-1]
    elif defect == 'negative':overlaps[0,0] = -1
    elif defect == 'fractional':overlaps[0,0] = .5
    elif defect == 'nonfinite':overlaps[0,0] = np.nan
    elif defect == 'too_many':overlaps[0,0] = 100
    else:overlaps[0,0] = 1
    fields['pair_overlap_counts'] = overlaps
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['raw_sha256'] = analysis.sha256(path)
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='[Pp]air-overlap'):
        analysis.load_validated_run(path,50)


def test_soft_overlap_profile_does_not_change_cavity_estimates(tmp_path):
    path,samples = save_graph_run(tmp_path)
    expected = analysis.analyze(path,50,20,57)[0]
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields['diameter'][0] = 1.
    fields['pair_penalty'] = np.array([2.,np.inf,np.inf,np.inf,np.inf])
    overlaps = np.zeros(samples.shape[:2],dtype=int)
    overlaps[:,0] = np.repeat([0,1,1,0],10)
    fields['pair_overlap_counts'] = overlaps
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['raw_sha256'] = analysis.sha256(path)
    path.with_suffix('.json').write_text(json.dumps(metadata))
    actual,_,diagnostics,_,_ = analysis.analyze(path,50,20,57)
    for name in expected:
        np.testing.assert_equal(actual[name],expected[name])
    profile = diagnostics['pair_overlap_profiles']['profiles'][0]
    assert profile['mean_overlap_count'] == .5
    assert profile['production_quarter_means'] == [0.,1.,1.,0.]
    assert profile['sampled_count_distribution'] == {'0':20,'1':20}
    assert not profile['hard_pair_interaction']


def replace_archive_fields(path, **updates):
    """Preserve the fixture's verified sidecar when changing its target fields."""
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields.update(updates)
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['raw_sha256'] = analysis.sha256(path)
    path.with_suffix('.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('defect',[
    'missing_preferred','missing_stiffness','stiffness_shape','preferred_shape',
    'negative_stiffness','negative_preferred','nonfinite_stiffness','nonfinite_preferred',
    'base_bias','interior_bias','endpoint_bias',
])
def test_number_umbrella_contract_rejects_invalid_or_physical_bias(tmp_path,defect):
    path,_ = save_graph_run(tmp_path)
    fields = {'number_stiffness':np.zeros(5),'number_preferred':np.zeros(5)}
    if defect == 'missing_preferred':fields.pop('number_preferred')
    elif defect == 'missing_stiffness':fields.pop('number_stiffness')
    elif defect == 'stiffness_shape':fields['number_stiffness'] = np.zeros(4)
    elif defect == 'preferred_shape':fields['number_preferred'] = np.zeros((5,1))
    elif defect == 'negative_stiffness':fields['number_stiffness'][0] = -1
    elif defect == 'negative_preferred':fields['number_preferred'][0] = -1
    elif defect == 'nonfinite_stiffness':fields['number_stiffness'][0] = np.inf
    elif defect == 'nonfinite_preferred':fields['number_preferred'][0] = np.nan
    else:fields['number_stiffness'][{'base_bias':1,'interior_bias':4,'endpoint_bias':2}[defect]] = 1
    replace_archive_fields(path,**fields)
    with pytest.raises(ValueError,match='[Nn]umber'):
        analysis.load_validated_run(path,50)


def test_declared_number_umbrella_cannot_silently_use_legacy_zero_default(tmp_path):
    path,_ = save_graph_run(tmp_path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    metadata['settings']['compact_number_stiffness'] = 4.
    path.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='target arrays are missing'):
        analysis.load_validated_run(path,50)
    replace_archive_fields(path,number_stiffness=np.zeros(5),number_preferred=np.zeros(5))
    with pytest.raises(ValueError,match='every recorded stiffness is zero'):
        analysis.load_validated_run(path,50)


def test_number_biased_auxiliaries_are_not_physical_bank_or_ideal_reservoir(tmp_path):
    path,_ = save_graph_run(tmp_path)
    original = analysis.analyze(path,50,20,57)
    replace_archive_fields(path,number_stiffness=np.array([4.,0.,0.,4.,0.]),
                           number_preferred=np.array([3.,0.,0.,3.,0.]))
    actual = analysis.analyze(path,50,20,57)
    for name in original[0]:
        np.testing.assert_equal(actual[0][name],original[0][name])
    assert actual[2]['ideal_reservoir_node'] is None
    bank = analysis.hard_phase_bank_diagnostics(actual[3],5)
    assert bank['nodes'] == [1]
    assert bank['including_cavity_stages']['nodes'] == [1,2,4]
    assert analysis.comparison_context(original[3],original[4])['physical_target'] == analysis.comparison_context(actual[3],actual[4])['physical_target']
    assert analysis.comparison_context(original[3],original[4])['auxiliary_graph_sha256'] != analysis.comparison_context(actual[3],actual[4])['auxiliary_graph_sha256']


def test_absent_number_umbrella_is_exactly_legacy_zero_target(tmp_path):
    path,_ = save_graph_run(tmp_path)
    original = analysis.analyze(path,50,20,57)
    replace_archive_fields(path,number_stiffness=np.zeros(5),number_preferred=np.zeros(5))
    actual = analysis.analyze(path,50,20,57)
    for name in original[0]:
        np.testing.assert_equal(actual[0][name],original[0][name])
    assert actual[1] == original[1]
    assert analysis.comparison_context(original[3],original[4])['auxiliary_graph_sha256'] == analysis.comparison_context(actual[3],actual[4])['auxiliary_graph_sha256']


def test_number_biased_physical_probe_is_rejected(tmp_path):
    path,_ = save_physical_probe_run(tmp_path)
    replace_archive_fields(path,number_stiffness=np.array([4.,0.,0.,0.,0.]),
                           number_preferred=np.full(5,3.))
    with pytest.raises(ValueError,match='unbiased radius-zero'):
        analysis.load_validated_run(path,50)


def test_round_trip_and_hysteresis_counts_are_complete_visits():
    ids = np.array([[0,1,2],[2,1,0],[0,1,2]])
    assert analysis.round_trips(ids,0,2) == (2,4,2)
    assert analysis.round_trips(ids,0,0) == (0,0,0)
    assert analysis.phase_transitions([.1,.5,.8,.6,.2]) == [1,1]


def test_inventory_reports_slow_quarter_drift_without_claiming_stationarity():
    counts = np.repeat([2,3,4,5],10)
    result = analysis.inventory_trace_summary(counts,5)
    assert result['production_quarter_means'] == [2.,3.,4.,5.]
    assert result['first_half_mean'] == 2.5
    assert result['second_half_mean'] == 4.5
    assert analysis.inventory_trace_summary(np.array([2]),1)['production_quarter_means'] == [2.,None,None,None]


def test_common_time_resampling_preserves_exact_endpoint_correlation(tmp_path):
    path,_ = save_run(tmp_path)
    values,errors,diagnostics,_,_ = analysis.analyze(path,block_sweeps=50,bootstrap=20,seed=57)
    assert values["response_direct"] == 0
    assert errors["response_direct"] == 0
    assert diagnostics["bootstrap_blocks"] == 8
    assert diagnostics["bootstrap_discarded_samples"] == 0
    assert diagnostics["bootstrap_seed"] == 57
    assert errors == analysis.analyze(path,50,20,57)[1]


def test_bootstrap_omits_auxiliary_copies_without_changing_common_block_estimates(tmp_path,monkeypatch):
    _,samples = save_graph_run(tmp_path)
    nodes = np.array([1,4,2])
    reference_estimates = analysis.estimates
    blocks,block_samples,draws,seed = 8,5,20,41
    full = samples.reshape(blocks,block_samples,*samples.shape[1:])
    rng = np.random.default_rng(seed)
    reference = {}
    for _ in range(draws):
        indices = rng.integers(0,blocks,blocks)
        values = reference_estimates(full[indices].reshape(samples.shape),1,20.,2.,cavity_nodes=nodes)
        for name,value in values.items():
            reference.setdefault(name,[]).append(value)
    expected = {name:float(np.std(values,ddof=1)) for name,values in reference.items()
                if np.isfinite(values).all()}
    observed_shapes = []
    def inspect_reduced(data,endpoint,*args,**kwargs):
        observed_shapes.append(data.shape)
        assert endpoint == 0
        np.testing.assert_array_equal(kwargs['cavity_nodes'],np.arange(len(nodes)))
        return reference_estimates(data,endpoint,*args,**kwargs)
    monkeypatch.setattr(analysis,'estimates',inspect_reduced)
    actual,count = analysis.block_bootstrap(samples,1,20.,2.,block_samples,draws,seed,cavity_nodes=nodes)
    assert count == blocks
    assert observed_shapes == [(len(samples),len(nodes),9)]*draws
    assert actual == expected


def test_custom_output_provenance_and_figures_are_local_to_output(tmp_path,monkeypatch,capsys):
    raw,_ = save_run(tmp_path/"inputs")
    original = {p:p.read_bytes() for p in (raw,raw.with_suffix(".json"))}
    output = tmp_path/"analysis"/"result.json"
    monkeypatch.setattr(analysis,"plot_audit",lambda data,metadata,path:path.write_bytes(b"figure fixture"))
    _,metadata_path = analysis.write_analysis([raw],output,50,20,seed=57)
    terminal_record = json.loads(capsys.readouterr().out)
    assert terminal_record["full_diagnostics"] == str(output)
    assert "mixing" not in terminal_record
    assert metadata_path == output.with_suffix(".metadata.json")
    figure = output.parent/"result_figures"/"run_audit.pdf"
    assert figure.is_file()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["settings"]["seed"] == 57
    assert metadata["script_sha256"] == analysis.EXECUTION_SOURCE_SHA256
    for name,digest in metadata["input_sha256"].items():
        assert digest == hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for name,digest in metadata["output_sha256"].items():
        assert digest == hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert all(path.read_bytes()==contents for path,contents in original.items())


def test_output_cannot_replace_execution_metadata(tmp_path):
    raw,_ = save_run(tmp_path)
    with pytest.raises(ValueError,match="overwrite raw"):
        analysis.write_analysis([raw],raw.with_suffix(".json"),50,20)


def test_duplicate_run_names_are_rejected_without_partial_output(tmp_path):
    first,_ = save_run(tmp_path/"first")
    second,_ = save_run(tmp_path/"second")
    output = tmp_path/"out"/"result.json"
    with pytest.raises(ValueError,match="duplicate stems"):
        analysis.write_analysis([first,second],output,50,20)
    assert not output.parent.exists()


def test_batch_preflight_fails_before_writing_any_artifact(tmp_path):
    first,_ = save_run(tmp_path/"inputs","first")
    second,_ = save_run(tmp_path/"inputs","second")
    second.with_suffix(".json").write_text('{"raw_sha256":"wrong"}')
    output = tmp_path/"out"/"result.json"
    with pytest.raises(ValueError,match="digest"):
        analysis.write_analysis([first,second],output,50,20)
    assert not output.parent.exists()


def test_exchange_plot_resolves_constant_diameter_soft_ladder(tmp_path,monkeypatch):
    raw,_ = save_run(tmp_path)
    data,metadata = analysis.load_validated_run(raw,50)
    original_close = analysis.plt.close
    monkeypatch.setattr(analysis.plt,"close",lambda figure:None)
    try:
        analysis.plot_audit(data,metadata,tmp_path/"diagnostics.pdf")
        figure = analysis.plt.gcf()
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(),np.arange(1,41)*10)
        exchange = figure.axes[3]
        np.testing.assert_array_equal(exchange.lines[0].get_xdata(),np.arange(len(data["swaps"])))
        assert exchange.get_xlabel() == "Neighbor replica-pair index"
        assert len(figure.legends) == 1
    finally:
        original_close("all")


def test_completed_continuum_raw_records_and_declared_snapshots_match():
    root = Path(__file__).resolve().parents[1]
    for path in (root/"data_raw").glob("continuum*.json"):
        record = json.loads(path.read_text())
        if "raw_sha256" not in record:
            continue
        assert analysis.sha256(path.with_suffix(".npz")) == record["raw_sha256"]
        # The earliest exploratory pilots predate source snapshots; do not
        # fabricate a matching historical source for those records.
        if "source_snapshot" in record:
            snapshot = root/record["source_snapshot"]
            assert analysis.sha256(snapshot) == record["source_sha256"]


def test_noncontiguous_graph_cavity_path_controls_all_inserted_statistics(tmp_path):
    path,samples = save_graph_run(tmp_path)
    values,errors,diagnostics,data,_ = analysis.analyze(path,50,20,57)
    assert data["exchange_edges"].shape == (5,2)
    assert values["mean_N_inserted"] == samples[:,2,0].mean()
    assert values["response_direct"] == 2
    assert errors["response_direct"] < 1e-14
    assert values["cavity_probability_stage"] == .75*.5
    assert diagnostics["cavity_nodes"] == [1,4,2]
    assert diagnostics["inserted_N_support"] == [4,5,6,7]
    assert diagnostics["ideal_to_inserted_round_trips"] == analysis.round_trips(samples[:,:,2],0,2)
    assert diagnostics["ideal_to_inserted_round_trips"][0] > 0
    assert analysis.round_trips(samples[:,:,2],0,4)[0] == 0


def test_graph_plot_follows_explicit_cavity_path(tmp_path,monkeypatch):
    path,samples = save_graph_run(tmp_path)
    data,metadata = analysis.load_validated_run(path,50)
    original_close = analysis.plt.close
    monkeypatch.setattr(analysis.plt,"close",lambda figure:None)
    try:
        analysis.plot_audit(data,metadata,tmp_path/"graph.pdf")
        figure = analysis.plt.gcf()
        np.testing.assert_array_equal(figure.axes[0].lines[1].get_ydata(),samples[:,2,0])
        np.testing.assert_allclose(figure.axes[5].lines[0].get_xdata(),[0.,.5])
        np.testing.assert_allclose(figure.axes[5].lines[0].get_ydata(),[.75,.5])
        assert len(figure.axes[3].lines) == 1  # No misleading contiguous-edge divider.
    finally:
        original_close("all")


@pytest.mark.parametrize("nodes", [[1,2,2],[0,4,2],[1,4,5],[1,2.5,4]])
def test_invalid_graph_cavity_paths_are_rejected(nodes):
    with pytest.raises(ValueError,match="cavity_nodes"):
        analysis.cavity_path(1,5,nodes)


@pytest.mark.parametrize("field,values", [
    ("pair_penalty",np.array([np.inf,np.inf,np.inf,np.inf,10.])),
    ("umbrella_stiffness",np.array([0.,0.,0.,0.,1.])),
])
def test_cavity_staging_rejects_nonphysical_graph_nodes(tmp_path,field,values):
    path,_ = save_graph_run(tmp_path)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields[field] = values
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="unbiased physical"):
        analysis.load_validated_run(path,50)


def save_inventory_run(directory, states):
    """Make marginals look mixed even when a phase-swap pair is inventory trapped."""
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/"inventory.npz"
    states = np.asarray(states)
    phases = np.column_stack((states//2,states%2))
    samples = np.zeros((len(states),2,9))
    samples[:,:,0] = 20+10*phases
    samples[:,:,1] = .2+.6*phases
    samples[:,:,2] = [0,1]
    samples[:,:,3] = 1
    samples[:,:,4] = 32-samples[:,:,0]
    counts = np.array([len(states),len(states)])
    accepts = np.zeros((2,8),dtype=int)
    accepts[0,6:8] = counts
    np.savez(path,samples=samples,endpoint=0,volume=40.,activity=np.array([2.,2.]),
             diameter=np.ones(2),radius=np.array([0.,1.]),swaps=np.array([[100,80]]),
             palm_move_counts=counts,accepts=accepts)
    path.with_suffix(".json").write_text(json.dumps({
        "settings":{"sample_stride":10},"raw_sha256":analysis.sha256(path)}))
    return path,samples


def test_phase_swaps_cannot_hide_a_trapped_joint_inventory(tmp_path):
    path,_ = save_inventory_run(tmp_path,np.tile([1,2],40))
    values,errors,diagnostics,_,_ = analysis.analyze(path,100,20,57)
    assert values["q_solid_base"] == values["q_solid_inserted"] == .5
    assert sum(diagnostics["physical_phase_transitions_low_to_high_high_to_low"]) == 79
    assert diagnostics["palm_transfer"]["acceptance_fraction"] == 1
    np.testing.assert_allclose([values[f"joint_phase_probability_{s}"] for s in analysis.PHASE_STATES],[0,.5,.5,0])
    np.testing.assert_allclose([values[f"joint_phase_product_{s}"] for s in analysis.PHASE_STATES],[.25]*4)
    assert values["joint_phase_correlation"] == -1
    assert values["endpoint_N_correlation"] == -1
    assert values["endpoint_N_covariance"] == -25
    assert values["endpoint_N_sum_variance"] == 0
    assert errors["endpoint_N_covariance"] == 0
    product = diagnostics["product_target"]
    assert product["both_endpoints_visit_both_basins"]
    assert product["strict_joint_inventory_coverage_failure"]
    assert product["missing_cells_with_positive_empirical_product"] == ["FF","SS"]
    assert product["insufficient_joint_state_visits"] == ["FF","SS"]
    assert not product["passes_all_reported_diagnostics"]
    assert product["endpoint_N_sum_support"] == [50]
    check = product["product_target_residual_checks"]["joint_phase_residual_FF"]
    assert check["zero_SE_with_nonzero_residual"] and check["exceeds_three_SE"]
    assert product["joint_phase_visits"]["FS"]["sampled_entries"] == 40
    assert product["joint_phase_visits"]["FS"]["complete_blocks_visited"] == 8


def test_balanced_product_inventory_has_zero_association_and_full_support(tmp_path):
    path,_ = save_inventory_run(tmp_path,np.tile([0,1,2,3],20))
    values,errors,diagnostics,_,_ = analysis.analyze(path,40,20,57)
    for state in analysis.PHASE_STATES:
        assert values[f"joint_phase_probability_{state}"] == .25
        assert values[f"joint_phase_residual_{state}"] == 0
        assert errors[f"joint_phase_residual_{state}"] == 0
    assert values["endpoint_N_covariance"] == 0
    assert values["endpoint_N_correlation"] == 0
    assert diagnostics["product_target"]["passes_all_reported_diagnostics"]
    assert diagnostics["product_target"]["endpoint_N_sum_support"] == [40,50,60]
    counts = diagnostics["product_target"]["occupancy_support_and_visits"]["unconstrained"]
    assert counts["solid"]["N"]["30"]["sample_count"] == 40
    assert counts["fluid"]["empty_reference_sites"]["12"]["sample_count"] == 40


def test_single_phase_is_not_misreported_as_product_support_failure(tmp_path):
    path,_ = save_inventory_run(tmp_path,np.zeros(80,dtype=int))
    values,_,diagnostics,_,_ = analysis.analyze(path,100,20,57)
    product = diagnostics["product_target"]
    assert values["joint_phase_probability_FF"] == 1
    assert not product["strict_joint_inventory_coverage_failure"]
    assert not product["both_endpoints_visit_both_basins"]
    assert not product["passes_all_reported_diagnostics"]


def test_deletion_phase_inventory_has_correct_palm_marginals(tmp_path):
    _,samples = save_run(tmp_path)
    values = analysis.estimates(samples,0,20.,2.)
    matrix = np.array([values[f"deletion_phase_probability_{s}"] for s in analysis.PHASE_STATES]).reshape(2,2)
    n = samples[:,0,0]
    b = samples[:,0,1]>.45
    np.testing.assert_allclose(matrix.sum(),1)
    np.testing.assert_allclose(matrix.sum(axis=1),[np.mean(n*~b)/n.mean(),np.mean(n*b)/n.mean()])
    np.testing.assert_allclose(matrix.sum(axis=0),[1-values["q_solid_Palm"],values["q_solid_Palm"]])
    np.testing.assert_allclose(matrix[0,1]+matrix[1,0],values["deletion_label_change_probability"])
    assert np.all(matrix >= 0)


def test_palm_counter_records_must_agree(tmp_path):
    path,_ = save_inventory_run(tmp_path,np.tile([1,2],40))
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields["accepts"][0,7] -= 1
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="Palm-transfer counters disagree"):
        analysis.load_validated_run(path,100)


def test_palm_target_requires_same_physical_activity(tmp_path):
    path,_ = save_inventory_run(tmp_path,np.tile([1,2],40))
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields["activity"][1] += .1
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="physical diameter/activity"):
        analysis.load_validated_run(path,100)


def test_phase_inventory_panel_exposes_product_target_mismatch(tmp_path,monkeypatch):
    path,_ = save_inventory_run(tmp_path,np.tile([1,2],40))
    data,metadata = analysis.load_validated_run(path,100)
    original_close = analysis.plt.close
    monkeypatch.setattr(analysis.plt,"close",lambda figure:None)
    try:
        analysis.plot_audit(data,metadata,tmp_path/"inventory.pdf")
        panel = analysis.plt.gcf().axes[2]
        np.testing.assert_allclose(panel.images[0].get_array(),[[-.25,.25],[.25,-.25]])
        assert "obs 0\nprod 0.25\nΔ -0.25" in [text.get_text() for text in panel.texts]
        assert panel.get_xlabel() == "Inserted endpoint"
        assert panel.get_ylabel() == "Unconstrained endpoint"
    finally:
        original_close("all")


@pytest.mark.parametrize("field,value,message",[
    ("volume",np.inf,"volume"),
    ("activity",np.array([np.inf,np.inf]),"finite"),
    ("palm_move_counts",np.array([1,2]),"attempt/accept"),
    ("accepts",np.zeros((2,6)),"replica-by-eight"),
])
def test_invalid_product_target_parameters_fail_before_analysis(tmp_path,field,value,message):
    path,_ = save_inventory_run(tmp_path,np.tile([1,2],40))
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields[field] = value
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match=message):
        analysis.load_validated_run(path,100)


def test_independent_marginal_law_checks_do_not_certify_a_product_control(tmp_path):
    path,_ = save_inventory_run(tmp_path,np.tile([0,1,2,3],20))
    _,_,diagnostics,_,_ = analysis.analyze(path,40,20,57)
    assert diagnostics["product_target"]["passes_all_reported_diagnostics"]
    checks = diagnostics["independent_identity_checks"]["checks"]
    assert not checks["direct_minus_Palm_N"]["exceeds_three_SE"]
    assert checks["direct_minus_Palm_q"]["exceeds_three_SE"]
    assert checks["stage_minus_GNZ_work"]["exceeds_three_SE"]
    assert "no overall equilibration certification" in diagnostics["independent_identity_checks"]["interpretation"]


def test_identity_warning_distinguishes_undefined_and_unknown_uncertainty():
    assert analysis.zero_residual_check(float("inf"),None)["status"] == "undefined estimate"
    assert analysis.zero_residual_check(2.,None)["status"] == "uncertainty unavailable"
    assert analysis.zero_residual_check(2.,0.)["zero_SE_with_nonzero_residual"]
    assert not analysis.zero_residual_check(1e-15,1e-17)["exceeds_three_SE"]


def test_exact_at_most_one_host_law_satisfies_all_three_marginal_controls():
    # At zV=1 a box admitting at most one hard host has P(N=0)=P(N=1)=1/2.
    # The particle-sized cavity is empty. The artificial basin b=1_{N=1}
    # changes on every deletion, so this also checks the finite-basin term.
    samples = np.zeros((40,2,9))
    samples[:,0,0] = np.tile([0.,1.],20)
    samples[:,0,1] = .2+.6*samples[:,0,0]
    samples[:,1,1] = .2
    samples[:,:,2] = [0,1]
    samples[:,0,3] = 1-samples[:,0,0]
    samples[:,0,8] = -samples[:,0,0]
    values = analysis.estimates(samples,0,1.,0.)
    errors,blocks = analysis.block_bootstrap(samples,0,1.,0.,2,20,57)
    assert blocks == 20
    for field in ("direct_minus_Palm_N","direct_minus_Palm_q","stage_minus_GNZ_work"):
        assert values[field] == 0
        assert errors[field] == 0
        assert analysis.zero_residual_check(values[field],errors[field])["status"] == "residual within three SE"
    assert values["deletion_label_change_probability"] == 1
    assert values["deletion_phase_probability_SF"] == 1


def save_physical_probe_run(directory):
    path,samples = save_graph_run(directory)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields["activity"][[0,3]] = [1.,3.]
    fields["diameter"][0] = 1.  # This variant replaces the ideal node by a hard probe.
    fields["probe_nodes"] = np.array([3,0])  # Archive order need not be activity order.
    fields["physical_probe_activity"] = np.array([3.,1.])
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path,samples


def test_physical_activity_transport_uses_lowest_probe_not_last_replica(tmp_path):
    path,samples = save_physical_probe_run(tmp_path)
    _,_,diagnostics,_,_ = analysis.analyze(path,50,20,57)
    probes = diagnostics["physical_activity_probes"]
    assert [row["node"] for row in probes["profiles"]] == [0,1,3]
    assert probes["lowest_activity_node"] == 0
    assert probes["highest_activity_node"] == 3
    assert probes["lowest_activity_is_distinct_from_base"]
    assert probes["lowest_activity_to_inserted_round_trips"] == analysis.round_trips(samples[:,:,2],0,2)
    assert probes["lowest_activity_to_inserted_round_trips"][0] > 0
    assert probes["lowest_to_highest_activity_round_trips"] == (0,0,0)


@pytest.mark.parametrize("field,value",[
    ("physical_probe_activity",np.array([3.,1.5])),
    ("umbrella_stiffness",np.array([1.,0.,0.,0.,0.])),
])
def test_physical_probe_diagnostics_reject_a_biased_or_mislabeled_target(tmp_path,field,value):
    path,_ = save_physical_probe_run(tmp_path)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields[field] = value
    np.savez(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="unbiased radius-zero"):
        analysis.load_validated_run(path,50)


def test_global_phase_bank_exposes_inventory_diluted_across_many_nodes():
    # One ordered configuration circulates over four unbiased hard nodes.
    # The cavity and biased auxiliary are deliberately ordered too; neither
    # belongs to the radius-zero physical inventory.
    samples = np.zeros((40,6,9))
    samples[:,:,0] = 25
    samples[:,:,1] = .2
    for row in range(40):
        samples[row,row%4,1] = .8
    samples[:,4:,1] = .8
    data = {"samples":samples,"activity":np.arange(6.),"diameter":np.ones(6),
            "radius":np.array([0.,0.,0.,0.,1.,0.]),
            "pair_penalty":np.full(6,np.inf),
            "umbrella_stiffness":np.array([0.,0.,0.,0.,0.,10.])}
    result = analysis.hard_phase_bank_diagnostics(data,4)
    assert result["nodes"] == [0,1,2,3]
    for key in ("operational_solid_inventory","ordered_high_density_inventory"):
        assert result[key]["support"] == [1]
        assert result[key]["sampled_count_changes"] == 0
        assert result[key]["sampled_count_trace"] == [1]*40
    assert result["host_number_inventory"]["support"] == [100]
    assert result["including_cavity_stages"]["nodes"] == [0,1,2,3,4]
    assert result["including_cavity_stages"]["operational_solid_inventory"]["support"] == [2]
    samples[-1,0,1] = .8  # A genuine second ordered member is created.
    changed = analysis.hard_phase_bank_diagnostics(data,4)
    assert changed["operational_solid_inventory"]["support"] == [1,2]
    assert changed["operational_solid_inventory"]["sampled_count_changes"] == 1


def test_alternative_phase_threshold_cannot_reuse_fixed_delete_one_labels(tmp_path):
    path,samples = save_run(tmp_path)
    with pytest.raises(ValueError,match="recomputing delete-one labels"):
        analysis.estimates(samples,0,20.,2.,threshold=.5)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["deletion_label_threshold"] = .5
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match="threshold is incompatible"):
        analysis.load_validated_run(path,50)


def save_thinned_probe_run(directory, indices=None, coordinate_shape=None):
    path,samples = save_physical_probe_run(directory)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    indices = np.arange(0,40,10) if indices is None else np.asarray(indices)
    fields["probe_sample_indices"] = indices
    shape = (len(indices),2,8,3) if coordinate_shape is None else coordinate_shape
    fields["physical_probe_configurations"] = np.zeros(shape)
    np.savez_compressed(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path,samples


def test_thinned_probe_alignment_uses_headers_without_loading_coordinates(tmp_path,monkeypatch):
    path,samples = save_thinned_probe_run(tmp_path)
    original_getitem = np.lib.npyio.NpzFile.__getitem__
    def guarded_getitem(archive,key):
        assert key != "physical_probe_configurations", "Full probe coordinates must not be loaded"
        return original_getitem(archive,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,"__getitem__",guarded_getitem)
    values,_,diagnostics,data,_ = analysis.analyze(path,50,20,57)
    assert data["probe_configuration_shape"] == (4,2,8,3)
    np.testing.assert_array_equal(data["probe_sample_indices"],[0,10,20,30])
    assert values["mean_N"] == samples[:,1,0].mean()
    assert diagnostics["physical_activity_probes"]["coordinate_sample_indices"] == [0,10,20,30]
    assert diagnostics["physical_activity_probes"]["coordinate_shape"] == [4,2,8,3]


@pytest.mark.parametrize("indices,shape,message",[
    ([0,10,10,30],None,"distinct increasing"),
    ([0,20,10,30],None,"distinct increasing"),
    ([0,10,20,40],None,"distinct increasing"),
    ([0,10,20,30],(40,2,8,3),"coordinate axes"),
    ([0,10,20,30],(4,1,8,3),"coordinate axes"),
])
def test_invalid_thinned_coordinate_mapping_fails_early(tmp_path,indices,shape,message):
    path,_ = save_thinned_probe_run(tmp_path,indices,shape)
    with pytest.raises(ValueError,match=message):
        analysis.load_validated_run(path,50)


def test_historical_full_probe_coordinates_default_to_all_diagnostic_frames(tmp_path):
    path,_ = save_thinned_probe_run(tmp_path,np.arange(40))
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files if name != "probe_sample_indices"}
    np.savez_compressed(path,**fields)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    data,_ = analysis.load_validated_run(path,50)
    np.testing.assert_array_equal(data["probe_sample_indices"],np.arange(40))
    assert data["probe_configuration_shape"] == (40,2,8,3)


def test_two_radius_zero_event_work_is_unresolved_and_no_ideal_node_is_invented(tmp_path):
    path,_ = save_run(tmp_path)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields["samples"][:,0,3] = 0
    np.savez(path,**fields)
    record = json.loads(path.with_suffix(".json").read_text())
    record["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(record))
    values,errors,diagnostics,_,_ = analysis.analyze(path,50,30,57)
    control = diagnostics["support_probability_control"]
    assert control["method"] == "direct rare-empty-event sampling"
    assert not control["has_intermediate_cavity_stages"]
    assert not control["work_estimate_resolved"]
    assert not control["work_difference_uncertainty_available"]
    assert control["stages"][0]["sampled_successes"] == 0
    assert control["stages"][0]["sampled_trials"] == 40
    assert "zero sampled support" in control["status"]
    assert values["cavity_probability_stage"] == 0
    assert "stage_minus_GNZ_work" not in errors
    check = diagnostics["independent_identity_checks"]["checks"]["stage_minus_GNZ_work"]
    assert check["status"] == "undefined estimate"
    assert not check["exceeds_three_SE"]
    assert diagnostics["ideal_reservoir_node"] is None
    assert diagnostics["ideal_to_inserted_round_trips"] is None
    assert analysis.json_ready(values)["work_stage"] is None
    json.dumps(analysis.json_ready(diagnostics),allow_nan=False)


def test_multistage_support_records_resolved_factors_and_bootstrap_zero_uncertainty(tmp_path):
    path,_ = save_graph_run(tmp_path)
    values,errors,diagnostics,data,_ = analysis.analyze(path,50,30,57)
    control = diagnostics["support_probability_control"]
    assert control["method"] == "multistage support product"
    assert control["has_intermediate_cavity_stages"]
    assert control["work_estimate_resolved"]
    assert control["work_difference_uncertainty_available"]
    assert [stage["sampled_successes"] for stage in control["stages"]] == [30,20]
    assert diagnostics["ideal_reservoir_node"] == 0
    # Finite empirical probability need not have a finite resampled work SE.
    control = analysis.support_probability_diagnostics(data,values,{},5)
    assert control["work_estimate_resolved"]
    assert not control["work_difference_uncertainty_available"]
    assert "uncertainty unresolved" in control["status"]
    data["samples"][:,4,3] = 0
    zero_values = analysis.estimates(data["samples"],1,20.,2.,cavity_nodes=data["cavity_nodes"])
    assert not analysis.support_probability_diagnostics(data,zero_values,{},5)["work_estimate_resolved"]


def test_boundary_basin_bootstrap_zero_is_not_physical_precision(tmp_path):
    _,samples = save_run(tmp_path)
    samples[:,:,1] = .8
    samples[:,:,8] = 0
    values = analysis.estimates(samples,0,20.,2.)
    errors,_ = analysis.block_bootstrap(samples,0,20.,2.,5,30,57)
    diagnostics = analysis.boundary_fraction_diagnostics(values,errors)
    for record in diagnostics["fractions"].values():
        assert record["estimate"] == 1
        assert record["common_block_SE"] == 0
        assert record["no_sampled_phase_variation"]
        assert record["zero_bootstrap_SE_not_equilibrium_precision"]


@pytest.mark.parametrize("value",[-1.,.5,2.])
def test_next_stage_support_requires_binary_measurements(tmp_path,value):
    path,_ = save_run(tmp_path)
    with np.load(path) as archive:
        fields = {name:archive[name] for name in archive.files}
    fields["samples"][0,0,3] = value
    np.savez(path,**fields)
    record = json.loads(path.with_suffix(".json").read_text())
    record["raw_sha256"] = analysis.sha256(path)
    path.with_suffix(".json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="binary event"):
        analysis.load_validated_run(path,50)


def comparison_record(tmp_path):
    path,_ = save_run(tmp_path)
    values,errors,_,data,metadata = analysis.analyze(path,50,20,57)
    metadata["settings"].update(cap=100,sites=32,seed=123,initialization="fluid")
    return {"estimates":values,"block_bootstrap_SE":errors,
            "comparison_context":analysis.comparison_context(data,metadata)}


def test_independent_start_difference_is_reported_without_pooling(tmp_path):
    first = comparison_record(tmp_path)
    second = copy.deepcopy(first)
    second["comparison_context"]["settings"].update(seed=456,initialization="crystal")
    second["comparison_context"]["raw_sha256"] = "different raw archive"
    first["estimates"]["q_solid_base"] = 0.
    second["estimates"]["q_solid_base"] = 1.
    first["block_bootstrap_SE"]["q_solid_base"] = 0.
    second["block_bootstrap_SE"]["q_solid_base"] = 0.
    second["estimates"]["mean_N"] += 3
    first["block_bootstrap_SE"]["mean_N"] = .3
    second["block_bootstrap_SE"]["mean_N"] = .4
    before = copy.deepcopy((first,second))
    result = analysis.compare_initializations("fluid",first,"crystal",second)
    assert result["comparison_eligible"]
    assert result["independent_fresh_random_histories_recorded"]
    assert result["different_initializations"]
    assert result["boundary_only_uncertainty_present"]
    assert result["any_difference_exceeds_three_nominal_SE"]
    check = result["observable_difference_checks"]["mean_N"]
    assert check["residual"] == -3
    assert check["common_block_SE"] == .5
    assert check["standardized_residual"] == -6
    assert result["observable_difference_checks"]["q_solid_base"]["boundary_only_uncertainty_present"]
    assert (first,second) == before


@pytest.mark.parametrize('scales',[np.array([.1]),np.array([.1,-.2]),np.array([.1,np.nan])])
def test_recorded_displacement_scales_are_validated(tmp_path,scales):
    path,_ = save_run(tmp_path)
    replace_archive_fields(path,displacement=scales)
    with pytest.raises(ValueError,match='displacement scales'):
        analysis.load_validated_run(path,50)


def test_full_recorded_displacement_arrays_expose_probe_tuning_difference(tmp_path):
    path,_ = save_physical_probe_run(tmp_path)
    replace_archive_fields(path,displacement=np.full(5,.12))
    data,metadata = analysis.load_validated_run(path,50)
    metadata['settings'].update(cap=100,sites=32,seed=123,initialization='fluid')
    first = comparison_record(tmp_path/'base')
    first['comparison_context'] = analysis.comparison_context(data,metadata)
    second = copy.deepcopy(first)
    second['comparison_context']['displacement_proposal_scales'][3] = .20
    result = analysis.compare_initializations('fluid',first,'crystal',second)
    proposal = result['displacement_proposal_comparison']
    assert result['same_physical_target'] and result['comparison_eligible']
    assert not result['different_auxiliary_graphs']
    assert not proposal['all_replica_scales_equal']
    assert proposal['different_scale_nodes'] == [3]
    assert result['different_settings'] == {}
    second['comparison_context']['displacement_proposal_scales'][3] = .12
    assert analysis.compare_initializations('a',first,'b',second)['displacement_proposal_comparison']['all_replica_scales_equal']


def test_missing_or_unmapped_displacement_arrays_do_not_certify_equal_tuning(tmp_path):
    first = comparison_record(tmp_path)
    second = copy.deepcopy(first)
    result = analysis.compare_initializations('a',first,'b',second)
    assert result['displacement_proposal_comparison']['all_replica_scales_equal'] is None
    first['comparison_context']['displacement_proposal_scales'] = [.1,.2]
    second['comparison_context']['displacement_proposal_scales'] = [.1,.2]
    second['comparison_context']['auxiliary_graph_sha256'] = 'different targets/order'
    result = analysis.compare_initializations('a',first,'b',second)
    assert result['displacement_proposal_comparison']['all_replica_scales_equal'] is None


@pytest.mark.parametrize("target",["x","V","sigma","R"])
def test_different_physical_targets_are_not_compared(tmp_path,target):
    first = comparison_record(tmp_path)
    second = copy.deepcopy(first)
    second["comparison_context"]["physical_target"][target] += .01
    result = analysis.compare_initializations("a",first,"b",second)
    assert not result["comparison_eligible"]
    assert result["target_mismatch_fields"] == [target]
    assert result["observable_difference_checks"] == {}


def test_capacity_proof_required_but_auxiliary_graph_and_proposals_may_differ(tmp_path):
    first = comparison_record(tmp_path)
    second = copy.deepcopy(first)
    second["comparison_context"]["auxiliary_graph_sha256"] = "different auxiliary graph"
    second["comparison_context"]["settings"]["birth_proposal"] = "vacancy"
    result = analysis.compare_initializations("a",first,"b",second)
    assert result["comparison_eligible"]
    assert result["different_auxiliary_graphs"]
    assert result["different_settings"]["birth_proposal"] == [None,"vacancy"]
    assert not result["independent_fresh_random_histories_recorded"]
    assert all(check["common_block_SE"] is None for check in result["observable_difference_checks"].values())
    second["comparison_context"]["capacity_proven_nontruncating"] = False
    result = analysis.compare_initializations("a",first,"b",second)
    assert not result["comparison_eligible"]
    assert result["observable_difference_checks"] == {}


def test_geometric_capacity_certificate_is_not_inferred_from_absence_of_cap_visits(tmp_path):
    path,_ = save_run(tmp_path)
    data,metadata = analysis.load_validated_run(path,50)
    assert not analysis.comparison_context(data,metadata)["capacity_proven_nontruncating"]
    required = int(np.ceil(6*float(data["volume"])/np.pi))
    metadata["settings"]["cap"] = required-1
    assert not analysis.comparison_context(data,metadata)["capacity_proven_nontruncating"]
    metadata["settings"]["cap"] = required
    assert analysis.comparison_context(data,metadata)["capacity_proven_nontruncating"]


def test_phase_label_mismatch_excludes_only_basin_fraction_comparisons(tmp_path):
    first = comparison_record(tmp_path)
    second = copy.deepcopy(first)
    second["comparison_context"]["phase_label"]["reference_sites"] = 108
    result = analysis.compare_initializations("a",first,"b",second)
    assert result["same_physical_target"]
    assert result["comparison_eligible"]
    assert not result["same_recorded_phase_label"]
    assert set(result["observable_difference_checks"]) == {"mean_N","Fano_factor","mean_N_inserted"}


def test_one_support_event_can_leave_bootstrap_work_uncertainty_unresolved(tmp_path):
    path,_ = save_run(tmp_path)
    data,_ = analysis.load_validated_run(path,50)
    data["samples"][:,0,3] = 0
    data["samples"][0,0,3] = 1
    values = analysis.estimates(data["samples"],0,20.,2.)
    errors,_ = analysis.block_bootstrap(data["samples"],0,20.,2.,5,100,57)
    assert values["cavity_probability_stage"] == 1/40
    assert np.isfinite(values["work_stage"])
    assert "work_stage" not in errors
    assert "stage_minus_GNZ_work" not in errors
    control = analysis.support_probability_diagnostics(data,values,errors,5)
    assert control["work_estimate_resolved"]
    assert not control["work_difference_uncertainty_available"]
    assert "uncertainty unresolved" in control["status"]


def test_rare_missing_joint_vacancies_qualify_nominal_covariance_exceedance():
    # Only ~1.2 coincident vacancies are expected from these marginals. An
    # empirical bootstrap cannot create the missing joint tail; a large
    # covariance/SE ratio must not be interpreted as a calibrated rejection.
    pairs = np.repeat([[30,31],[31,31],[32,30],[32,31]], [1,62,39,1898], axis=0)
    samples = np.zeros((2000,2,9))
    samples[:,:,0] = pairs
    samples[:,:,1] = .9
    samples[:,:,3] = 1
    values = analysis.estimates(samples,0,30.,14.8)
    assert np.isclose(values["endpoint_N_covariance"],-.000624)
    errors = {name:1. for name in values}
    errors["endpoint_N_covariance"] = .000115
    diagnostics = analysis.product_target_diagnostics(samples,0,np.array([0,1]),values,errors,100)
    support = diagnostics["joint_number_support"]
    cell = support["coarse_joint_cells"]["below_mode/below_mode"]
    assert cell["sample_count"] == 0
    assert np.isclose(cell["empirical_product_expected_sample_count"],1.2285)
    assert support["sparse_joint_tail_warning"]
    check = diagnostics["product_target_residual_checks"]["endpoint_N_covariance"]
    assert check["exceeds_three_SE"]  # Retain the numerical diagnostic.
    assert check["sparse_joint_support_warning"]
    assert "prevents a calibrated rejection" in check["inference_scope"]


def test_joint_number_support_detects_populated_tails_and_constant_histories():
    pairs = np.repeat([[i,j] for i in (31,32,33) for j in (30,31,32)],
                      [1,2,1,2,4,2,1,2,1],axis=0)
    pairs = np.tile(pairs,(25,1))
    support = analysis.joint_number_support_diagnostics(pairs[:,0],pairs[:,1],16)
    assert support["modal_particle_numbers"] == [32,31]
    assert not support["sparse_joint_tail_warning"]
    assert support["number_variation_sampled_at_both_endpoints"]
    constant = analysis.joint_number_support_diagnostics(np.ones(400),np.ones(400),16)
    assert constant["sparse_joint_tail_warning"]
    assert not constant["number_variation_sampled_at_both_endpoints"]
