"""Number-target staging, default transparency, and low-level setup guards."""

import numpy as np
import pytest
import hashlib
import json
from types import SimpleNamespace

from scripts.continuum_replica_exchange import (
    add_compact_number_bridge, fcc_reference, make_compact_ladder, run_chunk, seed_rng,
)
import scripts.continuum_replica_exchange as sampler


def tiny_fixture():
    positions = np.zeros((3, 5, 3))
    positions[:, 0] = [.4, .7, 1.1]
    return dict(positions=positions, counts=np.ones(3, dtype=np.int64),
                centers=np.zeros((3, 3)), diameter=np.ones(3),
                radius=np.array([0., 1., 0.]), activity=np.full(3, -2.2),
                identities=np.arange(3), length=3., sweeps=400,
                endpoint=0, sample_stride=5,
                reference=fcc_reference(4, 3.), displacement=np.full(3, .8),
                pair_penalty=np.array([np.inf, np.inf, .7]),
                umbrella_stiffness=np.array([0., 0., 2.5]),
                umbrella_preferred=np.array([0., 0., .4]),
                exchange_edges=np.array([[0, 1], [0, 2], [1, 2]]),
                cavity_nodes=np.array([0, 1]), snapshot_nodes=np.arange(3),
                palm_transfer=True, biased_birth=True, birth_proposal=1,
                birth_scope=1, birth_width=.3)


def test_explicit_zero_number_bias_preserves_every_state_and_random_draw():
    first, second = tiny_fixture(), tiny_fixture()
    seed_rng(2026092928)
    expected = run_chunk(**first)
    seed_rng(2026092928)
    actual = run_chunk(**second, number_stiffness=np.zeros(3),
                       number_preferred=np.array([0., 2., 5.]))
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    for key in ("positions", "counts", "centers", "identities"):
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize("strength,preferred", [
    ([-1.,0.,0.],[0.,0.,0.]), ([0.,np.nan,0.],[0.,0.,0.]),
    ([0.,np.inf,0.],[0.,0.,0.]), ([0.,0.],[0.,0.,0.]),
    ([0.,0.,0.],[0.,0.]), ([0.,0.,0.],[0.,np.nan,0.]),
    ([0.,0.,0.],[0.,-1.,0.]), ([0.,0.,0.],[0.,6.,0.]),
])
def test_number_bias_setup_validated_before_moves(strength, preferred):
    with pytest.raises(ValueError, match="Number-umbrella"):
        run_chunk.py_func(**tiny_fixture(), number_stiffness=np.array(strength),
                         number_preferred=np.array(preferred))


@pytest.mark.parametrize("node", [0,1])
def test_marked_transfer_rejects_number_biased_physical_endpoint(node):
    strength = np.zeros(3)
    strength[node] = .1
    with pytest.raises(ValueError, match="Physical cavity path requires zero"):
        run_chunk.py_func(**tiny_fixture(), number_stiffness=strength,
                         number_preferred=np.zeros(3))


def test_intermediate_cavity_number_bias_is_rejected_even_without_marked_moves():
    settings = tiny_fixture()
    settings.update(palm_transfer=False,cavity_nodes=np.array([0,2,1]))
    with pytest.raises(ValueError, match="Physical cavity path requires zero"):
        run_chunk.py_func(**settings,number_stiffness=np.array([0.,0.,.1]),
                         number_preferred=np.zeros(3))


def test_number_bridge_has_explicit_soft_on_and_hard_off_paths():
    old = make_compact_ladder(chemical_potential=14.8, order_penalty=2.,
                              low_order_return=True)
    new = add_compact_number_bridge(old,4.)
    volume,diameter,radius,activity,base,penalty,kappa,preferred,cavity,edges,nu,nstar = new
    assert len(diameter) == 134 and base == 70 and tuple(cavity) == (70,71)
    for i in (1,2,3,5,6,7):
        np.testing.assert_array_equal(new[i][:120],old[i])
    assert np.all(nu[:89] == 0) and np.all(nu[89:120] == 4)
    np.testing.assert_array_equal(nu[120:124],[.5,1.,2.,4.])
    np.testing.assert_array_equal(nu[124:],[3.,2.,1.5,1.,.7,.5,.3,.2,.1,0.])
    assert np.all(nstar[89:] == volume/1.06448)
    assert np.all(nu[cavity] == 0) and np.all(kappa[cavity] == 0)
    assert np.all(np.isposinf(penalty[cavity])) and np.all(activity[cavity] == 14.8)
    assert np.all(penalty[120:124] == 2) and np.all(kappa[120:124] == 120)
    assert np.all(penalty[124:] == 25) and np.all(kappa[124:] == 0)
    assert (20,120) in set(map(tuple,edges)) and (123,89) in set(map(tuple,edges))
    assert (119,124) in set(map(tuple,edges)) and (133,70) in set(map(tuple,edges))
    assert (20,89) not in set(map(tuple,edges)) and (119,70) not in set(map(tuple,edges))
    adjacency = {j:[] for j in range(len(diameter))}
    for left,right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    frontier,seen = [0],set()
    while frontier:
        node = frontier.pop()
        if node not in seen:
            seen.add(node)
            frontier.extend(adjacency[node])
    assert seen == set(range(len(diameter)))


@pytest.mark.parametrize("strength", [0.,-1.,np.inf,np.nan])
def test_number_bridge_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="Number-umbrella strength"):
        add_compact_number_bridge(make_compact_ladder(low_order_return=True),strength)


def test_number_bridge_cannot_silently_replace_the_old_unbiased_return():
    with pytest.raises(ValueError, match="compact low-order return"):
        add_compact_number_bridge(make_compact_ladder(),4.)


def test_state_checkpoint_preserves_number_targets_and_its_own_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(sampler,"ROOT",tmp_path)
    (tmp_path/"data_raw/source_snapshots").mkdir(parents=True)
    state = tiny_fixture()
    prefix = tmp_path/"test_number_state"
    nu,target = np.array([0.,0.,4.]),np.array([0.,0.,2.9])
    sampler.write_state_checkpoint(prefix,SimpleNamespace(compact_number_stiffness=4.),
        400,state["positions"],state["counts"],state["centers"],state["diameter"],
        state["radius"],state["activity"],state["pair_penalty"],state["umbrella_stiffness"],
        state["umbrella_preferred"],0,27.,state["displacement"],state["cavity_nodes"],
        state["exchange_edges"],nu,target)
    path = prefix.with_suffix(".checkpoint.npz")
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["number_stiffness"],nu)
        np.testing.assert_array_equal(saved["number_preferred"],target)
        assert int(saved["completed_sweeps"]) == 400
    metadata = json.loads(prefix.with_suffix(".checkpoint.json").read_text())
    assert metadata["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["settings"]["compact_number_stiffness"] == 4
    snapshot = tmp_path/metadata["source_snapshot"]
    assert metadata["source_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert not prefix.with_suffix(".checkpoint.npz.tmp").exists()
