import torch

from dynamode.minimiser import (
    calc_distances,
    calc_segment_segment_distances,
    get_topology,
    minimise_ca,
)


def test_ca_minimiser_reduces_simple_nonbonded_clash():
    ca = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    params = {
        "data": {"batch_size": 8},
        "top": {"min_sep": 2, "nb_centers_threshold": 10.0},
        "opt_ini": None,
        "opt": {
            "opt": "adam",
            "step_size": 0.01,
            "steps": 25,
            "nb_update_freq": 1,
            "bond_target_mode": "ideal",
            "energy_params": {
                "bond_const": 10.0,
                "angle_const": 0.0,
                "dihedral_const": 0.0,
                "nb_const": 1000.0,
                "nb_threshold": 3.5,
                "early_stopping_clash_score": None,
            },
        },
    }
    topology = get_topology(4, n_frames=1, device=ca.device)
    before = calc_distances(ca, topology["nb_centers"]["ids"]).min()

    out = minimise_ca(ca, params=params, protocol="custom", verbose=0)
    after = calc_distances(out, topology["nb_centers"]["ids"]).min()

    assert out.shape == ca.shape
    assert torch.isfinite(out).all()
    assert after > before


def test_ca_minimiser_preserves_shape_for_batched_trajectories():
    ca = torch.randn(2, 3, 5, 3)
    out = minimise_ca(
        ca,
        params={
            "data": {"batch_size": 2},
            "opt_ini": None,
            "opt": {
                "opt": "adam",
                "steps": 1,
                "step_size": 0.001,
                "energy_params": {"early_stopping_clash_score": None},
            },
        },
        protocol="custom",
        verbose=0,
    )
    assert out.shape == ca.shape
    assert out.dtype == ca.dtype


def test_ca_minimiser_masks_bad_initial_bonds_in_conservative_mode():
    ca = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [23.8, 0.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    params = {
        "data": {"batch_size": 8},
        "top": {"min_sep": 2, "nb_centers_threshold": 10.0},
        "opt_ini": None,
        "opt": {
            "opt": "adam",
            "step_size": 0.1,
            "steps": 3,
            "bond_target_mode": "initial_in_range_else_ignore",
            "bond_init_range": [3.57, 4.11],
            "energy_params": {
                "bond_const": 10000.0,
                "angle_const": 0.0,
                "dihedral_const": 0.0,
                "nb_const": 0.0,
                "segment_const": 0.0,
                "early_stopping_clash_score": None,
            },
        },
    }

    out = minimise_ca(ca, params=params, protocol="custom", verbose=0)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ca, atol=1e-5)


def test_ca_minimiser_reduces_segment_intersection_proxy():
    ca = torch.tensor(
        [[
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.1],
            [0.0, 1.0, 0.1],
        ]],
        dtype=torch.float32,
    )
    params = {
        "data": {"batch_size": 8},
        "top": {"min_sep": 2, "min_segment_sep": 2, "nb_centers_threshold": 10.0},
        "opt_ini": None,
        "opt": {
            "opt": "adam",
            "step_size": 0.02,
            "steps": 80,
            "nb_update_freq": 1,
            "bond_target_mode": "initial",
            "energy_params": {
                "bond_const": 10.0,
                "angle_const": 0.0,
                "dihedral_const": 0.0,
                "nb_const": 0.0,
                "segment_const": 1000.0,
                "segment_threshold": 1.0,
                "early_stopping_clash_score": None,
            },
        },
    }

    topology = get_topology(4, n_frames=1, min_segment_sep=2, device=ca.device)
    segment_ids = topology["segments"]["ids"]
    pair_ids = topology["segment_pairs"]["ids"]

    def min_segment_dist(x):
        first = segment_ids[pair_ids[:, 0]]
        second = segment_ids[pair_ids[:, 1]]
        return calc_segment_segment_distances(
            x[:, first[:, 0], :],
            x[:, first[:, 1], :],
            x[:, second[:, 0], :],
            x[:, second[:, 1], :],
        ).min()

    before = min_segment_dist(ca)
    out = minimise_ca(ca, params=params, protocol="custom", verbose=0)
    after = min_segment_dist(out)

    assert torch.isfinite(out).all()
    assert after > before


def test_ca_minimiser_keeps_partial_progress_when_caca_guard_trips():
    ca = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [3.8, 3.8, 0.0],
                [0.0, 3.8, 0.0],
                [0.2, 0.2, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [7.6, 0.0, 0.0],
                [11.4, 0.0, 0.0],
                [15.2, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    params = {
        "data": {"batch_size": 2},
        "top": {"min_sep": 2, "nb_centers_threshold": 10.0},
        "opt_ini": None,
        "opt": {
            "opt": "adam",
            "step_size": 0.2,
            "steps": 20,
            "nb_update_freq": 1,
            "bond_target_mode": "initial",
            "max_step_displacement": None,
            "min_caca_bond": 1.0,
            "max_caca_bond": 4.2,
            "energy_params": {
                "bond_const": 10.0,
                "angle_const": 0.0,
                "dihedral_const": 0.0,
                "nb_const": 1000.0,
                "nb_threshold": 3.5,
                "segment_const": 0.0,
                "early_stopping_clash_score": None,
            },
        },
    }
    topology = get_topology(5, n_frames=2, device=ca.device)
    before = calc_distances(ca, topology["nb_centers"]["ids"]).min()

    out = minimise_ca(ca, params=params, protocol="custom", verbose=0)
    after = calc_distances(out, topology["nb_centers"]["ids"]).min()
    bonds = torch.linalg.vector_norm(out[:, 1:] - out[:, :-1], dim=-1)

    assert torch.isfinite(out).all()
    assert after > before
    assert float(bonds.max()) <= 4.2 + 1e-5
