import json

import pytest
import torch

from app.engine.ann.model import build_network
from app.engine.ann.state_tree import build_backward_state_tree
from app.engine.training.weight_loss_surface import compute_weight_loss_surface
from app.schemas.network import LayerSpec, NetworkSpec


def _spec():
    return NetworkSpec(
        input_size=3,
        seed=42,
        layers=[
            LayerSpec(out_features=4, activation="relu"),
            LayerSpec(out_features=2, activation="sigmoid"),
        ],
    )


X = [1.0, -2.0, 0.5]
TARGET = [0.2, 0.8]
COORD1 = {"layer_index": 0, "row": 0, "col": 0}
COORD2 = {"layer_index": 1, "row": 1, "col": 2}


def test_grid_shape_matches_resolution():
    net = build_network(_spec())
    surface = compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-2, 2), (-2, 2), resolution=11)
    assert len(surface["x"]) == 11
    assert len(surface["y"]) == 11
    assert len(surface["z"]) == 11
    assert all(len(row) == 11 for row in surface["z"])


def test_original_weight_values_are_restored_after_computation():
    net = build_network(_spec())
    before_w1 = net.linears[0].weight[0, 0].item()
    before_w2 = net.linears[1].weight[1, 2].item()

    compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-5, 5), (-5, 5), resolution=9)

    assert net.linears[0].weight[0, 0].item() == pytest.approx(before_w1)
    assert net.linears[1].weight[1, 2].item() == pytest.approx(before_w2)


def test_weights_are_restored_even_if_loss_computation_fails_midway():
    net = build_network(_spec())
    before_w1 = net.linears[0].weight[0, 0].item()
    before_w2 = net.linears[1].weight[1, 2].item()

    with pytest.raises(ValueError):
        compute_weight_loss_surface(net, X, TARGET, "not_a_real_loss", COORD1, COORD2, (-2, 2), (-2, 2), resolution=5)

    assert net.linears[0].weight[0, 0].item() == pytest.approx(before_w1)
    assert net.linears[1].weight[1, 2].item() == pytest.approx(before_w2)


def test_grid_values_match_independent_backward_state_tree_computation():
    """Cross-check against a completely separate code path: manually set the
    two weights to a specific grid point, then compute loss via
    build_backward_state_tree (a different module) and confirm it matches
    the value the surface grid recorded for that same point.
    """
    net = build_network(_spec())
    surface = compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-1, 1), (-1, 1), resolution=5)

    probe_net = build_network(_spec())
    probe_net.load_state_dict(net.state_dict())
    with torch.no_grad():
        probe_net.linears[0].weight[0, 0] = surface["x"][2]
        probe_net.linears[1].weight[1, 2] = surface["y"][3]

    tree = build_backward_state_tree(probe_net, X, TARGET, "mse")
    assert tree["loss"] == pytest.approx(surface["z"][3][2], abs=1e-4)


def test_original_point_matches_unperturbed_loss():
    net = build_network(_spec())
    surface = compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-2, 2), (-2, 2), resolution=7)

    original_w1 = surface["original_point"][0]
    original_w2 = surface["original_point"][1]
    assert original_w1 == pytest.approx(net.linears[0].weight[0, 0].item())
    assert original_w2 == pytest.approx(net.linears[1].weight[1, 2].item())


def test_result_is_json_serializable():
    net = build_network(_spec())
    surface = compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-1, 1), (-1, 1), resolution=6)
    json.dumps(surface)


def test_bias_coordinate_supported():
    net = build_network(_spec())
    bias_coord = {"layer_index": 0, "row": 2, "col": None}
    surface = compute_weight_loss_surface(net, X, TARGET, "mse", bias_coord, COORD2, (-2, 2), (-2, 2), resolution=5)
    assert len(surface["z"]) == 5


@pytest.mark.parametrize("bad_coord", [
    {"layer_index": 99, "row": 0, "col": 0},
    {"layer_index": 0, "row": 99, "col": 0},
    {"layer_index": 0, "row": 0, "col": 99},
])
def test_invalid_coordinate_raises(bad_coord):
    net = build_network(_spec())
    with pytest.raises(ValueError):
        compute_weight_loss_surface(net, X, TARGET, "mse", bad_coord, COORD2, (-1, 1), (-1, 1), resolution=5)


def test_resolution_below_two_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-1, 1), (-1, 1), resolution=1)


def test_inverted_range_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (2, -2), (-1, 1), resolution=5)


def test_repeated_calls_are_idempotent():
    net = build_network(_spec())
    surface1 = compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-1, 1), (-1, 1), resolution=5)
    surface2 = compute_weight_loss_surface(net, X, TARGET, "mse", COORD1, COORD2, (-1, 1), (-1, 1), resolution=5)
    assert surface1["z"] == surface2["z"]
