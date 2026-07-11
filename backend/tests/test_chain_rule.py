import json

import pytest

from app.engine.ann.chain_rule import trace_weight_gradient
from app.engine.ann.model import build_network
from app.schemas.network import LayerSpec, NetworkSpec


def _spec():
    return NetworkSpec(
        input_size=3,
        seed=42,
        layers=[
            LayerSpec(out_features=4, activation="relu"),
            LayerSpec(out_features=3, activation="tanh"),
            LayerSpec(out_features=2, activation="sigmoid"),
        ],
    )


X = [1.0, -2.0, 0.5]
TARGET = [0.2, 0.8]


@pytest.mark.parametrize("layer_index,row,col", [
    (0, 0, 0), (0, 2, 1), (1, 1, 3), (2, 0, 2), (2, 1, 1),
])
def test_chain_product_matches_analytic_gradient_for_weights(layer_index, row, col):
    net = build_network(_spec())
    result = trace_weight_gradient(net, X, TARGET, "mse", layer_index, row, col)
    assert result["matches_analytic"], result
    assert result["chain_product"] == pytest.approx(result["analytic_gradient"], abs=1e-3)


@pytest.mark.parametrize("layer_index,row", [(0, 0), (1, 2), (2, 1)])
def test_chain_product_matches_analytic_gradient_for_biases(layer_index, row):
    net = build_network(_spec())
    result = trace_weight_gradient(net, X, TARGET, "mse", layer_index, row, None)
    assert result["matches_analytic"]
    assert result["chain_product"] == pytest.approx(result["analytic_gradient"], abs=1e-3)
    assert result["param_kind"] == "bias"


def test_downstream_breakdown_present_for_non_final_layer_and_sums_correctly():
    net = build_network(_spec())
    result = trace_weight_gradient(net, X, TARGET, "mse", 0, 1, 2)
    breakdown = result["downstream_breakdown"]
    assert breakdown is not None
    assert len(breakdown["terms"]) == 3  # layer 1 has 3 output neurons

    dL_da = result["chain_steps"][0]["value"]
    assert breakdown["sum"] == pytest.approx(dL_da, abs=1e-3)


def test_no_downstream_breakdown_for_final_layer():
    net = build_network(_spec())
    result = trace_weight_gradient(net, X, TARGET, "mse", 2, 0, 1)
    assert result["downstream_breakdown"] is None


def test_result_is_json_serializable():
    net = build_network(_spec())
    result = trace_weight_gradient(net, X, TARGET, "mse", 1, 0, 0)
    json.dumps(result)


def test_dead_relu_neuron_gives_zero_chain_product():
    net = build_network(_spec())
    # Find a layer-0 neuron whose pre-activation is negative (dead for this input)
    from app.engine.ann.state_tree import build_forward_state_tree

    tree = build_forward_state_tree(net, X)
    dead_neurons = [i for i, z in enumerate(tree["layers"][0]["pre_activation"]) if z < 0]
    assert dead_neurons, "test fixture assumption: expected at least one dead ReLU neuron"

    j = dead_neurons[0]
    result = trace_weight_gradient(net, X, TARGET, "mse", 0, j, 0)
    assert result["chain_product"] == pytest.approx(0.0, abs=1e-9)
    assert result["analytic_gradient"] == pytest.approx(0.0, abs=1e-9)


def test_out_of_range_layer_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        trace_weight_gradient(net, X, TARGET, "mse", 5, 0, 0)


def test_out_of_range_row_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        trace_weight_gradient(net, X, TARGET, "mse", 0, 99, 0)


def test_out_of_range_col_raises():
    net = build_network(_spec())
    with pytest.raises(ValueError):
        trace_weight_gradient(net, X, TARGET, "mse", 0, 0, 99)


def test_different_loss_functions_still_match_analytic():
    net = build_network(_spec())
    for loss_name in ["mse", "bce"]:
        result = trace_weight_gradient(net, X, TARGET, loss_name, 2, 0, 0)
        assert result["matches_analytic"], (loss_name, result)
